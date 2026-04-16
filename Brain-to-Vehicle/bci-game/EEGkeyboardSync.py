import socket
import struct
import time
import threading
import csv
import keyboard
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

# Configuration
EEG_HOST = "127.0.0.1"
EEG_PORT = 8712
CSV_FILENAME = f"eeg_keyboard_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CHANNEL_NAMES = [
    "T7", "T8", "TP7", "TP8", "P7", "P5", "P3", "Pz", "P4", "P6", 
    "P8", "PO7", "PO5", "PO3", "Pz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"
    # You can add more channels here
]

# Mapping keyboard keys to trigger codes
KEY_MAPPING = {
    'up': 1,
    'down': 2,
    'left': 3,
    'right': 4,
    'space': 5  # stop
}

# Shared trigger value
current_trigger = 0
trigger_lock = threading.Lock()

@dataclass
class EEGDataPoint:
    timestamp: float
    channels: List[float]  # Variable length for EEG channels
    trigger: int           # current keyboard trigger value

class EEGReceiver:
    def __init__(self, host=EEG_HOST, port=EEG_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.num_channels = len(CHANNEL_NAMES)  # Use the length of CHANNEL_NAMES
        self.bytes_per_sample = (self.num_channels + 1) * 4  # 4 bytes per float, +1 for trigger

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to EEG server at {self.host}:{self.port}")

    def start_receiving(self):
        self.running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def _receive_loop(self):
        buffer = b''

        # Open CSV file and write header
        with open(CSV_FILENAME, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + CHANNEL_NAMES + ['trigger'])

            while self.running:
                try:
                    data = self.socket.recv(4096)
                    if not data:
                        print("EEG connection closed.")
                        break

                    buffer += data

                    while len(buffer) >= self.bytes_per_sample:
                        sample_bytes = buffer[:self.bytes_per_sample]
                        buffer = buffer[self.bytes_per_sample:]

                        # Unpack EEG data dynamically based on the number of channels
                        values = struct.unpack(f'<{self.num_channels + 1}f', sample_bytes)  # +1 for trigger
                        channels = list(values[:self.num_channels])  # The first N values are EEG channels
                        # Ignore original trigger from device, use keyboard trigger instead
                        timestamp = time.time()

                        with trigger_lock:
                            trigger = current_trigger

                        # Write row to CSV
                        writer.writerow([timestamp] + channels + [trigger])
                except Exception as e:
                    print(f"Error receiving EEG data: {e}")
                    break

        self.running = False

def keyboard_listener():
    global current_trigger
    print("Listening for keys: up/down/left/right/space (stop). Press 'esc' to quit.")
    active_keys = set()

    while True:
        event = keyboard.read_event()

        # If a key is pressed down
        if event.event_type == keyboard.KEY_DOWN:
            key = event.name
            if key in KEY_MAPPING:
                active_keys.add(key)
                with trigger_lock:
                    current_trigger = KEY_MAPPING[key]
                print(f"Key pressed: {key} → Trigger: {current_trigger}")
            elif key == 'esc':
                print("Exiting keyboard listener.")
                break

        # If a key is released
        elif event.event_type == keyboard.KEY_UP:
            key = event.name
            if key in active_keys:
                active_keys.remove(key)
                if len(active_keys) == 0:
                    with trigger_lock:
                        current_trigger = 0
                    print("No keys pressed. Trigger set to 0.")

def main():
    receiver = EEGReceiver()

    try:
        receiver.connect()

        # Start keyboard thread
        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()

        # Start receiving EEG data
        receiver.start_receiving()

        # Wait for keyboard thread to finish (esc pressed)
        kb_thread.join()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        receiver.running = False
        print(f"EEG recording stopped. Data saved to: {CSV_FILENAME}")

if __name__ == "__main__":
    main()
