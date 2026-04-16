#!/usr/bin/env python3
"""
21通道Montage EEG数据接收器（修正版 - 每帧21通道+1触发信号）
特性：
- 正确解析数据格式: [Ch1, Ch2, ..., Ch21, Trigger]
- 每帧共22个float（88字节）
- 保证Trigger与当前通道对齐，无错位
- 自动限制缓冲区大小（默认10000样本）
- CSV保存格式: timestamp + 21通道 + trigger
"""

import socket
import struct
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

EEG_HOST = "127.0.0.1"
EEG_PORT = 8712

@dataclass
class EEGDataPoint:
    timestamp: float
    channels: List[float]
    trigger: float
    raw_bytes: bytes

class Ssvep21ChannelReceiver:
    def __init__(self, host=EEG_HOST, port=EEG_PORT, max_buffer_size: int = 10000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

        self.num_channels = 21
        self.bytes_per_float = 4
        self.bytes_per_sample = (self.num_channels + 1) * self.bytes_per_float  # 88字节/样本

        # 通道名称
        self.channel_names = [
            "T7", "T8", "TP7", "TP8", "P7", "P5", "P3", "Pz", "P4", "P6",
            "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"
        ]

        self.data_buffer: List[EEGDataPoint] = []
        self.latest_data: Optional[EEGDataPoint] = None
        self.lock = threading.Lock()
        self.total_samples = 0
        self.start_time = None
        self.sample_rate = 0
        self.receive_thread = None
        self.running = False
        self.max_buffer_size = max_buffer_size

        print(f"初始化21通道Montage接收器")
        print(f"预期数据格式: 21通道 + 1触发 = {self.bytes_per_sample}字节/样本")
        print(f"通道名称: {self.channel_names}")

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ 成功连接到EEG服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ 连接EEG服务器失败: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2)
        if self.socket:
            self.socket.close()
            self.connected = False
        print("连接已断开")

    def start_receiving(self):
        if not self.connected:
            print("未连接到服务器")
            return False
        self.running = True
        self.start_time = time.time()
        self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
        self.receive_thread.start()
        print("开始接收21通道Montage EEG数据...")
        return True

    def _receive_data(self):
        buffer = b''
        while self.running and self.connected:
            try:
                data = self.socket.recv(4096)
                if not data:
                    print("连接已关闭")
                    break
                buffer += data

                # 每次读取完整一帧 (21通道 + 1 trigger)
                while len(buffer) >= self.bytes_per_sample:
                    sample_bytes = buffer[:self.bytes_per_sample]
                    buffer = buffer[self.bytes_per_sample:]

                    try:
                        values = struct.unpack('<22f', sample_bytes)
                    except struct.error:
                        print("解析错误: 样本长度不匹配")
                        continue

                    channels = list(values[:self.num_channels])
                    trigger = values[self.num_channels]

                    sample = EEGDataPoint(
                        timestamp=time.time(),
                        channels=channels,
                        trigger=trigger,
                        raw_bytes=sample_bytes
                    )

                    with self.lock:
                        self.latest_data = sample
                        self.data_buffer.append(sample)
                        self.total_samples += 1

                        # 限制缓冲区大小
                        if len(self.data_buffer) > self.max_buffer_size:
                            self.data_buffer = self.data_buffer[-self.max_buffer_size:]

                        # 每100样本输出采样率
                        if self.total_samples % 100 == 0:
                            elapsed = time.time() - self.start_time
                            self.sample_rate = self.total_samples / elapsed
                            print(f"已接收 {self.total_samples} 个样本, 采样率: {self.sample_rate:.1f} Hz")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"接收数据时出错: {e}")
                break

        self.running = False

    def get_latest_data(self) -> Optional[EEGDataPoint]:
        with self.lock:
            return self.latest_data

    def get_data_buffer(self, max_samples: int = 1000) -> List[EEGDataPoint]:
        with self.lock:
            if max_samples:
                return self.data_buffer[-max_samples:]
            return self.data_buffer.copy()

    def get_statistics(self) -> Dict:
        with self.lock:
            if not self.data_buffer:
                return {}

            channel_stats = {}
            for i, name in enumerate(self.channel_names):
                data = [s.channels[i] for s in self.data_buffer]
                channel_stats[name] = {
                    'min': min(data),
                    'max': max(data),
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'samples': len(data)
                }

            trigger_data = [s.trigger for s in self.data_buffer]
            trigger_stats = {
                'min': min(trigger_data),
                'max': max(trigger_data),
                'mean': np.mean(trigger_data),
                'std': np.std(trigger_data),
                'samples': len(trigger_data)
            }

            return {
                'total_samples': self.total_samples,
                'sample_rate': self.sample_rate,
                'channels': channel_stats,
                'trigger': trigger_stats,
                'start_time': self.start_time,
                'current_time': time.time()
            }

    def save_data_to_csv(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"montage_21channel_{timestamp}.csv"

        with self.lock:
            if not self.data_buffer:
                print("没有数据可保存")
                return

            csv_data = [[s.timestamp] + s.channels + [s.trigger] for s in self.data_buffer]
            columns = ['timestamp'] + self.channel_names + ['trigger']
            df = pd.DataFrame(csv_data, columns=columns)
            df.to_csv(filename, index=False)
            print(f"数据已保存到: {filename}, 样本数: {len(csv_data)}")

def main():
    print("=== 21通道Montage EEG数据接收器 (修正版) ===")
    receiver = Ssvep21ChannelReceiver()
    try:
        if not receiver.connect():
            return
        if not receiver.start_receiving():
            return
        print("\n按 Ctrl+C 停止接收...\n")
        while receiver.running:
            time.sleep(2)
            stats = receiver.get_statistics()
            if stats:
                print(f"总样本数: {stats['total_samples']}, 采样率: {stats['sample_rate']:.1f} Hz")
                latest = receiver.get_latest_data()
                if latest:
                    print(f"最新 Trigger={latest.trigger:.3f}, 第一通道={latest.channels[0]:.3f}")
    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
    finally:
        if receiver.data_buffer:
            receiver.save_data_to_csv()
        receiver.disconnect()
        print("测试完成")

if __name__ == "__main__":
    main()