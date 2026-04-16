import socket
import struct
import time
import random
import threading
from datetime import datetime

# 服务器配置
MOCK_EEG_HOST = "127.0.0.1"  # 监听所有可用接口
MOCK_EEG_PORT = 8712  # 端口与Ssvep10ChannelReceiver中一致

# EEG数据配置 (与Ssvep10ChannelReceiver保持一致)
NUM_CHANNELS = 21
SAMPLE_RATE = 250  # 模拟的采样率，例如250Hz


class MockEEGServer:
    """
    模拟EEG数据服务器，向客户端发送10通道EEG数据和1个触发信号。
    数据格式: Channel-1, ..., Channel-10, Trigger (每个4字节浮点数)
    """

    def __init__(self, host=MOCK_EEG_HOST, port=MOCK_EEG_PORT, sample_rate=SAMPLE_RATE):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.server_socket = None
        self.client_sockets = []
        self.running = False
        self.listener_thread = None
        self.data_sender_thread = None
        self.lock = threading.Lock()  # 用于保护client_sockets

        print(f"初始化模拟EEG服务器在 {self.host}:{self.port}")
        print(f"模拟采样率: {self.sample_rate} Hz")
        print(f"通道数: {NUM_CHANNELS}")

    def start(self):
        """启动服务器并监听客户端连接"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1)  # 设置监听超时，方便检查running状态

        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_for_clients, daemon=True)
        self.listener_thread.start()

        self.data_sender_thread = threading.Thread(target=self._send_mock_data, daemon=True)
        self.data_sender_thread.start()

        print(f"模拟EEG服务器正在监听 {self.host}:{self.port}...")

    def _listen_for_clients(self):
        """在单独线程中监听新客户端连接"""
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                with self.lock:
                    self.client_sockets.append(conn)
                print(f"新客户端连接: {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"监听客户端时出错: {e}")
                break

    def _generate_eeg_sample(self) -> bytes:
        """生成一个模拟EEG数据样本 (10通道 + 1触发)"""
        channels_data = [random.uniform(-100, 100) for _ in range(NUM_CHANNELS)]  # 模拟-100到100uV
        # 模拟触发信号，例如每隔一段时间发送一个非零触发
        trigger = 0.0
        if random.random() < 0.01:  # 1%的概率发送一个触发
            trigger = float(random.choice([1, 2, 3, 4]))  # 模拟不同的事件触发

        # 打包成字节流
        # '<' 表示小端字节序，'f' 表示浮点数
        fmt = f"<{NUM_CHANNELS + 1}f"
        packed_data = struct.pack(fmt, *channels_data, trigger)
        return packed_data

    def _send_mock_data(self):
        """在单独线程中向所有连接的客户端发送模拟EEG数据"""
        sleep_interval = 1.0 / self.sample_rate  # 根据采样率计算发送间隔

        while self.running:
            if not self.client_sockets:
                time.sleep(0.1)  # 没有客户端连接时稍作等待
                continue

            sample_bytes = self._generate_eeg_sample()

            clients_to_remove = []
            with self.lock:
                for client_socket in self.client_sockets:
                    try:
                        client_socket.sendall(sample_bytes)
                    except Exception as e:
                        print(f"发送数据到客户端时出错，客户端可能已断开: {e}")
                        clients_to_remove.append(client_socket)

                # 移除已断开的客户端
                for client_socket in clients_to_remove:
                    self.client_sockets.remove(client_socket)
                    try:
                        client_socket.close()
                    except:
                        pass  # 忽略关闭错误

            time.sleep(sleep_interval)  # 控制发送速率

    def stop(self):
        """停止服务器"""
        print("正在停止模拟EEG服务器...")
        self.running = False
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2)
        if self.data_sender_thread and self.data_sender_thread.is_alive():
            self.data_sender_thread.join(timeout=2)

        if self.server_socket:
            self.server_socket.close()

        with self.lock:
            for client_socket in self.client_sockets:
                try:
                    client_socket.close()
                except:
                    pass
            self.client_sockets.clear()
        print("模拟EEG服务器已停止。")


def main():
    server = MockEEGServer()
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()