# core/hardware.py

import socket
import config

class CarController:
    """
    处理通过 UDP 协议与小车的通信。
    特性：无连接、低延迟。
    """

    def __init__(self, ip=config.CAR_IP, port=config.CAR_PORT, simulation=config.IS_SIMULATION):
        self.target_ip = ip
        self.target_port = port
        self.simulation = simulation
        self.sock = None

        # 定义从命令字符串到字节的映射
        # 这里为了兼容性保留了之前的字节码，你也可以根据小车端代码修改为字符串
        self.command_map = {
            'stop': b'\x00',
            'forward': b'\x01',
            'backward': b'\x02',
            'left': b'\x03',
            'right': b'\x04',
        }

    def connect(self):
        """
        UDP 是无连接协议，这里主要用于初始化 Socket 对象。
        """
        if self.simulation:
            print(f"[小车控制器-模拟] 虚拟 UDP 连接已就绪 -> {self.target_ip}:{self.target_port}")
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.1)
            print(f"[小车控制器] UDP Socket 已创建。目标: {self.target_ip}:{self.target_port}")
            return True
        except Exception as e:
            print(f"[小车控制器] Socket 创建失败: {e}")
            return False

    def disconnect(self):
        """关闭 Socket。"""
        if self.simulation:
            print("[小车控制器-模拟] 虚拟连接已关闭。")
            return

        if self.sock:
            self.sock.close()
            self.sock = None
            print(f"[小车控制器] UDP Socket 已关闭。")

    def send_command(self, command: str):
        """向小车发送一个 UDP 数据包。"""
        if command not in self.command_map:
            # print(f"[小车控制器] 未知命令: {command}")
            return

        byte_to_send = self.command_map[command]

        if self.simulation:
            print(f"[小车控制器-模拟] UDP 发送 -> {self.target_ip}:{self.target_port} | 数据: {byte_to_send.hex()} ({command})")
            return

        if self.sock:
            try:
                # sendto 不需要建立连接，直接指定地址发送
                self.sock.sendto(byte_to_send, (self.target_ip, self.target_port))
            except Exception as e:
                print(f"[小车控制器] 发送错误: {e}")
        else:
            print("[小车控制器] 无法发送，Socket 未初始化。")