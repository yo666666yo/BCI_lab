#!/usr/bin/env python3
"""
10通道Montage EEG数据接收器
专门用于接收SSVEP-10 montage的EEG数据
数据格式: Channel-1, Channel-2, ..., Channel-10, Trigger (每个4字节浮点数)
"""

import socket
import struct
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# EEG服务器配置
EEG_HOST = "127.0.0.1"
EEG_PORT = 8712

@dataclass
class EEGDataPoint:
    """单个EEG数据点"""
    timestamp: float
    channels: List[float]  # 10个通道的数据
    trigger: float         # 触发信号
    raw_bytes: bytes      # 原始字节数据

class Ssvep10ChannelReceiver:
    """10通道Montage EEG数据接收器"""
    
    def __init__(self, host=EEG_HOST, port=EEG_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # 数据配置
        self.num_channels = 10
        self.bytes_per_float = 4
        self.bytes_per_sample = (self.num_channels + 1) * self.bytes_per_float  # 10通道 + 1触发 = 44字节
        
        # 通道名称 (根据montage配置)
        self.channel_names = [
            "P4", "PO3", "PO4", "T5", "T6","PO5", "PO6", "Oz", "O1", "O2"
        ]
        
        # 数据缓冲区
        self.data_buffer: List[EEGDataPoint] = []
        self.latest_data: Optional[EEGDataPoint] = None
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_samples = 0
        self.start_time = None
        self.sample_rate = 0
        
        # 数据接收线程
        self.receive_thread = None
        self.running = False
        
        print(f"初始化10通道Montage接收器")
        print(f"预期数据格式: {self.num_channels}通道 + 1触发 = {self.bytes_per_sample}字节/样本")
        print(f"通道名称: {self.channel_names}")
    
    def connect(self) -> bool:
        """连接到EEG数据服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5秒超时
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ 成功连接到EEG服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ 连接EEG服务器失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2)
        
        if self.socket:
            self.socket.close()
            self.connected = False
        print("连接已断开")
    
    def start_receiving(self):
        """开始接收数据"""
        if not self.connected:
            print("未连接到服务器")
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # 启动数据接收线程
        self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
        self.receive_thread.start()
        
        print("开始接收10通道Montage EEG数据...")
        return True
    
    def _receive_data(self):
        """在后台线程中接收数据"""
        buffer = b''  # 数据缓冲区
        
        while self.running and self.connected:
            try:
                # 接收数据
                data = self.socket.recv(4096)
                if not data:
                    print("连接已关闭")
                    break
                
                # 将新数据添加到缓冲区
                buffer += data
                
                # 处理完整的样本
                while len(buffer) >= self.bytes_per_sample:
                    # 提取一个完整的样本
                    sample_bytes = buffer[:self.bytes_per_sample]
                    buffer = buffer[self.bytes_per_sample:]
                    
                    # 解析样本
                    sample = self._parse_sample(sample_bytes)
                    if sample:
                        with self.lock:
                            self.latest_data = sample
                            self.data_buffer.append(sample)
                            self.total_samples += 1
                            
                            # 计算采样率
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
    
    def _parse_sample(self, sample_bytes: bytes) -> Optional[EEGDataPoint]:
        """解析单个样本数据"""
        try:
            # 检查数据长度
            if len(sample_bytes) != self.bytes_per_sample:
                print(f"样本长度错误: 期望{self.bytes_per_sample}字节, 实际{len(sample_bytes)}字节")
                return None
            
            # 解析10个通道数据 + 1个触发信号
            num_values = self.num_channels + 1
            values = struct.unpack(f'<{num_values}f', sample_bytes)
            
            # 分离通道数据和触发信号
            channels = list(values[:self.num_channels])  # 转换为list
            trigger = values[self.num_channels]
            
            # 创建数据点
            data_point = EEGDataPoint(
                timestamp=time.time(),
                channels=channels,
                trigger=trigger,
                raw_bytes=sample_bytes
            )
            
            return data_point
            
        except Exception as e:
            print(f"解析样本数据失败: {e}")
            return None
    
    def get_latest_data(self) -> Optional[EEGDataPoint]:
        """获取最新的数据"""
        with self.lock:
            return self.latest_data
    
    def get_data_buffer(self, max_samples: int = 1000) -> List[EEGDataPoint]:
        """获取数据缓冲区"""
        with self.lock:
            if max_samples:
                return self.data_buffer[-max_samples:]
            return self.data_buffer.copy()
    
    def get_channel_data(self, channel_index: int, max_samples: int = 1000) -> List[float]:
        """获取指定通道的数据"""
        if not 0 <= channel_index < self.num_channels:
            raise ValueError(f"通道索引超出范围: {channel_index}, 有效范围: 0-{self.num_channels-1}")
        
        with self.lock:
            samples = self.data_buffer[-max_samples:] if max_samples else self.data_buffer
            return [sample.channels[channel_index] for sample in samples]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            if not self.data_buffer:
                return {}
            
            # 计算每个通道的统计信息
            channel_stats = {}
            for i, channel_name in enumerate(self.channel_names):
                channel_data = [sample.channels[i] for sample in self.data_buffer]
                channel_stats[channel_name] = {
                    'min': min(channel_data),
                    'max': max(channel_data),
                    'mean': np.mean(channel_data),
                    'std': np.std(channel_data),
                    'samples': len(channel_data)
                }
            
            # 触发信号统计
            trigger_data = [sample.trigger for sample in self.data_buffer]
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
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"montage_10channel_{timestamp}.csv"
        
        with self.lock:
            if not self.data_buffer:
                print("没有数据可保存")
                return
            
            # 准备CSV数据
            csv_data = []
            for sample in self.data_buffer:
                row = [sample.timestamp] + sample.channels + [sample.trigger]
                csv_data.append(row)
            
            # 创建列名
            columns = ['timestamp'] + self.channel_names + ['trigger']
            
            # 保存为CSV
            import pandas as pd
            df = pd.DataFrame(csv_data, columns=columns)
            df.to_csv(filename, index=False)
            print(f"数据已保存到: {filename}")
            print(f"保存了 {len(csv_data)} 个样本")

def main():
    """主函数 - 测试10通道Montage接收器"""
    print("=== 10通道Montage EEG数据接收器测试 ===")
    
    # 创建接收器
    receiver = Ssvep10ChannelReceiver()
    
    try:
        # 连接服务器
        if not receiver.connect():
            return
        
        # 开始接收数据
        if not receiver.start_receiving():
            return
        
        print("\n按 Ctrl+C 停止接收...")
        print("-" * 50)
        
        # 主循环 - 显示实时统计信息
        while receiver.running:
            time.sleep(2)
            
            # 显示统计信息
            stats = receiver.get_statistics()
            if stats:
                print(f"\n当前状态:")
                print(f"  总样本数: {stats['total_samples']}")
                print(f"  采样率: {stats['sample_rate']:.1f} Hz")
                
                # 显示最新数据
                latest = receiver.get_latest_data()
                if latest:
                    print(f"  最新数据:")
                    for i, (name, value) in enumerate(zip(receiver.channel_names, latest.channels)):
                        print(f"    {name}: {value:8.3f}")
                    print(f"    Trigger: {latest.trigger:8.3f}")
                
                print("-" * 30)
        
    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        # 保存数据
        if receiver.data_buffer:
            receiver.save_data_to_csv()
        
        # 断开连接
        receiver.disconnect()
        print("测试完成")

if __name__ == "__main__":
    main()
