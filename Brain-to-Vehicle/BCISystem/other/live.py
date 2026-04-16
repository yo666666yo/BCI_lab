import argparse
import logging
import time
import numpy as np
from typing import List

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

# Import SSVEP8 receiver
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import ssvep10_receiver
from ssvep10_receiver import Ssvep10ChannelReceiver, EEGDataPoint


class Graph:
    def __init__(self, ssvep10_receiver: Ssvep10ChannelReceiver):
        self.ssvep10_receiver = ssvep10_receiver
        self.num_channels = 10  # SSVEP8 有8个通道
        self.channel_names = ssvep10_receiver.channel_names
        self.sampling_rate = 1000  # 假设采样率为1000Hz
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        
        # 数据缓冲区，用于存储历史数据以显示时间序列
        self.data_buffer = [[] for _ in range(self.num_channels)]
        self.max_buffer_size = self.num_points

        # 尝试不同的QApplication导入方式
        try:
            self.app = QtWidgets.QApplication([])
        except AttributeError:
            try:
                self.app = QtGui.QApplication([])
            except AttributeError:
                # 如果都不行，直接从pyqtgraph获取
                self.app = pg.mkQApp()
        # 创建主窗口和布局
        try:
            # 尝试使用新的API
            self.win = pg.GraphicsLayoutWidget(title="SSVEP8 实时数据显示")
            self.win.resize(800, 600)
        except AttributeError:
            try:
                # 尝试使用旧的API
                self.win = pg.GraphicsWindow(title="SSVEP8 实时数据显示", size=(800, 600))
            except AttributeError:
                # 备用方案：手动创建
                self.win = pg.GraphicsLayoutWidget()
                self.win.setWindowTitle("SSVEP8 实时数据显示")
                self.win.resize(800, 600)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        
        # 显示窗口
        self.win.show()
        
        # 启动应用程序事件循环
        try:
            if hasattr(self.app, 'exec_'):
                self.app.exec_()
            else:
                self.app.exec()
        except AttributeError:
            # 备用方案
            pg.QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(self.num_channels):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis("left", True)
            p.setMenuEnabled("left", False)
            p.showAxis("bottom", True if i == self.num_channels - 1 else False)
            p.setMenuEnabled("bottom", False)
            
            # 设置通道标题
            channel_title = f"{self.channel_names[i]} (通道 {i+1})"
            if i == 0:
                channel_title = f"SSVEP8 实时数据 - {channel_title}"
            p.setTitle(channel_title)
            
            # 设置Y轴标签
            p.setLabel('left', 'Amplitude (μV)')
            if i == self.num_channels - 1:
                p.setLabel('bottom', 'Time (samples)')
            
            self.plots.append(p)
            curve = p.plot(pen=pg.mkPen(color=(255, 255, 255), width=1))
            self.curves.append(curve)

    def update(self):
        # 获取最新的SSVEP8数据
        latest_data = self.ssvep10_receiver.get_latest_data()
        
        if latest_data:
            # 将新数据添加到缓冲区
            for i in range(self.num_channels):
                if i < len(latest_data.channels):
                    self.data_buffer[i].append(latest_data.channels[i])
                else:
                    self.data_buffer[i].append(0.0)
                
                # 限制缓冲区大小
                if len(self.data_buffer[i]) > self.max_buffer_size:
                    self.data_buffer[i] = self.data_buffer[i][-self.max_buffer_size:]
        
        # 更新图表显示
        for i in range(self.num_channels):
            if len(self.data_buffer[i]) > 1:
                # 简单的数据处理
                data_array = np.array(self.data_buffer[i])
                
                # 去除直流分量（简单的高通滤波）
                if len(data_array) > 10:
                    data_array = data_array - np.mean(data_array[-100:])
                
                # 更新曲线
                self.curves[i].setData(data_array.tolist())
        
        # 显示连接状态和统计信息
        if hasattr(self, 'status_updated'):
            if time.time() - self.status_updated > 2:  # 每2秒更新一次状态
                self._update_status()
        else:
            self.status_updated = time.time()
            self._update_status()

        self.app.processEvents()
    
    def _update_status(self):
        """更新状态信息"""
        self.status_updated = time.time()
        
        # 获取统计信息
        stats = self.ssvep10_receiver.get_statistics()
        if stats:
            status_text = f"连接状态: {'已连接' if self.ssvep10_receiver.connected else '未连接'} | "
            status_text += f"采样率: {stats.get('sample_rate', 0):.1f} Hz | "
            status_text += f"总样本: {stats.get('total_samples', 0)}"
            
            # 在第一个图表上显示状态信息
            if self.plots:
                self.plots[0].setTitle(f"SSVEP8 实时数据 - {self.channel_names[0]} (通道 1) | {status_text}")
        
        print(f"[{time.strftime('%H:%M:%S')}] SSVEP8状态: 连接={self.ssvep10_receiver.connected}, 数据缓冲区大小={[len(buf) for buf in self.data_buffer]}")


def main():
    logging.basicConfig(level=logging.INFO)
    print("=== SSVEP8 实时数据可视化工具 ===")

    parser = argparse.ArgumentParser(description="SSVEP8 实时数据可视化")
    parser.add_argument(
        "--host",
        type=str,
        help="EEG服务器地址",
        required=False,
        default="127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="EEG服务器端口",
        required=False,
        default=8712
    )
    args = parser.parse_args()

    # 创建SSVEP8接收器
    ssvep10_receiver = Ssvep10ChannelReceiver(host=args.host, port=args.port)
    
    try:
        print(f"正在连接到SSVEP8服务器 {args.host}:{args.port}...")
        
        if ssvep10_receiver.connect():
            print("✓ 成功连接到SSVEP8服务器")
            print("启动实时数据可视化...")
            print("按 Ctrl+C 停止程序")
            
            # 启动图形界面
            Graph(ssvep10_receiver)
        else:
            print("✗ 连接失败，请检查:")
            print("  1. SSVEP8服务器是否已启动")
            print("  2. 网络连接是否正常")
            print("  3. 端口是否正确")
            
            # 如果连接失败，可以选择使用模拟数据
            print("\n使用模拟数据进行演示...")
            Graph(ssvep10_receiver)  # 即使未连接也可以显示模拟数据
            
    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
    except Exception as e:
        logging.error(f"运行出错: {e}", exc_info=True)
    finally:
        print("正在清理资源...")
        if ssvep10_receiver:
            ssvep10_receiver.disconnect()
        print("程序结束")


if __name__ == "__main__":
    main()
