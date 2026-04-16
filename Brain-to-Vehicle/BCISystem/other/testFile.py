import sys
# 导入线程管理相关的库
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
                             QLabel, QPushButton, QTextEdit, QComboBox,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QSplitter, QFrame, QScrollArea)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread  # QThread 用于后台任务
from PyQt6.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg

# 导入EEG接收器模块
from ssvep10_receiver import Ssvep10ChannelReceiver, EEGDataPoint
# 导入双端队列，用作缓冲区
from collections import deque
import numpy as np
# 导入线程安全的队列
import queue
# 检查串口库是否可用
try:
    import serial
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

# 运行EEG接收器的线程
class EEGReceiverWorker(QThread):
    """
    用于处理EEG数据接收的工作线程，以免阻塞图形用户界面（GUI）。
    """
    # 定义一个信号，当接收到新的EEG数据时，会发出一个包含通道值列表的信号
    data_received = pyqtSignal(list)
    # 定义一个信号，用于向主GUI记录消息
    log_message = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.receiver = Ssvep10ChannelReceiver(host, port)
        self._is_running = False

    def run(self):
        """
        当线程启动后，此函数会自动执行。
        """
        self._is_running = True
        self.log_message.emit(
            f"工作线程：正在尝试连接到EEG服务器，地址为 {self.receiver.host}:{self.receiver.port}...")

        if not self.receiver.connect():
            self.log_message.emit("工作线程：连接失败。线程停止。")
            self._is_running = False
            return

        # 启动接收器内部的数据收集线程
        self.receiver.start_receiving()
        self.log_message.emit("工作线程：连接成功。正在接收数据。")

        last_data_timestamp = 0
        while self._is_running:
            latest_data_point = self.receiver.get_latest_data()

            # 检查是否有新数据，以避免多次发出相同的数据点
            if latest_data_point and latest_data_point.timestamp > last_data_timestamp:
                self.data_received.emit(latest_data_point.channels)
                last_data_timestamp = latest_data_point.timestamp

            # 短暂休眠以防止此循环过度占用CPU
            self.msleep(10)  # 休眠10毫秒

        self.receiver.disconnect()
        self.log_message.emit("工作线程：已断开连接，线程停止。")

    def stop(self):
        """
        安全地停止线程。
        """
        self.log_message.emit("工作线程：收到停止信号。")
        self._is_running = False


class EEGWaveformWidget(QWidget):
    """
    用于显示EEG实时波形的组件。
    采用垂直堆叠布局，每个通道一个图表。
    """

    def __init__(self):
        super().__init__()
        # 从接收器类中获取标准的通道名称
        self.channel_names = Ssvep10ChannelReceiver().channel_names
        self.num_channels = len(self.channel_names)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距，让图表占满空间

        # 创建标题
        title = QLabel("EEG波形实时显示")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        # 设置标题的一些边距，使其不至于贴着边框
        title.setContentsMargins(10, 5, 10, 5)
        layout.addWidget(title)

        # CHANGED: 使用 GraphicsLayoutWidget 作为基础，用于容纳多个图表
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(None)  # 设置为透明背景，以便匹配窗口主题
        layout.addWidget(self.plot_widget)

        # 初始化曲线和子图表列表
        self.plots = []
        self.curves = []

        # 循环创建10个子图表
        for i in range(self.num_channels):
            # 添加一个新的PlotItem到布局中，row=i, col=0表示垂直排列
            plot_item = self.plot_widget.addPlot(row=i, col=0)

            # --- 样式设置 ---
            plot_item.getAxis('left').setLabel('幅值(μV)', color='#AAAAAA', fontSize='8pt')
            plot_item.getAxis('left').setWidth(50)  # 固定左轴宽度，使其对齐
            plot_item.showGrid(x=True, y=True, alpha=0.3)

            # 隐藏上下边框线
            plot_item.getAxis('top').setStyle(showValues=False)
            plot_item.getAxis('top').setHeight(0)
            plot_item.getAxis('bottom').setStyle(showValues=False)
            plot_item.getAxis('bottom').setHeight(0)

            # 在图表右侧显示通道名称作为标题
            text_item = pg.TextItem(f"{self.channel_names[i]} (通道 {i + 1})", anchor=(0, 0.5), color='#DDDDDD')
            plot_item.addItem(text_item)
            # 将文本放置在图表内部的右上角
            text_item.setPos(0.85, 0.5)

            # 只有最下面的图表显示X轴（时间轴）
            if i == self.num_channels - 1:
                plot_item.getAxis('bottom').setStyle(showValues=True)
                plot_item.getAxis('bottom').setHeight(30)
                plot_item.setLabel('bottom', '时间 (采样点)', color='#AAAAAA')

            self.plots.append(plot_item)

            # 创建曲线
            pen = pg.mkPen(color='#4ECDC4', width=2)  # 使用统一的颜色，更专业
            curve = plot_item.plot(pen=pen)
            self.curves.append(curve)

            # 链接所有图表的X轴，实现同步缩放和平移
            if i > 0:
                plot_item.setXLink(self.plots[0])

        # 初始化数据缓冲区，为10个通道分别创建，长度为500点以获得更平滑的滚动效果
        self.data = [np.zeros(500) for _ in range(self.num_channels)]

    def update_plot_from_data(self, channel_data: list):
        """
        槽（slot），用于接收新的EEG数据并更新所有子图表的波形。
        """
        if len(channel_data) != self.num_channels:
            return

        for i in range(self.num_channels):
            # 滚动缓冲区，在末尾添加新的数据点
            self.data[i] = np.roll(self.data[i], -1)
            self.data[i][-1] = channel_data[i]

            # 更新对应子图表的曲线
            self.curves[i].setData(self.data[i])
# 在后台运行推理的线程
class InferenceWorker(QThread):
    """
    在后台线程中处理推理（意图识别）部分，避免阻塞GUI。
    """
    # 定义信号，当推理完成时，发出推断的命令（字符串）和置信度（浮点数）
    inference_complete = pyqtSignal(str, float)
    log_message = pyqtSignal(str)

    def __init__(self, data_buffer):
        super().__init__()
        self.data_buffer = data_buffer  # 共享来自主线程的数据缓冲区
        self._is_running = False
        self.directions = ['forward', 'backward', 'left', 'right', 'stop']

    def run(self):
        self._is_running = True
        self.log_message.emit("推理工作线程启动...")

        # 在该线程内部使用一个QTimer来周期性地触发推理
        self.timer = QTimer()
        self.timer.moveToThread(self)  # 确保定时器在此线程中运行
        self.timer.timeout.connect(self.run_inference)
        self.timer.start(200)  # 每200毫秒触发一次推理

        # 启动线程的事件循环
        self.exec()

    def run_inference(self):
        """核心的推理逻辑（目前为模拟）。"""
        if len(self.data_buffer) < 100:  # 至少需要100个样本才能进行推理
            return

        # --- 这里应该是您真正的模型推理代码 ---
        # 目前，我们用一个简单的算法来模拟。
        # 让我们以分析第一个通道（P4）为例。

        # 将双端队列转换为NumPy数组以便于处理
        # 我们取最近的100个样本进行分析
        recent_data = np.array(list(self.data_buffer)[-100:])
        channel_p4_data = recent_data[:, 0]  # 获取第一个通道（P4）的数据

        # 简单的基于规则的模拟：
        # 如果平均绝对振幅很高，可能意图是'前进'
        mean_amplitude = np.mean(np.abs(channel_p4_data))

        inferred_direction = 'stop'  # 默认为'停止'
        confidence = 0.85

        if mean_amplitude > 50:  # 示例阈值
            inferred_direction = 'forward'
            confidence = min(0.99, 0.7 + (mean_amplitude - 50) / 100.0)
        elif mean_amplitude < 10:
            inferred_direction = 'stop'
            confidence = min(0.99, 0.8 + (10 - mean_amplitude) / 50.0)
        else:  # 为其他命令添加一些随机性
            if int(time.time()) % 5 == 0:
                inferred_direction = 'left'
            elif int(time.time()) % 5 == 2:
                inferred_direction = 'right'

        # --- 模拟逻辑结束 ---

        self.inference_complete.emit(inferred_direction, confidence)

    def stop(self):
        self.log_message.emit("推理工作线程正在停止...")
        self._is_running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.quit()  # 退出事件循环
        self.wait()  # 等待线程完全结束


# 新增：一个用于处理底层小车通信的类
class CarController:
    """处理通过串口与小车的实际通信。"""

    def __init__(self, port='COM3', baudrate=9600, simulation=True):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.simulation = simulation

        # 定义从命令字符串到字节的映射
        self.command_map = {
            'stop': b'\x00',
            'forward': b'\x01',
            'backward': b'\x02',
            'left': b'\x03',
            'right': b'\x04',
        }

    def connect(self):
        """连接到串口。"""
        if self.simulation:
            print(f"[小车控制器-模拟] 到 {self.port} 的虚拟连接已打开。")
            return True

        if not PYSERIAL_AVAILABLE:
            print("[小车控制器] Pyserial库未找到。无法连接。在模拟模式下运行。")
            self.simulation = True
            return True

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[小车控制器] 成功连接到 {self.port}。")
            return True
        except serial.SerialException as e:
            print(f"[小车控制器] 连接到 {self.port} 失败: {e}")
            return False

    def disconnect(self):
        """断开与串口的连接。"""
        if self.simulation:
            print("[小车控制器-模拟] 虚拟连接已关闭。")
            return

        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[小车控制器] 已从 {self.port} 断开。")

    def send_command(self, command: str):
        """向小车发送一个命令。"""
        if command not in self.command_map:
            print(f"[小车控制器] 未知命令: {command}")
            return

        byte_to_send = self.command_map[command]

        if self.simulation:
            print(f"[小车控制器-模拟] 将为命令 '{command}' 发送 {byte_to_send.hex()}")
            return

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(byte_to_send)
            except serial.SerialException as e:
                print(f"[小车控制器] 写入串口时出错: {e}")
        else:
            print("[小车控制器] 无法发送命令，未连接。")


# 新增：一个在单独线程中运行CarController的工作线程
class CarControlWorker(QThread):
    """用于向小车发送命令的工作线程。"""
    log_message = pyqtSignal(str)

    def __init__(self, port='COM3', baudrate=9600, simulation=True):
        super().__init__()
        self.controller = CarController(port, baudrate, simulation)
        self.command_queue = queue.Queue() # 使用线程安全的队列来接收命令
        self._is_running = False

    def run(self):
        self._is_running = True
        if not self.controller.connect():
            self.log_message.emit("小车控制器连接失败。工作线程将停止。")
            self._is_running = False
            return

        self.log_message.emit("小车控制工作线程已启动。")
        while self._is_running:
            try:
                # 等待命令被放入队列中
                command = self.command_queue.get(timeout=1)  # 设置超时以允许检查_is_running标志
                if command is None:  # 使用None作为停止线程的哨兵值
                    break
                self.controller.send_command(command)
            except queue.Empty:
                continue

        self.controller.disconnect()
        self.log_message.emit("小车控制工作线程已停止。")

    def submit_command(self, command: str):
        """线程安全的方法，用于将命令添加到队列中。"""
        self.command_queue.put(command)

    def stop(self):
        self._is_running = False
        self.command_queue.put(None)  # 发送哨兵值以解除queue.get()的阻塞
        self.wait()


class InferenceResultWidget(QWidget):
    """显示意图识别结果的组件"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)

        title = QLabel("控制意图识别结果")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title, 0, 0, 1, 3)

        # 创建方向按钮用于显示
        self.buttons = {
            'forward': self.create_direction_button("↑", "前进"),
            'backward': self.create_direction_button("↓", "后退"),
            'left': self.create_direction_button("←", "左转"),
            'right': self.create_direction_button("→", "右转"),
            'stop': self.create_direction_button("●", "停止")
        }

        # 布局方向按钮
        layout.addWidget(self.buttons['forward'], 1, 1)
        layout.addWidget(self.buttons['left'], 2, 0)
        layout.addWidget(self.buttons['stop'], 2, 1)
        layout.addWidget(self.buttons['right'], 2, 2)
        layout.addWidget(self.buttons['backward'], 3, 1)

        # 置信度显示标签
        self.confidence_label = QLabel("置信度: 0.00")
        self.confidence_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.confidence_label, 4, 0, 1, 3)

    def create_direction_button(self, symbol, text):
        """辅助函数，创建一个用于显示方向的自定义组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(symbol)
        label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(text_label)
        widget.setStyleSheet("""
                    QWidget {
                        border: 2px solid #CCCCCC;
                        border-radius: 10px;
                        padding: 10px;
                        background-color: #F8F9FA;
                    }
                    QLabel {
                        color: #333333; /* 设置文字和符号的颜色为深灰色 */
                    }
                """)
        return widget

    def update_inference(self, current_direction: str, confidence: float):
        """Slot to update the inference result display."""

        # 更新按钮高亮
        for direction, widget in self.buttons.items():
            if direction == current_direction:
                # 当按钮被选中（高亮）时的样式
                widget.setStyleSheet("""
                    QWidget {
                        border: 3px solid #007BFF;
                        border-radius: 10px;
                        padding: 10px;
                        background-color: #E3F2FD;
                    }
                    QLabel {
                        color: #333333; /* 确保高亮时文字也是深色 */
                    }
                """)
            else:
                # 当按钮未被选中时的样式
                widget.setStyleSheet("""
                    QWidget {
                        border: 2px solid #CCCCCC;
                        border-radius: 10px;
                        padding: 10px;
                        background-color: #F8F9FA;
                    }
                    QLabel {
                        color: #333333; /* 确保非高亮时文字也是深色 */
                    }
                """)

        # 更新置信度
        self.confidence_label.setText(f"置信度: {confidence:.2f}")


class CarStatusWidget(QWidget):
    """显示小车状态的组件"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("小车状态监控")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        status_group = QGroupBox("实时状态")
        status_layout = QGridLayout(status_group)
        status_layout.addWidget(QLabel("速度:"), 0, 0)
        self.speed_label = QLabel("0.0 m/s")
        self.speed_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        status_layout.addWidget(self.speed_label, 0, 1)
        status_layout.addWidget(QLabel("方向:"), 1, 0)
        self.direction_label = QLabel("停止")
        self.direction_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        status_layout.addWidget(self.direction_label, 1, 1)
        status_layout.addWidget(QLabel("电池电量:"), 2, 0)
        self.battery_label = QLabel("N/A")  # 如果小车没有回传数据，这里无法得知
        self.battery_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        status_layout.addWidget(self.battery_label, 2, 1)
        status_layout.addWidget(QLabel("连接状态:"), 3, 0)
        self.connection_label = QLabel("未连接")  # 将由主窗口更新
        self.connection_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.connection_label, 3, 1)
        layout.addWidget(status_group)

    def update_status_from_command(self, command: str):
        """根据发送给小车的命令更新状态显示。"""
        command_map_display = {'stop': "停止", 'forward': "前进", 'backward': "后退",
                               'left': "左转", 'right': "右转"}
        speed_map = {'stop': "0.0 m/s", 'forward': "1.5 m/s", 'backward': "1.0 m/s",
                     'left': "0.8 m/s", 'right': "0.8 m/s"}
        self.direction_label.setText(command_map_display.get(command, "未知"))
        self.speed_label.setText(speed_map.get(command, "0.0 m/s"))


class LogWidget(QWidget):
    """日志窗口组件"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("系统日志")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

    def add_log(self, message):
        """添加一条带时间戳的日志消息。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        # 自动滚动到日志底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())


class ControlButtonsWidget(QWidget):
    """包含紧急停止和人工控制按钮的组件"""
    emergency_stop_signal = pyqtSignal() # 定义紧急停止信号

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        # 紧急停止按钮
        self.emergency_stop_btn = QPushButton("紧急停止")
        self.emergency_stop_btn.setStyleSheet("""
            QPushButton { background-color: #DC3545; color: white; font-weight: bold;
                          border: none; padding: 10px 20px; border-radius: 5px; font-size: 14px; }
            QPushButton:hover { background-color: #C82333; }
            QPushButton:pressed { background-color: #BD2130; } """)
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        # 人工控制开关按钮
        self.manual_control_btn = QPushButton("开启人工控制")
        self.manual_control_btn.setCheckable(True)
        self.manual_control_btn.setStyleSheet("""
            QPushButton { background-color: #6C757D; color: white; font-weight: bold;
                          border: none; padding: 10px 20px; border-radius: 5px; font-size: 14px; }
            QPushButton:checked { background-color: #28A745; }
            QPushButton:hover { background-color: #5A6268; }
            QPushButton:checked:hover { background-color: #218838; } """)
        layout.addWidget(self.emergency_stop_btn)
        layout.addWidget(self.manual_control_btn)
        layout.addStretch()

    def emergency_stop(self):
        """当点击紧急停止时，发出信号。"""
        self.emergency_stop_signal.emit()


class SettingsWidget(QWidget):
    """设置界面组件"""
    # 定义一个信号，当点击“应用”时，会发出一个包含所有设置的字典
    settings_applied = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # --- 连接设置 ---
        connection_group = QGroupBox("连接设置")
        connection_layout = QGridLayout(connection_group)
        connection_layout.addWidget(QLabel("EEG服务器IP:"), 0, 0)
        self.eeg_ip = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.eeg_ip, 0, 1)
        connection_layout.addWidget(QLabel("EEG服务器端口:"), 0, 2)
        self.eeg_port = QSpinBox()
        self.eeg_port.setRange(1000, 65535)
        self.eeg_port.setValue(8712)
        connection_layout.addWidget(self.eeg_port, 0, 3)
        connection_layout.addWidget(QLabel("小车控制串口:"), 1, 0)
        self.control_serial = QLineEdit("COM3")
        connection_layout.addWidget(self.control_serial, 1, 1)
        # 新增：模拟模式复选框
        self.simulation_mode = QCheckBox("启用小车模拟模式")
        self.simulation_mode.setChecked(True)
        connection_layout.addWidget(self.simulation_mode, 1, 2, 1, 2)
        layout.addWidget(connection_group)
        # --- 参数设置 ---
        params_group = QGroupBox("参数设置")
        params_layout = QGridLayout(params_group)
        layout.addWidget(params_group)
        # --- 用户设置 ---
        user_group = QGroupBox("用户设置")
        user_layout = QGridLayout(user_group)
        layout.addWidget(user_group)
        # 应用按钮
        self.apply_btn = QPushButton("应用设置")
        self.apply_btn.setStyleSheet("""
            QPushButton { background-color: #007BFF; color: white; font-weight: bold;
                          border: none; padding: 10px 20px; border-radius: 5px; font-size: 14px; }
            QPushButton:hover { background-color: #0056B3; } """)
        self.apply_btn.clicked.connect(self.emit_settings)
        layout.addWidget(self.apply_btn)
        layout.addStretch()

    def emit_settings(self):
        """收集UI中的设置并通过信号发送出去。"""
        settings = {
            'eeg_ip': self.eeg_ip.text(),
            'eeg_port': self.eeg_port.value(),
            'car_serial_port': self.control_serial.text(),
            'car_simulation': self.simulation_mode.isChecked(),
        }
        self.log_widget.add_log(f"应用新设置: {settings}")
        self.settings_applied.emit(settings)


class MainWindow(QMainWindow):
    """主窗口类，整合所有组件。"""

    def __init__(self):
        super().__init__()
        self.eeg_data_buffer = deque(maxlen=1000) # 创建一个最大长度为1000的数据缓冲区
        self.eeg_worker = None
        self.inference_worker = None
        self.car_worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("脑机接口小车控制系统")
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        self.main_tab = self.create_main_tab()
        self.settings_tab = SettingsWidget()
        self.tabs.addTab(self.main_tab, "主界面")
        self.tabs.addTab(self.settings_tab, "设置")
        main_layout.addWidget(self.tabs)
        # --- 连接信号与槽 ---
        self.control_buttons.emergency_stop_signal.connect(self.emergency_stop)
        self.settings_tab.settings_applied.connect(self.handle_settings_applied)
        # 将日志组件实例传递给设置页面，以便其可以记录日志
        self.settings_tab.log_widget = self.log_widget
        # --- 初始启动 ---
        self.log_widget.add_log("系统启动，使用默认设置。")
        self.start_all_workers(self.get_default_settings())

    def get_default_settings(self):
        """返回一个包含默认设置的字典。"""
        return {
            'eeg_ip': '127.0.0.1',
            'eeg_port': 8712,
            'car_serial_port': 'COM3',
            'car_simulation': True,
        }

    def handle_settings_applied(self, settings):
        """处理来自设置页面的新设置的槽函数。"""
        self.log_widget.add_log("收到应用设置请求，正在重启所有服务...")
        self.stop_all_workers()
        self.start_all_workers(settings)

    def start_all_workers(self, settings):
        """使用给定的设置启动所有工作线程。"""
        # 启动EEG工作线程
        self.eeg_worker = EEGReceiverWorker(host=settings['eeg_ip'], port=settings['eeg_port'])
        self.eeg_worker.data_received.connect(self.handle_new_eeg_data)
        self.eeg_worker.log_message.connect(self.log_widget.add_log)
        self.eeg_worker.start()
        # 启动推理工作线程
        self.inference_worker = InferenceWorker(data_buffer=self.eeg_data_buffer)
        self.inference_worker.inference_complete.connect(self.process_inference_result)
        self.inference_worker.log_message.connect(self.log_widget.add_log)
        self.inference_worker.start()
        # 启动小车控制工作线程
        self.car_worker = CarControlWorker(port=settings['car_serial_port'], simulation=settings['car_simulation'])
        self.car_worker.log_message.connect(self.log_widget.add_log)
        self.car_worker.start()
        # 根据新的连接尝试更新小车状态UI
        conn_text = "已连接" if self.car_worker.controller.connect() else "连接失败"
        sim_text = " (模拟)" if settings['car_simulation'] else ""
        self.car_status_widget.connection_label.setText(f"{conn_text}{sim_text}")
        color = "green" if "已连接" in conn_text else "red"
        self.car_status_widget.connection_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def stop_all_workers(self):
        """安全地停止所有正在运行的工作线程。"""
        self.log_widget.add_log("正在停止所有服务...")
        if self.eeg_worker and self.eeg_worker.isRunning():
            self.eeg_worker.stop()
            self.eeg_worker.wait() # 等待线程结束
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait()
        if self.car_worker and self.car_worker.isRunning():
            self.car_worker.stop()
            self.car_worker.wait()
        self.log_widget.add_log("所有服务已停止。")

    def handle_new_eeg_data(self, channel_data: list):
        """处理新接收到的EEG数据的槽函数。"""
        self.eeg_data_buffer.append(channel_data)
        self.eeg_widget.update_plot_from_data(channel_data)

    def process_inference_result(self, command: str, confidence: float):
        """处理推理结果的槽函数。"""
        self.inference_widget.update_inference(command, confidence)
        self.car_status_widget.update_status_from_command(command)
        if self.car_worker:
            self.car_worker.submit_command(command)

    def create_main_tab(self):
        """创建主界面的布局和组件。"""
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        splitter = QSplitter(Qt.Orientation.Vertical)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        self.eeg_widget = EEGWaveformWidget()
        top_layout.addWidget(self.eeg_widget, 60) # 60% 宽度
        self.inference_widget = InferenceResultWidget()
        top_layout.addWidget(self.inference_widget, 40) # 40% 宽度
        splitter.addWidget(top_widget)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        self.car_status_widget = CarStatusWidget()
        bottom_layout.addWidget(self.car_status_widget, 30)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.log_widget = LogWidget()
        right_layout.addWidget(self.log_widget, 70)
        self.control_buttons = ControlButtonsWidget()
        right_layout.addWidget(self.control_buttons, 30)
        bottom_layout.addWidget(right_widget, 70)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([600, 300])
        layout.addWidget(splitter)
        return main_tab

    def emergency_stop(self):
        """紧急停止的槽函数。"""
        self.log_widget.add_log("!!! 紧急停止已触发 !!!")
        if self.car_worker:
            self.car_worker.submit_command('stop')
        self.car_status_widget.update_status_from_command('stop')

    def closeEvent(self, event):
        """在关闭窗口前，确保所有线程都已停止。"""
        self.stop_all_workers()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # 设置UI风格
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()