import sys
# 导入线程管理相关的库
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox, QCheckBox,
                             QSplitter, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QThread  # QThread 用于后台任务
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from eeg_utils import EEGPreprocessor

# 导入EEG接收器模块
from ssvep21_receiver import Ssvep21ChannelReceiver
# 导入双端队列，用作缓冲区
from collections import deque
import numpy as np
# 导入线程安全的队列
import queue
# 检查串口库是否可用
from eeg_utils import EEGModelHandler # 导入新写的类
from maze_calib import MazeCalibrationGame


# 运行EEG接收器的线程
class EEGReceiverWorker(QThread):
    """
    用于处理EEG数据接收的工作线程。
    新增功能：集成实时信号预处理（滤波）。
    """
    # 信号发出的是处理后的数据
    data_received = pyqtSignal(list)
    log_message = pyqtSignal(str)

    def __init__(self, host, port, fs=110):
        # 默认采样率改为 110 (或者更精确的 100/125/128，取决于你的硬件说明书，如果是 OpenBCI Cyton 蓝牙模式通常是 125Hz，如果是某些自制设备可能是 100Hz。既然日志说是 109.8，那设为 110 比较安全)。
        super().__init__()
        self.receiver = Ssvep21ChannelReceiver(host, port)
        self._is_running = False
        # 初始化预处理器，通道数为21 (不包含trigger)
        self.preprocessor = EEGPreprocessor(fs=fs, num_channels=21)

    def run(self):
        self._is_running = True
        self.log_message.emit(
            f"工作线程：正在尝试连接到EEG服务器 {self.receiver.host}:{self.receiver.port}...")

        if not self.receiver.connect():
            self.log_message.emit("工作线程：连接失败。线程停止。")
            self._is_running = False
            return

        self.receiver.start_receiving()
        self.log_message.emit("工作线程：连接成功。正在接收并预处理数据。")

        last_data_timestamp = 0
        while self._is_running:
            # 获取最新的一帧原始数据
            latest_data_point = self.receiver.get_latest_data()

            if latest_data_point and latest_data_point.timestamp > last_data_timestamp:
                # 1. 提取原始通道数据 (List -> Numpy)
                raw_channels = np.array(latest_data_point.channels)

                # 2. 进行实时滤波预处理
                # 注意：我们只处理EEG通道，不处理Trigger
                filtered_channels = self.preprocessor.process_sample(raw_channels)

                # 3. 将处理后的数据转换为列表并发送
                # 保持数据结构一致性，如果后续需要Trigger，可以将其附加回去，
                # 但这里 data_received 主要用于绘图和推理，仅包含 EEG 信号即可
                self.data_received.emit(filtered_channels.tolist())

                # 更新时间戳
                last_data_timestamp = latest_data_point.timestamp

            self.msleep(4)  # 250Hz大约每4ms一个点，适当休眠避免死循环占用过高

        self.receiver.disconnect()
        self.log_message.emit("工作线程：已断开连接，线程停止。")

    def stop(self):
        self.log_message.emit("工作线程：收到停止信号。")
        self._is_running = False

class EEGWaveformWidget(QWidget):
    """
    用于显示EEG实时波形的组件。
    采用垂直堆叠布局，每个通道一个图表。
    """

    doubleClicked = pyqtSignal() # 双击放大信号

    def __init__(self):
        super().__init__()
        # 从接收器类中获取标准的通道名称
        self.channel_names = Ssvep21ChannelReceiver().channel_names
        self.num_channels = len(self.channel_names)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距，让图表占满空间

        # 创建标题
        title = QLabel("EEG波形实时显示（双击图表可全屏）")
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
            plot_item.getAxis('left').setLabel('幅值(μV)', color="#4D4B4B", fontSize='8pt')
            plot_item.getAxis('left').setWidth(50)  # 固定左轴宽度，使其对齐
            plot_item.showGrid(x=True, y=True, alpha=0.3)

            # 隐藏上下边框线
            plot_item.getAxis('top').setStyle(showValues=False)
            plot_item.getAxis('top').setHeight(0)
            plot_item.getAxis('bottom').setStyle(showValues=False)
            plot_item.getAxis('bottom').setHeight(0)

            # 在图表右侧显示通道名称作为标题
            text_item = pg.TextItem(f"{self.channel_names[i]} (通道 {i + 1})", anchor=(0, 0.5), color="#BE0909")
            plot_item.addItem(text_item)
            # 将文本放置在图表内部的右上角
            text_item.setPos(0.85, 0.5)

            # 只有最下面的图表显示X轴（时间轴）
            if i == self.num_channels - 1:
                plot_item.getAxis('bottom').setStyle(showValues=True)
                plot_item.getAxis('bottom').setHeight(30)
                plot_item.setLabel('bottom', '时间 (采样点)', color="#100707EE")

            self.plots.append(plot_item)

            # 创建曲线
            pen = pg.mkPen(color='#4ECDC4', width=2)  # 使用统一的颜色，更专业
            curve = plot_item.plot(pen=pen)
            self.curves.append(curve)

            # 链接所有图表的X轴，实现同步缩放和平移
            if i > 0:
                plot_item.setXLink(self.plots[0])
                plot_item.setYLink(self.plots[0])

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

    def mouseDoubleClickEvent(self, event):
        """双击事件：触发全屏/恢复切换"""
        self.doubleClicked.emit()  # 发送信号给给主窗口
        super().mouseDoubleClickEvent(event)

    def save_original_layout(self):
        """保存当前窗口在父容器中的布局信息"""
        self.original_parent = self.parentWidget()
        if self.original_parent:
            self.original_layout = self.original_parent.layout()
            for i in range(self.original_layout.count()):
                item = self.original_layout.itemAt(i)
                if item.widget() == self:
                    self.original_index = i
                    break

    def restore_original_layout(self):
        """恢复到原始布局位置"""
        if self.original_parent and self.original_layout and self.original_index != -1:
            # 从当前父容器移除
            current_parent = self.parentWidget()
            if current_parent and current_parent.layout():
                current_parent.layout().removeWidget(self)
            # 添加回原始布局
            self.original_layout.insertWidget(self.original_index, self, 60)
# 在后台运行推理的线程
class InferenceWorker(QThread):
    inference_complete = pyqtSignal(str, float)
    log_message = pyqtSignal(str)

    def __init__(self, data_buffer, model_handler):  # 新增 model_handler 参数
        super().__init__()
        self.data_buffer = data_buffer
        self.model_handler = model_handler  # 保存句柄
        self._is_running = False
        # 类别索引到字符串的映射
        self.idx_to_cmd = {0: 'stop', 1: 'forward', 2: 'backward', 3: 'left', 4: 'right'}

    def run(self):
        self._is_running = True
        self.log_message.emit("AI 推理线程启动...")

        # 简单循环，或者使用 QTimer 均可
        while self._is_running:
            self.run_inference()
            self.msleep(200)  # 200ms 推理一次 (5Hz)

    def run_inference(self):
        # 1. 检查数据量是否足够
        required_len = self.model_handler.input_window  # 例如 500
        if len(self.data_buffer) < required_len:
            return

        # 2. 获取最新数据窗口
        # deque 不支持切片，需转 list 或 itertools.islice，这里转 list 比较简单但有性能开销
        # 优化：只取最后 required_len 个
        recent_data = list(self.data_buffer)[-required_len:]

        # 3. 调用模型进行预测
        try:
            # model_handler.predict 接受 (Time, Chan)
            idx, conf = self.model_handler.predict(recent_data)

            command = self.idx_to_cmd.get(idx, 'stop')

            # 只有置信度够高才发送 (阈值可设)
            if conf > 0.5:
                self.inference_complete.emit(command, conf)
            else:
                # 置信度过低，可视作 'stop' 或 维持上一状态，这里发 stop 安全点
                # 或者什么都不发
                pass

        except Exception as e:
            # 偶尔可能会因为数据形状问题报错，捕获之
            print(f"推理错误: {e}")

    def stop(self):
        self._is_running = False
        self.wait()


# 一个用于处理底层小车通信的类




# 一个在单独线程中运行CarController的工作线程
class CarControlWorker(QThread):
    """用于向小车发送命令的工作线程 (UDP版)。"""
    log_message = pyqtSignal(str)

    def __init__(self, ip=CAR_IP, port=CAR_PORT, simulation=IS_SIMULATION):
        super().__init__()
        self.controller = CarController(ip, port, simulation)
        self.command_queue = queue.Queue()
        self._is_running = False

    def run(self):
        self._is_running = True
        if not self.controller.connect():
            self.log_message.emit("小车控制器初始化失败。工作线程将停止。")
            self._is_running = False
            return

        self.log_message.emit(
            f"小车控制工作线程已启动 (UDP 目标: {self.controller.target_ip}:{self.controller.target_port})。")

        while self._is_running:
            try:
                command = self.command_queue.get(timeout=1)
                if command is None:
                    break
                self.controller.send_command(command)
            except queue.Empty:
                continue

        self.controller.disconnect()
        self.log_message.emit("小车控制工作线程已停止。")

    def submit_command(self, command: str):
        self.command_queue.put(command)

    def stop(self):
        self._is_running = False
        self.command_queue.put(None)
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

        # EEG 设置 (保持不变)
        connection_layout.addWidget(QLabel("EEG服务器IP:"), 0, 0)
        self.eeg_ip = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.eeg_ip, 0, 1)
        connection_layout.addWidget(QLabel("EEG端口:"), 0, 2)
        self.eeg_port = QSpinBox()
        self.eeg_port.setRange(1000, 65535)
        self.eeg_port.setValue(8712)
        connection_layout.addWidget(self.eeg_port, 0, 3)

        # 小车设置 (修改为 UDP)
        connection_layout.addWidget(QLabel("小车IP地址:"), 1, 0)
        self.car_ip = QLineEdit("192.168.4.1")  # 默认填一个常见的ESP32 AP模式IP
        connection_layout.addWidget(self.car_ip, 1, 1)

        connection_layout.addWidget(QLabel("小车UDP端口:"), 1, 2)
        self.car_port = QSpinBox()
        self.car_port.setRange(1000, 65535)
        self.car_port.setValue(3333)  # 假设小车监听3333
        connection_layout.addWidget(self.car_port, 1, 3)

        # 模拟模式
        self.simulation_mode = QCheckBox("启用小车模拟模式")
        self.simulation_mode.setChecked(True)
        connection_layout.addWidget(self.simulation_mode, 2, 0, 1, 4)  # 稍微调整位置

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
            'car_ip': self.car_ip.text(),        # 变更为 car_ip
            'car_port': self.car_port.value(),   # 变更为 car_port
            'car_simulation': self.simulation_mode.isChecked(),
        }
        self.log_widget.add_log(f"应用新设置: {settings}")
        self.settings_applied.emit(settings)


class OnlineTrainerWorker(QThread):
    """
    后台在线训练线程
    消费者：从队列中取数据 -> 调用模型训练一步
    """
    log_message = pyqtSignal(str)

    def __init__(self, model_handler, training_queue):
        super().__init__()
        self.model_handler = model_handler
        self.training_queue = training_queue
        self._is_running = False

    def run(self):
        self._is_running = True
        self.log_message.emit(">>> 在线训练线程已启动，等待游戏数据...")

        while self._is_running:
            try:
                # 阻塞等待队列中的数据，超时1秒以便检查停止标志
                data, label = self.training_queue.get(timeout=1)

                # 执行单步训练
                loss = self.model_handler.train_one_step(data, label)

                # 发送日志 (可选，太频繁可能会刷屏，可以设置每5次发一次)
                self.log_message.emit(f"在线学习中... Label:{label} | Loss:{loss:.4f}")

                self.training_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"在线训练出错: {e}")

    def stop(self):
        self._is_running = False
        self.wait()


class CalibrationWidget(QWidget):
    """
    在线模型校准界面 (Online Calibration)
    特点：单按钮流程，玩游戏即训练，无需后续操作。
    """
    calibration_finished = pyqtSignal()

    def __init__(self, data_buffer, model_handler):
        super().__init__()
        self.data_buffer = data_buffer
        self.model_handler = model_handler
        self.training_queue = queue.Queue()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(25)
        layout.setContentsMargins(40, 40, 40, 40)

        # --- 1. 顶部：协议标题 ---
        header_layout = QVBoxLayout()
        title = QLabel("ADAPTIVE CALIBRATION")
        title.setStyleSheet("color: #00d2ff; letter-spacing: 4px;")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("在线自适应神经校准协议")
        subtitle.setStyleSheet("color: #888; letter-spacing: 2px;")
        subtitle.setFont(QFont("Microsoft YaHei", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # --- 2. 中部：状态监控卡片 ---
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-left: 4px solid #00d2ff; /* 左侧亮条装饰 */
                border-radius: 6px;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #ccc;
                font-family: "Consolas", "Segoe UI";
                font-size: 13px;
                line-height: 160%;
            }
        """)
        card_layout = QVBoxLayout(status_card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        # 动态生成的富文本说明
        status_text = QLabel(
            "<b>SYSTEM STATUS:</b> <span style='color:#2ea043'>ONLINE</span><br>"
            "<b>MODE:</b> <span style='color:#00d2ff'>Real-time Incremental Learning</span><br>"
            "<br>"
            "<b>操作说明 / MISSION:</b><br>"
            "1. 点击启动按钮，进入神经链接迷宫。<br>"
            "2. 通过意图控制红点移动，系统将<span style='color:#ffc107'>实时更新</span>神经网络参数。<br>"
            "3. 游戏结束后，模型即刻校准完成，无需额外等待。"
        )
        status_text.setTextFormat(Qt.TextFormat.RichText)
        card_layout.addWidget(status_text)

        layout.addWidget(status_card)

        # --- 3. 核心交互：巨大的启动按钮 ---
        self.start_btn = QPushButton("INITIALIZE SEQUENCE [启动校准]")
        self.start_btn.setMinimumHeight(70)  # 很大，很显眼
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d1117;
                border: 2px solid #00d2ff;
                border-radius: 8px;
                color: #00d2ff;
                font-family: "Segoe UI";
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #00d2ff;
                color: #000000;
                box-shadow: 0 0 15px #00d2ff;
            }
            QPushButton:pressed {
                background-color: #00a0c2;
                border-color: #00a0c2;
            }
            QPushButton:disabled {
                border-color: #444;
                color: #666;
                background-color: #1a1a1a;
            }
        """)
        self.start_btn.clicked.connect(self.launch_maze_game)
        layout.addWidget(self.start_btn)

        # --- 4. 底部：实时终端日志 ---
        log_label = QLabel("NEURAL LINK LOGS")
        log_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        log_label.setStyleSheet("color: #555; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                border: 1px solid #333;
                color: #2ea043; /* 绿色终端字 */
                font-family: "Consolas", monospace;
                font-size: 11px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_text)

    def log(self, msg):
        self.log_text.append(f">> {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def launch_maze_game(self):
        if not self.data_buffer or len(self.data_buffer) < 100:
            self.log("ERROR: Data stream unstable. Check EEG connection.")
            return

        self.log("Initializing Neural Calibration Protocol...")
        self.start_btn.setEnabled(False)
        self.start_btn.setText("CALIBRATION IN PROGRESS...")
        self.start_btn.setStyleSheet("border-color: #2ea043; color: #2ea043;")  # 变成绿色状态

        # 1. 启动后台训练线程 (Online Learner)
        self.trainer_thread = OnlineTrainerWorker(self.model_handler, self.training_queue)
        # 连接日志：后台线程的输出直接显示在界面Log里
        self.trainer_thread.log_message.connect(self.log)
        self.trainer_thread.start()

        # 2. 启动游戏 (阻塞式，直到游戏窗口关闭)
        try:
            self.log("Launching simulation window...")
            game = MazeCalibrationGame(self.data_buffer,
                                       input_window=self.model_handler.input_window,
                                       training_queue=self.training_queue)

            # 运行游戏，并在结束后返回（虽然我们不需要返回值了，因为已经实时训练了）
            game.run_game()

            self.log("Simulation ended. Model parameters updated.")

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

        # 3. 游戏关闭后的清理工作
        self.trainer_thread.stop()

        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.start_btn.setText("INITIALIZE SEQUENCE [启动校准]")
        self.start_btn.setStyleSheet("")  # 恢复默认样式

        # 发送完成信号，通知主界面
        self.calibration_finished.emit()
        self.log("System Ready. Control parameters synchronized.")

class MainWindow(QMainWindow):
    """主窗口类，整合所有组件。"""

    def __init__(self):
        super().__init__()
        self.eeg_data_buffer = deque(maxlen=1000) # 创建一个最大长度为1000的数据缓冲区
        # 2. 初始化模型处理器 (ModelHandler)
        # 假设 21 通道，5 分类 (Stop, Forward, Backward, Left, Right)
        # 输入窗口设为 500 (2秒)
        self.model_handler = EEGModelHandler(n_chan=21, n_classes=5, input_window=220)
        self.eeg_worker = None
        self.inference_worker = None
        self.car_worker = None
        self.is_eeg_fullscreen = False  # EEG波形是否全屏，用于实现EEG波形全屏功能
        self.init_ui()

    def init_ui(self):
        self.stack = QStackedWidget()
        self.setWindowTitle("脑机接口小车控制系统")
        self.setGeometry(100, 100, 1400, 900)
        self.central_widget = QWidget()
        # self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.main_tab = self.create_main_tab()
        self.settings_tab = SettingsWidget()
        # 将 data_buffer 和 model_handler 传给校准界面
        self.calibration_tab = CalibrationWidget(self.eeg_data_buffer, self.model_handler)
        self.calibration_tab.calibration_finished.connect(self.on_calibration_finished)

        self.tabs.addTab(self.main_tab, "主界面")
        self.tabs.addTab(self.calibration_tab, "模型校准")  # 新增
        self.tabs.addTab(self.settings_tab, "设置")

        self.main_layout.addWidget(self.tabs)
        self.stack.addWidget(self.central_widget)

        # 创建 fullscreen page（空容器）
        self.fullscreen_page = QWidget()
        self.fullscreen_layout = QVBoxLayout(self.fullscreen_page)
        self.fullscreen_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.fullscreen_page)

        # 把 stack 设为主窗口中央控件
        self.setCentralWidget(self.stack)

        # --- 连接信号与槽 ---
        self.control_buttons.emergency_stop_signal.connect(self.emergency_stop)
        self.settings_tab.settings_applied.connect(self.handle_settings_applied)
        self.eeg_widget.doubleClicked.connect(self.toggle_eeg_fullscreen)  # 绑定双击事件
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
            'car_ip': '192.168.4.1', # 修改默认值
            'car_port': 3333,        # 修改默认值
            'car_simulation': True,
        }

    def handle_settings_applied(self, settings):
        """处理来自设置页面的新设置的槽函数。"""
        self.log_widget.add_log("收到应用设置请求，正在重启所有服务...")
        self.stop_all_workers()
        self.start_all_workers(settings)

    def on_calibration_finished(self):
        """校准完成后调用"""
        self.log_widget.add_log("校准完成，推理系统已切换到微调后的参数。")
        # 可以在这里做一些提示，例如切换回主界面
        # self.tabs.setCurrentIndex(0)

    def start_all_workers(self, settings):
        """使用给定的设置启动所有工作线程。"""
        # 启动EEG工作线程
        self.eeg_worker = EEGReceiverWorker(host=settings['eeg_ip'], port=settings['eeg_port'])
        self.eeg_worker.data_received.connect(self.handle_new_eeg_data)
        self.eeg_worker.log_message.connect(self.log_widget.add_log)
        self.eeg_worker.start()
        # 启动推理工作线程
        self.inference_worker = InferenceWorker(
            data_buffer=self.eeg_data_buffer,
            model_handler=self.model_handler
        )
        self.inference_worker.inference_complete.connect(self.process_inference_result)
        self.inference_worker.log_message.connect(self.log_widget.add_log)
        self.inference_worker.start()
        # 启动小车控制工作线程 (修改参数)
        self.car_worker = CarControlWorker(
            ip=settings['car_ip'],
            port=settings['car_port'],
            simulation=settings['car_simulation']
        )
        self.car_worker.log_message.connect(self.log_widget.add_log)
        self.car_worker.start()

        # 更新状态UI
        conn_text = "UDP就绪" if self.car_worker.controller.connect() else "Socket错误"
        sim_text = " (模拟)" if settings['car_simulation'] else ""
        self.car_status_widget.connection_label.setText(f"{conn_text}{sim_text}")
        color = "green" if "UDP" in conn_text else "red"
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

    def toggle_eeg_fullscreen(self):
        """切换EEG波形的全屏/恢复状态的槽函数"""
        if not self.is_eeg_fullscreen:
            # 1. 保存原始布局，进入全屏模式
            self.eeg_widget.save_original_layout()
            # 先从原布局移除 eeg_widget
            if self.eeg_widget.parentWidget() and self.eeg_widget.parentWidget().layout():
                self.eeg_widget.parentWidget().layout().removeWidget(self.eeg_widget)
            # 再加入fullscreen_page
            self.fullscreen_layout.addWidget(self.eeg_widget)
            self.stack.setCurrentIndex(1)
            self.is_eeg_fullscreen = True
        else:
            # 2. 恢复到原始布局
            self.eeg_widget.restore_original_layout()
            self.stack.setCurrentIndex(0)
            self.is_eeg_fullscreen = False


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # 设置UI风格
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()