# ui/widgets.py
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox, QCheckBox,
                             QSplitter, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QThread  # QThread 用于后台任务
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
# 导入线程安全的队列
import queue
from core.workers import OnlineTrainerWorker
from games.maze_calib import MazeCalibrationGame
from ssvep21_receiver import Ssvep21ChannelReceiver
import config

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




class CalibrationWidget(QWidget):
    """
    在线模型校准界面 (UI风格重构版)
    风格：深色扁平化，与主界面保持一致。
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
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # --- 1. 顶部标题 ---
        # 风格参考：主界面左上角的标题
        title = QLabel("在线模型校准")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        # --- 2. 操作说明区域 (仿照“小车状态监控”的风格) ---
        info_group = QGroupBox("操作指南")
        info_group.setFont(QFont("Microsoft YaHei", 10))
        info_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                color: #e0e0e0;
                background-color: #2b2b2b; /* 深色背景卡片 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #aaa;
            }
        """)

        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(20, 25, 20, 20)

        desc_label = QLabel(
            "1. 点击下方的蓝色按钮启动迷宫游戏。<br>"
            "2. 使用键盘方向键控制红点移动，系统将自动更新模型。<br>"
            "3. 游戏结束后，校准即刻完成。"
        )
        desc_label.setFont(QFont("Microsoft YaHei", 10))
        desc_label.setStyleSheet("color: #cccccc;")
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)

        layout.addWidget(info_group)

        # --- 3. 核心交互按钮 ---
        # 风格参考：设置界面的“应用设置”按钮 (蓝色)
        self.start_btn = QPushButton("启动校准程序")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))

        # 默认蓝色风格
        self.btn_style_normal = """
            QPushButton {
                background-color: #007BFF; /* 科技蓝 */
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004494;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """
        # 运行中绿色风格 (参考主界面的连接成功状态)
        self.btn_style_running = """
            QPushButton {
                background-color: #28a745; /* 成功绿 */
                color: white;
                border: none;
                border-radius: 5px;
            }
        """

        self.start_btn.setStyleSheet(self.btn_style_normal)
        self.start_btn.clicked.connect(self.launch_maze_game)
        layout.addWidget(self.start_btn)

        # --- 4. 底部日志区域 ---
        # 风格参考：主界面底部的“系统日志”
        log_label = QLabel("校准日志")
        log_label.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        log_label.setStyleSheet("color: #e0e0e0; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        # 这里的样式完全复刻主界面的日志样式
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; /* 比背景稍黑 */
                border: 1px solid #444;
                border-radius: 4px;
                color: #dcdcdc;            /* 浅灰色字体，不刺眼 */
                font-family: "Consolas", "Microsoft YaHei";
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.log_text.setReadOnly(True)
        # 让日志区域占据剩余空间
        layout.addWidget(self.log_text, stretch=1)

    def log(self, msg):
        # 简单的追加模式
        self.log_text.append(f">> {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def launch_maze_game(self):
        if not self.data_buffer or len(self.data_buffer) < 100:
            self.log("错误：数据流不稳定或数据不足，请检查 EEG 设备连接。")
            return

        self.log("正在初始化迷宫游戏...")
        self.start_btn.setEnabled(False)
        self.start_btn.setText("校准进行中... (请在游戏窗口操作)")
        # 切换为绿色，表示正在运行
        self.start_btn.setStyleSheet(self.btn_style_running)

        # 1. 启动后台训练线程
        self.trainer_thread = OnlineTrainerWorker(self.model_handler, self.training_queue)
        self.trainer_thread.log_message.connect(self.log)
        self.trainer_thread.start()

        # 2. 启动游戏
        try:
            self.log("正在启动游戏窗口...")
            # 必须传入 training_queue 以开启在线模式
            game = MazeCalibrationGame(self.data_buffer,
                                       input_window=self.model_handler.input_window,
                                       training_queue=self.training_queue)

            # 阻塞运行，直到窗口关闭
            game.run_game()

            self.log("游戏结束。模型参数已实时更新完毕。")

        except Exception as e:
            self.log(f"严重错误: {e}")
            import traceback
            traceback.print_exc()

        # 3. 清理
        self.trainer_thread.stop()

        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.start_btn.setText("启动校准程序")
        self.start_btn.setStyleSheet(self.btn_style_normal)  # 恢复蓝色

        self.calibration_finished.emit()
        self.log("系统就绪。参数已同步。")