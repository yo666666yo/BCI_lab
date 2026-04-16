# ui/main_window.py

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox, QCheckBox,
                             QSplitter, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
import config
from core.eeg_utils import EEGModelHandler
from core.workers import EEGReceiverWorker, InferenceWorker, CarControlWorker
from ui.widgets import (EEGWaveformWidget, CalibrationWidget, CarStatusWidget,
                        LogWidget, ControlButtonsWidget, SettingsWidget, InferenceResultWidget)
from ui.ssvep_interface import SSVEPInterface
from collections import deque




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 数据中心
        self.eeg_data_buffer = deque(maxlen=2000)
        self.model_handler = EEGModelHandler(
            model_path=config.MODEL_PATH,
            n_chan=config.CHANNELS,
            n_classes=config.CLASSES,
            input_window=config.INPUT_WINDOW
        )

        # 线程占位
        self.eeg_worker = None
        self.inference_worker = None
        self.car_worker = None

        self.init_ui()

        # 启动默认配置
       # self.start_all_workers()

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
        self.ssvep_tab = SSVEPInterface()
        self.tabs.addTab(self.ssvep_tab, "SSVEP控制")

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

        # 监听 Tab 切换事件
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        """当用户切换选项卡时触发"""
        tab_name = self.tabs.tabText(index)

        if self.inference_worker:
            if "SSVEP" in tab_name:
                # 如果切到了 SSVEP 界面
                self.inference_worker.set_mode('SSVEP')
                self.log_widget.add_log(">>> 系统切换至 SSVEP 视觉控制模式")
            elif "主界面" in tab_name:
                # 如果切回了主界面 (MI)
                self.inference_worker.set_mode('MI')
                self.log_widget.add_log(">>> 系统切换至 MI 运动想象模式")
            else:
                # 其他界面 (如设置、校准)，保持 MI 或暂停推理都可以
                pass

    def get_default_settings(self):
        """返回一个包含默认设置的字典。"""
        return {
            'eeg_ip': '127.0.0.1',
            'eeg_port': 8712,
            'car_ip': '192.168.4.1',  # 修改默认值
            'car_port': 3333,  # 修改默认值
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

    def start_all_workers(self, settings=None):
        """
        启动所有后台工作线程 (EEG接收, 推理, 小车控制)
        """
        # 1. 加载配置 (如果未提供，读取 config.py 默认值)
        if settings is None:
            settings = {
                'eeg_ip': config.EEG_IP,
                'eeg_port': config.EEG_PORT,
                'car_ip': config.CAR_IP,
                'car_port': config.CAR_PORT,
                'car_simulation': config.IS_SIMULATION,
            }

        # 2. 启动 EEG 接收线程
        # 负责连接 EEG 设备并进行实时滤波
        self.eeg_worker = EEGReceiverWorker(
            host=settings['eeg_ip'],
            port=settings['eeg_port'],
            fs=config.SAMPLE_RATE
        )
        self.eeg_worker.data_received.connect(self.handle_new_eeg_data)
        if hasattr(self, 'log_widget'):
            self.eeg_worker.log_message.connect(self.log_widget.add_log)
        self.eeg_worker.start()

        # 3. 启动 AI 推理线程
        # 负责从缓冲区读取数据并进行模型预测
        self.inference_worker = InferenceWorker(
            data_buffer=self.eeg_data_buffer,
            model_handler=self.model_handler
        )
        if hasattr(self, 'process_inference_result'):
            self.inference_worker.inference_complete.connect(self.process_inference_result)
        if hasattr(self, 'log_widget'):
            self.inference_worker.log_message.connect(self.log_widget.add_log)
        self.inference_worker.start()

        # 4. 启动小车控制线程 (UDP)
        # 负责将推理结果转换为 UDP 指令发送给小车
        self.car_worker = CarControlWorker(
            ip=settings['car_ip'],
            port=settings['car_port'],
            simulation=settings['car_simulation']
        )
        if hasattr(self, 'log_widget'):
            self.car_worker.log_message.connect(self.log_widget.add_log)
        self.car_worker.start()

        # 5. 更新状态栏连接状态
        if hasattr(self, 'car_status_widget'):
            is_connected = self.car_worker.controller.connect()
            conn_text = "UDP就绪" if is_connected else "Socket错误"
            sim_text = " (模拟)" if settings['car_simulation'] else ""

            self.car_status_widget.connection_label.setText(f"{conn_text}{sim_text}")
            color = "green" if "UDP" in conn_text else "red"
            self.car_status_widget.connection_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        self.log_widget.add_log(f"系统服务已启动。配置: {settings}")
    def stop_all_workers(self):
        """安全地停止所有正在运行的工作线程。"""
        self.log_widget.add_log("正在停止所有服务...")
        if self.eeg_worker and self.eeg_worker.isRunning():
            self.eeg_worker.stop()
            self.eeg_worker.wait()  # 等待线程结束
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

        if self.inference_worker and self.inference_worker.mode == 'SSVEP':
            self.trigger_ssvep_cooldown()

    def trigger_ssvep_cooldown(self):
        """触发 SSVEP 冷却"""
        self.log_widget.add_log(f"指令已发送，触发冷却，暂停接收 {config.SSVEP_COOLDOWN_TIME} 秒...")

        # 1. 暂停推理
        if self.inference_worker:
            self.inference_worker.set_cooldown(True)

        # 2. 暂停界面闪烁
        if hasattr(self, 'ssvep_tab') and self.ssvep_tab.is_running:
            for box in self.ssvep_tab.stimuli:
                box.stop_flashing()

        # 3. 启动定时器，3秒后恢复
        QTimer.singleShot(int(config.SSVEP_COOLDOWN_TIME * 1000), self.end_ssvep_cooldown)

    def end_ssvep_cooldown(self):
        """结束 SSVEP 冷却"""
        self.log_widget.add_log("冷却结束，恢复 SSVEP 闪烁和脑电识别。")

        # 1. 恢复推理
        if self.inference_worker:
            self.inference_worker.set_cooldown(False)

        # 2. 恢复界面闪烁 (判断用户是否还没有点击停止)
        if hasattr(self, 'ssvep_tab') and self.ssvep_tab.is_running:
            for box in self.ssvep_tab.stimuli:
                box.start_flashing()

    def create_main_tab(self):
        """创建主界面的布局和组件。"""
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        splitter = QSplitter(Qt.Orientation.Vertical)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        self.eeg_widget = EEGWaveformWidget()
        top_layout.addWidget(self.eeg_widget, 60)  # 60% 宽度
        self.inference_widget = InferenceResultWidget()
        top_layout.addWidget(self.inference_widget, 40)  # 40% 宽度
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