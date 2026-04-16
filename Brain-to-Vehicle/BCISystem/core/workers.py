# core/workers.py
from PyQt6.QtCore import QThread, pyqtSignal
from core.hardware import CarController
from core.eeg_utils import EEGPreprocessor
import config
# 导入EEG接收器模块
from ssvep21_receiver import Ssvep21ChannelReceiver
# 导入双端队列，用作缓冲区
import numpy as np
# 导入线程安全的队列
import queue
from core.ssvep_utils import SSVEPHandler

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

class InferenceWorker(QThread):
    inference_complete = pyqtSignal(str, float)
    log_message = pyqtSignal(str)

    def __init__(self, data_buffer, model_handler):
        super().__init__()
        self.data_buffer = data_buffer
        self.model_handler = model_handler  # MI 用的
        self._is_running = False
        self.is_cooldown = False

        # --- SSVEP 初始化 ---
        # 窗口长度建议比 MI 长一点，例如 2秒-3秒，越长 CCA 越准
        self.ssvep_handler = SSVEPHandler(
            sample_rate=config.SAMPLE_RATE,
            window_len_sec=2.0,
            target_freqs=config.SSVEP_FREQS
        )
        self.mode = 'MI'  # 默认模式 'MI' 或 'SSVEP'

    def set_mode(self, mode):
        """切换模式: 'MI' 或 'SSVEP'"""
        self.mode = mode
        self.log_message.emit(f"推理模式已切换为: {mode}")

    def set_cooldown(self, state: bool):
        self.is_cooldown = state
        if state:
            self.data_buffer.clear()

    def run(self):
        self._is_running = True
        self.log_message.emit("AI 推理线程启动...")

        while self._is_running:
            # 根据模式选择不同的处理间隔
            # MI 反应快，可以 200ms 一次；SSVEP 窗口长，可以 500ms 一次
            interval = 0.2 if self.mode == 'MI' else 0.5
            self.msleep(int(interval * 1000))

            if self.mode == 'MI':
                self.run_mi_inference()
            else:
                self.run_ssvep_inference()

    def run_mi_inference(self):
        # ... (这里是你原来的 run_inference 代码，逻辑不变) ...
        # 注意：为了代码整洁，把原来的 run_inference 里的内容搬到这里
        required_len = config.INPUT_WINDOW
        if len(self.data_buffer) < required_len: return
        recent_data = list(self.data_buffer)[-required_len:]

        try:
            idx, conf = self.model_handler.predict(recent_data)
            command = config.CMD_MAP.get(idx, 'stop')
            if conf > 0.5:
                self.inference_complete.emit(command, conf)
        except Exception as e:
            print(f"MI Error: {e}")

    def run_ssvep_inference(self):
        """SSVEP 推理逻辑"""
        # SSVEP 需要更长的数据，例如 2秒 (2 * 110 = 220点)
        if self.is_cooldown:
            return

        n_samples = self.ssvep_handler.n_samples
        if len(self.data_buffer) < n_samples: return

        # 获取最新数据
        # 优化：SSVEP 主要看枕叶 (O1, Oz, O2)，如果你知道它们是第几个通道，
        # 可以在这里切片，例如 recent_data = recent_data[:, [0, 1, 2]]
        # 目前假设用所有通道跑 CCA (CCA 会自动给无关通道低权重，所以全放进去也没问题)
        recent_data = np.array(list(self.data_buffer)[-n_samples:]).T  # (Chan, Time)

        try:
            # 调用 CCA
            best_idx, corr = self.ssvep_handler.classify(recent_data)

            if corr > config.SSVEP_THRESHOLD:
                label_idx = config.SSVEP_LABELS[best_idx]  # 映射回 1,2,3,4
                command = config.CMD_MAP.get(label_idx, 'stop')
                self.inference_complete.emit(command, corr)
                # 可选：打印高置信度日志
                # self.log_message.emit(f"SSVEP Det: {command} (Corr: {corr:.2f})")
            else:
                # 低于阈值认为在休息
                # self.inference_complete.emit('stop', corr)
                pass  # 也可以不发 stop，保持惯性

        except Exception as e:
            print(f"SSVEP Error: {e}")

    def stop(self):
        """安全停止推理线程"""
        self._is_running = False
        self.wait()

class CarControlWorker(QThread):
    """用于向小车发送命令的工作线程 (UDP版)。"""
    log_message = pyqtSignal(str)

    def __init__(self, ip=config.CAR_IP, port=config.CAR_PORT, simulation=config.IS_SIMULATION):
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
                # 阻塞等待队列中的数据，超时n秒，检查停止标志
                data, label = self.training_queue.get(timeout=1)

                # 单步训练
                loss = self.model_handler.train_one_step(data, label)

                # 发送日志
                self.log_message.emit(f"在线学习中... Label:{label} | Loss:{loss:.4f}")

                self.training_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"在线训练出错: {e}")

    def stop(self):
        self._is_running = False
        self.wait()
