# run_experiment.py
import threading
import time
from collections import deque
import numpy as np
from psychopy import visual, core, event

import config
from core.ssvep_utils import SSVEPHandler
from core.eeg_utils import EEGPreprocessor
from core.hardware import CarController
from ssvep21_receiver import Ssvep21ChannelReceiver  # 你的数据接收器

# ==========================================
# 1. 后台脑电接收线程 (复用你的处理逻辑)
# ==========================================
eeg_data_buffer = deque(maxlen=1000)  # 约 9 秒的缓冲区
is_running = True


def eeg_worker_thread():
    print("正在连接 EEG 设备...")
    receiver = Ssvep21ChannelReceiver(config.EEG_IP, config.EEG_PORT)
    if not receiver.connect():
        print("!!! EEG 连接失败 !!!")
        return
    receiver.start_receiving()
    print("EEG 连接成功，开始后台接收并滤波...")

    preprocessor = EEGPreprocessor(fs=config.SAMPLE_RATE, num_channels=21)
    last_ts = 0

    while is_running:
        latest = receiver.get_latest_data()
        if latest and latest.timestamp > last_ts:
            raw_channels = np.array(latest.channels)
            filtered = preprocessor.process_sample(raw_channels)
            eeg_data_buffer.append(filtered)
            last_ts = latest.timestamp
        time.sleep(0.002)  # 释放 CPU 资源

    receiver.disconnect()
    print("EEG 后台线程已安全退出。")


# 启动 EEG 采集线程
eeg_thread = threading.Thread(target=eeg_worker_thread, daemon=True)
eeg_thread.start()

# ==========================================
# 2. 核心控制逻辑初始化
# ==========================================
print("初始化小车控制器与 AI 模型...")
car_controller = CarController(config.CAR_IP, config.CAR_PORT, config.IS_SIMULATION)
car_controller.connect()

# 确保这里的频率和你的 config 一致:[6.0, 10.0, 15.0, 12.0, 7.5]
ssvep_handler = SSVEPHandler(
    sample_rate=config.SAMPLE_RATE,
    window_len_sec=2.0,  # 2秒的判定窗口
    target_freqs=config.SSVEP_FREQS
)

# ==========================================
# 3. PsychoPy 视觉界面与闪烁循环
# ==========================================
print("启动 PsychoPy 界面...")
win = visual.Window(fullscr=False, size=(800, 600), color=[-1, -1, -1], units='height')
ifi = win.monitorFramePeriod
if ifi is None: ifi = 1.0 / 60.0

# 定义方块
side_len = 0.2
offset = 0.35
aspect_ratio = win.size[0] / win.size[1]

pos_center = (0, 0)
pos_up = (0, offset)
pos_down = (0, -offset)
pos_left = (-offset * aspect_ratio, 0)
pos_right = (offset * aspect_ratio, 0)

rect_center = visual.Rect(win, width=side_len, height=side_len, pos=pos_center, fillColor='white')
rect_up = visual.Rect(win, width=side_len, height=side_len, pos=pos_up, fillColor='white')
rect_down = visual.Rect(win, width=side_len, height=side_len, pos=pos_down, fillColor='white')
rect_left = visual.Rect(win, width=side_len, height=side_len, pos=pos_left, fillColor='white')
rect_right = visual.Rect(win, width=side_len, height=side_len, pos=pos_right, fillColor='white')

# 定义文字
txt_center = visual.TextStim(win, text='STOP', pos=pos_center, height=0.08, color='black', bold=True)
txt_up = visual.TextStim(win, text='^', pos=pos_up, height=0.1, color='black', bold=True)
txt_down = visual.TextStim(win, text='v', pos=pos_down, height=0.1, color='black', bold=True)
txt_left = visual.TextStim(win, text='<', pos=pos_left, height=0.1, color='black', bold=True)
txt_right = visual.TextStim(win, text='>', pos=pos_right, height=0.1, color='black', bold=True)


def get_flicker_color(freq, current_time):
    """根据真实物理时间计算当前的颜色(-1 黑, 1 白)"""
    # 直接使用绝对时间计算正弦波，绝对不会因为掉帧而错乱
    lum = np.sin(2 * np.pi * freq * current_time)

    # 强行二值化：大于0为纯白(1)，小于等于0为纯黑(-1)
    val = 1 if lum > 0 else -1
    return [val, val, val]


# 状态控制变量
frame_num = 0
is_cooldown = False
cooldown_start_time = 0

print(">>> 一切就绪！请注视屏幕方块进行控制。(按 ESC 退出) <<<")

# 清除按键缓存
event.clearEvents()

while True:
    # 1. 检查退出按键
    keys = event.getKeys()
    if 'escape' in keys:
        break

    current_time = core.getTime()  # 获取当前真实的物理时间（秒）

    # 2. 状态判断：冷却期 vs 刺激期
    if is_cooldown:
        # 【冷却期】：全黑，休息，不进行推理 (注意这里改成了 setFillColor)
        rect_center.setFillColor([-1, -1, -1])
        rect_up.setFillColor([-1, -1, -1])
        rect_down.setFillColor([-1, -1, -1])
        rect_left.setFillColor([-1, -1, -1])
        rect_right.setFillColor([-1, -1, -1])

        # 检查冷却是否结束 (3秒)
        if current_time - cooldown_start_time > config.SSVEP_COOLDOWN_TIME:
            is_cooldown = False
            eeg_data_buffer.clear()  # ★ 核心：清空残影数据
            print(">>> 冷却结束，恢复脑电闪烁。")
    else:
        # 【刺激期】：进行闪烁 (传入当前真实时间)
        rect_center.setFillColor(get_flicker_color(6.0, current_time))  # Stop
        rect_up.setFillColor(get_flicker_color(10.0, current_time))  # Forward
        rect_down.setFillColor(get_flicker_color(15.0, current_time))  # Backward
        rect_left.setFillColor(get_flicker_color(12.0, current_time))  # Left
        rect_right.setFillColor(get_flicker_color(7.5, current_time))  # Right

        # ---------------------------------------------------
        # 推理逻辑 (为了不卡顿 UI，每 30 帧 ≈ 0.5秒 做一次推理)
        # ---------------------------------------------------
        if frame_num % 30 == 0 and len(eeg_data_buffer) >= ssvep_handler.n_samples:
            # 提取最新的 2 秒数据并转置为 (Channels, Time) 供 CCA 使用
            recent_data = np.array(list(eeg_data_buffer)[-ssvep_handler.n_samples:]).T

            best_idx, corr = ssvep_handler.classify(recent_data)

            # 如果突破置信度阈值
            if corr > config.SSVEP_THRESHOLD:
                label_idx = config.SSVEP_LABELS[best_idx]
                command = config.CMD_MAP.get(label_idx, 'stop')

                print(f"触发指令:[{command.upper()}] | 频率: {config.SSVEP_FREQS[best_idx]}Hz | 置信度: {corr:.3f}")

                car_controller.send_command(command)

                # 进入冷却状态
                is_cooldown = True
                cooldown_start_time = core.getTime()

    # 3. 绘制元素并刷新屏幕
    rect_center.draw();
    txt_center.draw()
    rect_up.draw();
    txt_up.draw()
    rect_down.draw();
    txt_down.draw()
    rect_left.draw();
    txt_left.draw()
    rect_right.draw();
    txt_right.draw()

    win.flip()
    frame_num += 1