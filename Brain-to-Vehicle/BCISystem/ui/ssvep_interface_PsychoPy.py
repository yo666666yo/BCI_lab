# ssvep_interface.py

import multiprocessing
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import config


# ==========================================
# 独立进程函数：只负责运行 PsychoPy
# (注意：psychopy 的导入必须放在函数内部，防止和 PyQt 冲突)
# ==========================================
def psychopy_process_runner(cooldown_event, quit_event, cooldown_time):
    # 在独立进程中导入硬件级渲染库
    from psychopy import visual, core, event
    import numpy as np

    # 1. 创建全屏窗口 (黑色背景)
    win = visual.Window(fullscr=True, color=[-1, -1, -1], units='height')
    ifi = win.monitorFramePeriod
    if ifi is None: ifi = 1.0 / 60.0

    # 2. 定义方块位置和大小
    side_len = 0.25
    offset = 0.35
    pos_up = (0, offset)
    pos_down = (0, -offset)
    pos_left = (-offset * win.size[0] / win.size[1], 0)
    pos_right = (offset * win.size[0] / win.size[1], 0)
    pos_center = (0, 0)

    # 创建方块
    rect_up = visual.Rect(win, width=side_len, height=side_len, pos=pos_up, fillColor='white')
    rect_down = visual.Rect(win, width=side_len, height=side_len, pos=pos_down, fillColor='white')
    rect_left = visual.Rect(win, width=side_len, height=side_len, pos=pos_left, fillColor='white')
    rect_right = visual.Rect(win, width=side_len, height=side_len, pos=pos_right, fillColor='white')
    rect_center = visual.Rect(win, width=side_len * 1.2, height=side_len * 1.2, pos=pos_center, fillColor='white')

    # 箭头文字
    txt_up = visual.TextStim(win, text='^\n(10 Hz)', pos=pos_up, height=0.05, color='black', bold=True)
    txt_down = visual.TextStim(win, text='v\n(15 Hz)', pos=pos_down, height=0.05, color='black', bold=True)
    txt_left = visual.TextStim(win, text='<\n(12 Hz)', pos=pos_left, height=0.05, color='black', bold=True)
    txt_right = visual.TextStim(win, text='>\n(7.5 Hz)', pos=pos_right, height=0.05, color='black', bold=True)
    txt_center = visual.TextStim(win, text='STOP\n(6.0 Hz)', pos=pos_center, height=0.05, color='black', bold=True)

    # 休息提示
    txt_rest = visual.TextStim(win, text='Command Sent!\nCooldown...', pos=(0, 0), height=0.1, color='green')

    def get_color(freq, frame, ifi):
        # 方波闪烁
        lum = np.sin(2 * np.pi * freq * frame * ifi)
        val = np.sign(lum)
        return [val, val, val]

    frame_num = 0
    event.clearEvents()

    # 3. 主渲染循环
    while not quit_event.is_set():
        # 检查是否按下了 ESC 键退出
        keys = event.getKeys()
        if 'escape' in keys:
            break

        # 检查是否收到了来自主进程的“冷却”信号
        if cooldown_event.is_set():
            # 渲染休息画面 (黑屏+绿字)
            win.color = [-1, -1, -1]
            txt_rest.draw()
            win.flip()

            # 阻塞等待冷却时间结束
            core.wait(cooldown_time)

            # 冷却结束，清除信号，恢复状态
            cooldown_event.clear()
            frame_num = 0
            event.clearEvents()
            continue

        # 正常闪烁计算 (对应 config.py 里的 10, 15, 12, 7.5, 6)
        rect_up.fillColor = get_color(10.0, frame_num, ifi)
        rect_down.fillColor = get_color(15.0, frame_num, ifi)
        rect_left.fillColor = get_color(12.0, frame_num, ifi)
        rect_right.fillColor = get_color(7.5, frame_num, ifi)
        rect_center.fillColor = get_color(6.0, frame_num, ifi)

        # 绘制所有元素
        rect_up.draw();
        txt_up.draw()
        rect_down.draw();
        txt_down.draw()
        rect_left.draw();
        txt_left.draw()
        rect_right.draw();
        txt_right.draw()
        rect_center.draw();
        txt_center.draw()

        win.flip()
        frame_num += 1

    win.close()
    core.quit()


# ==========================================
# PyQt 界面：用于控制启动/停止 PsychoPy 进程
# ==========================================
class SSVEPInterface(QWidget):
    def __init__(self):
        super().__init__()
        # 进程间通信对象 (IPC)
        self.cooldown_event = multiprocessing.Event()
        self.quit_event = multiprocessing.Event()
        self.psy_process = None
        self.is_running = False

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("PsychoPy SSVEP 独立渲染引擎")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = QLabel("点击下方按钮将弹出全屏高精度闪烁窗口。\n控制过程中，按键盘 [ESC] 键即可退出全屏。")
        desc.setFont(QFont("Arial", 14))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_start = QPushButton("启动全屏 SSVEP")
        self.btn_start.setFixedSize(300, 80)
        self.btn_start.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; border-radius: 10px;")
        self.btn_start.clicked.connect(self.toggle_process)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(desc)
        layout.addSpacing(50)
        layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignCenter)

    def toggle_process(self):
        if self.psy_process is None or not self.psy_process.is_alive():
            self.start_psychopy()
        else:
            self.stop_psychopy()

    def start_psychopy(self):
        # 启动外部进程
        self.quit_event.clear()
        self.cooldown_event.clear()

        self.psy_process = multiprocessing.Process(
            target=psychopy_process_runner,
            args=(self.cooldown_event, self.quit_event, config.SSVEP_COOLDOWN_TIME)
        )
        self.psy_process.start()

        self.is_running = True
        self.btn_start.setText("停止 SSVEP 进程")
        self.btn_start.setStyleSheet("background-color: #dc3545; color: white; border-radius: 10px;")

    def stop_psychopy(self):
        # 安全停止外部进程
        if self.psy_process and self.psy_process.is_alive():
            self.quit_event.set()
            self.psy_process.join(timeout=2)
            if self.psy_process.is_alive():
                self.psy_process.terminate()  # 强制杀死

        self.is_running = False
        self.btn_start.setText("启动全屏 SSVEP")
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; border-radius: 10px;")