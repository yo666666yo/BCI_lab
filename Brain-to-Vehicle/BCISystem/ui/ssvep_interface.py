from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor


class SSVEPStimulusBox(QLabel):
    """
    单个闪烁刺激块
    支持点击事件，可兼作按钮使用
    """
    clicked = pyqtSignal()  # 新增点击信号

    def __init__(self, text, frequency, size=150):
        super().__init__(text)
        self.frequency = frequency
        self.original_text = text
        self.is_on = True
        self.is_flashing_active = False  # 是否处于闪烁模式

        # 基础样式：黑白高对比度
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Arial", 40, QFont.Weight.Bold))
        self.setFixedSize(size, size)

        self.default_style = "background-color: white; color: black; border: 2px solid #333;"
        self.off_style = "background-color: black; color: white; border: 2px solid #333;"

        # 初始状态：全黑
        self.setStyleSheet(self.off_style)

        # 闪烁定时器
        self.timer = QTimer()
        if frequency > 0:
            # 计算翻转间隔 (ms) = 1000 / (2 * freq)
            interval = int(1000 / (2 * self.frequency))
            self.timer.timeout.connect(self.toggle)
            self.interval = interval
        else:
            self.interval = 0

    def mousePressEvent(self, event):
        """重写鼠标点击事件"""
        self.clicked.emit()
        super().mousePressEvent(event)

    def toggle(self):
        self.is_on = not self.is_on
        if self.is_on:
            self.setStyleSheet(self.default_style)
        else:
            self.setStyleSheet(self.off_style)

    def start_flashing(self):
        if self.interval > 0:
            self.is_flashing_active = True
            self.timer.start(self.interval)
            self.is_on = True
            self.setStyleSheet(self.default_style)

    def stop_flashing(self):
        self.is_flashing_active = False
        self.timer.stop()
        # 停止时恢复黑色背景，不刺眼
        self.setStyleSheet(self.off_style)


class SSVEPInterface(QWidget):
    """
    SSVEP 主界面
    中间方块现在既是【开始开关】，也是【停止指令(Stop)】的闪烁源
    """

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.init_ui()

    def init_ui(self):
        # 全屏黑色背景，减少视觉干扰
        self.setStyleSheet("background-color: black;")

        layout = QGridLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        # --- 配置刺激块 ---
        # 频率分配建议 (避免倍频干扰):
        # Up(前): 10Hz
        # Left(左): 12Hz
        # Right(右): 8Hz
        # Down(后): 15Hz
        # Center(停): 6Hz

        self.box_up = SSVEPStimulusBox("^", 10.0)
        self.box_left = SSVEPStimulusBox("<", 12.0)
        self.box_right = SSVEPStimulusBox(">", 7.5)
        self.box_down = SSVEPStimulusBox("v", 15.0)

        # 中间方块：稍微大一点，默认显示 Start
        self.box_center = SSVEPStimulusBox("START", 6.0, size=200)  # 13Hz 用于停止
        self.box_center.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        # 初始样式：灰色，显眼一点
        self.box_center.setStyleSheet("background-color: #333; color: yellow; border: 2px solid yellow;")
        self.box_center.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # 连接点击信号
        self.box_center.clicked.connect(self.toggle_test)

        # --- 布局 (3x3 Grid) ---
        # (Row, Col, Alignment)
        layout.addWidget(self.box_up, 0, 1, Qt.AlignmentFlag.AlignCenter)  # 上
        layout.addWidget(self.box_left, 1, 0, Qt.AlignmentFlag.AlignCenter)  # 左
        layout.addWidget(self.box_center, 1, 1, Qt.AlignmentFlag.AlignCenter)  # 中 (停止/开关)
        layout.addWidget(self.box_right, 1, 2, Qt.AlignmentFlag.AlignCenter)  # 右
        layout.addWidget(self.box_down, 2, 1, Qt.AlignmentFlag.AlignCenter)  # 下

        # 调整拉伸比例，保持居中
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        # 放在列表里方便批量操作
        self.stimuli = [self.box_up, self.box_left, self.box_right, self.box_down, self.box_center]

    def toggle_test(self):
        if self.is_running:
            self.stop_stimulation()
        else:
            self.start_stimulation()

    def start_stimulation(self):
        self.is_running = True

        # 修改中间方块的文字为 STOP，并开始闪烁
        self.box_center.setText("STOP")
        # 恢复标准 SSVEP 样式 (黑白)
        self.box_center.default_style = "background-color: white; color: black; border: none;"
        self.box_center.off_style = "background-color: black; color: white; border: none;"

        for box in self.stimuli:
            box.start_flashing()

    def stop_stimulation(self):
        self.is_running = False

        for box in self.stimuli:
            box.stop_flashing()

        # 恢复中间方块为“按钮”样式
        self.box_center.setText("CLICK TO\nSTART")
        self.box_center.setStyleSheet("background-color: #333; color: yellow; border: 2px solid yellow;")