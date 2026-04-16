from psychopy import visual, core, event
import numpy as np

# --- 1. 初始化设置 ---
# units='height' 表示以屏幕高度为单位。
# 屏幕顶部是 0.5，底部是 -0.5。这样无论什么分辨率，位置都固定。
win = visual.Window(fullscr=True, color=[-1, -1, -1], units='height')
# 注意：PsychoPy颜色通常是 -1(黑) 到 1(白)

# 获取刷新率 (用于计算闪烁)
ifi = win.monitorFramePeriod
if ifi is None: ifi = 1.0 / 60.0  # 默认 60Hz

# --- 2. 定义参数 (使用相对单位，无需计算像素) ---
# 方块大小 (屏幕高度的 20%)
side_len = 0.2
# 偏移量 (方块中心距离屏幕中心的距离)
offset = 0.35

# 定义四个方块的位置 (x, y)
pos_up = (0, offset)
pos_down = (0, -offset)
pos_left = (-offset * win.size[0] / win.size[1], 0)  # 修正宽屏比例
pos_right = (offset * win.size[0] / win.size[1], 0)

# 创建方块对象 (初始化时只需定义一次)
rect_up = visual.Rect(win, width=side_len, height=side_len, pos=pos_up, fillColor='white')
rect_down = visual.Rect(win, width=side_len, height=side_len, pos=pos_down, fillColor='white')
rect_left = visual.Rect(win, width=side_len, height=side_len, pos=pos_left, fillColor='white')
rect_right = visual.Rect(win, width=side_len, height=side_len, pos=pos_right, fillColor='white')

# 创建箭头文字对象 (自动居中)
# height=0.1 设置字号
txt_up = visual.TextStim(win, text='^', pos=pos_up, height=0.1, color='black', bold=True)
txt_down = visual.TextStim(win, text='v', pos=pos_down, height=0.1, color='black', bold=True)
txt_left = visual.TextStim(win, text='<', pos=pos_left, height=0.1, color='black', bold=True)
txt_right = visual.TextStim(win, text='>', pos=pos_right, height=0.1, color='black', bold=True)

# 创建引导语背景和文字
intro_bg = visual.Rect(win, width=0.8, height=0.4, fillColor=[0, 0, 0], lineColor='white')  # 灰色背景
intro_txt = visual.TextStim(win, text="Press Any Key\nTo Start Test",
                            color='gold', height=0.08, alignText='center')


# --- 3. 辅助函数：模拟你的 flicker 函数 ---
def get_flicker_color(freq, frame_num, phase, ifi):
    # SSVEP 公式: Luminance = (1 + sin(2*pi*f*t + phase)) / 2
    # 结果 0~1，映射到 PsychoPy 的颜色空间 -1~1
    t = frame_num * ifi
    lum = np.sin(2 * np.pi * freq * t + phase)
    # 这里简单返回黑白闪烁，lum > 0 为白(1)，lum < 0 为黑(-1)
    # 如果你是模拟正弦光栅变化，可以直接用 lum
    # 这是一个方波闪烁示例 (比较强烈)，如果是正弦波请去掉 np.sign
    val = np.sign(lum)
    return [val, val, val]


# --- 4. 引导语阶段 ---
while not event.getKeys():
    # 绘制静态背景
    rect_up.draw()
    rect_down.draw()
    rect_left.draw()
    rect_right.draw()

    # 绘制静态箭头
    txt_up.draw()
    txt_down.draw()
    txt_left.draw()
    txt_right.draw()

    # 绘制引导
    intro_bg.draw()
    intro_txt.draw()

    win.flip()

# --- 5. 正式测试 (闪烁阶段) ---
frame_num = 0
run_test = True

# 清除之前的按键缓存
event.clearEvents()

while run_test:
    # 检查退出按键
    keys = event.getKeys()
    if keys:
        run_test = False  # 按键退出

    # 计算颜色 (频率可自己改)
    # 注意：Python 的 sin 是弧度制，和你 MATLAB 代码一致
    rect_up.fillColor = get_flicker_color(8.0, frame_num, 0.0 * np.pi, ifi)
    rect_down.fillColor = get_flicker_color(10.0, frame_num, 0.5 * np.pi, ifi)
    rect_left.fillColor = get_flicker_color(12.0, frame_num, 1.0 * np.pi, ifi)
    rect_right.fillColor = get_flicker_color(15.8, frame_num, 1.5 * np.pi, ifi)

    # 绘制方块
    rect_up.draw()
    rect_down.draw()
    rect_left.draw()
    rect_right.draw()

    # 绘制箭头 (文字保持黑色)
    txt_up.draw()
    txt_down.draw()
    txt_left.draw()
    txt_right.draw()

    win.flip()
    frame_num += 1

win.close()
core.quit()