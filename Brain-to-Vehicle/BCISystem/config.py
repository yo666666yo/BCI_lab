# config.py

# --- EEG 连接设置 ---
EEG_IP = "127.0.0.1"
EEG_PORT = 8712
SAMPLE_RATE = 110      # 采样率
INPUT_WINDOW = 220     # 模型输入窗口 (2秒)

# --- 小车连接设置 ---
CAR_IP = "192.168.4.1"
CAR_PORT = 3333
IS_SIMULATION = True   # 是否模拟模式

# --- 模型设置 ---
MODEL_PATH = "pretrained_model.pth"
CHANNELS = 21
CLASSES = 5

# --- 映射关系 ---
# 0:Stop, 1:Forward, 2:Backward, 3:Left, 4:Right
CMD_MAP = {0: 'stop', 1: 'forward', 2: 'backward', 3: 'left', 4: 'right'}

# --- SSVEP 配置 ---
# 对应: 停(Center), 前(Up), 后(Down), 左(Left), 右(Right)
# 注意：频率必须和界面闪烁的频率完全一致！
# 假设界面 SSVEPStimulusBox 设定的频率是:
# Center(无), Up(10), Left(12), Right(8), Down(15)
# 我们需要定义一个列表，顺序要和 label id 对应
# 0:Stop, 1:Forward, 2:Backward, 3:Left, 4:Right

# 因为 Stop 通常是不闪烁或看中间，CCA 很难识别 "不看"，
# 所以通常设定：如果所有频率的相关系数都低于阈值，就是 Stop。
# 这里的列表只包含有频率的指令：[1, 2, 3, 4]
# 假设我们把频率分配给: Forward(10), Backward(15), Left(12), Right(8), Stop(6.0)
SSVEP_FREQS = [6.0, 10.0, 15.0, 12.0, 7.5]
SSVEP_LABELS = [0, 1, 2, 3, 4] # 对应 CMD_MAP 的键

SSVEP_THRESHOLD = 0.5 # CCA 相关系数阈值，低于这个认为在休息(Stop)\
SSVEP_COOLDOWN_TIME = 5.0
