import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os



class EEGPreprocessor:
    """
    实时EEG数据预处理器
    功能：
    1. 50Hz 陷波滤波器 (去除工频干扰)
    2. 4-40Hz 带通滤波器 (保留运动想象/SSVEP常用频段)
    """

    def __init__(self, fs=250, num_channels=21, notch_freq=50.0, low_cut=4.0, high_cut=40.0):
        self.fs = fs
        self.num_channels = num_channels

        # --- 设计陷波滤波器 (Notch Filter) ---
        # Q值决定带宽，Q=30在50Hz处大约有1.5Hz的带宽
        b_notch, a_notch = signal.iirnotch(notch_freq, Q=30, fs=fs)
        self.sos_notch = signal.tf2sos(b_notch, a_notch)
        # 初始化滤波器状态 (zi)
        self.zi_notch = signal.sosfilt_zi(self.sos_notch)
        # 扩展状态以匹配通道数: (n_sections, n_channels, 2)
        self.zi_notch = np.repeat(self.zi_notch[:, np.newaxis, :], num_channels, axis=1)

        # --- 设计带通滤波器 (Bandpass Filter) ---
        # 使用Butterworth滤波器，4阶
        self.sos_bp = signal.butter(4, [low_cut, high_cut], btype='band', fs=fs, output='sos')
        self.zi_bp = signal.sosfilt_zi(self.sos_bp)
        self.zi_bp = np.repeat(self.zi_bp[:, np.newaxis, :], num_channels, axis=1)

    def process_sample(self, sample_data):
        """
        处理单个时间点的多通道样本
        Input: list or np.array of shape (num_channels,)
        Output: filtered np.array of shape (num_channels,)
        """
        data = np.array(sample_data)

        # 维度变换适配 scipy: (num_channels, 1)
        # scipy sosfilt 如果处理单个样本，通常需要 loop 或者巧妙的维度设计
        # 这里为了实时性，我们按通道维度进行计算

        # 1. 应用陷波滤波
        # out: (n_channels, 1), new_zi: (n_sections, n_channels, 2)
        res_notch, self.zi_notch = signal.sosfilt(self.sos_notch, data[:, np.newaxis], axis=1, zi=self.zi_notch)

        # 2. 应用带通滤波
        res_bp, self.zi_bp = signal.sosfilt(self.sos_bp, res_notch, axis=1, zi=self.zi_bp)

        return res_bp.flatten()

    def process_batch(self, batch_data):
        """
        处理一批数据 (用于校准时的离线处理)
        Input: (n_samples, n_channels)
        Output: (n_samples, n_channels)
        """
        # 离线处理可以使用 filtfilt 实现零相位滤波(无延迟)，或者沿用 sosfilt
        # 这里为了保持和实时处理一致的相位特性，建议使用 sosfilt 但不更新保存的实时状态
        # 或者直接使用 filtfilt 获得更好波形（但会造成训练和推理的微小差异）
        # 这里演示使用 filtfilt (零相位)
        data = np.array(batch_data)

        # 1. 陷波
        b_notch, a_notch = signal.iirnotch(50.0, Q=30, fs=self.fs)
        data = signal.filtfilt(b_notch, a_notch, data, axis=0)

        # 2. 带通
        b_bp, a_bp = signal.butter(4, [4.0, 40.0], btype='band', fs=self.fs)
        data = signal.filtfilt(b_bp, a_bp, data, axis=0)

        return data


# EEG-TCNet
class TCN_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCN_ResidualBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.left_pad = dilation * (kernel_size - 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                dilation=(1, dilation)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                dilation=(1, dilation)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.pad(x, (self.left_pad, 0, 0, 0), mode='constant', value=0)
        x = self.conv1(x)
        x = F.pad(x, (self.left_pad, 0, 0, 0), mode='constant', value=0)
        x = self.conv2(x)
        return x + residual


class EEG_TCNet(nn.Module):
    def __init__(self, F1=8, F2=16, F_T=16, K_E=64, K_T=25, n_chan=22, n_cls=5, dropout_E=0.5, dropout_T=0.5, L=2):
        """
        Args:
            n_chan: 输入通道数 (接收器发来是21，但Trigger也可能被算作通道，或者只用EEG。这里默认由外部传入)
            n_cls: 分类类别数
        """
        super(EEG_TCNet, self).__init__()
        self.blk_1 = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, K_E),
                padding=(0, K_E // 2)
            ),
            nn.BatchNorm2d(F1)
        )

        self.blk_2 = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * 2,
                kernel_size=(n_chan, 1),
                groups=F1
            ),
            nn.BatchNorm2d(F1 * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_E)
        )

        self.blk_3 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(
                F1 * 2,
                F1 * 2,
                kernel_size=(1, 16),
                groups=F1 * 2,
                padding=(0, 8)
            ),
            # Pointwise conv
            nn.Conv2d(
                F1 * 2,
                F2,
                kernel_size=(1, 1)
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_E)
        )

        self.tcn_blocks = nn.ModuleList()
        for i in range(L):
            dilation = 2 ** i
            in_channels = F2 if i == 0 else F_T
            self.tcn_blocks.append(
                TCN_ResidualBlock(
                    in_channels=in_channels,
                    out_channels=F_T,
                    kernel_size=K_T,
                    dilation=dilation,
                    dropout=dropout_T
                )
            )
        self.blk_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F_T, n_cls)
        )

    def forward(self, x):
        # x shape: (batch, 1, channels, time)
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        x = self.blk_5(x)
        return x



class  EEGModelHandler:
    """
    负责管理 EEG-TCNet 模型：加载、微调、推理
    """

    def __init__(self, model_path='pretrained_model.pth', n_chan=21, n_classes=5, input_window=500, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.input_window = input_window  # 输入窗口长度，例如 2秒*250Hz = 500点

        # 1. 初始化模型结构
        # 这里的超参数 F1, F2等需要与训练时保持一致
        self.model = EEG_TCNet(n_chan=n_chan, n_cls=n_classes).to(self.device)
        self.model_path = model_path
        self.is_calibrated = False

        # 2. 尝试加载预训练权重
        self.load_pretrained()

        # 持久化优化器和损失函数，供在线学习使用
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _init_optimizer(self):
        """初始化优化器，只训练分类层"""
        if self.optimizer is None:
            # 锁特征层，只解锁分类层
            self.prepare_for_finetuning()
            # 单样本更新
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0005)

    def train_one_step(self, X_sample, y_label):
        """
        在线学习：执行一次反向传播
        X_sample: 单个样本 (Channels, Time) 或 Batch
        y_label: 单个标签 (int) 或 Batch
        """
        self.model.train()
        self._init_optimizer()  # 确保优化器已创建

        # 1. 数据处理
        # 输入是 (Channels, Time)，增加 Batch 维度 -> (1, 1, C, T)
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0).unsqueeze(0)
        elif X_tensor.ndim == 3:  # (Batch, C, T)
            X_tensor = X_tensor.unsqueeze(1)

        # 标签处理
        if isinstance(y_label, int):
            y_label = [y_label]
        y_tensor = torch.LongTensor(y_label).to(self.device)

        # 2. 训练步骤
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def load_pretrained(self):
        if os.path.exists(self.model_path):
            try:
                # map_location 确保在只有 CPU 的机器上也能加载 GPU 训练的模型
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[ModelHandler] 成功加载预训练模型: {self.model_path}")
            except Exception as e:
                print(f"[ModelHandler] 加载模型失败: {e}。将使用随机初始化权重。")
        else:
            print(f"[ModelHandler] 未找到预训练权重文件 {self.model_path}。将使用随机初始化权重。")

    def prepare_for_finetuning(self):
        """
        校准准备：锁特征提取层，只让分类头可训练
        """
        # 1.锁住所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 2.解锁最后一层
        for param in self.model.blk_5.parameters():
            param.requires_grad = True

        print("[ModelHandler] 模型已锁定，仅分类层开放训练。")

    def fine_tune(self, X_data, y_labels, epochs=10, batch_size=16, lr=0.001):
        """
        使用校准数据进行微调
        X_data: shape (N, Channels, Time) or (N, Time, Channels) -> 会自动修正
        y_labels: shape (N,)
        """
        self.model.train()
        self.prepare_for_finetuning()

        # 数据格式(N, 1, Channels, Time)
        X_tensor = torch.FloatTensor(X_data).to(self.device)
        if X_tensor.ndim == 3:
            # 输入是 (N, Channels, Time)，增加 Channel 维度 -> (N, 1, C, T)
            X_tensor = X_tensor.unsqueeze(1)

        y_tensor = torch.LongTensor(y_labels).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"[ModelHandler] 开始微调... 数据量: {len(X_data)}")

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()

            acc = 100 * correct / len(dataset)
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

        self.is_calibrated = True
        print("[ModelHandler] 微调完成！")

        # 恢复，在下一次微调前需要重新reset
        # 这里保持现状推理模式用 eval()

    def predict(self, raw_data_window):
        """
        对实时窗口数据进行推理
        input: numpy array (Time, Channels) (例如 500x21)
        output: (predicted_class_index, confidence)
        """
        self.model.eval()

        # 1. 数据形状变换
        # 输入 (Time, Chan) -> 转置为 (Chan, Time) -> 增加batch和dim -> (1, 1, Chan, Time)
        data = np.array(raw_data_window).T
        data_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(data_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item()