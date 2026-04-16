import numpy as np
from sklearn.cross_decomposition import CCA
import config


class SSVEPHandler:
    """
    SSVEP 信号处理器
    核心算法：CCA (Canonical Correlation Analysis)
    """

    def __init__(self, sample_rate, window_len_sec, target_freqs):
        """
        :param sample_rate: 采样率 (例如 110Hz)
        :param window_len_sec: 窗口长度 (例如 2.0秒)
        :param target_freqs: 目标频率列表 [8.0, 10.0, 12.0, ...]
        """
        self.fs = sample_rate
        self.n_samples = int(window_len_sec * sample_rate)
        self.target_freqs = target_freqs
        self.n_harmonics = 2  # 使用 2 个谐波 (f, 2f) 能提高准确率

        # 预先生成参考模板 (Reference Signals)
        # 这是一个优化的关键：不要每次推理都重新生成正弦波，算一次存起来
        self.reference_signals = self._generate_reference_templates()

        # 初始化 CCA 模型
        self.cca = CCA(n_components=1)

    def _generate_reference_templates(self):
        """生成正弦/余弦参考信号模板"""
        t = np.linspace(0, self.n_samples / self.fs, self.n_samples, endpoint=False)
        templates = []

        for freq in self.target_freqs:
            temp = []
            for h in range(1, self.n_harmonics + 1):
                temp.append(np.sin(2 * np.pi * h * freq * t))
                temp.append(np.cos(2 * np.pi * h * freq * t))
            templates.append(np.array(temp).T)  # Shape: (Time, 2*Harmonics)

        return templates

    def classify(self, eeg_data):
        """
        对传入的脑电数据进行分类
        :param eeg_data: numpy array (Channels, Time)
        :return: (best_freq_index, correlation_score)
        """
        # 1. 检查数据长度
        n_channels, n_points = eeg_data.shape
        if n_points < self.n_samples:
            # 如果数据不够长，就取目前的长度重新生成模板，或者直接返回 None
            # 为了稳定性，我们截取或填充，这里简单处理：只处理长度足够的数据
            return -1, 0.0

        # 截取最新的 n_samples 个点
        # 注意：eeg_data 应该是 (Channels, Time)，我们需要转置为 (Time, Channels) 给 sklearn 用
        X = eeg_data[:, -self.n_samples:].T

        scores = []
        for i, Y_ref in enumerate(self.reference_signals):
            # Y_ref shape: (Time, 4) if 2 harmonics
            # X shape: (Time, Channels)

            # 这里的 X 和 Y_ref 的 Time 维度必须一致
            if X.shape[0] != Y_ref.shape[0]:
                # 重新生成临时模板适配长度 (容错处理)
                t = np.linspace(0, X.shape[0] / self.fs, X.shape[0], endpoint=False)
                Y_temp = []
                for h in range(1, self.n_harmonics + 1):
                    Y_temp.append(np.sin(2 * np.pi * h * self.target_freqs[i] * t))
                    Y_temp.append(np.cos(2 * np.pi * h * self.target_freqs[i] * t))
                Y_ref = np.array(Y_temp).T

            try:
                self.cca.fit(X, Y_ref)
                # 获取第一对典型变量的相关系数
                xc, yc = self.cca.transform(X, Y_ref)
                corr = np.corrcoef(xc[:, 0], yc[:, 0])[0, 1]
                scores.append(corr)
            except:
                scores.append(0.0)

        # 找出最大相关系数
        max_corr = max(scores)
        best_idx = scores.index(max_corr)

        return best_idx, max_corr