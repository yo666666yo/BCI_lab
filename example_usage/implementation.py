import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import pickle
from scipy import interpolate, signal

class _TCN_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(_TCN_ResidualBlock, self).__init__()
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

class _EEG_TCNet(nn.Module):
    def __init__(self, F1=8, F2=16, F_T=32, K_E=64, K_T=4, n_chan=19, n_cls=4, 
                 dropout_E=0.3, dropout_T=0.2, L=2):
        super(_EEG_TCNet, self).__init__()
        
        self.blk_1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, K_E), padding=(0, K_E//2)),
            nn.BatchNorm2d(F1)
        )

        self.blk_2 = nn.Sequential(
            nn.Conv2d(F1, F1 * 2, kernel_size=(n_chan, 1), groups=F1),
            nn.BatchNorm2d(F1 * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_E)
        )

        self.blk_3 = nn.Sequential(
            nn.Conv2d(F1 * 2, F1 * 2, kernel_size=(1, 16), groups=F1 * 2, padding=(0, 8)),
            nn.Conv2d(F1 * 2, F2, kernel_size=(1, 1)),
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
                _TCN_ResidualBlock(
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
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        x = self.blk_5(x)
        return x

class HiddenBCIModel:
    def __init__(self, n_channels=19, n_times=256, n_classes=4, sampling_rate=128):
        self.n_channels = n_channels
        self.n_times = n_times  
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate

        self.model = _EEG_TCNet(
            n_chan=n_channels,
            n_cls=n_classes
        )

        self.scaler = StandardScaler()
        self._is_trained = False
        
    def load_weights(self, model_path: str):
        try:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                print("reloading...")
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self._is_trained = True
            if 'n_channels' in checkpoint:
                self.n_channels = checkpoint['n_channels']
            if 'n_times' in checkpoint:
                self.n_times = checkpoint['n_times']
            if 'n_classes' in checkpoint:
                self.n_classes = checkpoint['n_classes']
            if 'sampling_rate' in checkpoint:
                self.sampling_rate = checkpoint['sampling_rate']
            if 'scaler_params' in checkpoint:
                scaler_params = checkpoint['scaler_params']
                self.scaler.mean_ = scaler_params['mean']
                self.scaler.var_ = scaler_params['var']
                self.scaler.scale_ = scaler_params['scale']
                self.scaler.n_samples_seen_ = scaler_params['n_samples_seen']
                
        except Exception as e:
            raise RuntimeError(f"weight load fail: {str(e)}")
    
    def predict(self, eeg_data: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("model is not trained")
        eeg_tensor = self._preprocess_data(eeg_data)
        with torch.no_grad():
            outputs = self.model(eeg_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()
            
        return probabilities
    
    def _preprocess_data(self, eeg_data: np.ndarray) -> torch.Tensor:
        n_trials, n_channels, n_times = eeg_data.shape
        nyquist = self.sampling_rate / 2.0
        low_freq = 8 / nyquist
        high_freq = 30 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        eeg_filtered = np.copy(eeg_data)
        for i in range(n_trials):
            for j in range(n_channels):
                eeg_filtered[i, j, :] = signal.filtfilt(b, a, eeg_filtered[i, j, :])
        eeg_data = eeg_filtered
        eeg_clean = np.zeros((n_trials, n_channels, n_times))
        
        for i in range(n_trials):
            trial_data = eeg_data[i].T
            data_mean = np.mean(trial_data, axis=0, keepdims=True)
            data_centered = trial_data - data_mean
            ica = FastICA(n_components=n_channels, random_state=42, max_iter=200)
            sources = ica.fit_transform(data_centered)  # (n_times, n_channels)
            var_sources = np.var(sources, axis=0)
            mean_var = np.mean(var_sources)
            threshold = 3 * mean_var
            artifact_components = np.where(var_sources > threshold)[0]
            
            print(f"Trial {i+1}: Detected {len(artifact_components)} artifact components (indices: {artifact_components})")
            sources_clean = sources.copy()
            sources_clean[:, artifact_components] = 0
            reconstructed_centered = ica.inverse_transform(sources_clean)  # (n_times, n_channels)
            reconstructed = reconstructed_centered + data_mean
            eeg_clean[i] = reconstructed.T
        eeg_data = eeg_clean
        if n_times != self.n_times:
            eeg_resampled = np.zeros((n_trials, n_channels, self.n_times))
            
            for i in range(n_trials):
                for j in range(n_channels):
                    x_old = np.linspace(0, 1, n_times)
                    x_new = np.linspace(0, 1, self.n_times)
                    f = interpolate.interp1d(x_old, eeg_data[i, j, :], kind='linear')
                    eeg_resampled[i, j, :] = f(x_new)
            
            eeg_data = eeg_resampled
        original_shape = eeg_data.shape
        eeg_reshaped = eeg_data.reshape(original_shape[0], -1)
        
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            eeg_scaled = self.scaler.transform(eeg_reshaped)
        else:
            eeg_scaled = (eeg_reshaped - eeg_reshaped.mean(axis=1, keepdims=True)) / (eeg_reshaped.std(axis=1, keepdims=True) + 1e-8)
        
        eeg_scaled = eeg_scaled.reshape(original_shape)
        eeg_tensor = torch.FloatTensor(eeg_scaled).unsqueeze(1)
        
        return eeg_tensor
