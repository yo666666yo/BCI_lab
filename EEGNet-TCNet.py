import torch
from torch import nn

class EEG_TCNet(nn.Module):
    def __init__(self,
            F1, F2,F_T, K_E, K_T, n_chan, n_cls, dropout_E, dropout_T):
       
        self.blk_1 = nn.Sequential(
                nn.Conv2d(
                    1,
                    F1,
                    kernel_size(1, K_E),
                    padding=(0, K_E // 2))
                nn.BatchNorm2d(F1))

        self.blk_2 = nn.Sequential(
                nn.Conv2d( # Depthwise Conv
                    F1,
                    2 * F1,
                    kernel_size=(n_chan, 1),
                    groups=F1),
                nn.BtachNorm2d(2 * F1),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                nn.Dropout(p=dropout_E))

        self.blk_3 = nn.Sequential(
                # Separable Conv
                nn.Conv(
                    F1 * 2,
                    F1 * 2,
                    kernel_size=(1, 16),
                    groups=F1 * 2,
                    padding=(0, 8)),
                nn.Conv(
                    F1 * 2,
                    F2,
                    kernel_size=1),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                nn.Dropout(p=dropout_E))

        # Residual Block4
        self.conv_4 = nn.Seuential(
                nn.Conv2d( # Dilated Conv
                    F2,
                    F2,
                    kernel_size=K_T,
                    dilation=d),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.Dropout(p=dropout_T),
                nn.Conv2d(
                    F2,
                    F_T,
                    kernel_size=K_T,
                    dilation=d),
                nn.BatchNorm2d(F_T),
                nn.ELU(),
                nn.Dropout(p=dropout_T))

        self.shortcut_4 = nn.Conv2d(
                    F2,
                    F_T,
                    kernel_size=1)

        self.blk_5 = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_cls),
                nn.Softmax())

    def forward(self, x):
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        x = self.conv_4(x) + self.shortcut_4(x)
        x = self.conv_4(x) + self.shortcut_4(x)
        x = self.blk_5
        return x
