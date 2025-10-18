import torch
from torch import nn
from torch.nn import functional as F

class EEG_TCNet(nn.Module):
    def __init__(self, F1, F2, F_T, K_E, K_T, n_chan, n_cls, dropout_E, dropout_T, L=2):
        super(EEG_TCNet, self).__init__()
        self.blk_1 = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, K_E),
                padding=(0, K_E//2)
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
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        x = self.blk_5(x)
        return x


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
