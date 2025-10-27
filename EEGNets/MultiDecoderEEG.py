import torch
from torch import nn
from torch.nn import functional as F

class TCN_ResidualBlock(nn.Module): # temporal convolutional network from EEGTCNet
    def __init__(self,
            in_chan, out_chan, K, dilation, dropout):
        super(TCN_ResidualBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = K
        self.left_pad = dilation * (K - 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=(1, self.kernel_size),
                dilation=(1, dilation)
            ),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_chan,
                out_chan,
                kernel_size=(1, self.kernel_size),
                dilation=(1, dilation)
            ),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        if in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_chan)
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

class EEGEncoder(nn.Module): # a general encoder using TCNet architecture
    def __init__(self,
            n_chan, F, F_T, K_T, T, L=2):
        super(EEGEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
                nn.Conv2d(
                    1,
                    25,
                    kernel_size=(1, 10),
                    padding=(0, 4)),
                nn.Conv2d(
                    25,
                    25,
                    kernel_size=(n_chan,1)),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5))

        self.short_1 = nn.Sequential(
                nn.Conv2d(
                    1,
                    25,
                    kernel_size=(n_chan, 1)),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.BatchNorm2d(25))

        # deepwise separable convolutional blocks
        self.conv_2 = nn.Sequential(
                nn.Conv2d(
                    25,
                    25,
                    kernel_size=(1, 5),
                    padding=(0, 2),
                    groups=25,
                    bias=False),
                nn.Conv2d(
                    25,
                    50,
                    kernel_size=1,
                    bias=False),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
                nn.Dropout(p=0.5))

        self.short_2 = nn.Sequential(
                nn.Conv2d(
                    25,
                    50,
                    kernel_size=1),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.BatchNorm2d(50))

        self.conv_3 = nn.Sequential(
                nn.Conv2d(
                    50,
                    50,
                    kernel_size=(1, 5),
                    padding=(0, 2),
                    groups=50,
                    bias=False),
                nn.Conv2d(
                    50,
                    100,
                    kernel_size=1,
                    bias=False),
                nn.BatchNorm2d(100, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5))

        self.short_3 = nn.Sequential(
                nn.Conv2d(
                    50,
                    100,
                    kernel_size=1),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.BatchNorm2d(100))

        self.conv_4 = nn.Sequential(
                nn.Conv2d(
                    100,
                    100,
                    kernel_size=(1, 5),
                    padding=(0, 2),
                    groups=100,
                    bias=False),
                nn.Conv2d(
                    100,
                    200,
                    kernel_size=1,
                    bias=False),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5))
        
        self.short_4 = nn.Sequential(
                nn.Conv2d(
                    100,
                    200,
                    kernel_size=1),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.BatchNorm2d(200))

        self.tcn_blocks = nn.ModuleList()
        for i in range(L):
            dilation = 2 ** i
            in_channels = F if i == 0 else F_T
            self.tcn_blocks.append(
                TCN_ResidualBlock(
                    in_chan=in_channels,
                    out_chan=F_T,
                    K=K_T,
                    dilation=dilation,
                    dropout=0.25
                )
            )

    def forward(self, x):
        x = F.elu(self.conv_1(x) + self.short_1(x))
        x = F.elu(self.conv_2(x) + self.short_2(x))
        x = F.elu(self.conv_3(x) + self.short_3(x))
        x = F.elu(self.conv_4(x) + self.short_4(x))
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        return x

class DecoderTemplate(nn.Module): # decoder for single task
    def __init__(self,
            n_cls, dropout):
        super(DecoderTemplate, self).__init__()
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(n_cls)
        self.norm = nn.BatchNorm2d(n_cls)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(self.flat(x))
        x = self.fc(x)
        x = self.drop(x)
        return x
