import torch
import torch.nn.functional as f 
from torch import nn

class ResEEG(nn.Module):
    def __init__(self,
            n_chan, n_cls, F, T):
        super().__init__()
        self.n_chan = n_chan
        self.n_cls = n_cls
        self.F = F
        self.T = T
        
        # blocks
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

        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_cls))

        
    def forward(self, x):
        # residual connect
        x = f.elu(self.conv_1(x) + self.short_1(x))
        x = f.elu(self.conv_2(x) + self.short_2(x))
        x = f.elu(self.conv_3(x) + self.short_3(x))
        x = f.elu(self.conv_4(x) + self.short_4(x))
        x = self.fc(x)
        return x
