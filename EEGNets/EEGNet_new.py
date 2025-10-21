import torch
from torch import nn

class EEGNet(nn.Module):
    def __init__(self,
            n_class,channels=64,samples=128,kernel_len=64,
            F_1=8,D=2,F_2=16):
        super().__init__()
        self.blk_1=nn.Sequential(
                nn.Conv2d(
                    1,
                    F_1,
                    kernel_size=(1,kernel_len),
                    padding=(0,kernel_len//2),
                    bias=False),
                nn.BatchNorm2d(F_1),
                nn.Conv2d(
                    F_1,
                    F_1*D,
                    kernel_size=(channels,1),
                    bias=False,
                    groups=F_1,
                    ),
                nn.BatchNorm(F_1*D),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,4)),
                nn.Dropout(p=0.5))
        self.blk_2=nn.Sequential(
                nn.Conv2d(
                    F_1*D,
                    F_1*D,
                    kernel_size=(1,16),
                    bias=False,
                    padding=(0,6),
                    groups=F_1*D,
                    ),
                nn.Conv2d(
                    F_1*D,
                    F_2,
                    kernel_size=1,
                    bias=True
                    ),
                nn.BatchNorm2d(F_2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,8)),
                nn.Dropout(p=0.5))
        self.fc=nn.Sequential(
                nn.Flatten(),
                nn.Linear(F_2*((samples//4-3)//8),n_class))

    def forward(self,x):
        x=self.blk_1(x)
        x=self.blk_2(x)
        x=self.fc(x)
        return x





