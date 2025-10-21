import torch
from torch import nn

class DeepEEG(nn.Module):
    def __init__(self,channels,n_class):
        super().__init__()
        self.blk_1=nn.Sequential(
                nn.Conv2d(
                    1,
                    25,
                    kernel_size=(1,10),
                    padding=(0,2)),
                nn.Conv2d(
                    25,
                    25,
                    kernel_size=(channels,1)),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)),
                nn.Dropout(p=0.5))
        self.blk_2=nn.Sequential(
                nn.Conv2d(
                    25,
                    50,
                    kernel_size=(1,5),
                    padding=(0,2)),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)),
                nn.Dropout(p=0.5))
        self.blk_3=nn.Sequential(
                nn.Conv2d(
                    50,
                    100,
                    kernel_size=(1,5),
                    padding=(0,2)),
                nn.BatchNorm2d(100,eps=1e-05,momentum=0.9),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)),
                nn.Dropout(p=0.5)),
        self.blk_4=nn.Sequential(
                nn.Conv2d(
                    100,
                    200,
                    kernel_size=(1,5),
                    padding=(0,2)),
                nn.BatchNorm2d(200,eps=1e-05,momentum=0.9),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)),
                nn.Dropout(p=0.5))
        self.fc=nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_class))

    def forward(self,x):
        x=self.blk_1(x)
        x=self.blk_2(x)
        x=self.blk_3(x)
        x=self.blk_4(x)
        x=self.fc(x)
        return x


