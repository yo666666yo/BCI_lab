import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniViT(nn.Module):
    def __init__(self, in_channels=22, img_size=16, n_classes=4):
        super().__init__()
        self.patch_size = 2
        self.num_patches = (img_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, 32, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, 32))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, 
            nhead=4, 
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        # patch embedding
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # single transformer layer
        x = self.transformer(x)
        
        # classification
        x = x[:, 0]
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=22, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*2*2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.classifier(x)
        return x

class MiniEEGProcessor:
    def __init__(self, target_size=(16, 16)):
        self.target_size = target_size
        
    def eeg_to_image_fast(self, eeg_data):
        batch_size, channels, time_points = eeg_data.shape
        height = self.target_size[0]
        width = self.target_size[1]
        if time_points < height * width:
            repeats = (height * width) // time_points + 1
            padded = eeg_data.repeat(1, 1, repeats)
            images = padded[:, :, :height*width].reshape(batch_size, channels, height, width)
        else:
            images = eeg_data[:, :, :height*width].reshape(batch_size, channels, height, width)
        
        return images
    
    def scale_signal(self, images, alpha=1e4):
        return images * alpha

