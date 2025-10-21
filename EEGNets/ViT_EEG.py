from typing import Optional, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
import math

class EEGTransform(nn.Module):
    def __init__(self, target_size: Tuple[int, int] = (224, 224), repeat_n: int = 1, alpha: float = 1.0):
        super().__init__()
        self.H, self.W = target_size
        self.repeat_n = max(1, repeat_n)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,T)
        elif x.dim() == 2:
            if x.size(0) <= x.size(1):
                # (C, T)
                x = x.unsqueeze(0)  # (1, C, T)
            else:
                x = x.unsqueeze(1)  # (B, 1, T)
        elif x.dim() == 3:
            pass  # (B, C, T)
        else:
            raise ValueError("tensor shape error")

        B, C, T = x.shape

        x_flat = x.view(B, -1)
        vmin = x_flat.min(dim=1, keepdim=True)[0]  # (B,1)
        vmax = x_flat.max(dim=1, keepdim=True)[0]  # (B,1)
        denom = (vmax - vmin).clamp(min=1e-9)
        x_norm = (x_flat - vmin) / denom  # (B, C*T)
        x_norm = x_norm.view(B, C, T)

        concat = x_norm.view(B, -1, 1)  # (B, C*T, 1)
        concat = concat.transpose(1, 2)  # (B,1, C*T)
        concat_rep = concat.repeat(1, self.repeat_n, 1)  # (B, repeat_n, C*T)

        B_, H_src, W_src = concat_rep.size()
        img = concat_rep.unsqueeze(1)  # (B,1,H_src,W_src)

        img_resized = F.interpolate(img, size=(self.H, self.W), mode='bilinear', align_corners=False)

        img_out = img_resized * self.alpha

        return img_out  # (B,1,H,W)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int,int]=(224,224), patch_size: Tuple[int,int]=(16,16),
                 in_chans: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, gh, gw)
        x = self.flatten(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat((cls, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim cant be divisible"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B,N,3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B,N,3,heads,hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, hd)

        # compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,heads,N,N)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        context = torch.matmul(attn_probs, v)  # (B,heads,N,hd)
        context = context.transpose(1, 2).reshape(B, N, D)  # (B,N,D)
        out = self.out_proj(context)
        out = self.proj_drop(out)
        return out


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0., drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout=attn_drop, proj_dropout=drop)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLPBlock(embed_dim, hidden_dim, dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.attn(y)
        x = x + self.drop_path(y)
        z = self.norm2(x)
        z = self.mlp(z)
        x = x + self.drop_path(z)
        return x


class ViT_EEG(nn.Module):
    def __init__(self,
                 img_size: Tuple[int,int] = (224,224),
                 patch_size: Tuple[int,int] = (16,16),
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed_module = PositionEmbedding(num_patches=num_patches, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        # encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads,
                                    mlp_ratio=mlp_ratio, attn_drop=attn_dropout, drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # initialize head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, N, D)
        x = self.pos_embed_module(x)  # (B, N+1, D)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token = x[:, 0]  # (B, D)
        logits = self.head(cls_token)  # (B, num_classes)
        return logits
