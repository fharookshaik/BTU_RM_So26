"""SpecTNT model: ResNet front-end + SpecTNT blocks + linear projection."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetFrontEnd(nn.Module):
    """2D conv front-end with temporal downsampling.

    Input:  (B, 1, 80, T_native)  @ ~21.5 fps
    Output: (B, T', 20, 96)       @ ~5.4 fps
    """

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 3, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(c * 3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 3, 2, 1)
        return x


class SpectralEncoder(nn.Module):
    """Self-attention across frequency bins within each time step."""

    def __init__(self, dim, n_heads=4, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x


class TemporalEncoder(nn.Module):
    """Self-attention across time steps for each frequency bin."""

    def __init__(self, dim, n_heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x


class SpecTNTBlock(nn.Module):
    """Spectral encoder → Temporal encoder with residual."""

    def __init__(self, dim, n_heads_spec=4, n_heads_temp=8):
        super().__init__()
        self.spectral = SpectralEncoder(dim, n_heads_spec)
        self.temporal = TemporalEncoder(dim, n_heads_temp)

    def forward(self, x):
        B, T, F, D = x.shape
        x_spec = x.reshape(B * T, F, D)
        x_spec = self.spectral(x_spec)
        x_spec = x_spec.reshape(B, T, F, D)

        x_temp = x_spec.permute(0, 2, 1, 3).reshape(B * F, T, D)
        x_temp = self.temporal(x_temp)
        x_temp = x_temp.reshape(B, F, T, D).permute(0, 2, 1, 3)

        return x_spec + x_temp


class SpecTNT(nn.Module):
    """Full SpecTNT model.

    Input:  (B, 1, 80, T_native)
    Output: (boundary_logits, function_logits)
        boundary_logits:  (B, T', 1)
        function_logits:  (B, T', 7)
    """

    def __init__(self, dim=96, n_blocks=5, n_heads_spec=4, n_heads_temp=8):
        super().__init__()
        self.frontend = ResNetFrontEnd()
        self.blocks = nn.ModuleList([
            SpecTNTBlock(dim, n_heads_spec, n_heads_temp)
            for _ in range(n_blocks)
        ])
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(dim, 8)

    def forward(self, x):
        x = self.frontend(x)
        for block in self.blocks:
            x = block(x)
        x = self.freq_pool(x)
        x = x.squeeze(2)
        x = self.proj(x)
        boundary_logits = x[..., :1]
        function_logits = x[..., 1:]
        return boundary_logits, function_logits
