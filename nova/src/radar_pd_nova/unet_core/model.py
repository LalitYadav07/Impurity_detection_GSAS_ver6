"""Small local-feature 1D U-Net for 2 x 156 diffraction focus windows."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FocusUNet(nn.Module):
    """Return attention and eleven independent feature-type logits per position.

    External input/output length is 156. Four values are padded internally so
    three pooling stages divide evenly; predictions are cropped back to 156.
    No full-curve context, instrument metadata, or residual channel is used.
    """

    def __init__(self, base_channels: int = 16, feature_types: int = 11):
        super().__init__()
        if base_channels % 4:
            raise ValueError("base_channels must be divisible by four")
        a, b, c, d = (base_channels * scale for scale in (1, 2, 4, 6))
        self.enc0 = ConvBlock(2, a)
        self.enc1 = ConvBlock(a, b)
        self.enc2 = ConvBlock(b, c)
        self.bottleneck = ConvBlock(c, d)
        self.dec2 = ConvBlock(d + c, c)
        self.dec1 = ConvBlock(c + b, b)
        self.dec0 = ConvBlock(b + a, a)
        self.head = nn.Conv1d(a, 1 + feature_types, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2)

    @staticmethod
    def up_to(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, 156):
            raise ValueError(f"Expected [batch, 2, 156], got {tuple(x.shape)}")
        x = F.pad(x, (2, 2), mode="replicate")
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        center = self.bottleneck(self.pool(e2))
        y = self.dec2(torch.cat((self.up_to(center, e2), e2), dim=1))
        y = self.dec1(torch.cat((self.up_to(y, e1), e1), dim=1))
        y = self.dec0(torch.cat((self.up_to(y, e0), e0), dim=1))
        return self.head(y)[..., 2:-2]
