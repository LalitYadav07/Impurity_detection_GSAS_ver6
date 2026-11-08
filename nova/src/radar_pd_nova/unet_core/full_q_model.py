"""Frozen architecture of the 3×1024 whole-Q diffraction model.

Keep this definition in step with the reviewed training checkpoint.  PyTorch is
an optional NOVA dependency, so only the inference adapter imports this module.
"""

import torch
from torch import nn
from torch.nn import functional as F


Q_MIN, Q_MAX, BINS = 0.25, 12.0, 1024
LOCAL_NAMES = ("attention", "peak_emergence", "relative_intensity_change",
               "peak_broadening", "peak_shift")
CAUSE_NAMES = ("lattice_change_candidate", "magnetic_order_candidate",
               "new_phase_candidate", "structural_change_candidate")


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel,
                               padding=kernel // 2)
        self.norm1 = nn.GroupNorm(8 if output_channels % 8 == 0 else 4, output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, 5, padding=2)
        self.norm2 = nn.GroupNorm(8 if output_channels % 8 == 0 else 4, output_channels)
        self.skip = (nn.Identity() if input_channels == output_channels else
                     nn.Conv1d(input_channels, output_channels, 1))

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class FullCurveUNet(nn.Module):
    """The selected plain, base-32 U-Net with local and whole-pair heads."""

    def __init__(self):
        super().__init__()
        widths = [32, 64, 128, 192, 256]
        self.stem = ResidualBlock(3, widths[0])
        self.down = nn.ModuleList([ResidualBlock(widths[i], widths[i + 1])
                                   for i in range(4)])
        self.middle = ResidualBlock(widths[-1], widths[-1])
        self.q_embed = nn.Sequential(nn.Conv1d(3, widths[-1], 1), nn.SiLU(),
                                     nn.Conv1d(widths[-1], widths[-1], 1))
        self.up = nn.ModuleList([ResidualBlock(widths[i + 1] + widths[i], widths[i])
                                 for i in reversed(range(4))])
        self.local = nn.Conv1d(widths[0], len(LOCAL_NAMES), 1)
        self.global_head = nn.Sequential(
            nn.Linear(widths[-1] * 4, widths[-1]), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(widths[-1], len(CAUSE_NAMES)))
        q = F.avg_pool1d(torch.linspace(Q_MIN, Q_MAX, BINS).view(1, 1, BINS), 16)
        q_norm = q / Q_MAX
        self.register_buffer("q_features", torch.cat((q_norm,
            torch.sin(2 * torch.pi * q_norm), torch.cos(2 * torch.pi * q_norm)), 1),
            persistent=False)
        self.register_buffer("q_bottleneck", q, persistent=False)

    def forward(self, x, coverage):
        skips = []
        y = self.stem(x)
        skips.append(y)
        for block in self.down:
            y = block(F.avg_pool1d(y, 2))
            skips.append(y)
        y = self.middle(y)
        y = y + self.q_embed(self.q_features.expand(len(y), -1, -1))
        mask = F.max_pool1d(coverage[:, None, :], 16).squeeze(1) > 0.5
        denom = mask.sum(1).clamp_min(1).unsqueeze(1)
        mean = (y * mask[:, None, :]).sum(-1) / denom
        maximum = y.masked_fill(~mask[:, None, :], -1e4).amax(-1)
        low = mask & (self.q_bottleneck[:, 0, :] < 3.0)
        high = mask & (self.q_bottleneck[:, 0, :] >= 3.0)
        low_mean = (y * low[:, None, :]).sum(-1) / low.sum(1).clamp_min(1).unsqueeze(1)
        high_mean = (y * high[:, None, :]).sum(-1) / high.sum(1).clamp_min(1).unsqueeze(1)
        cause = self.global_head(torch.cat((mean, maximum, low_mean, high_mean), 1))
        for block, skip in zip(self.up, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-1], mode="linear", align_corners=False)
            y = block(torch.cat((y, skip), 1))
        return self.local(y), cause
