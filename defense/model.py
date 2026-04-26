from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        groups = min(16, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, 1)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, channel_mults: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        channels = [base_channels * m for m in channel_mults]
        self.stem = ResBlock(in_channels, channels[0])
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev = channels[0]
        for ch in channels:
            self.down_blocks.append(ResBlock(prev, ch))
            self.downsamples.append(ConvNormAct(ch, ch, 3, 2))
            prev = ch

        self.bottleneck = nn.Sequential(
            ResBlock(prev, prev),
            ResBlock(prev, prev),
        )
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        h = self.stem(x)
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips


class LatentPurifier(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(channels, channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, z_adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.head(self.blocks(z_adv))
        z_hat = z_adv - residual
        return z_hat, residual


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = ResBlock(in_ch + skip_ch, out_ch)
        self.refine = ResBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, base_channels: int = 32, channel_mults: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        channels = [base_channels * m for m in channel_mults]
        rev_channels = list(reversed(channels))
        current = rev_channels[0]
        self.up_blocks = nn.ModuleList()
        for skip_ch in reversed(channels):
            out_ch = skip_ch
            self.up_blocks.append(UpBlock(current, skip_ch, out_ch))
            current = out_ch
        self.tail = nn.Sequential(
            ResBlock(current, current),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_hat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        h = z_hat
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip)
        return torch.sigmoid(self.tail(h))


@dataclass
class DefenseConfig:
    in_channels: int = 3
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    purifier_blocks: int = 4


class LatentSemanticDefense(nn.Module):
    def __init__(self, cfg: DefenseConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
        )
        self.purifier = LatentPurifier(
            channels=self.encoder.out_channels,
            num_blocks=cfg.purifier_blocks,
        )
        self.decoder = Decoder(
            out_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
        )

    def forward(self, x_adv: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_adv, skips = self.encoder(x_adv)
        z_hat, residual = self.purifier(z_adv)
        x_hat = self.decoder(z_hat, skips)
        return {
            "x_hat": x_hat,
            "z_adv": z_adv,
            "z_hat": z_hat,
            "latent_residual": residual,
        }
