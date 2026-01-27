import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleStyleEncoder(nn.Module):
    """Multi-Scale Style Encoder (MSSE) for extracting style features at different scales."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_scales: int = 5):
        super().__init__()
        self.num_scales = num_scales
        
        # Store the actual output channels for each scale
        self.output_channels = []
        
        self.encoders = nn.ModuleList()
        for i in range(num_scales):
            out_channels = base_channels * (2**i)  # 64, 128, 256, 512, 1024
            self.output_channels.append(out_channels)
            
            encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((48 // (2**i), 48 // (2**i))),
            )
            self.encoders.append(encoder)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for encoder in self.encoders:
            features.append(encoder(x))
        return features

    def get_output_channels(self) -> list[int]:
        """Return the actual output channels for each scale."""
        return self.output_channels


class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
