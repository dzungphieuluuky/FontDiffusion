import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleStyleEncoder(nn.Module):
    """
    CNN-based encoder to extract multi-scale style features from a glyph image.
    Corresponds to the style encoder E_s in FSTDiff paper (Section 3.1).

    Args:
        in_channels: Input channels (e.g., 1 for grayscale).
        base_channels: Base number of channels.
        num_scales: Number of feature scales (n_s in paper, default 5).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_scales: int = 5,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.base_channels = base_channels

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=7, stride=2, padding=3
        )
        self.initial_norm = nn.InstanceNorm2d(base_channels)

        # Downsampling Residual Blocks with consistent channel progression
        self.down_blocks = nn.ModuleList()
        self.feature_channels = []

        current_channels = base_channels

        for i in range(num_scales):
            # Consistent channel progression: 64, 128, 256, 512, 1024
            next_channels = base_channels * (2 ** i)
            use_stride = 2 if i < num_scales - 1 else 1

            block = ResidualBlock(
                current_channels,
                next_channels,
                downsample=(use_stride == 2),
            )
            self.down_blocks.append(block)
            self.feature_channels.append(next_channels)
            current_channels = next_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: Input glyph image tensor (B, C, H, W).
        Returns:
            A list of multi-scale style features [f^{s,1}, f^{s,2}, ..., f^{s,n_s}].
            Each feature shape: (B, C_i, H_i, W_i).
        """
        features: list[torch.Tensor] = []

        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = F.leaky_relu(x, 0.2)

        # Extract features at each scale
        for down_block in self.down_blocks:
            x = down_block(x)
            features.append(x)

        return features

    def get_feature_channels(self) -> list[int]:
        """Return the channel dimensions for each scale."""
        return self.feature_channels


class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)

        # Shortcut connection
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.InstanceNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out, 0.2, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = F.leaky_relu(out, 0.2, inplace=True)

        return out