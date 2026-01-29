import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleStyleEncoder(nn.Module):
    """Enhanced MSSE with residual connections and attention."""
    def __init__(
        self, 
        in_channels: int = 1,  # Grayscale for fonts
        base_channels: int = 64, 
        num_scales: int = 5,
        use_attention: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales
        self.output_channels = []
        self.encoders = nn.ModuleList()
        
        for i in range(num_scales):
            out_channels = base_channels * (2 ** i)
            self.output_channels.append(out_channels)
            
            layers = [
                # Initial conv with larger receptive field
                nn.Conv2d(in_channels, out_channels // 2, 5, 1, 2),
                nn.InstanceNorm2d(out_channels // 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Residual block
                ResidualBlock(out_channels // 2, out_channels, downsample=False),
            ]
            
            # Add attention at higher scales (512, 1024)
            if use_attention and i >= 3:
                layers.append(SELayer(out_channels, reduction=16))
            
            # Adaptive pooling to target resolution
            target_size = 48 // (2 ** i)
            layers.append(nn.AdaptiveAvgPool2d((target_size, target_size)))
            
            self.encoders.append(nn.Sequential(*layers))
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, 1, H, W) → list of (B, C_i, H_i, W_i)"""
        features = []
        for encoder in self.encoders:
            feat = encoder(x)
            features.append(feat)
        return features

    def get_output_channels(self) -> list[int]:
        """Return the actual output channels for each scale."""
        return self.output_channels

class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)
    
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
