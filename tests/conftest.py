"""
Shared pytest configuration and fixtures for FontDiffuser DL tests.

Key patterns:
- Determinism: torch.manual_seed() ensures reproducible tests
- Device agnostic: fixtures handle cpu/cuda detection
- Hardware efficiency: use smaller models for CPU testing
- Isolation: each test gets a fresh, seeded environment
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Set module-level seed before any imports
torch.manual_seed(42)
import numpy as np

np.random.seed(42)


@dataclass
class TestConfig:
    """Minimal config for testing."""

    seed: int = 42
    resolution: int = 96
    batch_size: int = 2
    unet_channels: tuple[int, ...] = (64, 128, 256, 512)
    style_start_channel: int = 64
    content_start_channel: int = 64
    style_image_size: tuple[int, ...] = (96, 96)
    content_encoder_downsample_size: int = 4
    channel_attn: bool = True
    num_neg: int = 4
    data_root: str = None


@pytest.fixture(scope="session")
def device():
    """Return available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def use_small_model():
    """Return True if testing on CPU to use smaller models."""
    return not torch.cuda.is_available()


@pytest.fixture(autouse=True)
def seed_everything():
    """Auto-use fixture to ensure determinism before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def test_config(use_small_model):
    """Create a minimal test config."""
    config = TestConfig(
        batch_size=2 if not use_small_model else 1,
        unet_channels=(32, 64, 128, 256) if use_small_model else (64, 128, 256, 512),
    )
    return config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def dummy_checkpoint_dir(temp_dir):
    """Create a minimal checkpoint directory structure for testing."""
    ckpt_dir = temp_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    return str(ckpt_dir)


@pytest.fixture
def dummy_data_dir(temp_dir):
    """Create a minimal data directory structure for testing."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create minimal train/test structure
    for split in ["train", "test"]:
        split_dir = data_dir / split / "TargetImage"
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal style folder
        style_dir = split_dir / "style_test"
        style_dir.mkdir(exist_ok=True)

    return str(data_dir)


@pytest.fixture
def sample_noisy_latents(test_config, device):
    """Create sample noisy latents for diffusion: (B, 4, H//8, W//8)."""
    B = test_config.batch_size
    H, W = test_config.resolution, test_config.resolution
    # UNet expects latents at 1/8 resolution
    return torch.randn(B, 4, H // 8, W // 8, device=device)


@pytest.fixture
def sample_timestep(test_config, device):
    """Create sample timestep tensor: (B,)."""
    B = test_config.batch_size
    return torch.randint(0, 1000, (B,), device=device)


@pytest.fixture
def sample_content_image(test_config, device):
    """Create sample content image: (B, 1, H, W)."""
    B = test_config.batch_size
    H, W = test_config.resolution, test_config.resolution
    return torch.rand(B, 1, H, W, device=device)


@pytest.fixture
def sample_style_image(test_config, device):
    """Create sample style image: (B, 1, H, W)."""
    B = test_config.batch_size
    H, W = test_config.resolution, test_config.resolution
    return torch.rand(B, 1, H, W, device=device)


@pytest.fixture
def sample_target_image(test_config, device):
    """Create sample target image: (B, 1, H, W)."""
    B = test_config.batch_size
    H, W = test_config.resolution, test_config.resolution
    return torch.rand(B, 1, H, W, device=device)


@pytest.fixture
def dummy_style_features(test_config, device):
    """Create dummy multi-scale style features mimicking MSSE output."""
    B = test_config.batch_size
    # Typical: [64, 128, 256, 512, 1024] channels across scales
    feature_channels = [64, 128, 256, 512, 1024]
    features = []

    spatial_size = 96
    for ch in feature_channels:
        spatial_size = spatial_size // 2
        feat = torch.randn(B, ch, spatial_size, spatial_size, device=device)
        features.append(feat)

    return features


@pytest.fixture
def dummy_unet_config():
    """Create minimal UNet config for testing."""
    config = {
        "sample_size": 96,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": (
            "DownBlock2D",
            "MCADownBlock2D",
            "MCADownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "StyleRSIUpBlock2D",
            "StyleRSIUpBlock2D",
            "UpBlock2D",
        ),
        "block_out_channels": (64, 128, 256, 512),
        "layers_per_block": 2,
        "cross_attention_dim": 1024,
        "attention_head_dim": 1,
    }
    return config


class SimpleEncoder(nn.Module):
    """Minimal encoder for testing."""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SimpleUNet(nn.Module):
    """Minimal UNet for testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 3, padding=1)
        self.decoder = nn.Conv2d(64, 3, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
    ) -> Dict:
        """Forward pass."""
        x = self.encoder(x)
        x = self.decoder(x)
        return {"sample": x}


@pytest.fixture
def simple_encoder(device):
    """Create a simple encoder for testing."""
    return SimpleEncoder().to(device)


@pytest.fixture
def simple_unet(device):
    """Create a simple UNet for testing."""
    return SimpleUNet().to(device)


@pytest.fixture
def optimizer_factory():
    """Factory to create optimizers for tests."""

    def _create_optimizer(model, lr=1e-3):
        return torch.optim.Adam(model.parameters(), lr=lr)

    return _create_optimizer


@pytest.fixture
def loss_fn():
    """Create a simple MSE loss for testing."""
    return nn.MSELoss()


@pytest.fixture
def scheduler_factory(optimizer_factory):
    """Factory to create schedulers for tests."""

    def _create_scheduler(model, num_training_steps=100, lr=1e-3):
        optimizer = optimizer_factory(model, lr=lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps
        )
        return optimizer, scheduler

    return _create_scheduler


@pytest.fixture
def assert_no_nan():
    """Helper to assert no NaNs in tensor."""

    def _assert_no_nan(tensor: torch.Tensor, name: str = "tensor"):
        assert not torch.isnan(tensor).any(), f"NaN detected in {name}"

    return _assert_no_nan


@pytest.fixture
def assert_no_inf():
    """Helper to assert no Infs in tensor."""

    def _assert_no_inf(tensor: torch.Tensor, name: str = "tensor"):
        assert not torch.isinf(tensor).any(), f"Inf detected in {name}"

    return _assert_no_inf


@pytest.fixture
def assert_shape():
    """Helper to assert tensor shape."""

    def _assert_shape(
        tensor: torch.Tensor, expected_shape: Tuple, name: str = "tensor"
    ):
        assert tensor.shape == expected_shape, (
            f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"
        )

    return _assert_shape


@pytest.fixture
def assert_gradients_exist():
    """Helper to assert that all parameters have gradients."""

    def _assert_gradients_exist(model: nn.Module, require_grad_only: bool = True):
        """
        Args:
            model: PyTorch model to check
            require_grad_only: If True, only check parameters with requires_grad=True
        """
        for name, param in model.named_parameters():
            if require_grad_only and not param.requires_grad:
                continue
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after backward()"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradients"
            )

    return _assert_gradients_exist
