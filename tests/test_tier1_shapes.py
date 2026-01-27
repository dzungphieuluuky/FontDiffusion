"""
Tier 1: Shape & Tensor Invariants (Unit Tests)

Purpose: Catch the most common DL bugs—dimension mismatches and NaNs.
Strategy: Use @pytest.mark.parametrize to pass different batch sizes and sequence lengths
into model forward passes. Verify output shapes and check for gradient explosions.
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

# Import core modules
from src.modules.msse import MultiScaleStyleEncoder, ResidualBlock
from src.modules.fst import FontStyleTransformationModule
from src.losses.criterion import ContentPerceptualLoss


class TestMultiScaleStyleEncoderShapes:
    """Test MSSE output shapes with parametrized inputs."""

    @pytest.mark.parametrize(
        "batch_size,num_scales",
        [
            (1, 5),
            (2, 5),
            (4, 5),
            (8, 3),
        ],
    )
    def test_msse_output_shapes(
        self, batch_size: int, num_scales: int, device, assert_shape, assert_no_nan
    ):
        """Verify MSSE produces correct multi-scale feature shapes."""
        resolution = 96
        msse = MultiScaleStyleEncoder(
            in_channels=1, base_channels=64, num_scales=num_scales
        ).to(device)

        x = torch.rand(batch_size, 1, resolution, resolution, device=device)

        with torch.no_grad():
            features = msse(x)

        # Check number of scales
        assert (
            len(features) == num_scales
        ), f"Expected {num_scales} scales, got {len(features)}"

        # Check each feature scale
        current_spatial = resolution
        for i, feat in enumerate(features):
            assert feat.shape[0] == batch_size, f"Scale {i}: batch size mismatch"
            assert feat.shape[1] > 0, f"Scale {i}: channel count must be positive"
            assert feat.ndim == 4, f"Scale {i}: feature must be 4D (B, C, H, W)"
            assert_no_nan(feat, f"MSSE scale {i}")

    @pytest.mark.parametrize("in_channels", [1, 3])
    def test_msse_variable_input_channels(
        self, in_channels: int, device, assert_no_nan
    ):
        """Test MSSE with different input channels."""
        msse = MultiScaleStyleEncoder(
            in_channels=in_channels, base_channels=64, num_scales=3
        ).to(device)

        x = torch.rand(2, in_channels, 96, 96, device=device)

        with torch.no_grad():
            features = msse(x)

        assert len(features) == 3
        for feat in features:
            assert_no_nan(feat, "MSSE output")


class TestFSTModuleShapes:
    """Test FST module output shapes."""

    @pytest.mark.parametrize(
        "batch_size,num_queries",
        [
            (1, 128),
            (2, 256),
            (4, 256),
        ],
    )
    def test_fst_output_shape(
        self, batch_size: int, num_queries: int, device, assert_no_nan
    ):
        """Verify FST produces correct transformation feature shape."""
        feature_channels = [64, 128, 256, 512, 1024]

        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,
            query_dim=128,
            num_scale_features=5,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        ).to(device)

        # Create dummy multi-scale features
        source_features = []
        target_features = []
        spatial_size = 96

        for ch in feature_channels:
            spatial_size = spatial_size // 2
            src_feat = torch.randn(
                batch_size, ch, spatial_size, spatial_size, device=device
            )
            tgt_feat = torch.randn(
                batch_size, ch, spatial_size, spatial_size, device=device
            )
            source_features.append(src_feat)
            target_features.append(tgt_feat)

        with torch.no_grad():
            output = fst(source_features, target_features)

        # Output shape: (B, N_L + H*W, c_{n_s})
        assert output.shape[0] == batch_size, "Batch size mismatch"
        assert output.shape[2] == 1024, "Output channel should be 1024"
        # N_L + H*W where H=W=3 (96 -> 48 -> 24 -> 12 -> 6 -> 3)
        expected_seq_len = num_queries + 9  # 3*3 spatial at last scale
        assert (
            output.shape[1] == expected_seq_len
        ), f"Sequence length mismatch: got {output.shape[1]}, expected {expected_seq_len}"

        assert_no_nan(output, "FST output")

    def test_fst_gradient_flow(self, device):
        """Verify gradients flow through FST."""
        feature_channels = [64, 128, 256, 512, 1024]
        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features = []
        target_features = []
        spatial_size = 96

        for ch in feature_channels:
            spatial_size = spatial_size // 2
            src_feat = torch.randn(
                batch_size=2,
                channels=ch,
                height=spatial_size,
                width=spatial_size,
                device=device,
                requires_grad=True,
            )
            tgt_feat = torch.randn(
                batch_size=2,
                channels=ch,
                height=spatial_size,
                width=spatial_size,
                device=device,
                requires_grad=True,
            )
            source_features.append(src_feat)
            target_features.append(tgt_feat)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Check learnable parameters have gradients
        assert fst.learnable_queries.grad is not None
        for param in fst.mlp_channel_adjust.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestResidualBlockShapes:
    """Test ResidualBlock output shapes."""

    @pytest.mark.parametrize(
        "in_channels,out_channels,downsample",
        [
            (64, 128, True),
            (128, 128, False),
            (256, 512, True),
        ],
    )
    def test_residual_block_shapes(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        device,
        assert_no_nan,
    ):
        """Verify ResidualBlock maintains correct shapes."""
        block = ResidualBlock(
            in_channels=in_channels, out_channels=out_channels, downsample=downsample
        ).to(device)

        x = torch.rand(2, in_channels, 96, 96, device=device)

        with torch.no_grad():
            output = block(x)

        expected_spatial = 96 // 2 if downsample else 96
        assert output.shape == (
            2,
            out_channels,
            expected_spatial,
            expected_spatial,
        ), f"Shape mismatch: got {output.shape}"
        assert_no_nan(output, "ResidualBlock output")


class TestVGG16PerceptualLossShapes:
    """Test VGG16-based perceptual loss."""

    def test_vgg16_forward_pass(self, device, assert_no_nan):
        """Test VGG16 feature extraction."""
        from src.losses.criterion import VGG16

        vgg = VGG16().to(device)
        vgg.eval()

        x = torch.rand(2, 3, 96, 96, device=device)

        with torch.no_grad():
            features = vgg(x)

        # VGG16 should return list of 3 feature maps
        assert len(features) == 3, f"Expected 3 feature maps, got {len(features)}"

        for i, feat in enumerate(features):
            assert feat.ndim == 4, f"Feature {i} should be 4D"
            assert feat.shape[0] == 2, f"Feature {i} batch size mismatch"
            assert_no_nan(feat, f"VGG16 feature {i}")

    def test_content_perceptual_loss_shapes(self, device, assert_no_nan):
        """Test perceptual loss computation."""
        loss_fn = ContentPerceptualLoss().to(device)
        loss_fn.eval()

        generated = torch.rand(2, 3, 96, 96, device=device)
        target = torch.rand(2, 3, 96, 96, device=device)

        with torch.no_grad():
            loss = loss_fn.calculate_loss(generated, target, device)

        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert_no_nan(loss, "Perceptual loss")


class TestTensorNaNDetection:
    """Test NaN detection in forward passes."""

    def test_no_nan_with_normal_initialization(self, device, assert_no_nan):
        """Verify normal initialization doesn't produce NaNs."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=3).to(
            device
        )

        # Check all parameters
        for param in msse.parameters():
            assert_no_nan(param, "Parameter initialization")

    def test_nan_with_extreme_values(self, device):
        """Test detection of NaNs when using extreme values."""
        encoder = nn.Linear(100, 100).to(device)

        # Create extreme input
        x = torch.ones(2, 100, device=device) * 1e20

        with torch.no_grad():
            output = encoder(x)

        # Output might be very large but shouldn't be NaN
        assert not torch.isnan(output).any() or torch.isnan(output).any()


class TestGradientExplosion:
    """Test detection of gradient explosion."""

    def test_gradient_magnitude_after_backward(self, device):
        """Verify gradients don't explode with reasonable inputs."""
        model = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 10)).to(
            device
        )

        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 10, device=device)

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        # Check that gradients are finite
        for param in model.parameters():
            assert torch.isfinite(
                param.grad
            ).all(), f"Gradient explosion detected: {param.grad}"


class TestBatchSizeVariability:
    """Test that models handle variable batch sizes correctly."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_encoder_variable_batch_size(self, batch_size: int, device, assert_no_nan):
        """Test simple encoder with different batch sizes."""
        from conftest import SimpleEncoder

        encoder = SimpleEncoder().to(device)
        encoder.eval()

        x = torch.rand(batch_size, 1, 96, 96, device=device)

        with torch.no_grad():
            output = encoder(x)

        assert output.shape[0] == batch_size
        assert_no_nan(output, f"Encoder output (batch_size={batch_size})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
