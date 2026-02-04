"""
Pytest suite for Multi-Scale Style Encoder (MSSE).
Tests feature extraction at multiple scales with proper channel dimensions.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.msse import MultiScaleStyleEncoder, ResidualBlock


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    @pytest.fixture
    def residual_block_no_downsample(self, device):
        """Create ResidualBlock without downsampling."""
        return ResidualBlock(in_channels=64, out_channels=64, downsample=False).to(
            device
        )

    @pytest.fixture
    def residual_block_with_downsample(self, device):
        """Create ResidualBlock with downsampling."""
        return ResidualBlock(in_channels=64, out_channels=128, downsample=True).to(
            device
        )

    def test_initialization_no_downsample(self, residual_block_no_downsample):
        """Test initialization without downsampling."""
        assert isinstance(residual_block_no_downsample.conv1, nn.Conv2d)
        assert isinstance(residual_block_no_downsample.norm1, nn.InstanceNorm2d)
        assert len(residual_block_no_downsample.shortcut) == 0

    def test_initialization_with_downsample(self, residual_block_with_downsample):
        """Test initialization with downsampling."""
        assert len(residual_block_with_downsample.shortcut) > 0
        assert isinstance(residual_block_with_downsample.shortcut[0], nn.Conv2d)

    def test_forward_no_downsample(
        self, residual_block_no_downsample, batch_size, device
    ):
        """Test forward pass without downsampling."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        output = residual_block_no_downsample(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_with_downsample(
        self, residual_block_with_downsample, batch_size, device
    ):
        """Test forward pass with downsampling."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        output = residual_block_with_downsample(x)

        # Spatial dimensions halved, channels doubled
        assert output.shape == (batch_size, 128, 16, 16)
        assert not torch.isnan(output).any()

    def test_residual_connection(self, residual_block_no_downsample, batch_size, device):
        """Test that residual connection works."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        # Store input
        x_original = x.clone()

        output = residual_block_no_downsample(x)

        # Output should be different from input (not identity)
        assert not torch.allclose(output, x_original)

        # But should not be purely from convolution (residual adds structure)
        assert output.shape == x_original.shape

    def test_gradient_flow(self, residual_block_no_downsample, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 64, 32, 32, device=device, requires_grad=True)

        output = residual_block_no_downsample(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMultiScaleStyleEncoder:
    """Test suite for MultiScaleStyleEncoder."""

    @pytest.fixture
    def msse(self, device):
        """Create MultiScaleStyleEncoder instance."""
        return MultiScaleStyleEncoder(
            in_channels=1, base_channels=64, num_scales=5
        ).to(device)

    def test_initialization(self, msse):
        """Test model initialization."""
        assert msse.num_scales == 5
        assert len(msse.encoders) == 5
        assert len(msse.output_channels) == 5

    def test_output_channels(self, msse):
        """Test that output channels are correct."""
        expected_channels = [64, 128, 256, 512, 1024]
        assert msse.get_output_channels() == expected_channels
        assert msse.output_channels == expected_channels

    def test_forward_output_count(self, msse, batch_size, device):
        """Test that forward returns correct number of scales."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        features = msse(x)

        assert len(features) == 5
        assert isinstance(features, list)

    def test_forward_output_shapes(self, msse, batch_size, device):
        """Test that each scale has correct shape."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        features = msse(x)

        expected_shapes = [
            (batch_size, 64, 48, 48),
            (batch_size, 128, 24, 24),
            (batch_size, 256, 12, 12),
            (batch_size, 512, 6, 6),
            (batch_size, 1024, 3, 3),
        ]

        for i, (feat, expected_shape) in enumerate(zip(features, expected_shapes)):
            assert (
                feat.shape == expected_shape
            ), f"Scale {i}: got {feat.shape}, expected {expected_shape}"

    def test_forward_no_nan(self, msse, batch_size, device):
        """Test that outputs have no NaN values."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        features = msse(x)

        for i, feat in enumerate(features):
            assert not torch.isnan(feat).any(), f"Scale {i} contains NaN"
            assert not torch.isinf(feat).any(), f"Scale {i} contains Inf"

    def test_gradient_flow(self, msse, batch_size, device):
        """Test gradient backpropagation through all scales."""
        x = torch.randn(batch_size, 1, 96, 96, device=device, requires_grad=True)

        features = msse(x)

        # Compute loss from all scales
        loss = sum(feat.mean() for feat in features)
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_channel_progression(self, msse, batch_size, device):
        """Test that channels double at each scale."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        features = msse(x)

        for i in range(len(features) - 1):
            current_channels = features[i].shape[1]
            next_channels = features[i + 1].shape[1]
            assert next_channels == current_channels * 2

    def test_spatial_reduction(self, msse, batch_size, device):
        """Test that spatial dimensions halve at each scale."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        features = msse(x)

        for i in range(len(features) - 1):
            current_h, current_w = features[i].shape[2:]
            next_h, next_w = features[i + 1].shape[2:]
            assert next_h == current_h // 2
            assert next_w == current_w // 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_1(self, device):
        """Test with batch size of 1."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )
        x = torch.randn(1, 1, 96, 96, device=device)

        features = msse(x)

        assert len(features) == 5
        assert features[0].shape[0] == 1

    def test_large_batch_size(self, device):
        """Test with large batch size."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )
        x = torch.randn(16, 1, 96, 96, device=device)

        features = msse(x)

        assert len(features) == 5
        assert features[0].shape[0] == 16

    def test_different_input_sizes(self, device):
        """Test with different input resolutions."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        # Test different resolutions
        for resolution in [64, 96, 128]:
            x = torch.randn(2, 1, resolution, resolution, device=device)
            features = msse(x)

            assert len(features) == 5
            # First scale should be resolution // 2
            assert features[0].shape[2:] == (48, 48)

    def test_rgb_input(self, device):
        """Test with RGB input instead of grayscale."""
        msse = MultiScaleStyleEncoder(in_channels=3, base_channels=64, num_scales=5).to(
            device
        )
        x = torch.randn(4, 3, 96, 96, device=device)

        features = msse(x)

        assert len(features) == 5
        assert not torch.isnan(features[0]).any()


class TestIntegration:
    """Integration tests with other components."""

    def test_msse_with_fst_compatibility(self, device):
        """Test that MSSE output is compatible with FST input."""
        from src.modules.fst import FontStyleTransformationModule

        batch_size = 4

        # Create MSSE
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        # Get output channels
        msse_channels = msse.get_output_channels()

        # Create FST with matching channels
        fst = FontStyleTransformationModule(
            msse_output_channels=msse_channels,
            num_queries=220,
            query_dim=128,
        ).to(device)

        # Generate style features
        style_img = torch.randn(batch_size, 1, 96, 96, device=device)
        features = msse(style_img)

        # Verify FST can process them
        source_features = features
        target_features = [f.clone() for f in features]

        transformation = fst(source_features, target_features)

        assert transformation.shape[0] == batch_size
        assert not torch.isnan(transformation).any()

    def test_msse_feature_diversity(self, device):
        """Test that different scales capture different information."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        # Create two different style images
        style_A = torch.randn(1, 1, 96, 96, device=device)
        style_B = torch.randn(1, 1, 96, 96, device=device)

        features_A = msse(style_A)
        features_B = msse(style_B)

        # Features from different styles should be different at all scales
        for i, (feat_A, feat_B) in enumerate(zip(features_A, features_B)):
            diff = torch.norm(feat_A - feat_B)
            assert diff > 0.0, f"Scale {i}: features are identical"


@pytest.mark.parametrize("num_scales", [3, 4, 5, 6])
def test_various_scale_counts(device, batch_size, num_scales):
    """Parametrized test for different numbers of scales."""
    msse = MultiScaleStyleEncoder(
        in_channels=1, base_channels=64, num_scales=num_scales
    ).to(device)

    x = torch.randn(batch_size, 1, 96, 96, device=device)
    features = msse(x)

    assert len(features) == num_scales
    assert len(msse.get_output_channels()) == num_scales


@pytest.mark.parametrize("base_channels", [32, 64, 128])
def test_various_base_channels(device, batch_size, base_channels):
    """Parametrized test for different base channel counts."""
    msse = MultiScaleStyleEncoder(
        in_channels=1, base_channels=base_channels, num_scales=5
    ).to(device)

    x = torch.randn(batch_size, 1, 96, 96, device=device)
    features = msse(x)

    # Check channel progression
    expected_channels = [base_channels * (2**i) for i in range(5)]
    actual_channels = [feat.shape[1] for feat in features]

    assert actual_channels == expected_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])