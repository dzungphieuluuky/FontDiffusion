"""
Pytest suite for DualChannelContentEncoder integration with ContentEncoder.
Tests skeleton-distance transform preprocessing for content images.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.skeleton_distance_transform import (
    SkeletonDistanceTransform,
    DualChannelContentEncoder,
)
from src.modules.content_encoder import ContentEncoder


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def content_encoder(device):
    """Create real ContentEncoder instance."""
    encoder = ContentEncoder(
        G_ch=64,
        resolution=96,
        input_nc=1,  # Grayscale input
    )
    return encoder.to(device)


@pytest.fixture
def skeleton_transform():
    """Create SkeletonDistanceTransform instance."""
    return SkeletonDistanceTransform(
        method="medial_axis",
        distance_method="hybrid",
        max_distance=10.0,
        output_mode="dual_channel",
        normalize=True,
    )


@pytest.fixture(params=["concat", "add", "weighted"])
def fusion_method(request):
    """Parametrize fusion methods."""
    return request.param


class TestDualChannelContentEncoder:
    """Test suite for DualChannelContentEncoder wrapper."""

    def test_initialization_concat(self, content_encoder):
        """Test initialization with concat fusion method."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        )

        assert wrapper.fusion_method == "concat"
        assert hasattr(wrapper, "fusion_conv")
        assert isinstance(wrapper.fusion_conv, nn.Conv2d)
        assert wrapper.fusion_conv.in_channels == 2
        assert wrapper.fusion_conv.out_channels == 1

    def test_initialization_weighted(self, content_encoder):
        """Test initialization with weighted fusion method."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="weighted",
            learnable_weights=True,
        )

        assert wrapper.fusion_method == "weighted"
        assert hasattr(wrapper, "skeleton_weight")
        assert hasattr(wrapper, "distance_weight")
        assert isinstance(wrapper.skeleton_weight, nn.Parameter)
        assert isinstance(wrapper.distance_weight, nn.Parameter)

    def test_1channel_input(self, content_encoder, device):
        """Test forward pass with 1-channel input (normal mode)."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x_1ch = torch.randn(2, 1, 96, 96, device=device)
        h, residuals = wrapper(x_1ch)

        # Check output shapes
        assert h.dim() == 4
        assert isinstance(residuals, list)
        assert len(residuals) > 0
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_2channel_input(self, content_encoder, device, fusion_method):
        """Test forward pass with 2-channel input (skeleton mode)."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method=fusion_method,
            learnable_weights=True,
        ).to(device)

        x_2ch = torch.randn(2, 2, 96, 96, device=device)
        h, residuals = wrapper(x_2ch)

        # Check output shapes
        assert h.dim() == 4
        assert isinstance(residuals, list)
        assert len(residuals) > 0
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_output_consistency_between_modes(self, content_encoder, device):
        """Test that 1-channel and 2-channel modes produce consistent shapes."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        # 1-channel input
        x_1ch = torch.randn(2, 1, 96, 96, device=device)
        h_1ch, res_1ch = wrapper(x_1ch)

        # 2-channel input
        x_2ch = torch.randn(2, 2, 96, 96, device=device)
        h_2ch, res_2ch = wrapper(x_2ch)

        # Check shapes match
        assert h_1ch.shape == h_2ch.shape, "Final feature shapes should match"
        assert len(res_1ch) == len(res_2ch), "Residual counts should match"

        for i, (r1, r2) in enumerate(zip(res_1ch, res_2ch)):
            assert r1.shape == r2.shape, f"Residual {i} shapes should match"

    def test_invalid_channel_count(self, content_encoder, device):
        """Test that invalid channel counts raise appropriate errors."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        # Test with 3 channels (invalid)
        x_3ch = torch.randn(2, 3, 96, 96, device=device)

        with pytest.raises(ValueError, match="expects 1 or 2 channels"):
            wrapper(x_3ch)

    def test_gradient_flow_1channel(self, content_encoder, device):
        """Test gradient backpropagation with 1-channel input."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.randn(2, 1, 96, 96, device=device, requires_grad=True)
        h, _ = wrapper(x)
        loss = h.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_2channel(self, content_encoder, device):
        """Test gradient backpropagation with 2-channel input."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.randn(2, 2, 96, 96, device=device, requires_grad=True)
        h, _ = wrapper(x)
        loss = h.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert wrapper.fusion_conv.weight.grad is not None

    def test_fusion_weights_learnable(self, content_encoder, device):
        """Test that fusion weights are learnable when enabled."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="weighted",
            learnable_weights=True,
        ).to(device)

        # Store initial weights
        initial_skeleton_weight = wrapper.skeleton_weight.item()
        initial_distance_weight = wrapper.distance_weight.item()

        # Forward and backward pass
        x = torch.randn(4, 2, 96, 96, device=device, requires_grad=True)
        h, _ = wrapper(x)
        loss = h.mean()
        loss.backward()

        # Check gradients exist
        assert wrapper.skeleton_weight.grad is not None
        assert wrapper.distance_weight.grad is not None

    def test_fusion_weights_fixed(self, content_encoder, device):
        """Test that fusion weights are fixed when learnable=False."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="weighted",
            learnable_weights=False,
        ).to(device)

        # Check weights are buffers, not parameters
        assert not isinstance(wrapper.skeleton_weight, nn.Parameter)
        assert not isinstance(wrapper.distance_weight, nn.Parameter)

        # Forward and backward pass
        x = torch.randn(4, 2, 96, 96, device=device, requires_grad=True)
        h, _ = wrapper(x)
        loss = h.mean()
        loss.backward()

        # Buffers should not have gradients
        assert wrapper.skeleton_weight.grad is None
        assert wrapper.distance_weight.grad is None

    def test_repr(self, content_encoder):
        """Test string representation."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        )

        repr_str = repr(wrapper)

        assert "DualChannelContentEncoder" in repr_str
        assert "fusion_method=concat" in repr_str
        assert "1-channel (normal) or 2-channel (skeleton-distance)" in repr_str


class TestSkeletonIntegration:
    """Integration tests with SkeletonDistanceTransform."""

    def test_full_pipeline(self, content_encoder, skeleton_transform, device):
        """Test full skeleton transform → encoder pipeline."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="weighted",
            learnable_weights=True,
        ).to(device)

        # Simulate content image (normalized to [-1, 1])
        content_1ch = torch.randn(4, 1, 96, 96, device=device) * 0.5 + 0.5
        content_1ch = (content_1ch > 0.5).float()  # Binarize
        content_1ch = content_1ch * 2.0 - 1.0  # Normalize to [-1, 1]

        # Apply skeleton transform
        content_2ch = skeleton_transform(content_1ch)

        assert content_2ch.shape == (4, 2, 96, 96)
        assert not torch.isnan(content_2ch).any()

        # Pass through encoder
        h, residuals = wrapper(content_2ch)

        assert h.dim() == 4
        assert len(residuals) > 0
        assert not torch.isnan(h).any()

    def test_skeleton_channel_ranges(self, skeleton_transform, device):
        """Test that skeleton channels have expected value ranges."""
        # Create binary-like input
        content = torch.rand(2, 1, 96, 96, device=device)
        content = (content > 0.5).float()

        transformed = skeleton_transform(content)

        # Skeleton channel (0) should be binary-ish
        skeleton_ch = transformed[:, 0]
        assert skeleton_ch.min() >= 0.0
        assert skeleton_ch.max() <= 1.0

        # Distance channel (1) should be normalized
        distance_ch = transformed[:, 1]
        assert distance_ch.min() >= 0.0
        assert distance_ch.max() <= 1.0

    def test_gradient_through_full_pipeline(
        self, content_encoder, skeleton_transform, device
    ):
        """Test gradient flow through skeleton transform and encoder."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        # Create input
        content = torch.rand(2, 1, 96, 96, device=device, requires_grad=True)
        content_binary = (content > 0.5).float()

        # Apply skeleton transform (no gradients through skeletonization itself)
        with torch.no_grad():
            content_skeleton = skeleton_transform(content_binary)

        # Pass through encoder
        h, _ = wrapper(content_skeleton)
        loss = h.mean()
        loss.backward()

        # Check fusion layer has gradients
        assert wrapper.fusion_conv.weight.grad is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_1(self, content_encoder, device):
        """Test with batch size of 1."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.randn(1, 2, 96, 96, device=device)
        h, residuals = wrapper(x)

        assert h.shape[0] == 1
        assert not torch.isnan(h).any()

    def test_large_batch_size(self, content_encoder, device):
        """Test with large batch size."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.randn(16, 2, 96, 96, device=device)
        h, residuals = wrapper(x)

        assert h.shape[0] == 16
        assert not torch.isnan(h).any()

    def test_all_zeros_input(self, content_encoder, device):
        """Test with all-zero input (edge case)."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.zeros(2, 2, 96, 96, device=device)
        h, residuals = wrapper(x)

        # Should not crash, though output may be near-zero
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_all_ones_input(self, content_encoder, device):
        """Test with all-ones input (edge case)."""
        wrapper = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method="concat",
            learnable_weights=True,
        ).to(device)

        x = torch.ones(2, 2, 96, 96, device=device)
        h, residuals = wrapper(x)

        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_various_batch_sizes(content_encoder, device, batch_size):
    """Parametrized test for different batch sizes."""
    wrapper = DualChannelContentEncoder(
        original_encoder=content_encoder,
        fusion_method="concat",
        learnable_weights=True,
    ).to(device)

    x = torch.randn(batch_size, 2, 96, 96, device=device)
    h, residuals = wrapper(x)

    assert h.shape[0] == batch_size
    assert not torch.isnan(h).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])