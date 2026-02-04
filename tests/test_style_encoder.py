"""
Pytest suite for StyleEncoder module.
Tests all components: StyleEncoder, DBlock, GBlock, GBlock2, and utility layers.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.style_encoder import (
    StyleEncoder,
    DBlock,
    GBlock,
    GBlock2,
    LinearBlock,
    MLP,
    SNConv2d,
    SNLinear,
    style_encoder_textedit_addskip_arch,
)


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


class TestStyleEncoder:
    """Test suite for StyleEncoder."""

    @pytest.fixture
    def style_encoder_96(self, device):
        """Create StyleEncoder for 96x96 resolution."""
        return StyleEncoder(
            G_ch=64,
            resolution=96,
            input_nc=1,
            G_wide=True,
            G_activation=nn.ReLU(inplace=False),
        ).to(device)

    @pytest.fixture
    def style_encoder_128(self, device):
        """Create StyleEncoder for 128x128 resolution."""
        return StyleEncoder(
            G_ch=64,
            resolution=128,
            input_nc=1,
            G_wide=True,
            G_activation=nn.ReLU(inplace=False),
        ).to(device)

    def test_initialization_96(self, style_encoder_96):
        """Test model initialization for 96x96 resolution."""
        assert style_encoder_96.resolution == 96
        assert style_encoder_96.ch == 64
        assert len(style_encoder_96.blocks) == 6  # 5 DBlocks + 1 final layer
        assert style_encoder_96.save_featrues == [0, 1, 2, 3, 4]

    def test_initialization_128(self, style_encoder_128):
        """Test model initialization for 128x128 resolution."""
        assert style_encoder_128.resolution == 128
        assert style_encoder_128.ch == 64
        assert len(style_encoder_128.blocks) == 6

    def test_forward_96(self, style_encoder_96, batch_size, device):
        """Test forward pass for 96x96 resolution."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        style_emd, h, residual_features = style_encoder_96(x)

        # Check style embedding shape (spatial)
        assert style_emd.dim() == 4
        assert style_emd.shape[0] == batch_size

        # Check pooled vector shape
        assert h.dim() == 2
        assert h.shape[0] == batch_size

        # Check residual features
        assert isinstance(residual_features, list)
        assert len(residual_features) > 0

        # Check no NaN/Inf
        assert not torch.isnan(style_emd).any()
        assert not torch.isnan(h).any()
        assert not torch.isinf(style_emd).any()

    def test_forward_128(self, style_encoder_128, batch_size, device):
        """Test forward pass for 128x128 resolution."""
        x = torch.randn(batch_size, 1, 128, 128, device=device)

        style_emd, h, residual_features = style_encoder_128(x)

        assert style_emd.shape[0] == batch_size
        assert h.shape[0] == batch_size
        assert not torch.isnan(h).any()

    def test_adaptive_pooling(self, style_encoder_96, batch_size, device):
        """Test that adaptive average pooling produces 1x1 output."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        style_emd, h, _ = style_encoder_96(x)

        # h should be flattened from (B, C, 1, 1) to (B, C)
        assert h.dim() == 2

    def test_residual_features_count(self, style_encoder_96, batch_size, device):
        """Test that residual features are properly collected."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        _, _, residual_features = style_encoder_96(x)

        # Should have input + intermediate features
        expected_count = len(style_encoder_96.save_featrues)
        assert len(residual_features) == expected_count

    def test_gradient_flow(self, style_encoder_96, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 1, 96, 96, device=device, requires_grad=True)

        style_emd, h, residual_features = style_encoder_96(x)

        # Compute combined loss
        loss = style_emd.mean() + h.mean() + sum(rf.mean() for rf in residual_features)
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_spatial_reduction(self, style_encoder_96, batch_size, device):
        """Test that spatial dimensions are properly reduced."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        style_emd, h, residual_features = style_encoder_96(x)

        # Final style embedding should be much smaller than input
        assert style_emd.shape[2] < 96
        assert style_emd.shape[3] < 96

        # Residual features should show progressive downsampling
        for i in range(len(residual_features) - 1):
            current_h = residual_features[i].shape[2]
            next_h = residual_features[i + 1].shape[2]
            assert next_h <= current_h

    def test_channel_progression(self, style_encoder_96, batch_size, device):
        """Test that channels increase as spatial dims decrease."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        _, _, residual_features = style_encoder_96(x)

        # Channels should generally increase
        for i in range(len(residual_features) - 1):
            current_channels = residual_features[i].shape[1]
            next_channels = residual_features[i + 1].shape[1]
            assert next_channels >= current_channels

    def test_fp16_mode(self, device):
        """Test FP16 support."""
        encoder = StyleEncoder(
            G_ch=64,
            resolution=96,
            input_nc=1,
            G_fp16=True,
        ).to(device)

        assert encoder.fp16 is True

    @pytest.mark.parametrize("input_nc", [1, 3])
    def test_different_input_channels(self, device, batch_size, input_nc):
        """Test with different input channel counts."""
        encoder = StyleEncoder(
            G_ch=64,
            resolution=96,
            input_nc=input_nc,
        ).to(device)

        x = torch.randn(batch_size, input_nc, 96, 96, device=device)
        style_emd, h, residual_features = encoder(x)

        assert style_emd.shape[0] == batch_size
        assert h.shape[0] == batch_size
        assert not torch.isnan(h).any()

    def test_output_dimensions(self, style_encoder_96, batch_size, device):
        """Test that output dimensions match expectations."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        style_emd, h, residual_features = style_encoder_96(x)

        # Style embedding should be 4D (B, C, H, W)
        assert style_emd.dim() == 4

        # Pooled vector should be 2D (B, C)
        assert h.dim() == 2

        # All residuals should be 4D
        for rf in residual_features:
            assert rf.dim() == 4


class TestArchitecture:
    """Test style_encoder_textedit_addskip_arch function."""

    def test_arch_96(self):
        """Test architecture for 96x96 resolution."""
        arch = style_encoder_textedit_addskip_arch(
            ch=64, out_channel_multiplier=1, input_nc=1
        )

        assert 96 in arch
        assert "in_channels" in arch[96]
        assert "out_channels" in arch[96]
        assert "resolution" in arch[96]

    def test_arch_128(self):
        """Test architecture for 128x128 resolution."""
        arch = style_encoder_textedit_addskip_arch(
            ch=64, out_channel_multiplier=1, input_nc=1
        )

        assert 128 in arch
        assert len(arch[128]["in_channels"]) == len(arch[128]["out_channels"])

    def test_arch_256(self):
        """Test architecture for 256x256 resolution."""
        arch = style_encoder_textedit_addskip_arch(
            ch=64, out_channel_multiplier=1, input_nc=1
        )

        assert 256 in arch


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_1(self, device):
        """Test with batch size of 1."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.randn(1, 1, 96, 96, device=device)

        style_emd, h, residuals = encoder(x)

        assert style_emd.shape[0] == 1
        assert h.shape[0] == 1
        assert not torch.isnan(h).any()

    def test_large_batch_size(self, device):
        """Test with large batch size."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.randn(16, 1, 96, 96, device=device)

        style_emd, h, residuals = encoder(x)

        assert style_emd.shape[0] == 16
        assert h.shape[0] == 16

    def test_all_zeros_input(self, device):
        """Test with all-zero input."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.zeros(2, 1, 96, 96, device=device)

        style_emd, h, residuals = encoder(x)

        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_all_ones_input(self, device):
        """Test with all-ones input."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.ones(2, 1, 96, 96, device=device)

        style_emd, h, residuals = encoder(x)

        assert not torch.isnan(h).any()


class TestIntegration:
    """Integration tests with other modules."""

    def test_style_encoder_with_content_encoder(self, device):
        """Test StyleEncoder and ContentEncoder output compatibility."""
        from src.modules.content_encoder import ContentEncoder

        batch_size = 2

        style_encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        content_encoder = ContentEncoder(G_ch=64, resolution=96, input_nc=1).to(device)

        style_img = torch.randn(batch_size, 1, 96, 96, device=device)
        content_img = torch.randn(batch_size, 1, 96, 96, device=device)

        # Extract features
        style_emd, style_vec, style_residuals = style_encoder(style_img)
        content_h, content_residuals = content_encoder(content_img)

        # Both should have residual features
        assert len(style_residuals) > 0
        assert len(content_residuals) > 0

        # Both should produce valid outputs
        assert not torch.isnan(style_vec).any()
        assert not torch.isnan(content_h).any()

    def test_style_encoder_with_msse(self, device):
        """Test StyleEncoder compatibility with MSSE."""
        from src.modules.msse import MultiScaleStyleEncoder

        batch_size = 2

        style_encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        style_img = torch.randn(batch_size, 1, 96, 96, device=device)

        # Both encoders should process the same input
        style_emd, style_vec, style_residuals = style_encoder(style_img)
        msse_features = msse(style_img)

        # Both should produce valid multi-scale features
        assert len(style_residuals) > 0
        assert len(msse_features) == 5
        assert not torch.isnan(style_vec).any()

    def test_style_vector_for_conditioning(self, device):
        """Test that style vector can be used for conditioning."""
        batch_size = 4
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)

        style_img = torch.randn(batch_size, 1, 96, 96, device=device)
        _, style_vec, _ = encoder(style_img)

        # Style vector should be suitable for linear projection
        projection = nn.Linear(style_vec.shape[1], 768).to(device)
        projected = projection(style_vec)

        assert projected.shape == (batch_size, 768)
        assert not torch.isnan(projected).any()


class TestConsistency:
    """Test consistency and determinism."""

    def test_deterministic_output(self, device):
        """Test that same input produces same output."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        encoder.eval()

        x = torch.randn(2, 1, 96, 96, device=device)

        with torch.no_grad():
            style_emd1, h1, _ = encoder(x)
            style_emd2, h2, _ = encoder(x)

        assert torch.allclose(style_emd1, style_emd2, atol=1e-6)
        assert torch.allclose(h1, h2, atol=1e-6)

    def test_different_inputs_different_outputs(self, device):
        """Test that different inputs produce different outputs."""
        encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1).to(device)

        x1 = torch.randn(2, 1, 96, 96, device=device)
        x2 = torch.randn(2, 1, 96, 96, device=device)

        style_emd1, h1, _ = encoder(x1)
        style_emd2, h2, _ = encoder(x2)

        # Outputs should be different
        assert not torch.allclose(h1, h2)
        assert not torch.allclose(style_emd1, style_emd2)


@pytest.mark.parametrize("resolution", [96, 128, 256])
def test_various_resolutions(device, batch_size, resolution):
    """Parametrized test for different resolutions."""
    encoder = StyleEncoder(G_ch=64, resolution=resolution, input_nc=1).to(device)

    x = torch.randn(batch_size, 1, resolution, resolution, device=device)
    style_emd, h, residuals = encoder(x)

    assert style_emd.shape[0] == batch_size
    assert h.shape[0] == batch_size
    assert not torch.isnan(h).any()


@pytest.mark.parametrize("G_ch", [32, 64, 128])
def test_various_channel_counts(device, batch_size, G_ch):
    """Parametrized test for different base channel counts."""
    encoder = StyleEncoder(G_ch=G_ch, resolution=96, input_nc=1).to(device)

    x = torch.randn(batch_size, 1, 96, 96, device=device)
    style_emd, h, residuals = encoder(x)

    assert style_emd.shape[0] == batch_size
    assert h.shape[0] == batch_size
    assert not torch.isnan(h).any()


@pytest.mark.parametrize("G_wide", [True, False])
def test_wide_mode(device, batch_size, G_wide):
    """Parametrized test for wide mode."""
    encoder = StyleEncoder(G_ch=64, resolution=96, input_nc=1, G_wide=G_wide).to(
        device
    )

    x = torch.randn(batch_size, 1, 96, 96, device=device)
    style_emd, h, residuals = encoder(x)

    assert not torch.isnan(h).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])