"""
Pytest suite for ContentEncoder module.
Tests all components: ContentEncoder, DBlock, GBlock, GBlock2, Attention, and utility layers.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.content_encoder import (
    ContentEncoder,
    DBlock,
    GBlock,
    GBlock2,
    Attention,
    LinearBlock,
    MLP,
    SNConv2d,
    SNLinear,
    content_encoder_arch,
)


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


class TestContentEncoder:
    """Test suite for ContentEncoder."""

    @pytest.fixture
    def content_encoder_96(self, device):
        """Create ContentEncoder for 96x96 resolution."""
        return ContentEncoder(
            G_ch=64,
            resolution=96,
            input_nc=1,
            G_wide=True,
            G_activation=nn.ReLU(inplace=False),
        ).to(device)

    @pytest.fixture
    def content_encoder_128(self, device):
        """Create ContentEncoder for 128x128 resolution."""
        return ContentEncoder(
            G_ch=64,
            resolution=128,
            input_nc=1,
            G_wide=True,
            G_activation=nn.ReLU(inplace=False),
        ).to(device)

    def test_initialization_96(self, content_encoder_96):
        """Test model initialization for 96x96 resolution."""
        assert content_encoder_96.resolution == 96
        assert content_encoder_96.ch == 64
        assert len(content_encoder_96.blocks) == 3
        assert content_encoder_96.save_featrues == [0, 1, 2, 3, 4]

    def test_initialization_128(self, content_encoder_128):
        """Test model initialization for 128x128 resolution."""
        assert content_encoder_128.resolution == 128
        assert content_encoder_128.ch == 64
        assert len(content_encoder_128.blocks) == 5

    def test_forward_96(self, content_encoder_96, batch_size, device):
        """Test forward pass for 96x96 resolution."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        h, residual_features = content_encoder_96(x)

        # Check output shapes
        assert h.dim() == 4
        assert h.shape[0] == batch_size
        assert isinstance(residual_features, list)
        assert len(residual_features) > 0
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_forward_128(self, content_encoder_128, batch_size, device):
        """Test forward pass for 128x128 resolution."""
        x = torch.randn(batch_size, 1, 128, 128, device=device)

        h, residual_features = content_encoder_128(x)

        # Check output shapes
        assert h.dim() == 4
        assert h.shape[0] == batch_size
        assert isinstance(residual_features, list)
        assert not torch.isnan(h).any()

    def test_residual_features_count(self, content_encoder_96, batch_size, device):
        """Test that residual features are properly collected."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        h, residual_features = content_encoder_96(x)

        # Should have input + intermediate features
        expected_count = len(content_encoder_96.save_featrues)
        assert len(residual_features) == expected_count

    def test_gradient_flow(self, content_encoder_96, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 1, 96, 96, device=device, requires_grad=True)

        h, residual_features = content_encoder_96(x)
        loss = h.mean() + sum(rf.mean() for rf in residual_features)
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_spatial_reduction(self, content_encoder_96, batch_size, device):
        """Test that spatial dimensions are properly reduced."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        h, residual_features = content_encoder_96(x)

        # Final feature map should be smaller than input
        assert h.shape[2] < 96
        assert h.shape[3] < 96

        # Residual features should show progressive downsampling
        for i in range(len(residual_features) - 1):
            current_h = residual_features[i].shape[2]
            next_h = residual_features[i + 1].shape[2]
            assert next_h <= current_h

    def test_channel_progression(self, content_encoder_96, batch_size, device):
        """Test that channels increase as spatial dims decrease."""
        x = torch.randn(batch_size, 1, 96, 96, device=device)

        h, residual_features = content_encoder_96(x)

        # Channels should generally increase
        for i in range(len(residual_features) - 1):
            current_channels = residual_features[i].shape[1]
            next_channels = residual_features[i + 1].shape[1]
            assert next_channels >= current_channels

    def test_fp16_mode(self, device):
        """Test FP16 support."""
        encoder = ContentEncoder(
            G_ch=64,
            resolution=96,
            input_nc=1,
            G_fp16=True,
        ).to(device)

        assert encoder.fp16 is True

    @pytest.mark.parametrize("input_nc", [1, 3])
    def test_different_input_channels(self, device, batch_size, input_nc):
        """Test with different input channel counts."""
        encoder = ContentEncoder(
            G_ch=64,
            resolution=96,
            input_nc=input_nc,
        ).to(device)

        x = torch.randn(batch_size, input_nc, 96, 96, device=device)
        h, residual_features = encoder(x)

        assert h.shape[0] == batch_size
        assert not torch.isnan(h).any()


class TestDBlock:
    """Test suite for DBlock (downsampling block)."""

    @pytest.fixture
    def dblock(self, device):
        """Create DBlock instance."""
        return DBlock(
            in_channels=64,
            out_channels=128,
            which_conv=SNConv2d,
            wide=True,
            activation=nn.ReLU(inplace=False),
            downsample=nn.AvgPool2d(2),
        ).to(device)

    def test_initialization(self, dblock):
        """Test DBlock initialization."""
        assert dblock.in_channels == 64
        assert dblock.out_channels == 128
        assert dblock.learnable_sc is True

    def test_forward_with_downsample(self, dblock, batch_size, device):
        """Test forward pass with downsampling."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        output = dblock(x)

        # Channels should increase, spatial dims should decrease
        assert output.shape == (batch_size, 128, 16, 16)
        assert not torch.isnan(output).any()

    def test_forward_without_downsample(self, batch_size, device):
        """Test forward pass without downsampling."""
        dblock = DBlock(
            in_channels=64,
            out_channels=64,
            which_conv=SNConv2d,
            activation=nn.ReLU(inplace=False),
            downsample=None,
        ).to(device)

        x = torch.randn(batch_size, 64, 32, 32, device=device)
        output = dblock(x)

        # Spatial dims should stay same
        assert output.shape == (batch_size, 64, 32, 32)

    def test_residual_connection(self, dblock, batch_size, device):
        """Test that residual connection works."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        output = dblock(x)

        # Output should not be zeros (residual adds structure)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self, dblock, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 64, 32, 32, device=device, requires_grad=True)

        output = dblock(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGBlock:
    """Test suite for GBlock (upsampling block with BatchNorm)."""

    @pytest.fixture
    def gblock(self, device):
        """Create GBlock instance."""
        return GBlock(
            in_channels=128,
            out_channels=64,
            which_conv=nn.Conv2d,
            which_bn=nn.BatchNorm2d,
            activation=nn.ReLU(inplace=False),
            upsample=nn.Upsample(scale_factor=2, mode="nearest"),
        ).to(device)

    def test_initialization(self, gblock):
        """Test GBlock initialization."""
        assert gblock.in_channels == 128
        assert gblock.out_channels == 64
        assert gblock.learnable_sc is True

    def test_forward_with_upsample(self, gblock, batch_size, device):
        """Test forward pass with upsampling."""
        x = torch.randn(batch_size, 128, 16, 16, device=device)

        output = gblock(x)

        # Channels should decrease, spatial dims should increase
        assert output.shape == (batch_size, 64, 32, 32)
        assert not torch.isnan(output).any()

    def test_forward_without_upsample(self, batch_size, device):
        """Test forward pass without upsampling."""
        gblock = GBlock(
            in_channels=64,
            out_channels=64,
            which_conv=nn.Conv2d,
            which_bn=nn.BatchNorm2d,
            activation=nn.ReLU(inplace=False),
            upsample=None,
        ).to(device)

        x = torch.randn(batch_size, 64, 32, 32, device=device)
        output = gblock(x)

        assert output.shape == (batch_size, 64, 32, 32)

    def test_gradient_flow(self, gblock, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 128, 16, 16, device=device, requires_grad=True)

        output = gblock(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGBlock2:
    """Test suite for GBlock2 (upsampling block without BatchNorm)."""

    @pytest.fixture
    def gblock2(self, device):
        """Create GBlock2 instance."""
        return GBlock2(
            in_channels=128,
            out_channels=64,
            which_conv=nn.Conv2d,
            activation=nn.ReLU(inplace=False),
            upsample=nn.Upsample(scale_factor=2, mode="nearest"),
            skip_connection=True,
        ).to(device)

    def test_initialization(self, gblock2):
        """Test GBlock2 initialization."""
        assert gblock2.in_channels == 128
        assert gblock2.out_channels == 64
        assert gblock2.skip_connection is True

    def test_forward_with_skip(self, gblock2, batch_size, device):
        """Test forward pass with skip connection."""
        x = torch.randn(batch_size, 128, 16, 16, device=device)

        output = gblock2(x)

        assert output.shape == (batch_size, 64, 32, 32)
        assert not torch.isnan(output).any()

    def test_forward_without_skip(self, batch_size, device):
        """Test forward pass without skip connection."""
        gblock2 = GBlock2(
            in_channels=128,
            out_channels=64,
            which_conv=nn.Conv2d,
            activation=nn.ReLU(inplace=False),
            upsample=nn.Upsample(scale_factor=2, mode="nearest"),
            skip_connection=False,
        ).to(device)

        x = torch.randn(batch_size, 128, 16, 16, device=device)
        output = gblock2(x)

        assert output.shape == (batch_size, 64, 32, 32)


class TestAttention:
    """Test suite for self-attention module."""

    @pytest.fixture
    def attention(self, device):
        """Create Attention instance."""
        return Attention(ch=256, which_conv=SNConv2d).to(device)

    def test_initialization(self, attention):
        """Test Attention initialization."""
        assert attention.ch == 256
        assert hasattr(attention, "gamma")
        assert attention.gamma.requires_grad is True

    def test_forward_shape(self, attention, batch_size, device):
        """Test forward pass output shape."""
        x = torch.randn(batch_size, 256, 32, 32, device=device)

        output = attention(x)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, attention, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 256, 32, 32, device=device, requires_grad=True)

        output = attention(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert attention.gamma.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gamma_learnable(self, attention, batch_size, device):
        """Test that gamma parameter is learnable."""
        initial_gamma = attention.gamma.item()

        x = torch.randn(batch_size, 256, 32, 32, device=device)
        output = attention(x)
        loss = output.mean()
        loss.backward()

        # Gradient should exist
        assert attention.gamma.grad is not None


class TestLinearBlock:
    """Test suite for LinearBlock."""

    @pytest.fixture
    def linear_block(self, device):
        """Create LinearBlock instance."""
        return LinearBlock(in_dim=256, out_dim=512, norm="bn", act="relu").to(device)

    def test_initialization(self, linear_block):
        """Test LinearBlock initialization."""
        assert isinstance(linear_block.fc, nn.Linear)
        assert isinstance(linear_block.norm, nn.BatchNorm1d)
        assert isinstance(linear_block.activation, nn.ReLU)

    def test_forward(self, linear_block, batch_size, device):
        """Test forward pass."""
        x = torch.randn(batch_size, 256, device=device)

        output = linear_block(x)

        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()

    def test_without_norm(self, batch_size, device):
        """Test LinearBlock without normalization."""
        block = LinearBlock(in_dim=256, out_dim=512, norm="none", act="relu").to(
            device
        )
        x = torch.randn(batch_size, 256, device=device)

        output = block(x)

        assert output.shape == (batch_size, 512)


class TestMLP:
    """Test suite for MLP."""

    @pytest.fixture
    def mlp(self, device):
        """Create MLP instance."""
        return MLP(
            nf_in=256, nf_out=512, nf_mlp=1024, num_blocks=3, norm="bn", act="relu"
        ).to(device)

    def test_initialization(self, mlp):
        """Test MLP initialization."""
        assert isinstance(mlp.model, nn.Sequential)
        assert len(mlp.model) == 3  # num_blocks

    def test_forward(self, mlp, batch_size, device):
        """Test forward pass."""
        x = torch.randn(batch_size, 256, device=device)

        output = mlp(x)

        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, mlp, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 256, device=device, requires_grad=True)

        output = mlp(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None


class TestSNConv2d:
    """Test suite for Spectral Normalized Conv2d."""

    @pytest.fixture
    def sn_conv(self, device):
        """Create SNConv2d instance."""
        return SNConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            num_svs=1,
            num_itrs=1,
        ).to(device)

    def test_initialization(self, sn_conv):
        """Test SNConv2d initialization."""
        assert sn_conv.in_channels == 64
        assert sn_conv.out_channels == 128
        assert hasattr(sn_conv, "u0")
        assert hasattr(sn_conv, "sv0")

    def test_forward(self, sn_conv, batch_size, device):
        """Test forward pass with spectral normalization."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        output = sn_conv(x)

        assert output.shape == (batch_size, 128, 32, 32)
        assert not torch.isnan(output).any()

    def test_spectral_norm_effect(self, sn_conv, batch_size, device):
        """Test that spectral normalization is applied."""
        x = torch.randn(batch_size, 64, 32, 32, device=device)

        # Forward with SN
        output_sn = sn_conv(x)

        # Forward without SN
        output_no_sn = sn_conv.forward_wo_sn(x)

        # Outputs should be different
        assert not torch.allclose(output_sn, output_no_sn)


class TestSNLinear:
    """Test suite for Spectral Normalized Linear."""

    @pytest.fixture
    def sn_linear(self, device):
        """Create SNLinear instance."""
        return SNLinear(in_features=256, out_features=512, num_svs=1, num_itrs=1).to(
            device
        )

    def test_initialization(self, sn_linear):
        """Test SNLinear initialization."""
        assert sn_linear.in_features == 256
        assert sn_linear.out_features == 512
        assert hasattr(sn_linear, "u0")

    def test_forward(self, sn_linear, batch_size, device):
        """Test forward pass."""
        x = torch.randn(batch_size, 256, device=device)

        output = sn_linear(x)

        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()


class TestArchitecture:
    """Test content_encoder_arch function."""

    def test_arch_96(self):
        """Test architecture for 96x96 resolution."""
        arch = content_encoder_arch(ch=64, out_channel_multiplier=1, input_nc=1)

        assert 96 in arch
        assert "in_channels" in arch[96]
        assert "out_channels" in arch[96]
        assert "resolution" in arch[96]

    def test_arch_128(self):
        """Test architecture for 128x128 resolution."""
        arch = content_encoder_arch(ch=64, out_channel_multiplier=1, input_nc=1)

        assert 128 in arch
        assert len(arch[128]["in_channels"]) == len(arch[128]["out_channels"])

    def test_arch_256(self):
        """Test architecture for 256x256 resolution."""
        arch = content_encoder_arch(ch=64, out_channel_multiplier=1, input_nc=1)

        assert 256 in arch


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_1(self, device):
        """Test with batch size of 1."""
        encoder = ContentEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.randn(1, 1, 96, 96, device=device)

        h, residuals = encoder(x)

        assert h.shape[0] == 1
        assert not torch.isnan(h).any()

    def test_large_batch_size(self, device):
        """Test with large batch size."""
        encoder = ContentEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.randn(16, 1, 96, 96, device=device)

        h, residuals = encoder(x)

        assert h.shape[0] == 16

    def test_all_zeros_input(self, device):
        """Test with all-zero input."""
        encoder = ContentEncoder(G_ch=64, resolution=96, input_nc=1).to(device)
        x = torch.zeros(2, 1, 96, 96, device=device)

        h, residuals = encoder(x)

        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()


class TestIntegration:
    """Integration tests with other modules."""

    def test_content_encoder_with_unet(self, device):
        """Test ContentEncoder output compatibility with U-Net."""
        batch_size = 2
        encoder = ContentEncoder(G_ch=64, resolution=96, input_nc=1).to(device)

        x = torch.randn(batch_size, 1, 96, 96, device=device)
        h, residuals = encoder(x)

        # Check that residual features can be used for skip connections
        assert len(residuals) > 0
        for res in residuals:
            assert res.dim() == 4
            assert res.shape[0] == batch_size


@pytest.mark.parametrize("resolution", [96, 128, 256])
def test_various_resolutions(device, batch_size, resolution):
    """Parametrized test for different resolutions."""
    encoder = ContentEncoder(G_ch=64, resolution=resolution, input_nc=1).to(device)

    x = torch.randn(batch_size, 1, resolution, resolution, device=device)
    h, residuals = encoder(x)

    assert h.shape[0] == batch_size
    assert not torch.isnan(h).any()


@pytest.mark.parametrize("G_ch", [32, 64, 128])
def test_various_channel_counts(device, batch_size, G_ch):
    """Parametrized test for different base channel counts."""
    encoder = ContentEncoder(G_ch=G_ch, resolution=96, input_nc=1).to(device)

    x = torch.randn(batch_size, 1, 96, 96, device=device)
    h, residuals = encoder(x)

    assert h.shape[0] == batch_size
    assert not torch.isnan(h).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])