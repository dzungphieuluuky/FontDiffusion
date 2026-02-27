"""
Pytest suite for Font Style Transformation (FST) module.
Tests all components: CrossAttentionBlock, SelfAttentionBlock, FontStyleTransformationModule.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.fst import (
    CrossAttentionBlock,
    SelfAttentionBlock,
    SelfAttention,
    CrossAttention,
    AdaptivePositionalEncoding,
    FontStyleTransformationModule,
    TransformerBlock,
)


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def msse_channels():
    """Standard MSSE output channels (5 scales)."""
    return [64, 128, 256, 512, 1024]


class TestCrossAttentionBlock:
    """Test suite for CrossAttentionBlock."""

    @pytest.fixture
    def cross_attention(self, device):
        """Create CrossAttentionBlock instance."""
        return CrossAttentionBlock(
            query_dim=256,
            key_dim=512,
            value_dim=512,
            num_heads=8,
            dropout=0.1,
        ).to(device)

    def test_initialization(self, cross_attention):
        """Test model initialization."""
        assert cross_attention.num_heads == 8
        assert cross_attention.head_dim == 32  # 256 // 8
        assert cross_attention.scale == 32**-0.5

    def test_forward_shape(self, cross_attention, batch_size, device):
        """Test forward pass output shapes."""
        query = torch.randn(batch_size, 64, 256, device=device)
        key = torch.randn(batch_size, 128, 512, device=device)
        value = torch.randn(batch_size, 128, 512, device=device)

        output = cross_attention(query, key, value)

        assert output.shape == (batch_size, 64, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, cross_attention, batch_size, device):
        """Test gradient backpropagation."""
        query = torch.randn(batch_size, 64, 256, device=device, requires_grad=True)
        key = torch.randn(batch_size, 128, 512, device=device, requires_grad=True)
        value = torch.randn(batch_size, 128, 512, device=device, requires_grad=True)

        output = cross_attention(query, key, value)
        loss = output.mean()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not torch.isnan(query.grad).any()

    @pytest.mark.parametrize("use_flash_attn", [True, False])
    def test_flash_attention_modes(self, device, batch_size, use_flash_attn):
        """Test both flash attention and standard attention."""
        cross_attention = CrossAttentionBlock(
            query_dim=256,
            key_dim=512,
            value_dim=512,
            num_heads=8,
            use_flash_attn=use_flash_attn,
        ).to(device)

        query = torch.randn(batch_size, 64, 256, device=device)
        key = torch.randn(batch_size, 128, 512, device=device)
        value = torch.randn(batch_size, 128, 512, device=device)

        output = cross_attention(query, key, value)

        assert output.shape == (batch_size, 64, 256)
        assert not torch.isnan(output).any()


class TestSelfAttentionBlock:
    """Test suite for SelfAttentionBlock."""

    @pytest.fixture
    def self_attention_block(self, device):
        """Create SelfAttentionBlock instance."""
        return SelfAttentionBlock(dim=256, num_heads=8, dropout=0.1).to(device)

    def test_initialization(self, self_attention_block):
        """Test model initialization."""
        assert isinstance(self_attention_block.attn, SelfAttention)
        assert isinstance(self_attention_block.norm, nn.LayerNorm)
        assert isinstance(self_attention_block.ffn, nn.Sequential)

    def test_forward_shape(self, self_attention_block, batch_size, device):
        """Test forward pass output shapes."""
        x = torch.randn(batch_size, 64, 256, device=device)

        output = self_attention_block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_residual_connection(self, self_attention_block, batch_size, device):
        """Test that residual connections work properly."""
        x = torch.randn(batch_size, 64, 256, device=device)

        # Zero out FFN weights to test residual path
        with torch.no_grad():
            for module in self_attention_block.ffn:
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()

        output = self_attention_block(x)

        # Output should still have structure due to attention + residual
        assert output.shape == x.shape
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self, self_attention_block, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 64, 256, device=device, requires_grad=True)

        output = self_attention_block(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAdaptivePositionalEncoding:
    """Test suite for AdaptivePositionalEncoding."""

    @pytest.fixture
    def pos_encoding(self, device):
        """Create AdaptivePositionalEncoding instance."""
        return AdaptivePositionalEncoding(channels=256, max_h=48, max_w=48).to(device)

    def test_initialization(self, pos_encoding):
        """Test model initialization."""
        assert pos_encoding.channels == 256
        assert pos_encoding.height_embed.shape == (48, 128)
        assert pos_encoding.width_embed.shape == (48, 128)

    def test_forward_same_size(self, pos_encoding, batch_size, device):
        """Test forward with input size matching max_h, max_w."""
        x = torch.randn(batch_size, 256, 48, 48, device=device)

        output = pos_encoding(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_different_size(self, pos_encoding, batch_size, device):
        """Test forward with input size different from max_h, max_w."""
        x = torch.randn(batch_size, 256, 32, 32, device=device)

        output = pos_encoding(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_learnable_scale(self, pos_encoding, batch_size, device):
        """Test that scale parameter is learnable."""
        x = torch.randn(batch_size, 256, 48, 48, device=device, requires_grad=True)

        output = pos_encoding(x)
        loss = output.mean()
        loss.backward()

        assert pos_encoding.scale.grad is not None


class TestFontStyleTransformationModule:
    """Test suite for FontStyleTransformationModule."""

    @pytest.fixture
    def fst_module(self, msse_channels, device):
        """Create FontStyleTransformationModule instance."""
        return FontStyleTransformationModule(
            msse_output_channels=msse_channels,
            num_queries=220,
            query_dim=128,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        ).to(device)

    def test_initialization(self, fst_module, msse_channels):
        """Test model initialization."""
        assert fst_module.num_queries == 220
        assert fst_module.query_dim == 128
        assert fst_module.num_scales == 5
        assert fst_module.msse_channels == msse_channels
        assert fst_module.learnable_queries.shape == (220, 128)

    def test_forward_shape(self, fst_module, batch_size, device):
        """Test forward pass output shapes."""
        # Create mock MSSE features (5 scales with decreasing spatial size)
        source_features = [
            torch.randn(batch_size, 64, 48, 48, device=device),
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]
        target_features = [
            torch.randn(batch_size, 64, 48, 48, device=device),
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]

        output = fst_module(source_features, target_features)

        # Output should be (B, N_L + H*W, 1024)
        # where N_L = 220, H*W = 3*3 = 9
        expected_seq_len = 220 + 9
        assert output.shape == (batch_size, expected_seq_len, 1024)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, fst_module, batch_size, device):
        """Test gradient backpropagation through FST."""
        source_features = [
            torch.randn(batch_size, 64, 48, 48, device=device, requires_grad=True),
            torch.randn(batch_size, 128, 24, 24, device=device, requires_grad=True),
            torch.randn(batch_size, 256, 12, 12, device=device, requires_grad=True),
            torch.randn(batch_size, 512, 6, 6, device=device, requires_grad=True),
            torch.randn(batch_size, 1024, 3, 3, device=device, requires_grad=True),
        ]
        target_features = [
            torch.randn(batch_size, 64, 48, 48, device=device, requires_grad=True),
            torch.randn(batch_size, 128, 24, 24, device=device, requires_grad=True),
            torch.randn(batch_size, 256, 12, 12, device=device, requires_grad=True),
            torch.randn(batch_size, 512, 6, 6, device=device, requires_grad=True),
            torch.randn(batch_size, 1024, 3, 3, device=device, requires_grad=True),
        ]

        output = fst_module(source_features, target_features)
        loss = output.mean()
        loss.backward()

        # Check gradients exist for all scales
        for src_feat in source_features:
            assert src_feat.grad is not None
            assert not torch.isnan(src_feat.grad).any()

    def test_learnable_queries_updated(self, fst_module, batch_size, device):
        """Test that learnable queries receive gradients."""
        source_features = [
            torch.randn(batch_size, 64, 48, 48, device=device),
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]
        target_features = [
            torch.randn(batch_size, 64, 48, 48, device=device),
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]

        output = fst_module(source_features, target_features)
        loss = output.mean()
        loss.backward()

        assert fst_module.learnable_queries.grad is not None
        assert not torch.isnan(fst_module.learnable_queries.grad).any()

    def test_invalid_input_channels(self, fst_module, batch_size, device):
        """Test error handling for mismatched channels."""
        # Wrong channels for first scale (should be 64, not 128)
        source_features = [
            torch.randn(batch_size, 128, 48, 48, device=device),  # Wrong!
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]
        target_features = [
            torch.randn(batch_size, 64, 48, 48, device=device),
            torch.randn(batch_size, 128, 24, 24, device=device),
            torch.randn(batch_size, 256, 12, 12, device=device),
            torch.randn(batch_size, 512, 6, 6, device=device),
            torch.randn(batch_size, 1024, 3, 3, device=device),
        ]

        with pytest.raises(ValueError, match="Expected.*channels"):
            fst_module(source_features, target_features)


class TestSelfAttention:
    """Test suite for SelfAttention module."""

    @pytest.fixture
    def self_attention(self, device):
        """Create SelfAttention instance."""
        return SelfAttention(dim=256, num_heads=8, use_flash_attn=True).to(device)

    def test_forward_shape(self, self_attention, batch_size, device):
        """Test forward pass output shapes."""
        x = torch.randn(batch_size, 64, 256, device=device)

        output = self_attention(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, self_attention, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 64, 256, device=device, requires_grad=True)

        output = self_attention(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCrossAttention:
    """Test suite for CrossAttention module."""

    @pytest.fixture
    def cross_attention(self, device):
        """Create CrossAttention instance."""
        return CrossAttention(dim=256, num_heads=8, use_flash_attn=True).to(device)

    def test_forward_with_same_context_value(self, cross_attention, batch_size, device):
        """Test forward when context and value are the same."""
        x = torch.randn(batch_size, 64, 256, device=device)
        context = torch.randn(batch_size, 128, 256, device=device)

        output = cross_attention(x, context, value=None)

        assert output.shape == (batch_size, 64, 256)
        assert not torch.isnan(output).any()

    def test_forward_with_different_context_value(
        self, cross_attention, batch_size, device
    ):
        """Test forward when context and value are different."""
        x = torch.randn(batch_size, 64, 256, device=device)
        context = torch.randn(batch_size, 128, 256, device=device)
        value = torch.randn(batch_size, 128, 256, device=device)

        output = cross_attention(x, context, value)

        assert output.shape == (batch_size, 64, 256)
        assert not torch.isnan(output).any()


class TestIntegration:
    """Integration tests for FST components."""

    def test_fst_with_msse_pipeline(self, device, batch_size):
        """Test FST module with realistic MSSE output."""
        from src.modules.msse import MultiScaleStyleEncoder

        # Create MSSE encoder
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        # Get actual output channels
        msse_channels = msse.get_output_channels()

        # Create FST module with matching channels
        fst = FontStyleTransformationModule(
            msse_output_channels=msse_channels,
            num_queries=220,
            query_dim=128,
        ).to(device)

        # Simulate style images
        source_style = torch.randn(batch_size, 1, 96, 96, device=device)
        target_style = torch.randn(batch_size, 1, 96, 96, device=device)

        # Extract multi-scale features
        source_features = msse(source_style)
        target_features = msse(target_style)

        # FST transformation
        transformation = fst(source_features, target_features)

        # Check output
        assert transformation.shape[0] == batch_size
        assert transformation.shape[2] == 1024  # Final channel dimension
        assert not torch.isnan(transformation).any()

    def test_fst_different_content_same_style(self, device):
        """Test that FST handles different content with same style."""
        from src.modules.msse import MultiScaleStyleEncoder

        batch_size = 2

        # Create models
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )
        fst = FontStyleTransformationModule(
            msse_output_channels=msse.get_output_channels(),
            num_queries=220,
            query_dim=128,
        ).to(device)

        # Same style, different content
        style_A = torch.randn(batch_size, 1, 96, 96, device=device)
        style_B = style_A.clone()  # Same style

        source_features = msse(style_A)
        target_features = msse(style_B)

        transformation = fst(source_features, target_features)

        # Transformation should be close to identity (small magnitude)
        # Since source and target are the same style
        transformation_norm = torch.norm(transformation, dim=-1).mean()

        # This is a weak constraint, just checking it doesn't explode
        assert transformation_norm < 100.0


@pytest.mark.parametrize("num_queries", [128, 220, 256])
def test_various_query_sizes(device, batch_size, num_queries, msse_channels):
    """Parametrized test for different query sizes."""
    fst = FontStyleTransformationModule(
        msse_output_channels=msse_channels,
        num_queries=num_queries,
        query_dim=128,
    ).to(device)

    source_features = [
        torch.randn(batch_size, ch, 48 // (2**i), 48 // (2**i), device=device)
        for i, ch in enumerate(msse_channels)
    ]
    target_features = [
        torch.randn(batch_size, ch, 48 // (2**i), 48 // (2**i), device=device)
        for i, ch in enumerate(msse_channels)
    ]

    output = fst(source_features, target_features)

    # Check that num_queries is reflected in output
    expected_seq_len = num_queries + (3 * 3)  # Last scale spatial size
    assert output.shape == (batch_size, expected_seq_len, 1024)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])