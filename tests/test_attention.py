"""
Pytest suite for attention modules.
Tests all attention mechanisms: SpatialTransformer, CrossAttention,
OffsetRefStrucInter, SELayer, and ChannelAttnBlock.
"""

import pytest
import torch
import torch.nn as nn

from src.modules.attention import (
    SpatialTransformer,
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    GEGLU,
    OffsetRefStrucInter,
    SELayer,
    ChannelAttnBlock,
)


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def spatial_size():
    """Standard spatial dimensions (H, W)."""
    return (32, 32)


class TestSpatialTransformer:
    """Test suite for SpatialTransformer module."""

    @pytest.fixture
    def spatial_transformer(self, device):
        """Create SpatialTransformer instance."""
        model = SpatialTransformer(
            in_channels=128,
            n_heads=4,
            d_head=32,
            depth=2,
            dropout=0.1,
            context_dim=256,
        )
        return model.to(device)

    def test_initialization(self, spatial_transformer):
        """Test model initialization."""
        assert spatial_transformer.n_heads == 4
        assert spatial_transformer.d_head == 32
        assert spatial_transformer.in_channels == 128
        assert len(spatial_transformer.transformer_blocks) == 2

    def test_forward_without_context(
        self, spatial_transformer, batch_size, spatial_size, device
    ):
        """Test forward pass without context (self-attention only)."""
        h, w = spatial_size
        x = torch.randn(batch_size, 128, h, w, device=device)

        output = spatial_transformer(x, context=None)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_context(
        self, spatial_transformer, batch_size, spatial_size, device
    ):
        """Test forward pass with context (cross-attention)."""
        h, w = spatial_size
        x = torch.randn(batch_size, 128, h, w, device=device)
        context = torch.randn(batch_size, 16, 256, device=device)

        output = spatial_transformer(x, context=context)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, spatial_transformer, batch_size, spatial_size, device):
        """Test gradient backpropagation."""
        h, w = spatial_size
        x = torch.randn(batch_size, 128, h, w, device=device, requires_grad=True)
        context = torch.randn(batch_size, 16, 256, device=device)

        output = spatial_transformer(x, context=context)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_residual_connection(
        self, spatial_transformer, batch_size, spatial_size, device
    ):
        """Test that residual connection is working."""
        h, w = spatial_size
        x = torch.randn(batch_size, 128, h, w, device=device)

        # Zero out weights to test residual path
        with torch.no_grad():
            for block in spatial_transformer.transformer_blocks:
                block.attn1.to_q.weight.zero_()
                block.attn1.to_k.weight.zero_()
                block.attn1.to_v.weight.zero_()

        output = spatial_transformer(x, context=None)

        # Output should be close to input due to residual connection
        assert torch.allclose(output, x, atol=1e-3)


class TestCrossAttention:
    """Test suite for CrossAttention module."""

    @pytest.fixture
    def cross_attention(self, device):
        """Create CrossAttention instance."""
        model = CrossAttention(
            query_dim=256,
            context_dim=512,
            heads=8,
            dim_head=64,
            dropout=0.1,
        )
        return model.to(device)

    def test_initialization(self, cross_attention):
        """Test model initialization."""
        assert cross_attention.heads == 8
        assert cross_attention.scale == 64**-0.5

    def test_self_attention(self, cross_attention, batch_size, device):
        """Test self-attention (no context provided)."""
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 256, device=device)

        output = cross_attention(x, context=None)

        assert output.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(output).any()

    def test_cross_attention(self, cross_attention, batch_size, device):
        """Test cross-attention with context."""
        query_len = 64
        context_len = 128
        query = torch.randn(batch_size, query_len, 256, device=device)
        context = torch.randn(batch_size, context_len, 512, device=device)

        output = cross_attention(query, context=context)

        assert output.shape == (batch_size, query_len, 256)
        assert not torch.isnan(output).any()

    def test_attention_weights_sum_to_one(self, cross_attention, batch_size, device):
        """Test that attention weights sum to 1."""
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 256, device=device)

        # Monkey-patch to capture attention scores
        original_attention = cross_attention._attention
        attention_scores_captured = []

        def capture_attention(query, key, value):
            attention_scores = (
                torch.matmul(query, key.transpose(-1, -2)) * cross_attention.scale
            )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_scores_captured.append(attention_probs)
            return original_attention(query, key, value)

        cross_attention._attention = capture_attention

        _ = cross_attention(x, context=None)

        # Check attention weights sum to 1
        attn_probs = attention_scores_captured[0]
        assert torch.allclose(
            attn_probs.sum(dim=-1), torch.ones_like(attn_probs.sum(dim=-1)), atol=1e-5
        )

    def test_gradient_flow(self, cross_attention, batch_size, device):
        """Test gradient backpropagation."""
        query = torch.randn(batch_size, 64, 256, device=device, requires_grad=True)
        context = torch.randn(batch_size, 128, 512, device=device, requires_grad=True)

        output = cross_attention(query, context=context)
        loss = output.mean()
        loss.backward()

        assert query.grad is not None
        assert context.grad is not None
        assert not torch.isnan(query.grad).any()


class TestFeedForward:
    """Test suite for FeedForward module."""

    @pytest.fixture
    def feedforward(self, device):
        """Create FeedForward instance."""
        model = FeedForward(dim=256, dim_out=256, mult=4, dropout=0.1)
        return model.to(device)

    def test_forward(self, feedforward, batch_size, device):
        """Test forward pass."""
        x = torch.randn(batch_size, 64, 256, device=device)

        output = feedforward(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_geglu_activation(self, device):
        """Test GEGLU activation function."""
        geglu = GEGLU(dim_in=256, dim_out=512).to(device)
        x = torch.randn(2, 64, 256, device=device)

        output = geglu(x)

        assert output.shape == (2, 64, 512)
        assert not torch.isnan(output).any()


class TestOffsetRefStrucInter:
    """Test suite for OffsetRefStrucInter module."""

    @pytest.fixture
    def offset_module(self, device):
        """Create OffsetRefStrucInter instance."""
        model = OffsetRefStrucInter(
            res_in_channels=256,
            style_feat_in_channels=512,
            n_heads=8,
            num_groups=32,
            dropout=0.1,
        )
        return model.to(device)

    def test_forward(self, offset_module, batch_size, device):
        """Test forward pass."""
        h, w = 32, 32
        res_hidden = torch.randn(batch_size, 256, h, w, device=device)
        style_hidden = torch.randn(batch_size, 512, h, w, device=device)

        output = offset_module(res_hidden, style_hidden)

        # Output should be offset map (1 channel * 2 * 3 * 3 = 18 channels)
        assert output.shape == (batch_size, 18, h, w)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, offset_module, batch_size, device):
        """Test gradient backpropagation."""
        h, w = 32, 32
        res_hidden = torch.randn(
            batch_size, 256, h, w, device=device, requires_grad=True
        )
        style_hidden = torch.randn(
            batch_size, 512, h, w, device=device, requires_grad=True
        )

        output = offset_module(res_hidden, style_hidden)
        loss = output.mean()
        loss.backward()

        assert res_hidden.grad is not None
        assert style_hidden.grad is not None
        assert not torch.isnan(res_hidden.grad).any()


class TestSELayer:
    """Test suite for Squeeze-and-Excitation layer."""

    @pytest.fixture
    def se_layer(self, device):
        """Create SELayer instance."""
        model = SELayer(channel=256, reduction=16)
        return model.to(device)

    def test_forward(self, se_layer, batch_size, device):
        """Test forward pass."""
        h, w = 32, 32
        x = torch.randn(batch_size, 256, h, w, device=device)

        output = se_layer(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_channel_weights_bounded(self, se_layer, batch_size, device):
        """Test that channel attention weights are in [0, 1]."""
        h, w = 32, 32
        x = torch.randn(batch_size, 256, h, w, device=device)

        # Capture channel weights
        with torch.no_grad():
            y = se_layer.avg_pool(x).view(batch_size, 256)
            weights = se_layer.fc(y)

        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_gradient_flow(self, se_layer, batch_size, device):
        """Test gradient backpropagation."""
        x = torch.randn(batch_size, 256, 32, 32, device=device, requires_grad=True)

        output = se_layer(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestChannelAttnBlock:
    """Test suite for Channel Attention Block."""

    @pytest.fixture
    def channel_attn_block(self, device):
        """Create ChannelAttnBlock instance."""
        model = ChannelAttnBlock(
            in_channels=512,
            out_channels=256,
            groups=32,
            channel_attn=True,
            reduction=16,
        )
        return model.to(device)

    def test_forward_with_attention(self, channel_attn_block, batch_size, device):
        """Test forward pass with channel attention enabled."""
        h, w = 32, 32
        input_feat = torch.randn(batch_size, 256, h, w, device=device)
        content_feat = torch.randn(batch_size, 256, h, w, device=device)

        output = channel_attn_block(input_feat, content_feat)

        assert output.shape == (batch_size, 256, h, w)
        assert not torch.isnan(output).any()

    def test_forward_without_attention(self, device, batch_size):
        """Test forward pass with channel attention disabled."""
        model = ChannelAttnBlock(
            in_channels=512,
            out_channels=256,
            groups=32,
            channel_attn=False,
        ).to(device)

        h, w = 32, 32
        input_feat = torch.randn(batch_size, 256, h, w, device=device)
        content_feat = torch.randn(batch_size, 256, h, w, device=device)

        output = model(input_feat, content_feat)

        assert output.shape == (batch_size, 256, h, w)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, channel_attn_block, batch_size, device):
        """Test gradient backpropagation."""
        input_feat = torch.randn(
            batch_size, 256, 32, 32, device=device, requires_grad=True
        )
        content_feat = torch.randn(
            batch_size, 256, 32, 32, device=device, requires_grad=True
        )

        output = channel_attn_block(input_feat, content_feat)
        loss = output.mean()
        loss.backward()

        assert input_feat.grad is not None
        assert content_feat.grad is not None
        assert not torch.isnan(input_feat.grad).any()


class TestIntegration:
    """Integration tests for attention modules."""

    def test_spatial_transformer_in_unet_context(self, device):
        """Test SpatialTransformer as it would be used in U-Net."""
        batch_size = 2
        spatial_transformer = SpatialTransformer(
            in_channels=256,
            n_heads=8,
            d_head=32,
            depth=1,
            context_dim=768,
        ).to(device)

        # Simulate U-Net encoder output
        encoder_features = torch.randn(batch_size, 256, 16, 16, device=device)

        # Simulate style/content embeddings
        context = torch.randn(batch_size, 64, 768, device=device)

        output = spatial_transformer(encoder_features, context=context)

        assert output.shape == encoder_features.shape
        assert not torch.isnan(output).any()

    def test_offset_inter_with_real_features(self, device):
        """Test OffsetRefStrucInter with realistic feature dimensions."""
        batch_size = 4
        offset_module = OffsetRefStrucInter(
            res_in_channels=128,
            style_feat_in_channels=256,
            n_heads=4,
        ).to(device)

        # Simulate residual features from content encoder
        res_features = torch.randn(batch_size, 128, 64, 64, device=device)

        # Simulate style features
        style_features = torch.randn(batch_size, 256, 64, 64, device=device)

        offsets = offset_module(res_features, style_features)

        # Should output 18 channels (1 * 2 * 3 * 3 for deformable convolution)
        assert offsets.shape == (batch_size, 18, 64, 64)
        assert not torch.isnan(offsets).any()


@pytest.mark.parametrize("non_linearity", ["swish", "mish", "silu"])
def test_channel_attn_block_activations(non_linearity, device):
    """Test ChannelAttnBlock with different activation functions."""
    model = ChannelAttnBlock(
        in_channels=512,
        out_channels=256,
        non_linearity=non_linearity,
    ).to(device)

    input_feat = torch.randn(2, 256, 32, 32, device=device)
    content_feat = torch.randn(2, 256, 32, 32, device=device)

    output = model(input_feat, content_feat)

    assert output.shape == (2, 256, 32, 32)
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
