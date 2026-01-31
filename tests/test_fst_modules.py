import pytest
import torch
import torch.nn as nn

# Import your modules
# Ensure msse.py and fst.py are in the python path
from src.modules.msse import MultiScaleStyleEncoder, ResidualBlock
from src.modules.fst import (
    CrossAttentionBlock,
    AdaptivePositionalEncoding,
    FontStyleTransformationModule,
)

# --- Fixtures ---


@pytest.fixture
def device():
    """Returns the available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def img_size():
    return 96


# --- MSSE Tests ---


def test_residual_block_shape(device, batch_size):
    """Test ResidualBlock maintains or changes shape correctly."""
    # Case 1: No downsample, same channels
    block = ResidualBlock(64, 64, downsample=False).to(device)
    x = torch.randn(batch_size, 64, 32, 32).to(device)
    out = block(x)
    assert out.shape == (batch_size, 64, 32, 32)

    # Case 2: Downsample, double channels
    block_down = ResidualBlock(64, 128, downsample=True).to(device)
    out_down = block_down(x)
    # Stride 2 should halve dimensions: 32 -> 16
    assert out_down.shape == (batch_size, 128, 16, 16)


def test_msse_output_shapes(device, batch_size, img_size):
    """Test MultiScaleStyleEncoder outputs correct pyramid shapes."""
    in_channels = 3
    base_channels = 64
    num_scales = 4

    encoder = MultiScaleStyleEncoder(
        in_channels=in_channels, base_channels=base_channels, num_scales=num_scales
    ).to(device)

    x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    features = encoder(x)

    # 1. Check number of outputs
    assert len(features) == num_scales

    # 2. Check shapes per scale
    # Expected resolutions based on: 48 // (2**i)
    expected_resolutions = [48, 24, 12, 6]

    for i, feat in enumerate(features):
        expected_ch = base_channels * (2**i)
        expected_res = expected_resolutions[i]

        print(
            f"Scale {i}: Expected ({expected_ch}, {expected_res}), Got {feat.shape[1:]}"
        )

        assert feat.shape[0] == batch_size
        assert feat.shape[1] == expected_ch
        assert feat.shape[2] == expected_res
        assert feat.shape[3] == expected_res


def test_msse_get_channels():
    """Verify get_output_channels returns correct list."""
    encoder = MultiScaleStyleEncoder(base_channels=32, num_scales=3)
    expected = [32, 64, 128]
    assert encoder.get_output_channels() == expected


# --- FST Tests ---


def test_adaptive_positional_encoding(device, batch_size):
    """Test Positional Encoding adds to tensor without changing shape."""
    channels = 64
    h, w = 24, 24
    ape = AdaptivePositionalEncoding(channels, max_h=48, max_w=48).to(device)

    x = torch.randn(batch_size, channels, h, w).to(device)
    out = ape(x)

    assert out.shape == x.shape
    # Verify it has learned parameters (height/width embeds)
    assert ape.height_embed.requires_grad
    assert ape.width_embed.requires_grad


def test_cross_attention_block(device, batch_size):
    """Test Cross Attention inputs and outputs."""
    query_dim = 64
    context_dim = 128
    seq_len_q = 10
    seq_len_kv = 20

    attn = CrossAttentionBlock(query_dim, context_dim, context_dim, num_heads=4).to(
        device
    )

    q = torch.randn(batch_size, seq_len_q, query_dim).to(device)
    k = torch.randn(batch_size, seq_len_kv, context_dim).to(device)
    v = torch.randn(batch_size, seq_len_kv, context_dim).to(device)

    out = attn(q, k, v)

    # Output should match query shape
    assert out.shape == (batch_size, seq_len_q, query_dim)


def test_fst_module_flow(device, batch_size):
    """
    Test the full FontStyleTransformationModule integration.
    Verifies that source/target features flow through to the correct output shape.
    """
    # Configuration based on MSSE output logic
    msse_channels = [64, 128, 256]
    spatial_sizes = [48, 24, 12]

    num_queries = 50
    query_dim = 32

    fst = FontStyleTransformationModule(
        msse_output_channels=msse_channels,
        num_queries=num_queries,
        query_dim=query_dim,
        num_cross_attn_blocks=1,
        num_self_attn_blocks=1,
    ).to(device)

    # Create dummy inputs mimicking MSSE output
    source_feats = []
    target_feats = []

    for ch, size in zip(msse_channels, spatial_sizes):
        src = torch.randn(batch_size, ch, size, size).to(device)
        tgt = torch.randn(batch_size, ch, size, size).to(device)
        source_feats.append(src)
        target_feats.append(tgt)

    # Run Forward Pass
    output = fst(source_feats, target_feats)

    # Expected Output Calculation:
    # 1. Learnable queries part: num_queries
    # 2. Residual part: Last scale spatial dim (12 * 12 = 144)
    # 3. Output dim: Last channel dim (256)

    expected_seq_len = num_queries + (spatial_sizes[-1] * spatial_sizes[-1])
    expected_dim = msse_channels[-1]

    print(f"FST Output Shape: {output.shape}")

    assert output.shape[0] == batch_size
    assert output.shape[1] == expected_seq_len
    assert output.shape[2] == expected_dim


def test_fst_channel_mismatch_error(device):
    """Test that FST raises ValueError if input channels don't match config."""
    msse_channels = [64, 128]
    fst = FontStyleTransformationModule(msse_output_channels=msse_channels).to(device)

    # Pass inputs with 32 channels instead of 64
    bad_src = [
        torch.randn(1, 32, 48, 48).to(device),
        torch.randn(1, 128, 24, 24).to(device),
    ]
    bad_tgt = [
        torch.randn(1, 32, 48, 48).to(device),
        torch.randn(1, 128, 24, 24).to(device),
    ]

    # Use pytest.raises to assert exception
    with pytest.raises(ValueError):
        fst(bad_src, bad_tgt)
