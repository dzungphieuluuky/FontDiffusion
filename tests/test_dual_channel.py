"""
Test corrected skeleton-distance transform integration with ContentEncoder.
"""
import torch
import torch.nn as nn
from src.modules.skeleton_distance_transform import (
    SkeletonDistanceTransform,
    DualChannelContentEncoder,
)
from src.modules.content_encoder import ContentEncoder


def test_dual_channel_wrapper_with_real_encoder():
    """Test wrapper with actual ContentEncoder from codebase."""
    
    # Create real ContentEncoder
    content_encoder = ContentEncoder(
        G_ch=64,
        resolution=96,
        input_nc=1,  # Grayscale input
    )
    
    # Wrap it
    wrapper = DualChannelContentEncoder(
        original_encoder=content_encoder,
        fusion_method="concat",
        learnable_weights=True,
    )
    
    print("Testing DualChannelContentEncoder with real ContentEncoder...")
    print("=" * 60)
    
    # Test 1: 1-channel input (normal mode)
    print("\nTest 1: 1-channel input (normal mode)")
    x_1ch = torch.randn(2, 1, 96, 96)
    h_1ch, res_1ch = wrapper(x_1ch)
    print(f"  Input shape: {x_1ch.shape}")
    print(f"  Output h shape: {h_1ch.shape}")
    print(f"  Num residual features: {len(res_1ch)}")
    print(f"  Residual shapes: {[r.shape for r in res_1ch]}")
    print("  ✓ 1-channel input works!")
    
    # Test 2: 2-channel input (skeleton mode)
    print("\nTest 2: 2-channel input (skeleton mode)")
    x_2ch = torch.randn(2, 2, 96, 96)
    h_2ch, res_2ch = wrapper(x_2ch)
    print(f"  Input shape: {x_2ch.shape}")
    print(f"  Output h shape: {h_2ch.shape}")
    print(f"  Num residual features: {len(res_2ch)}")
    print(f"  Residual shapes: {[r.shape for r in res_2ch]}")
    print("  ✓ 2-channel input works!")
    
    # Test 3: Verify output shapes match
    print("\nTest 3: Verify output consistency")
    assert h_1ch.shape == h_2ch.shape, "Output shapes should match!"
    assert len(res_1ch) == len(res_2ch), "Residual feature counts should match!"
    for i, (r1, r2) in enumerate(zip(res_1ch, res_2ch)):
        assert r1.shape == r2.shape, f"Residual {i} shapes should match!"
    print("  ✓ Output shapes are consistent between 1-ch and 2-ch modes!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")


def test_full_pipeline_with_skeleton():
    """Test full skeleton transform → encoder pipeline with real ContentEncoder."""
    
    print("\nTesting full skeleton → ContentEncoder pipeline...")
    print("=" * 60)
    
    # Create skeleton transform
    skeleton_transform = SkeletonDistanceTransform(
        method="medial_axis",
        distance_method="hybrid",
        max_distance=10.0,
        output_mode="dual_channel",
    )
    
    # Create real ContentEncoder
    content_encoder = ContentEncoder(
        G_ch=64,
        resolution=96,
        input_nc=1,
    )
    
    # Wrap it
    wrapper = DualChannelContentEncoder(
        original_encoder=content_encoder,
        fusion_method="weighted",
        learnable_weights=True,
    )
    
    # Simulate content image (normalized to [-1, 1] like in training)
    content_1ch = torch.randn(4, 1, 96, 96) * 0.5 + 0.5  # [0, 1] range
    content_1ch = (content_1ch > 0.5).float()  # Binarize
    content_1ch = content_1ch * 2.0 - 1.0  # Normalize to [-1, 1]
    
    print(f"Original content shape: {content_1ch.shape}")
    print(f"Original content range: [{content_1ch.min():.2f}, {content_1ch.max():.2f}]")
    
    # Apply skeleton transform
    content_2ch = skeleton_transform(content_1ch)
    
    print(f"After skeleton transform: {content_2ch.shape}")
    print(f"Skeleton channel range: [{content_2ch[:, 0].min():.2f}, {content_2ch[:, 0].max():.2f}]")
    print(f"Distance channel range: [{content_2ch[:, 1].min():.2f}, {content_2ch[:, 1].max():.2f}]")
    
    # Pass through wrapper
    h, residuals = wrapper(content_2ch)
    
    print(f"\nEncoder output h shape: {h.shape}")
    print(f"Number of residual features: {len(residuals)}")
    print(f"Residual feature shapes: {[r.shape for r in residuals]}")
    
    # Verify gradient flow
    loss = h.mean()
    loss.backward()
    
    print("\n✓ Gradient flow verified!")
    print(f"Fusion conv gradient norm: {wrapper.fusion_conv.weight.grad.norm().item():.6f}")
    
    print("=" * 60)
    print("✓ Full pipeline works correctly!")


if __name__ == "__main__":
    test_dual_channel_wrapper_with_real_encoder()
    test_full_pipeline_with_skeleton()