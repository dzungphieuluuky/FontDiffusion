"""
Test script for FontDiffuserWithFST integration.
Tests tensor shapes and forward pass correctness.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def print_tensor_info(name, tensor):
    """Print detailed tensor information."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name:40s} | Shape: {str(tuple(tensor.shape)):20s} | "
              f"dtype: {str(tensor.dtype):10s} | device: {tensor.device}")
    elif isinstance(tensor, list):
        print(f"{name:40s} | List of {len(tensor)} tensors:")
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}] {str(tuple(t.shape)):20s} | dtype: {str(t.dtype):10s}")


def create_mock_fontdiffuser():
    """Create a mock FontDiffuser model for testing."""
    from src.modules.content_encoder import ContentEncoder
    from src.modules.style_encoder import StyleEncoder
    from src.modules.unet import UNet
    
    # Create mock config
    class MockUNetConfig:
        cross_attention_dim = 1280
    
    # Create components
    content_encoder = ContentEncoder(
        G_ch=16,  # Reduced for testing
        resolution=96,
        input_nc=1,
    )
    
    style_encoder = StyleEncoder(
        G_ch=64,
        resolution=96,
        input_nc=1,
    )
    
    # Create mock U-Net
    unet = UNet(
        sample_size=24,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "MCAUpBlock2D", "MCAUpBlock2D", "UpBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=1,
        cross_attention_dim=1280,
        attention_head_dim=8,
        content_encoder_downsample_size=4,
    )
    unet.config = MockUNetConfig()
    
    # Create mock FontDiffuser
    class MockFontDiffuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.content_encoder = content_encoder
            self.style_encoder = style_encoder
            self.unet = unet
    
    return MockFontDiffuser()


def test_msse_module():
    """Test Multi-Scale Style Encoder."""
    print_separator("Testing MSSE Module")
    
    from src.modules.msse import MultiScaleStyleEncoder
    
    msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 1, 96, 96)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    features = msse(x)
    
    print(f"\nOutput: List of {len(features)} feature maps")
    for i, feat in enumerate(features):
        print_tensor_info(f"  Scale {i}", feat)
    
    # Verify shapes
    expected_shapes = [
        (batch_size, 64, 48, 48),
        (batch_size, 128, 24, 24),
        (batch_size, 256, 12, 12),
        (batch_size, 512, 6, 6),
        (batch_size, 1024, 6, 6),
    ]
    
    print("\nShape verification:")
    all_correct = True
    for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
        is_correct = feat.shape == expected
        status = "✓" if is_correct else "✗"
        print(f"  Scale {i}: {status} Expected {expected}, Got {feat.shape}")
        all_correct = all_correct and is_correct
    
    return all_correct


def test_fst_module():
    """Test Font Style Transformation Module."""
    print_separator("Testing FST Module")
    
    from src.modules.fst import FontStyleTransformationModule
    
    feature_channels = [64, 128, 256, 512, 1024]
    fst = FontStyleTransformationModule(
        feature_channels=feature_channels,
        num_queries=256,
        query_dim=128,
        num_scale_features=5,
        num_cross_attn_blocks=2,
        num_self_attn_blocks=2
    )
    
    # Create mock multi-scale features
    batch_size = 2
    source_features = [
        torch.randn(batch_size, 64, 48, 48),
        torch.randn(batch_size, 128, 24, 24),
        torch.randn(batch_size, 256, 12, 12),
        torch.randn(batch_size, 512, 6, 6),
        torch.randn(batch_size, 1024, 6, 6),
    ]
    
    target_features = [
        torch.randn(batch_size, 64, 48, 48),
        torch.randn(batch_size, 128, 24, 24),
        torch.randn(batch_size, 256, 12, 12),
        torch.randn(batch_size, 512, 6, 6),
        torch.randn(batch_size, 1024, 6, 6),
    ]
    
    print("Input features:")
    print_tensor_info("Source features", source_features)
    print_tensor_info("Target features", target_features)
    
    # Forward pass
    output = fst(source_features, target_features)
    
    print(f"\nOutput shape: {output.shape}")
    expected_shape = (batch_size, 256 + 6*6, 1024)  # (B, N_L + H*W, 1024)
    
    is_correct = output.shape == expected_shape
    status = "✓" if is_correct else "✗"
    print(f"Shape verification: {status} Expected {expected_shape}, Got {output.shape}")
    
    return is_correct


def test_integrated_model():
    """Test the complete FontDiffuserWithFST model."""
    print_separator("Testing Integrated FontDiffuserWithFST Model")
    
    # Import after defining
    import sys
    sys.path.insert(0, '.')
    
    # Create mock original FontDiffuser
    print("Creating mock FontDiffuser...")
    mock_fontdiffuser = create_mock_fontdiffuser()
    
    # Create enhanced model
    print("Creating FontDiffuserWithFST...")
    from src.model import FontDiffuserWithFST
    model = FontDiffuserWithFST(mock_fontdiffuser)
    
    # Prepare test inputs
    batch_size = 2
    print(f"\nPreparing test inputs (batch_size={batch_size})...")
    
    inputs = {
        'noisy_latents': torch.randn(batch_size, 4, 24, 24),
        'timestep': torch.randint(0, 1000, (batch_size,)),
        'content_img': torch.randn(batch_size, 1, 96, 96),
        'style_source_img': torch.randn(batch_size, 1, 96, 96),
        'style_target_img': torch.randn(batch_size, 1, 96, 96),
    }
    
    print("\nInput tensor shapes:")
    for name, tensor in inputs.items():
        print_tensor_info(f"  {name}", tensor)
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        
        print("\n✓ Forward pass successful!")
        
        print("\nOutput tensor shapes:")
        for name, tensor in outputs.items():
            print_tensor_info(f"  {name}", tensor)
        
        # Verify key outputs
        print("\nKey output verification:")
        checks = [
            ('noise_pred', (batch_size, 4, 24, 24)),
            ('transformation_features', (batch_size, 292, 1024)),
            ('fst_condition', (batch_size, 292, 1280)),
        ]
        
        all_correct = True
        for name, expected_shape in checks:
            if name in outputs:
                actual_shape = outputs[name].shape
                is_correct = actual_shape == expected_shape
                status = "✓" if is_correct else "✗"
                print(f"  {name}: {status} Expected {expected_shape}, Got {actual_shape}")
                all_correct = all_correct and is_correct
        
        return all_correct
        
    except Exception as e:
        print(f"\n✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient flow through the model."""
    print_separator("Testing Gradient Flow")
    
    # Create mock model
    mock_fontdiffuser = create_mock_fontdiffuser()
    from src.model import FontDiffuserWithFST
    model = FontDiffuserWithFST(mock_fontdiffuser)
    
    # Prepare inputs
    batch_size = 2
    inputs = {
        'noisy_latents': torch.randn(batch_size, 4, 24, 24, requires_grad=True),
        'timestep': torch.randint(0, 1000, (batch_size,)),
        'content_img': torch.randn(batch_size, 1, 96, 96),
        'style_source_img': torch.randn(batch_size, 1, 96, 96),
        'style_target_img': torch.randn(batch_size, 1, 96, 96),
    }
    
    # Forward pass
    outputs = model(**inputs, return_dict=True)
    
    # Create dummy loss
    noise_pred = outputs['noise_pred']
    target = torch.randn_like(noise_pred)
    loss = nn.functional.mse_loss(noise_pred, target)
    
    print(f"Loss value: {loss.item():.6f}")
    
    # Backward pass
    try:
        loss.backward()
        print("✓ Backward pass successful!")
        
        # Check gradients
        print("\nGradient statistics:")
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats.append((name, grad_norm))
        
        # Show top 10 gradients by magnitude
        grad_stats.sort(key=lambda x: x[1], reverse=True)
        print("Top 10 gradient magnitudes:")
        for name, norm in grad_stats[:10]:
            print(f"  {name:60s} | grad_norm: {norm:.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Backward pass failed:")
        print(f"  {type(e).__name__}: {str(e)}")
        return False


def test_loss_computation():
    """Test loss computation utilities."""
    print_separator("Testing Loss Computation")
    
    mock_fontdiffuser = create_mock_fontdiffuser()
    from src.model import FontDiffuserWithFST
    model = FontDiffuserWithFST(mock_fontdiffuser)
    
    # Prepare inputs
    batch_size = 2
    inputs = {
        'noisy_latents': torch.randn(batch_size, 4, 24, 24),
        'timestep': torch.randint(0, 1000, (batch_size,)),
        'content_img': torch.randn(batch_size, 1, 96, 96),
        'style_source_img': torch.randn(batch_size, 1, 96, 96),
        'style_target_img': torch.randn(batch_size, 1, 96, 96),
    }
    
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    
    # Compute losses
    target_noise = torch.randn(batch_size, 4, 24, 24)
    losses = model.get_loss_dict(outputs, target_noise)
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name:20s}: {value.item():.6f}")
    
    return True


def run_all_tests():
    """Run all test functions."""
    print_separator("FontDiffuserWithFST Integration Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results = OrderedDict()
    
    # Run individual module tests
    results['MSSE Module'] = test_msse_module()
    results['FST Module'] = test_fst_module()
    
    # Run integrated tests
    results['Integrated Model'] = test_integrated_model()
    results['Gradient Flow'] = test_gradient_flow()
    results['Loss Computation'] = test_loss_computation()
    
    # Print summary
    print_separator("Test Summary")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} : {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("  ✓ All tests passed!")
    else:
        print("  ✗ Some tests failed. Please review the output above.")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)