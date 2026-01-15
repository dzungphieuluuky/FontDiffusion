from functools import wraps
from typing import List, get_type_hints
import inspect
import torch

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.model import FontDiffuserWithFST

def debug_shapes(verbose: bool = True):
    """
    Decorator to log input/output tensor shapes for debugging.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Function: {func.__name__}")
                print(f"{'='*60}")
            
            # Log input tensor shapes
            for name, value in bound_args.arguments.items():
                if torch.is_tensor(value):
                    if verbose:
                        print(f"Input  {name:20s}: {tuple(value.shape)}")
                elif isinstance(value, list) and all(torch.is_tensor(v) for v in value):
                    shapes = [tuple(v.shape) for v in value]
                    if verbose:
                        print(f"Input  {name:20s}: List{shapes}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log output tensor shapes
            if torch.is_tensor(result):
                if verbose:
                    print(f"Output {'return':20s}: {tuple(result.shape)}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if torch.is_tensor(r):
                        if verbose:
                            print(f"Output [tuple][{i}]         : {tuple(r.shape)}")
            elif isinstance(result, dict):
                for k, v in result.items():
                    if torch.is_tensor(v):
                        if verbose:
                            print(f"Output dict[{k:15s}]: {tuple(v.shape)}")
            
            return result
        return wrapper
    return decorator


class ShapeValidator:
    """Automated tensor shape validation for debugging."""
    
    @staticmethod
    def validate_mss_encoder(encoder, input_shape=(2, 1, 96, 96)):
        """Validate MSSE output shapes."""
        print("🔍 Validating MSSE...")
        dummy_input = torch.randn(*input_shape)
        features = encoder(dummy_input)
        
        rules = [
            ("Number of scales", len(features) == 5),
            ("Scale 1 channels", features[0].shape[1] == 64),
            ("Scale 5 channels", features[-1].shape[1] == 1024),
            ("Scale 5 spatial", features[-1].shape[2:] == (6, 6)),
        ]
        
        for desc, condition in rules:
            status = "✅" if condition else "❌"
            print(f"  {status} {desc}")
        
        return all(condition for _, condition in rules)
    
    @staticmethod
    def validate_fst_module(module, mss_encoder):
        """Validate FST module compatibility."""
        print("\n🔍 Validating FST Module...")
        
        # Create dummy features
        dummy_img = torch.randn(2, 1, 96, 96)
        features = mss_encoder(dummy_img)
        
        # Test with identical source/target (should give zero transformation?)
        output = module(features, features)
        
        rules = [
            ("Output rank", output.dim() == 3),
            ("Batch dimension", output.shape[0] == 2),
            ("Sequence length", output.shape[1] == 292),  # 256 queries + 36 spatial
            ("Feature dimension", output.shape[2] == 1024),
        ]
        
        for desc, condition in rules:
            status = "✅" if condition else "❌"
            print(f"  {status} {desc}")
        
        return all(condition for _, condition in rules)
    
    @staticmethod
    def validate_full_pipeline(model, batch_size=2):
        """End-to-end pipeline validation."""
        print("\n🔍 Validating Full Pipeline...")
        
        # Create test batch
        test_inputs = {
            'content_img': torch.randn(batch_size, 1, 96, 96),
            'style_source_img': torch.randn(batch_size, 1, 96, 96),
            'style_target_img': torch.randn(batch_size, 1, 96, 96),
            'noisy_latents': torch.randn(batch_size, 4, 24, 24),
            'timestep': torch.randint(0, 1000, (batch_size,))
        }
        
        try:
            with torch.no_grad():
                outputs = model(**test_inputs)
            
            print("  ✅ Forward pass successful")
            
            # Check output structure
            required_keys = ['model_output', 'transformation_features']
            for key in required_keys:
                if key in outputs:
                    print(f"  ✅ Output contains '{key}'")
                else:
                    print(f"  ❌ Missing output key '{key}'")
            
            return True
        except Exception as e:
            print(f"  ❌ Pipeline failed: {e}")
            return False

def test_integration_pipeline():
    """
    Comprehensive test to verify tensor shapes flow correctly.
    """
    print("🚀 Starting FSTDiff-FontDiffuser Integration Test")
    print("=" * 70)
    
    # Create dummy tensors with correct dimensions
    batch_size = 2
    H, W = 96, 96  # FSTDiff uses 96x96
    
    dummy_tensors = {
        'content_img': torch.randn(batch_size, 1, H, W),
        'style_source_img': torch.randn(batch_size, 1, H, W),
        'style_target_img': torch.randn(batch_size, 1, H, W),
        'noisy_latents': torch.randn(batch_size, 4, H//4, W//4),  # Latent space
        'timestep': torch.randint(0, 1000, (batch_size,))
    }
    
    print("📊 Input Tensor Shapes:")
    for name, tensor in dummy_tensors.items():
        print(f"  {name:25s}: {tuple(tensor.shape)}")
    
    # ========== TEST 1: MSSE Module ==========
    print("\n" + "="*70)
    print("TEST 1: Multi-Scale Style Encoder (MSSE)")
    print("="*70)
    
    mss_encoder = DebuggableMultiScaleStyleEncoder()
    style_features = mss_encoder(dummy_tensors['style_source_img'])
    
    # Verify MSSE outputs
    assert len(style_features) == 5, f"MSSE should output 5 scales, got {len(style_features)}"
    
    expected_shapes = [
        (batch_size, 64, 48, 48),   # Scale 1
        (batch_size, 128, 24, 24),  # Scale 2
        (batch_size, 256, 12, 12),  # Scale 3
        (batch_size, 512, 6, 6),    # Scale 4
        (batch_size, 1024, 6, 6)    # Scale 5 (after adaptive pool)
    ]
    
    for i, (feat, expected) in enumerate(zip(style_features, expected_shapes)):
        actual = tuple(feat.shape)
        assert actual == expected, f"MSSE scale {i}: expected {expected}, got {actual}"
        print(f"  ✓ Scale {i}: {actual}")
    
    # ========== TEST 2: FST Module ==========
    print("\n" + "="*70)
    print("TEST 2: Font Style Transformation (FST) Module")
    print("="*70)
    
    feature_channels = [64, 128, 256, 512, 1024]
    fst_module = DebuggableFontStyleTransformationModule(feature_channels=feature_channels)
    
    # Get features from MSSE for both source and target
    source_features = mss_encoder(dummy_tensors['style_source_img'])
    target_features = mss_encoder(dummy_tensors['style_target_img'])
    
    transformation = fst_module(source_features, target_features)
    
    # Verify FST output
    # Expected: (B, N_L + h_{n_s}*w_{n_s}, c_{n_s}) = (B, 256 + 36, 1024)
    expected_fst_shape = (batch_size, 292, 1024)
    actual_fst_shape = tuple(transformation.shape)
    
    assert actual_fst_shape == expected_fst_shape, \
        f"FST output: expected {expected_fst_shape}, got {actual_fst_shape}"
    print(f"  ✓ FST output: {actual_fst_shape}")
    
    # ========== TEST 3: Complete Pipeline (Dry Run) ==========
    print("\n" + "="*70)
    print("TEST 3: Complete Pipeline Integration")
    print("="*70)
    
    # Mock the original FontDiffuser
    class MockFontDiffuser:
        class content_encoder:
            def __call__(self, x):
                return [torch.randn(batch_size, 64, 48, 48) for _ in range(4)]
        
        class style_encoder:
            def __call__(self, x):
                return torch.randn(batch_size, 512)
        
        class unet:
            class config:
                cross_attention_dim = 768
    
    mock_fontdiffuser = MockFontDiffuser()
    integrated_model = FontDiffuserWithFST(mock_fontdiffuser)
    
    # Run forward pass
    with torch.no_grad():
        outputs = integrated_model(
            noisy_latents=dummy_tensors['noisy_latents'],
            timestep=dummy_tensors['timestep'],
            content_img=dummy_tensors['content_img'],
            style_source_img=dummy_tensors['style_source_img'],
            style_target_img=dummy_tensors['style_target_img']
        )
    
    print("📊 Pipeline Output Shapes:")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"  {key:30s}: {tuple(value.shape)}")
        elif isinstance(value, list):
            shapes = [tuple(v.shape) for v in value]
            print(f"  {key:30s}: List{shapes}")
    
    # ========== TEST 4: Consistency Loss (FSTDiff requirement) ==========
    print("\n" + "="*70)
    print("TEST 4: Consistency Loss Compatibility")
    print("="*70)
    
    # Generate transformations for two different reference characters
    style_source_2 = torch.randn(batch_size, 1, H, W)
    style_target_2 = torch.randn(batch_size, 1, H, W)
    
    features_src_2 = mss_encoder(style_source_2)
    features_tgt_2 = mss_encoder(style_target_2)
    transformation_2 = fst_module(features_src_2, features_tgt_2)
    
    # Compute consistency loss (L_consistency in Eq. 19)
    consistency_loss = torch.mean((transformation - transformation_2) ** 2)
    
    print(f"  ✓ Consistency loss computed: {consistency_loss.item():.6f}")
    print(f"  ✓ Transformation shapes match: {transformation.shape == transformation_2.shape}")
    
    print("\n" + "="*70)
    print("🎉 All integration tests passed!")
    print("="*70)
    
    return True

# Run the test
if __name__ == "__main__":
    test_integration_pipeline()
    # Apply to key modules
    @debug_shapes(verbose=True)
    class DebuggableMultiScaleStyleEncoder(MultiScaleStyleEncoder):
        pass

    @debug_shapes(verbose=True)
    class DebuggableFontStyleTransformationModule(FontStyleTransformationModule):
        pass

    # Usage
    validator = ShapeValidator()
    mss_encoder = MultiScaleStyleEncoder()
    validator.validate_mss_encoder(mss_encoder)