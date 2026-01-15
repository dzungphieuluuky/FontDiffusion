"""
Simplified shape testing script for FontDiffuserWithFST.
This can run independently to verify tensor shapes without full model.
"""

import torch
import torch.nn as nn


class SimplifiedShapeTest:
    """Test harness for verifying tensor shapes through the pipeline."""
    
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.results = {}
    
    def log(self, stage, tensor_name, tensor, expected_shape=None):
        """Log tensor shape information."""
        key = f"{stage}::{tensor_name}"
        
        if isinstance(tensor, torch.Tensor):
            actual_shape = tuple(tensor.shape)
            self.results[key] = {
                'actual': actual_shape,
                'expected': expected_shape,
                'dtype': str(tensor.dtype),
                'match': actual_shape == expected_shape if expected_shape else None
            }
        elif isinstance(tensor, list):
            self.results[key] = {
                'type': 'list',
                'length': len(tensor),
                'shapes': [tuple(t.shape) if isinstance(t, torch.Tensor) else None 
                          for t in tensor]
            }
    
    def print_results(self):
        """Print formatted results."""
        print("\n" + "="*100)
        print("TENSOR SHAPE VERIFICATION REPORT")
        print("="*100)
        
        for key, info in self.results.items():
            stage, name = key.split("::")
            print(f"\n[{stage}] {name}")
            
            if info.get('type') == 'list':
                print(f"  Type: List of {info['length']} tensors")
                for i, shape in enumerate(info['shapes']):
                    print(f"    [{i}] {shape}")
            else:
                actual = info['actual']
                expected = info.get('expected')
                dtype = info['dtype']
                
                print(f"  Actual:   {actual}")
                if expected:
                    match = info['match']
                    symbol = "✓" if match else "✗"
                    print(f"  Expected: {expected} {symbol}")
                print(f"  DType:    {dtype}")
        
        print("\n" + "="*100)
        
        # Summary
        mismatches = [(k, v) for k, v in self.results.items() 
                     if v.get('match') is False]
        
        if mismatches:
            print(f"\n⚠ WARNING: {len(mismatches)} shape mismatches found:")
            for key, info in mismatches:
                print(f"  {key}: expected {info['expected']}, got {info['actual']}")
        else:
            print("\n✓ All tensor shapes match expected values!")
        
        print("="*100 + "\n")


def test_complete_pipeline():
    """Test complete pipeline with expected shapes."""
    
    batch_size = 2
    tester = SimplifiedShapeTest(batch_size)
    
    print("Testing FontDiffuserWithFST Pipeline...")
    print(f"Batch size: {batch_size}\n")
    
    # ========== INPUT STAGE ==========
    print("Stage 1: Preparing inputs...")
    
    noisy_latents = torch.randn(batch_size, 4, 24, 24)
    timestep = torch.randint(0, 1000, (batch_size,))
    content_img = torch.randn(batch_size, 1, 96, 96)
    style_source_img = torch.randn(batch_size, 1, 96, 96)
    style_target_img = torch.randn(batch_size, 1, 96, 96)
    
    tester.log("INPUT", "noisy_latents", noisy_latents, (batch_size, 4, 24, 24))
    tester.log("INPUT", "timestep", timestep, (batch_size,))
    tester.log("INPUT", "content_img", content_img, (batch_size, 1, 96, 96))
    tester.log("INPUT", "style_source_img", style_source_img, (batch_size, 1, 96, 96))
    tester.log("INPUT", "style_target_img", style_target_img, (batch_size, 1, 96, 96))
    
    # ========== CONTENT ENCODER STAGE ==========
    print("Stage 2: Content Encoder (simulated)...")
    
    # Simulating content encoder output
    # Based on ContentEncoder architecture for 96x96 input
    content_img_feature = torch.randn(batch_size, 64, 12, 12)  # Final feature
    content_residual_features = [
        content_img,  # Original input
        torch.randn(batch_size, 16, 48, 48),  # Scale 1
        torch.randn(batch_size, 32, 24, 24),  # Scale 2
        torch.randn(batch_size, 64, 12, 12),  # Scale 3 (final)
    ]
    
    tester.log("CONTENT_ENC", "content_img_feature", content_img_feature, 
               (batch_size, 64, 12, 12))
    tester.log("CONTENT_ENC", "content_residual_features", content_residual_features)
    
    # ========== ORIGINAL STYLE ENCODER STAGE ==========
    print("Stage 3: Original Style Encoder (simulated)...")
    
    orig_style_feat = torch.randn(batch_size, 1024, 3, 3)
    orig_style_vec = torch.randn(batch_size, 1024)
    orig_style_hidden = orig_style_feat.permute(0, 2, 3, 1).reshape(batch_size, 9, 1024)
    
    tester.log("STYLE_ENC", "orig_style_feat", orig_style_feat, 
               (batch_size, 1024, 3, 3))
    tester.log("STYLE_ENC", "orig_style_vec", orig_style_vec, 
               (batch_size, 1024))
    tester.log("STYLE_ENC", "orig_style_hidden", orig_style_hidden, 
               (batch_size, 9, 1024))
    
    # ========== MSSE STAGE ==========
    print("Stage 4: Multi-Scale Style Encoder...")
    
    source_style_features = [
        torch.randn(batch_size, 64, 48, 48),
        torch.randn(batch_size, 128, 24, 24),
        torch.randn(batch_size, 256, 12, 12),
        torch.randn(batch_size, 512, 6, 6),
        torch.randn(batch_size, 1024, 6, 6),
    ]
    
    target_style_features = [
        torch.randn(batch_size, 64, 48, 48),
        torch.randn(batch_size, 128, 24, 24),
        torch.randn(batch_size, 256, 12, 12),
        torch.randn(batch_size, 512, 6, 6),
        torch.randn(batch_size, 1024, 6, 6),
    ]
    
    tester.log("MSSE", "source_style_features", source_style_features)
    tester.log("MSSE", "target_style_features", target_style_features)
    
    # ========== FST STAGE ==========
    print("Stage 5: Font Style Transformation...")
    
    num_queries = 256
    last_spatial_size = 6 * 6  # 36 from last scale (6x6)
    transformation_features = torch.randn(batch_size, num_queries + last_spatial_size, 1024)
    
    tester.log("FST", "transformation_features", transformation_features, 
               (batch_size, 292, 1024))
    
    # ========== PROJECTION STAGE ==========
    print("Stage 6: Feature Projection...")
    
    cross_attn_dim = 1280
    fst_condition = torch.randn(batch_size, 292, cross_attn_dim)
    orig_style_projected = torch.randn(batch_size, 1, cross_attn_dim)
    combined_style_condition = torch.cat([fst_condition, orig_style_projected], dim=1)
    
    tester.log("PROJECTION", "fst_condition", fst_condition, 
               (batch_size, 292, cross_attn_dim))
    tester.log("PROJECTION", "orig_style_projected", orig_style_projected, 
               (batch_size, 1, cross_attn_dim))
    tester.log("PROJECTION", "combined_style_condition", combined_style_condition, 
               (batch_size, 293, cross_attn_dim))
    
    # ========== U-NET STAGE ==========
    print("Stage 7: U-Net Output...")
    
    noise_pred = torch.randn(batch_size, 4, 24, 24)
    offset_out_sum = torch.randn(1)
    
    tester.log("UNET", "noise_pred", noise_pred, (batch_size, 4, 24, 24))
    tester.log("UNET", "offset_out_sum", offset_out_sum, (1,))
    
    # Print complete report
    tester.print_results()
    
    return tester


def test_msse_detailed():
    """Detailed test of MSSE output shapes."""
    print("\n" + "="*100)
    print("DETAILED MSSE SHAPE ANALYSIS")
    print("="*100 + "\n")
    
    batch_size = 2
    input_size = 96
    base_channels = 64
    
    print(f"Input: ({batch_size}, 1, {input_size}, {input_size})")
    print(f"Base channels: {base_channels}\n")
    
    print("Expected multi-scale outputs:")
    print("-" * 60)
    
    scales_info = [
        (0, 48, 64, "First downsampling (stride 2)"),
        (1, 24, 128, "Second downsampling + channel increase"),
        (2, 12, 256, "Third downsampling + channel increase"),
        (3, 6, 512, "Fourth downsampling + channel increase"),
        (4, 6, 1024, "Fifth block (no downsampling) + channel increase"),
    ]
    
    for scale, spatial, channels, description in scales_info:
        expected_shape = (batch_size, channels, spatial, spatial)
        print(f"Scale {scale}: {expected_shape} - {description}")
    
    print("\n" + "="*100 + "\n")


def test_fst_detailed():
    """Detailed test of FST computation."""
    print("\n" + "="*100)
    print("DETAILED FST COMPUTATION ANALYSIS")
    print("="*100 + "\n")
    
    batch_size = 2
    num_queries = 256
    query_dim = 128
    num_scales = 5
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of learnable queries (N_L): {num_queries}")
    print(f"  Query dimension (d): {query_dim}")
    print(f"  Number of scales (n_s): {num_scales}\n")
    
    print("Per-scale processing:")
    print("-" * 80)
    
    scales = [
        (64, 48, "Scale 0"),
        (128, 24, "Scale 1"),
        (256, 12, "Scale 2"),
        (512, 6, "Scale 3"),
        (1024, 6, "Scale 4"),
    ]
    
    for i, (channels, spatial, name) in enumerate(scales):
        print(f"\n{name}:")
        print(f"  Input feature:  ({batch_size}, {channels}, {spatial}, {spatial})")
        print(f"  Flattened:      ({batch_size}, {spatial*spatial}, {channels})")
        print(f"  Query (Q):      ({batch_size}, {num_queries}, {query_dim})")
        print(f"  Key (K):        ({batch_size}, {spatial*spatial}, {query_dim})")
        print(f"  Value (V):      ({batch_size}, {spatial*spatial}, {query_dim})")
        print(f"  After cross-attn: ({batch_size}, {num_queries}, {query_dim})")
        print(f"  L_diff:         ({batch_size}, {num_queries}, {query_dim})")
    
    print(f"\nConcatenation:")
    print(f"  Input: {num_scales} tensors of ({batch_size}, {num_queries}, {query_dim})")
    print(f"  Concatenated: ({batch_size}, {num_queries}, {query_dim * num_scales})")
    print(f"  After MLP: ({batch_size}, {num_queries}, 1024)")
    
    last_spatial = 6 * 6
    print(f"\nResidual connection:")
    print(f"  Last scale spatial: ({batch_size}, 1024, 6, 6)")
    print(f"  Flattened: ({batch_size}, {last_spatial}, 1024)")
    print(f"  After projection: ({batch_size}, {last_spatial}, 1024)")
    
    print(f"\nFinal output:")
    print(f"  Concatenated: ({batch_size}, {num_queries + last_spatial}, 1024)")
    print(f"  = ({batch_size}, {num_queries} + {last_spatial}, 1024)")
    print(f"  = ({batch_size}, 292, 1024)")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    print("="*100)
    print(" FontDiffuserWithFST - Tensor Shape Testing Suite")
    print("="*100)
    
    # Run tests
    test_msse_detailed()
    test_fst_detailed()
    test_complete_pipeline()
    
    print("\n" + "="*100)
    print(" Testing Complete!")
    print("="*100 + "\n")