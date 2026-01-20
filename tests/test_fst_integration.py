"""
FST Module Integration Tests

Comprehensive tests for FontDiffuserWithFST, MSSE, and FST components.
Tests all tiers: shapes, overfit, data, and gradient flow.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple
from unittest.mock import MagicMock, patch

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.models.fst_model import FontDiffuserWithFST


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return torch device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_feature_channels():
    """Default feature channels for MSSE output."""
    return [64, 128, 256, 512, 1024]


@pytest.fixture
def expected_spatial_sizes():
    """Expected spatial sizes after MSSE processing (input 512x512)."""
    return [256, 128, 64, 32, 16]


@pytest.fixture
def msse_features(default_feature_channels, expected_spatial_sizes, device):
    """Generate MSSE-style features for testing."""
    def _create_features(batch_size: int = 2, requires_grad: bool = False):
        source = []
        target = []
        for ch, size in zip(default_feature_channels, expected_spatial_sizes):
            src = torch.randn(
                batch_size, ch, size, size,
                device=device,
                requires_grad=requires_grad
            )
            tgt = torch.randn(
                batch_size, ch, size, size,
                device=device,
                requires_grad=requires_grad
            )
            source.append(src)
            target.append(tgt)
        return source, target
    return _create_features


@pytest.fixture
def dummy_unet_config():
    """Create a mock UNet config with required attributes."""
    config = MagicMock()
    config.cross_attention_dim = 1280
    config.in_channels = 4
    config.out_channels = 4
    return config


@pytest.fixture
def base_model_mock(dummy_unet_config):
    """Create a mock base FontDiffuser model."""
    mock_unet = MagicMock()
    mock_unet.config = dummy_unet_config
    
    base_model = MagicMock()
    base_model.unet = mock_unet
    base_model.style_encoder = MagicMock()
    base_model.content_encoder = MagicMock()
    
    return base_model


# ============================================================================
# Helper Functions
# ============================================================================

def assert_gradients_exist(module: nn.Module, module_name: str = ""):
    """Verify all trainable parameters in module have gradients."""
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, \
                f"{module_name}.{name} has no gradient"
            assert torch.isfinite(param.grad).all(), \
                f"{module_name}.{name} gradient contains inf/nan"


def assert_features_match_channels(
    features: List[torch.Tensor],
    expected_channels: List[int],
    expected_spatial_sizes: List[int],
    batch_size: int = 2,
    msg_prefix: str = ""
):
    """Verify feature shapes match expected dimensions."""
    assert len(features) == len(expected_channels), \
        f"{msg_prefix}: Number of features mismatch"
    
    for i, (feat, ch, size) in enumerate(
        zip(features, expected_channels, expected_spatial_sizes)
    ):
        expected_shape = (batch_size, ch, size, size)
        assert feat.shape == expected_shape, \
            f"{msg_prefix} scale {i}: {feat.shape} != {expected_shape}"


# ============================================================================
# Test Classes
# ============================================================================

class TestFontDiffuserWithFST:
    """Tests for FontDiffuserWithFST model."""
    
    def test_initialization(self, base_model_mock, device):
        """Test FontDiffuserWithFST initialization."""
        model = FontDiffuserWithFST(
            original_fontdiffuser=base_model_mock,
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=256,
            query_dim=128,
            num_scales=5,
        ).to(device)

        assert model.num_queries == 256
        assert model.feature_channels == [64, 128, 256, 512, 1024]
        assert model.cross_attn_dim == 1280
        assert model.mss_encoder is not None
        assert model.fst_module is not None
        
        # Verify FST module is initialized with correct parameters
        assert model.fst_module.num_queries == 256
        assert model.fst_module.query_dim == 128


class TestFSTModule:
    """Integration tests for FST module."""
    
    @pytest.mark.parametrize(
        "num_queries,query_dim,batch_size",
        [
            (64, 64, 1),
            (128, 128, 2),
            (256, 128, 4),
        ]
    )
    def test_forward_pass(
        self,
        device,
        default_feature_channels,
        expected_spatial_sizes,
        num_queries: int,
        query_dim: int,
        batch_size: int
    ):
        """Test FST forward pass with different configurations."""
        fst = FontStyleTransformationModule(
            feature_channels=default_feature_channels,
            num_queries=num_queries,
            query_dim=query_dim,
            num_scale_features=5,
        ).to(device)

        # Create test features
        source_features = []
        target_features = []
        for ch, spatial_size in zip(default_feature_channels, expected_spatial_sizes):
            src = torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        # Forward pass
        with torch.no_grad():
            output = fst(source_features, target_features)

        # Verify output dimensions
        assert output.shape[0] == batch_size
        assert output.shape[2] == 1024  # Last channel dimension
        
        # Sequence length = num_queries + last_spatial_size (16x16 = 256)
        expected_seq_len = num_queries + 256
        assert output.shape[1] == expected_seq_len, \
            f"Expected seq len {expected_seq_len}, got {output.shape[1]}"
        
        # Verify no NaNs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_backward_flow(self, msse_features, device):
        """Test complete forward and backward pass through FST."""
        batch_size = 2
        num_queries = 128
        
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=num_queries,
            query_dim=128,
            num_scale_features=5,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        ).to(device)

        # Get features with gradients
        source_features, target_features = msse_features(
            batch_size=batch_size,
            requires_grad=True
        )

        # Forward pass
        output = fst(source_features, target_features)

        # Loss and backward
        loss = output.sum()
        loss.backward()

        # Verify output shape
        expected_seq_len = num_queries + (16 * 16)  # num_queries + 256
        assert output.shape == (batch_size, expected_seq_len, 1024)
        
        # Verify gradients exist
        assert fst.learnable_queries.grad is not None
        assert_gradients_exist(fst, "FST")
        
        # Verify input feature gradients
        for i, (src, tgt) in enumerate(zip(source_features, target_features)):
            assert src.grad is not None, f"Source feature {i} has no gradient"
            assert tgt.grad is not None, f"Target feature {i} has no gradient"

    def test_fst_with_incorrect_feature_length(self, device):
        """Test FST raises error with incorrect number of features."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256],
            num_queries=128,
            query_dim=128,
            num_scale_features=3,
        ).to(device)

        # Create mismatched features (only 2 instead of 3)
        source_features = [
            torch.randn(2, 64, 128, 128, device=device),
            torch.randn(2, 128, 64, 64, device=device),
            # Missing the 256-channel feature
        ]
        target_features = [
            torch.randn(2, 64, 128, 128, device=device),
            torch.randn(2, 128, 64, 64, device=device),
        ]

        with pytest.raises((RuntimeError, ValueError, IndexError)):
            fst(source_features, target_features)

    def test_fst_with_varying_batch_sizes(self, device):
        """Test FST handles varying batch sizes correctly."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128],
            num_queries=64,
            query_dim=64,
            num_scale_features=2,
        ).to(device)

        # Different batch sizes should fail
        source_features = [
            torch.randn(2, 64, 128, 128, device=device),
            torch.randn(2, 128, 64, 64, device=device),
        ]
        target_features = [
            torch.randn(3, 64, 128, 128, device=device),  # Different batch size
            torch.randn(3, 128, 64, 64, device=device),
        ]

        with pytest.raises(RuntimeError):
            fst(source_features, target_features)


class TestMultiScaleStyleEncoder:
    """Integration tests for MultiScaleStyleEncoder."""
    
    @pytest.mark.parametrize("input_channels", [1, 3])
    @pytest.mark.parametrize("base_channels", [32, 64])
    def test_msse_initialization(self, device, input_channels: int, base_channels: int):
        """Test MSSE initialization with different parameters."""
        num_scales = 4
        
        msse = MultiScaleStyleEncoder(
            in_channels=input_channels,
            base_channels=base_channels,
            num_scales=num_scales
        ).to(device)

        assert msse.in_channels == input_channels
        assert msse.base_channels == base_channels
        assert msse.num_scales == num_scales
        
        # Verify feature channels
        expected_channels = [base_channels * (2 ** i) for i in range(num_scales)]
        assert msse.get_feature_channels() == expected_channels

    def test_msse_output_consistency_eval(self, device):
        """Test MSSE produces identical outputs in eval mode."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        msse.eval()

        x = torch.randn(2, 1, 512, 512, device=device)

        with torch.no_grad():
            features1 = msse(x)
            features2 = msse(x)

        # In eval mode, outputs should be identical
        assert len(features1) == len(features2) == 5
        
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            assert torch.allclose(f1, f2, atol=1e-6), \
                f"Scale {i} outputs not consistent in eval mode"

    def test_msse_training_mode(self, device):
        """Test MSSE behavior in training mode."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        msse.train()
        x = torch.randn(2, 1, 512, 512, device=device)

        # InstanceNorm should be deterministic even in training mode
        # for a given input in the same forward pass
        with torch.no_grad():
            features = msse(x)
        
        # Verify output shapes
        expected_channels = [64, 128, 256, 512, 1024]
        expected_sizes = [256, 128, 64, 32, 16]
        
        assert_features_match_channels(
            features,
            expected_channels,
            expected_sizes,
            msg_prefix="MSSE training mode"
        )

    def test_msse_gradient_flow(self, device):
        """Test gradient propagation through MSSE."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        x = torch.randn(2, 1, 512, 512, device=device, requires_grad=True)
        
        features = msse(x)
        
        # Compute loss on all features
        loss = 0
        for feat in features:
            loss = loss + feat.sum()
        
        loss.backward()
        
        # Verify input gradient
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Verify module gradients
        assert_gradients_exist(msse, "MSSE")


class TestFSTWithMSSEPipeline:
    """Integration tests for FST and MSSE working together."""
    
    def test_complete_pipeline(self, device):
        """Test complete MSSE -> FST pipeline."""
        # Initialize components
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        feature_channels = msse.get_feature_channels()
        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Create style images
        batch_size = 2
        style_source = torch.randn(batch_size, 1, 512, 512, device=device)
        style_target = torch.randn(batch_size, 1, 512, 512, device=device)

        # Extract features
        msse.eval()
        with torch.no_grad():
            source_features = msse(style_source)
            target_features = msse(style_target)

        # Verify MSSE output shapes
        expected_sizes = [256, 128, 64, 32, 16]
        assert_features_match_channels(
            source_features,
            feature_channels,
            expected_sizes,
            batch_size=batch_size,
            msg_prefix="MSSE source"
        )
        assert_features_match_channels(
            target_features,
            feature_channels,
            expected_sizes,
            batch_size=batch_size,
            msg_prefix="MSSE target"
        )

        # Process through FST
        fst_output = fst(source_features, target_features)

        # Verify FST output
        assert fst_output.shape[0] == batch_size
        assert fst_output.shape[2] == 1024
        
        expected_seq_len = 128 + 256  # num_queries + 16x16
        assert fst_output.shape[1] == expected_seq_len
        
        # Verify no NaNs
        assert not torch.isnan(fst_output).any()

    def test_pipeline_with_gradients(self, device):
        """Test pipeline with gradient computation."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Enable training mode
        msse.train()
        fst.train()

        # Create inputs with gradients
        style_source = torch.randn(2, 1, 512, 512, device=device, requires_grad=True)
        style_target = torch.randn(2, 1, 512, 512, device=device, requires_grad=True)

        # Forward pass through pipeline
        source_features = msse(style_source)
        target_features = msse(style_target)
        output = fst(source_features, target_features)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Verify gradients
        assert style_source.grad is not None
        assert style_target.grad is not None
        assert_gradients_exist(msse, "MSSE")
        assert_gradients_exist(fst, "FST")


class TestFSTGradientFlow:
    """Detailed gradient flow tests for FST."""
    
    def test_learnable_queries_gradients(self, msse_features, device):
        """Verify learnable queries receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features, target_features = msse_features(batch_size=2)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Learnable queries should have gradient
        assert fst.learnable_queries.grad is not None
        assert (fst.learnable_queries.grad.abs() > 1e-8).any(), \
            "Learnable queries gradient is effectively zero"
        assert torch.isfinite(fst.learnable_queries.grad).all()

    def test_positional_encoding_gradients(self, msse_features, device):
        """Verify positional encodings receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features, target_features = msse_features(batch_size=2)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Each positional encoding should have gradient
        for i, pe in enumerate(fst.pos_encodings):
            assert pe.grad is not None, f"PosEncoding {i} has no gradient"
            assert (pe.grad.abs() > 1e-8).any(), \
                f"PosEncoding {i} gradient is effectively zero"

    def test_projection_layers_gradients(self, msse_features, device):
        """Verify all projection layers in FST receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features, target_features = msse_features(batch_size=2)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Check q_projs
        for i, proj in enumerate(fst.q_projs):
            assert proj.weight.grad is not None, f"q_projs[{i}] has no gradient"
            assert (proj.weight.grad.abs() > 1e-8).any(), \
                f"q_projs[{i}] gradient is effectively zero"

        # Check k_projs
        for i, proj in enumerate(fst.k_projs):
            assert proj.weight.grad is not None, f"k_projs[{i}] has no gradient"

        # Check v_projs
        for i, proj in enumerate(fst.v_projs):
            assert proj.weight.grad is not None, f"v_projs[{i}] has no gradient"

        # Check MLP layers
        for name, param in fst.mlp_channel_adjust.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"mlp_channel_adjust.{name} has no gradient"

        # Check residual projection
        if fst.residual_proj.weight.requires_grad:
            assert fst.residual_proj.weight.grad is not None, \
                "residual_proj.weight has no gradient"


class TestOverfitting:
    """Tests that verify models can overfit small datasets."""
    
    def test_fst_overfitting(self, msse_features, device):
        """Test FST can memorize and overfit a small batch."""
        torch.manual_seed(42)  # For reproducibility
        
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Create fixed features
        source_features, target_features = msse_features(batch_size=2)

        # Create target output
        batch_size = 2
        num_queries = 128
        last_spatial_size = 256  # 16x16
        target_output = torch.randn(
            batch_size,
            num_queries + last_spatial_size,
            1024,
            device=device
        )

        # Training loop
        optimizer = torch.optim.Adam(fst.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = fst(source_features, target_features)
            loss = loss_fn(output, target_output)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if loss.item() < 1e-4:  # Early stopping
                break

        # Loss should decrease significantly
        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        print(f"\nFST Overfitting: Initial={initial_loss:.6f}, "
              f"Final={final_loss:.6f}, Reduction={reduction:.1%}")

        assert final_loss < initial_loss, "Loss should decrease"
        assert reduction > 0.5, f"Loss reduction should be >50%, got {reduction:.1%}"

    def test_msse_overfitting(self, device):
        """Test MSSE can learn to map inputs to target features."""
        torch.manual_seed(42)
        
        base_channels = 64
        num_scales = 5
        
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=base_channels,
            num_scales=num_scales,
        ).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 1, 512, 512, device=device)

        # Create target features
        target_features = []
        spatial_size = 256  # After initial conv stride=2
        for scale_idx in range(num_scales):
            ch = base_channels * (2 ** scale_idx)
            target_features.append(
                torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            )
            spatial_size = spatial_size // 2

        optimizer = torch.optim.Adam(msse.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(50):
            optimizer.zero_grad()
            features = msse(x)
            
            # Compute loss for all scales
            loss = 0
            for feat, target in zip(features, target_features):
                loss += loss_fn(feat, target)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if loss.item() < 1e-3:  # Early stopping
                break

        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        print(f"\nMSSE Overfitting: Initial={initial_loss:.6f}, "
              f"Final={final_loss:.6f}, Reduction={reduction:.1%}")

        assert final_loss < initial_loss, "Loss should decrease"
        assert reduction > 0.3, f"Loss reduction should be >30%, got {reduction:.1%}"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_fst_single_sample_batch(self, device):
        """Test FST with batch size of 1."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128],
            num_queries=64,
            query_dim=64,
            num_scale_features=2,
        ).to(device)

        source_features = [
            torch.randn(1, 64, 128, 128, device=device),
            torch.randn(1, 128, 64, 64, device=device),
        ]
        target_features = [
            torch.randn(1, 64, 128, 128, device=device),
            torch.randn(1, 128, 64, 64, device=device),
        ]

        output = fst(source_features, target_features)
        assert output.shape[0] == 1  # Batch size 1

    @pytest.mark.parametrize("num_scales", [1, 2, 3, 5])
    def test_msse_varying_scales(self, device, num_scales: int):
        """Test MSSE with different numbers of scales."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=32,
            num_scales=num_scales
        ).to(device)

        x = torch.randn(2, 1, 512, 512, device=device)
        features = msse(x)
        
        assert len(features) == num_scales
        
        # Verify decreasing spatial dimensions
        for i in range(1, num_scales):
            assert features[i].shape[2] < features[i-1].shape[2], \
                f"Scale {i} should have smaller spatial size than scale {i-1}"
            assert features[i].shape[3] < features[i-1].shape[3]

    def test_fst_high_precision(self, device):
        """Test FST with float64 precision."""
        if device.type == "cuda":
            pytest.skip("float64 not well-supported on CUDA for all operations")
            
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        
        try:
            fst = FontStyleTransformationModule(
                feature_channels=[64, 128],
                num_queries=64,
                query_dim=64,
                num_scale_features=2,
            ).to(device).double()

            source_features = [
                torch.randn(2, 64, 128, 128, dtype=torch.float64, device=device),
                torch.randn(2, 128, 64, 64, dtype=torch.float64, device=device),
            ]
            target_features = [
                torch.randn(2, 64, 128, 128, dtype=torch.float64, device=device),
                torch.randn(2, 128, 64, 64, dtype=torch.float64, device=device),
            ]

            output = fst(source_features, target_features)
            assert output.dtype == torch.float64
        finally:
            torch.set_default_dtype(original_dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])