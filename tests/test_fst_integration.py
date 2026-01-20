"""
FST Module Integration Tests

Comprehensive tests for FontDiffuserWithFST, MSSE, and FST components.
Tests all tiers: shapes, overfit, data, and gradient flow.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule


class TestFSTModuleIntegration:
    """Integration tests for FST module."""

    def test_fst_forward_backward_flow(self, device):
        """Test complete forward and backward pass through FST."""
        feature_channels = [64, 128, 256, 512, 1024]
        batch_size = 2
        num_queries = 128

        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,
            query_dim=128,
            num_scale_features=5,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        ).to(device)

        # Create input features with correct spatial sizes
        # MSSE: input 512x512 -> initial_conv stride=2 -> 256x256
        # Then 4 more downsamples: 128x128, 64x64, 32x32, 16x16
        source_features = []
        target_features = []
        spatial_sizes = [256, 128, 64, 32, 16]

        for ch, spatial_size in zip(feature_channels, spatial_sizes):
            src = torch.randn(
                batch_size,
                ch,
                spatial_size,
                spatial_size,
                device=device,
                requires_grad=True,
            )
            tgt = torch.randn(
                batch_size,
                ch,
                spatial_size,
                spatial_size,
                device=device,
                requires_grad=True,
            )
            source_features.append(src)
            target_features.append(tgt)

        # Forward pass
        output = fst(source_features, target_features)

        # Loss and backward
        loss = output.sum()
        loss.backward()

        # Verify output shape: (B, N_L + H_last*W_last, 1024)
        assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
        assert output.shape[2] == 1024, f"Output channels mismatch: {output.shape[2]} != 1024"
        
        # Last scale is 16x16 = 256
        expected_seq_len = num_queries + (16 * 16)
        assert output.shape[1] == expected_seq_len, f"Sequence length mismatch: {output.shape[1]} != {expected_seq_len}"

        # Verify gradients exist
        assert fst.learnable_queries.grad is not None
        for param in fst.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter has no gradient"

    @pytest.mark.parametrize(
        "num_queries,query_dim",
        [
            (64, 64),
            (128, 128),
            (256, 128),
        ],
    )
    def test_fst_variable_configs(self, device, num_queries: int, query_dim: int):
        """Test FST with different configurations."""
        feature_channels = [64, 128, 256, 512, 1024]
        
        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,
            query_dim=query_dim,
            num_scale_features=5,
        ).to(device)

        batch_size = 2
        spatial_sizes = [256, 128, 64, 32, 16]
        source_features = []
        target_features = []

        for ch, spatial_size in zip(feature_channels, spatial_sizes):
            src = torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        with torch.no_grad():
            output = fst(source_features, target_features)

        # Verify output dimensions
        assert output.shape[0] == batch_size, f"Batch size: {output.shape[0]} != {batch_size}"
        
        # Sequence length = num_queries + last_spatial_size (16x16 = 256)
        expected_seq_len = num_queries + 256
        assert output.shape[1] == expected_seq_len, f"Seq len: {output.shape[1]} != {expected_seq_len}"
        
        assert output.shape[2] == 1024, f"Output channels: {output.shape[2]} != 1024"


class TestMSSEIntegration:
    """Integration tests for MultiScaleStyleEncoder."""

    def test_msse_output_consistency(self, device):
        """Test MSSE produces consistent outputs."""
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
        assert len(features1) == len(features2) == 5, "MSSE should output 5 scales"
        
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            assert torch.allclose(f1, f2, atol=1e-6), f"Scale {i} outputs not consistent"

    def test_msse_feature_channels(self, device):
        """Test MSSE outputs correct channel dimensions."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        expected_channels = [64, 128, 256, 512, 1024]
        actual_channels = msse.get_feature_channels()
        
        assert actual_channels == expected_channels, \
            f"Channel mismatch: {actual_channels} != {expected_channels}"

    def test_msse_training_mode_consistency(self, device):
        """Test MSSE behavior in training mode."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        msse.train()
        x = torch.randn(2, 1, 512, 512, device=device)

        # InstanceNorm should be deterministic even in training
        with torch.no_grad():
            features1 = msse(x)
            features2 = msse(x)

        for i, (f1, f2) in enumerate(zip(features1, features2)):
            assert torch.allclose(f1, f2, atol=1e-5), \
                f"InstanceNorm should be deterministic in scale {i}"


class TestFSTWithMSSEPipeline:
    """Test FST and MSSE working together."""

    def test_msse_to_fst_pipeline(self, device):
        """Test complete MSSE -> FST pipeline."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=64,
            num_scales=5
        ).to(device)

        feature_channels = msse.get_feature_channels()
        assert feature_channels == [64, 128, 256, 512, 1024]

        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Source and target style images (512x512)
        style_source = torch.randn(2, 1, 512, 512, device=device)
        style_target = torch.randn(2, 1, 512, 512, device=device)

        # Extract features
        msse.eval()
        with torch.no_grad():
            source_features = msse(style_source)
            target_features = msse(style_target)

        # Verify feature shapes
        expected_spatial_sizes = [256, 128, 64, 32, 16]
        for i, (src, tgt, expected_ch, expected_size) in enumerate(
            zip(source_features, target_features, feature_channels, expected_spatial_sizes)
        ):
            assert src.shape == (2, expected_ch, expected_size, expected_size), \
                f"Source scale {i} shape mismatch"
            assert tgt.shape == (2, expected_ch, expected_size, expected_size), \
                f"Target scale {i} shape mismatch"

        # Process through FST
        fst_output = fst(source_features, target_features)

        # Verify shapes
        assert fst_output.shape[0] == 2, "Batch size mismatch"
        assert fst_output.shape[2] == 1024, "Output channels mismatch"
        
        # Expected: num_queries + last_spatial_size (16x16 = 256)
        expected_seq_len = 128 + 256
        assert fst_output.shape[1] == expected_seq_len, "Sequence length mismatch"

        # Verify no NaNs
        assert not torch.isnan(fst_output).any(), "Output contains NaN"


class TestFSTGradientFlow:
    """Test gradient flow specific to FST."""

    def test_fst_learnable_queries_gradients(self, device):
        """Verify learnable queries receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Create dummy inputs with correct spatial sizes
        source_features = []
        target_features = []
        spatial_sizes = [256, 128, 64, 32, 16]

        for ch, spatial_size in zip([64, 128, 256, 512, 1024], spatial_sizes):
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Learnable queries should have gradient
        assert fst.learnable_queries.grad is not None, "Learnable queries have no gradient"
        assert (fst.learnable_queries.grad.abs() > 0).any(), "Learnable queries gradient is zero"

    def test_fst_positional_encoding_gradients(self, device):
        """Verify positional encodings receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features = []
        target_features = []
        spatial_sizes = [256, 128, 64, 32, 16]

        for ch, spatial_size in zip([64, 128, 256, 512, 1024], spatial_sizes):
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Each positional encoding should have gradient
        for i, pe in enumerate(fst.pos_encodings):
            assert pe.grad is not None, f"PosEncoding {i} has no gradient"
            assert (pe.grad.abs() > 0).any(), f"PosEncoding {i} gradient is zero"

    def test_fst_projection_layers_gradients(self, device):
        """Verify all projection layers in FST receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features = []
        target_features = []
        spatial_sizes = [256, 128, 64, 32, 16]

        for ch, spatial_size in zip([64, 128, 256, 512, 1024], spatial_sizes):
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Check q_projs
        for i, proj in enumerate(fst.q_projs):
            assert proj.weight.grad is not None, f"q_projs[{i}] has no gradient"

        # Check k_projs
        for i, proj in enumerate(fst.k_projs):
            assert proj.weight.grad is not None, f"k_projs[{i}] has no gradient"

        # Check v_projs
        for i, proj in enumerate(fst.v_projs):
            assert proj.weight.grad is not None, f"v_projs[{i}] has no gradient"

        # Check MLP layers
        for name, param in fst.mlp_channel_adjust.named_parameters():
            assert param.grad is not None, f"mlp_channel_adjust.{name} has no gradient"

        # Check residual projection
        assert fst.residual_proj.weight.grad is not None, "residual_proj.weight has no gradient"


class TestFSTOverfitting:
    """Test that FST can overfit on small batch."""

    def test_fst_batch_overfitting(self, device):
        """Test FST can memorize and overfit a small batch."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Create fixed batch with correct spatial sizes
        source_features = []
        target_features = []
        spatial_sizes = [256, 128, 64, 32, 16]

        for ch, spatial_size in zip([64, 128, 256, 512, 1024], spatial_sizes):
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

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
        optimizer = torch.optim.Adam(fst.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            output = fst(source_features, target_features)
            loss = loss_fn(output, target_output)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        initial_loss = losses[0]
        final_loss = losses[-1]

        print(f"\nFST overfitting - Initial: {initial_loss:.6f}, Final: {final_loss:.6f}")

        assert final_loss < initial_loss, "FST loss should decrease during overfitting"
        assert (initial_loss - final_loss) / initial_loss > 0.2, (
            "Loss reduction should be > 20%"
        )


class TestMSSEOverfitting:
    """Test that MSSE can overfit."""

    def test_msse_can_memorize(self, device):
        """Test MSSE can learn to map inputs to target features."""
        base_channels = 64
        num_scales = 5
        
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=base_channels,
            num_scales=num_scales,
        ).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 1, 512, 512, device=device)

        # Create target features matching MSSE output
        target_features = []
        spatial_size = 256  # After initial conv stride=2
        for scale_idx in range(num_scales):
            ch = base_channels * (2 ** scale_idx)
            target_features.append(
                torch.randn(batch_size, ch, spatial_size, spatial_size, device=device)
            )
            spatial_size = spatial_size // 2

        optimizer = torch.optim.Adam(msse.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        losses = []
        for _ in range(30):
            optimizer.zero_grad()

            features = msse(x)

            # Compute loss for all scales
            loss = 0
            for feat, target in zip(features, target_features):
                loss += loss_fn(feat, target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"\nMSSE overfitting - Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}")

        assert losses[-1] < losses[0], "MSSE loss should decrease"
        assert (losses[0] - losses[-1]) / losses[0] > 0.1, "Loss reduction should be > 10%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])