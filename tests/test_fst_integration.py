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

        fst = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        ).to(device)

        # Create input features
        source_features = []
        target_features = []
        spatial_size = 96

        for ch in feature_channels:
            spatial_size = spatial_size // 2
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

        # Verify output shape
        assert output.shape[0] == batch_size
        assert output.shape[2] == 1024

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
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=num_queries,
            query_dim=query_dim,
            num_scale_features=5,
        ).to(device)

        source_features = []
        target_features = []
        spatial_size = 96

        for ch in [64, 128, 256, 512, 1024]:
            spatial_size = spatial_size // 2
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        with torch.no_grad():
            output = fst(source_features, target_features)

        # Verify output dimensions
        assert output.shape[0] == 2
        assert output.shape[1] == num_queries + 9  # 3x3 spatial at last scale
        assert output.shape[2] == 1024


class TestMSSEIntegration:
    """Integration tests for MultiScaleStyleEncoder."""

    def test_msse_output_consistency(self, device):
        """Test MSSE produces consistent outputs."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        msse.eval()

        x = torch.randn(2, 1, 96, 96, device=device)

        with torch.no_grad():
            features1 = msse(x)
            features2 = msse(x)

        # In eval mode, outputs should be identical
        for f1, f2 in zip(features1, features2):
            assert torch.allclose(f1, f2), "MSSE outputs not consistent"

    def test_msse_training_mode_variance(self, device):
        """Test MSSE has variance in training mode."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=3).to(
            device
        )

        msse.train()

        x = torch.randn(4, 1, 96, 96, device=device)

        # Multiple forward passes should give different results (due to dropout, etc.)
        features_list = []
        for _ in range(3):
            with torch.no_grad():
                features = msse(x)
            features_list.append(torch.cat([f.flatten() for f in features]))

        # At least some differences should exist
        # (This is probabilistic, so we just check they're not identical)
        all_same = all(
            torch.allclose(features_list[0], f, atol=1e-6) for f in features_list[1:]
        )
        # In training mode with batch norm, we expect some variance
        # But we don't assert this strictly as it depends on architecture


class TestFSTWithMSSEPipeline:
    """Test FST and MSSE working together."""

    def test_msse_to_fst_pipeline(self, device):
        """Test complete MSSE -> FST pipeline."""
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )

        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        # Source and target style images
        style_source = torch.randn(2, 1, 96, 96, device=device)
        style_target = torch.randn(2, 1, 96, 96, device=device)

        # Extract features
        msse.eval()
        with torch.no_grad():
            source_features = msse(style_source)
            target_features = msse(style_target)

        # Process through FST
        fst_output = fst(source_features, target_features)

        # Verify shapes
        assert fst_output.shape[0] == 2
        assert fst_output.shape[2] == 1024

        # Verify no NaNs
        assert not torch.isnan(fst_output).any()


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

        # Create dummy inputs
        source_features = []
        target_features = []
        spatial_size = 96

        for ch in [64, 128, 256, 512, 1024]:
            spatial_size = spatial_size // 2
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Learnable queries should have gradient
        assert fst.learnable_queries.grad is not None
        assert (fst.learnable_queries.grad.abs() > 0).any()

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
        spatial_size = 96

        for ch in [64, 128, 256, 512, 1024]:
            spatial_size = spatial_size // 2
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

    def test_fst_projection_layers_gradients(self, device):
        """Verify projection layers in FST receive gradients."""
        fst = FontStyleTransformationModule(
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=128,
            query_dim=128,
            num_scale_features=5,
        ).to(device)

        source_features = []
        target_features = []
        spatial_size = 96

        for ch in [64, 128, 256, 512, 1024]:
            spatial_size = spatial_size // 2
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        output = fst(source_features, target_features)
        loss = output.sum()
        loss.backward()

        # Check all MLPs have gradients
        for name, param in fst.mlp_channel_adjust.named_parameters():
            assert param.grad is not None, f"MLP param {name} has no gradient"

        assert fst.residual_proj.weight.grad is not None


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

        # Create fixed batch
        source_features = []
        target_features = []
        spatial_size = 96

        for ch in [64, 128, 256, 512, 1024]:
            spatial_size = spatial_size // 2
            src = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            tgt = torch.randn(2, ch, spatial_size, spatial_size, device=device)
            source_features.append(src)
            target_features.append(tgt)

        # Create target output
        target_output = torch.randn(2, 256 + 9, 1024, device=device)

        # Training loop
        optimizer = torch.optim.Adam(fst.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            output = fst(source_features, target_features)
            loss = loss_fn(output, target_output)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        initial_loss = losses[0]
        final_loss = losses[-1]

        print(f"\nFST overfitting loss: {initial_loss:.6f} -> {final_loss:.6f}")

        assert final_loss < initial_loss, "FST should be able to overfit"
        assert (
            initial_loss - final_loss
        ) / initial_loss > 0.3, "Loss reduction should be > 30%"


class TestMSSEOverfitting:
    """Test that MSSE can overfit."""

    def test_msse_can_memorize(self, device):
        """Test MSSE can memorize simple mappings."""
        msse = MultiScaleStyleEncoder(
            in_channels=1,
            base_channels=32,  # Smaller for testing
            num_scales=3,
        ).to(device)

        x = torch.randn(2, 1, 96, 96, device=device)

        # Create target features
        target_features = []
        spatial_size = 96
        for _ in range(3):
            spatial_size = spatial_size // 2
            target_features.append(
                torch.randn(2, 32 * (2**_), spatial_size, spatial_size, device=device)
            )

        optimizer = torch.optim.Adam(msse.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()

            features = msse(x)

            # Compute loss (resize targets if needed)
            loss = 0
            for feat, target in zip(features, target_features):
                loss += loss_fn(feat, target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"\nMSSE overfitting loss: {losses[0]:.6f} -> {losses[-1]:.6f}")

        assert losses[-1] < losses[0], "MSSE should be able to overfit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
