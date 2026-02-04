"""
Pytest suite for Identity Mapping Loss module.
Tests all loss variants: IdentityMappingLoss, PooledIdentityMappingLoss, AdaptiveIdentityMappingLoss.
"""
import pytest
import torch
import torch.nn as nn

from src.modules.identity_mapping_loss import (
    IdentityMappingLoss,
    PooledIdentityMappingLoss,
    AdaptiveIdentityMappingLoss,
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
def matrix_size():
    """Standard transformation matrix size."""
    return 256


@pytest.fixture
def feature_dim():
    """Feature dimension for FST outputs."""
    return 1024


class TestIdentityMappingLoss:
    """Test suite for IdentityMappingLoss."""

    @pytest.fixture
    def identity_loss(self, matrix_size, device):
        """Create IdentityMappingLoss instance."""
        return IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="frobenius",
            regularization="orthogonal",
            reg_weight=0.01,
        ).to(device)

    def test_initialization(self, identity_loss, matrix_size):
        """Test model initialization."""
        assert identity_loss.matrix_size == matrix_size
        assert identity_loss.loss_type == "frobenius"
        assert identity_loss.regularization == "orthogonal"
        assert identity_loss.identity_matrix.shape == (matrix_size, matrix_size)

    def test_extract_features_3d(self, identity_loss, batch_size, matrix_size, device):
        """Test feature extraction from 3D tensor."""
        features = torch.randn(batch_size, matrix_size + 100, 1024, device=device)

        extracted = identity_loss._extract_features(features)

        assert extracted.shape == (batch_size, matrix_size, 1024)

    def test_extract_features_4d(self, identity_loss, batch_size, matrix_size, device):
        """Test feature extraction from 4D tensor (flattening batch dims)."""
        features = torch.randn(2, batch_size // 2, matrix_size + 50, 1024, device=device)

        extracted = identity_loss._extract_features(features)

        assert extracted.shape == (batch_size, matrix_size, 1024)

    def test_compute_transformation_matrix(
        self, identity_loss, batch_size, matrix_size, device
    ):
        """Test transformation matrix computation."""
        source = torch.randn(batch_size, matrix_size, 1024, device=device)
        target = torch.randn(batch_size, matrix_size, 1024, device=device)

        T = identity_loss.compute_transformation_matrix(source, target)

        assert T.shape == (batch_size, matrix_size, matrix_size)
        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()

    def test_identity_distance_loss_frobenius(
        self, batch_size, matrix_size, device
    ):
        """Test Frobenius norm distance from identity."""
        loss_module = IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="frobenius",
        ).to(device)

        # Create near-identity matrix
        T = (
            torch.eye(matrix_size, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        )
        T = T + torch.randn_like(T) * 0.01

        loss = loss_module.identity_distance_loss(T)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss).any()

    def test_identity_distance_loss_mse(self, batch_size, matrix_size, device):
        """Test MSE distance from identity."""
        loss_module = IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="mse",
        ).to(device)

        T = torch.randn(batch_size, matrix_size, matrix_size, device=device)

        loss = loss_module.identity_distance_loss(T)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss).any()

    def test_identity_distance_loss_cosine(self, batch_size, matrix_size, device):
        """Test cosine distance from identity."""
        loss_module = IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="cosine",
        ).to(device)

        T = torch.randn(batch_size, matrix_size, matrix_size, device=device)

        loss = loss_module.identity_distance_loss(T)

        assert loss.item() >= 0.0
        assert loss.item() <= 2.0  # Cosine distance bounded by [0, 2]
        assert not torch.isnan(loss).any()

    def test_orthogonality_regularization(
        self, identity_loss, batch_size, matrix_size, device
    ):
        """Test orthogonality regularization."""
        T = torch.randn(batch_size, matrix_size, matrix_size, device=device)

        reg_loss = identity_loss.orthogonality_regularization(T)

        assert reg_loss.item() >= 0.0
        assert not torch.isnan(reg_loss).any()

    def test_spectral_regularization(
        self, identity_loss, batch_size, matrix_size, device
    ):
        """Test spectral regularization."""
        T = torch.randn(batch_size, matrix_size, matrix_size, device=device)

        reg_loss = identity_loss.spectral_regularization(T)

        assert reg_loss.item() >= 0.0
        assert not torch.isnan(reg_loss).any()

    def test_forward_same_style_pair(
        self, identity_loss, batch_size, matrix_size, device
    ):
        """Test forward pass with same-style pair."""
        # Simulate FST features from same style
        source = torch.randn(batch_size, matrix_size + 100, 1024, device=device)
        target = source + torch.randn_like(source) * 0.1  # Small perturbation

        loss, metrics = identity_loss(source, target)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss).any()
        assert "identity_loss" in metrics
        assert "total_identity_loss" in metrics
        assert "diagonal_mean" in metrics

    def test_forward_different_style_pair(
        self, identity_loss, batch_size, matrix_size, device
    ):
        """Test forward pass with different-style pair."""
        source = torch.randn(batch_size, matrix_size + 100, 1024, device=device)
        target = torch.randn(batch_size, matrix_size + 100, 1024, device=device)

        loss, metrics = identity_loss(source, target)

        # Loss should be positive (not identity)
        assert loss.item() > 0.0
        assert not torch.isnan(loss).any()

    def test_gradient_flow(self, identity_loss, batch_size, matrix_size, device):
        """Test gradient backpropagation."""
        source = torch.randn(
            batch_size, matrix_size, 1024, device=device, requires_grad=True
        )
        target = torch.randn(
            batch_size, matrix_size, 1024, device=device, requires_grad=True
        )

        loss, _ = identity_loss(source, target)
        loss.backward()

        assert source.grad is not None
        assert target.grad is not None
        assert not torch.isnan(source.grad).any()

    def test_perfect_identity_matrix(self, identity_loss, batch_size, matrix_size, device):
        """Test loss is zero for perfect identity transformation."""
        # Create identical features
        features = torch.randn(batch_size, matrix_size, 1024, device=device)

        # Normalize for numerical stability
        features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)

        loss, metrics = identity_loss(features_norm, features_norm)

        # Loss should be very small (near zero)
        assert loss.item() < 0.1  # Allow small numerical error


class TestPooledIdentityMappingLoss:
    """Test suite for PooledIdentityMappingLoss."""

    @pytest.fixture
    def pooled_loss(self, matrix_size, device):
        """Create PooledIdentityMappingLoss instance."""
        return PooledIdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="frobenius",
            reduction="mean",
        ).to(device)

    def test_initialization(self, pooled_loss):
        """Test model initialization."""
        assert pooled_loss.reduction == "mean"
        assert isinstance(pooled_loss.base_loss, IdentityMappingLoss)

    def test_forward_single_pair(self, pooled_loss, batch_size, matrix_size, device):
        """Test forward with single pair."""
        pair = [
            (
                torch.randn(batch_size, matrix_size, 1024, device=device),
                torch.randn(batch_size, matrix_size, 1024, device=device),
            )
        ]

        loss, metrics = pooled_loss(pair)

        assert loss.item() >= 0.0
        assert metrics["num_pairs"] == 1

    def test_forward_multiple_pairs(self, pooled_loss, batch_size, matrix_size, device):
        """Test forward with multiple pairs."""
        pairs = [
            (
                torch.randn(batch_size, matrix_size, 1024, device=device),
                torch.randn(batch_size, matrix_size, 1024, device=device),
            )
            for _ in range(5)
        ]

        loss, metrics = pooled_loss(pairs)

        assert loss.item() >= 0.0
        assert metrics["num_pairs"] == 5
        assert "avg_identity_loss" in metrics

    def test_forward_empty_pairs(self, pooled_loss):
        """Test forward with no pairs."""
        pairs = []

        loss, metrics = pooled_loss(pairs)

        assert loss.item() == 0.0
        assert metrics["num_pairs"] == 0

    def test_reduction_modes(self, batch_size, matrix_size, device):
        """Test different reduction modes."""
        pairs = [
            (
                torch.randn(batch_size, matrix_size, 1024, device=device),
                torch.randn(batch_size, matrix_size, 1024, device=device),
            )
            for _ in range(3)
        ]

        # Test mean reduction
        loss_mean_module = PooledIdentityMappingLoss(
            matrix_size=matrix_size, reduction="mean"
        ).to(device)
        loss_mean, _ = loss_mean_module(pairs)

        # Test sum reduction
        loss_sum_module = PooledIdentityMappingLoss(
            matrix_size=matrix_size, reduction="sum"
        ).to(device)
        loss_sum, _ = loss_sum_module(pairs)

        # Sum should be approximately 3x mean
        assert abs(loss_sum.item() - loss_mean.item() * 3) < 0.1

    def test_gradient_flow_multiple_pairs(
        self, pooled_loss, batch_size, matrix_size, device
    ):
        """Test gradient flow through multiple pairs."""
        pairs = [
            (
                torch.randn(
                    batch_size, matrix_size, 1024, device=device, requires_grad=True
                ),
                torch.randn(
                    batch_size, matrix_size, 1024, device=device, requires_grad=True
                ),
            )
            for _ in range(3)
        ]

        loss, _ = pooled_loss(pairs)
        loss.backward()

        for source, target in pairs:
            assert source.grad is not None
            assert target.grad is not None


class TestAdaptiveIdentityMappingLoss:
    """Test suite for AdaptiveIdentityMappingLoss."""

    @pytest.fixture
    def adaptive_loss(self, matrix_size, device):
        """Create AdaptiveIdentityMappingLoss instance."""
        return AdaptiveIdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type="frobenius",
            similarity_threshold=0.8,
            max_weight=1.0,
            min_weight=0.1,
        ).to(device)

    def test_initialization(self, adaptive_loss):
        """Test model initialization."""
        assert adaptive_loss.similarity_threshold == 0.8
        assert adaptive_loss.max_weight == 1.0
        assert adaptive_loss.min_weight == 0.1

    def test_compute_style_similarity_identical(
        self, adaptive_loss, batch_size, matrix_size, device
    ):
        """Test similarity computation for identical styles."""
        features = torch.randn(batch_size, matrix_size, 1024, device=device)

        similarity = adaptive_loss.compute_style_similarity(features, features)

        # Should be very close to 1.0
        assert torch.allclose(similarity, torch.ones_like(similarity), atol=1e-5)

    def test_compute_style_similarity_different(
        self, adaptive_loss, batch_size, matrix_size, device
    ):
        """Test similarity computation for different styles."""
        source = torch.randn(batch_size, matrix_size, 1024, device=device)
        target = torch.randn(batch_size, matrix_size, 1024, device=device)

        similarity = adaptive_loss.compute_style_similarity(source, target)

        # Should be between -1 and 1
        assert (similarity >= -1.0).all()
        assert (similarity <= 1.0).all()

    def test_forward_high_similarity(
        self, adaptive_loss, batch_size, matrix_size, device
    ):
        """Test forward with high style similarity."""
        # Similar styles
        source = torch.randn(batch_size, matrix_size, 1024, device=device)
        target = source + torch.randn_like(source) * 0.01  # Very similar

        loss, metrics = adaptive_loss(source, target)

        # Weight should be close to max_weight
        assert metrics["adaptive_weight"] > 0.8
        assert metrics["style_similarity"] > 0.9

    def test_forward_low_similarity(
        self, adaptive_loss, batch_size, matrix_size, device
    ):
        """Test forward with low style similarity."""
        # Different styles
        source = torch.randn(batch_size, matrix_size, 1024, device=device)
        target = torch.randn(batch_size, matrix_size, 1024, device=device)

        loss, metrics = adaptive_loss(source, target)

        # Weight should be lower
        assert "adaptive_weight" in metrics
        assert "style_similarity" in metrics

    def test_gradient_flow(self, adaptive_loss, batch_size, matrix_size, device):
        """Test gradient backpropagation."""
        source = torch.randn(
            batch_size, matrix_size, 1024, device=device, requires_grad=True
        )
        target = torch.randn(
            batch_size, matrix_size, 1024, device=device, requires_grad=True
        )

        loss, _ = adaptive_loss(source, target)
        loss.backward()

        assert source.grad is not None
        assert target.grad is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_matrix_size_mismatch(self, device):
        """Test error handling for mismatched matrix sizes."""
        loss_module = IdentityMappingLoss(matrix_size=256).to(device)

        source = torch.randn(2, 128, 1024, device=device)  # Wrong size
        target = torch.randn(2, 256, 1024, device=device)

        # Should extract only first matrix_size features
        loss, _ = loss_module(source, target)
        assert not torch.isnan(loss).any()

    def test_batch_size_1(self, matrix_size, device):
        """Test with batch size of 1."""
        loss_module = IdentityMappingLoss(matrix_size=matrix_size).to(device)

        source = torch.randn(1, matrix_size, 1024, device=device)
        target = torch.randn(1, matrix_size, 1024, device=device)

        loss, _ = loss_module(source, target)
        assert not torch.isnan(loss).any()

    def test_large_batch_size(self, matrix_size, device):
        """Test with large batch size."""
        loss_module = IdentityMappingLoss(matrix_size=matrix_size).to(device)

        source = torch.randn(32, matrix_size, 1024, device=device)
        target = torch.randn(32, matrix_size, 1024, device=device)

        loss, _ = loss_module(source, target)
        assert not torch.isnan(loss).any()


class TestIntegration:
    """Integration tests with FST module."""

    def test_with_fst_output(self, device):
        """Test identity loss with realistic FST output."""
        from src.modules.fst import FontStyleTransformationModule
        from src.modules.msse import MultiScaleStyleEncoder

        batch_size = 2
        matrix_size = 220  # FST num_queries

        # Create models
        msse = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5).to(
            device
        )
        fst = FontStyleTransformationModule(
            msse_output_channels=msse.get_output_channels(),
            num_queries=matrix_size,
            query_dim=128,
        ).to(device)

        identity_loss = IdentityMappingLoss(matrix_size=matrix_size).to(device)

        # Simulate same-style pairs
        style_img = torch.randn(batch_size, 1, 96, 96, device=device)

        source_features = msse(style_img)
        target_features = [f.clone() for f in source_features]

        # FST transformation
        source_transform = fst(source_features, target_features)
        target_transform = fst(target_features, source_features)

        # Compute identity loss
        loss, metrics = identity_loss(source_transform, target_transform)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss).any()
        assert "identity_loss" in metrics


@pytest.mark.parametrize("loss_type", ["frobenius", "mse", "cosine"])
def test_various_loss_types(device, batch_size, matrix_size, loss_type):
    """Parametrized test for different loss types."""
    loss_module = IdentityMappingLoss(
        matrix_size=matrix_size,
        loss_type=loss_type,
    ).to(device)

    source = torch.randn(batch_size, matrix_size, 1024, device=device)
    target = torch.randn(batch_size, matrix_size, 1024, device=device)

    loss, _ = loss_module(source, target)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss).any()


@pytest.mark.parametrize("regularization", ["orthogonal", "spectral", None])
def test_various_regularizations(device, batch_size, matrix_size, regularization):
    """Parametrized test for different regularization types."""
    loss_module = IdentityMappingLoss(
        matrix_size=matrix_size,
        regularization=regularization,
        reg_weight=0.01,
    ).to(device)

    source = torch.randn(batch_size, matrix_size, 1024, device=device)
    target = torch.randn(batch_size, matrix_size, 1024, device=device)

    loss, metrics = loss_module(source, target)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])