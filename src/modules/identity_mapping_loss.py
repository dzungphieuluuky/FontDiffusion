"""
Identity Mapping Loss for Font Style Transformation (FST) Module.

Enforces that when source and target styles are identical, the learned
transformation should approximate the identity mapping.

Key Concept:
    For pairs (A_style1, B_style1), (C_style2, D_style2), etc., where both
    images in each pair share the same style but have different content,
    the style transformation matrix should be identity (no change needed).

This encourages the FST module to:
1. Recognize when styles are identical
2. Learn content-invariant style representations
3. Produce disentangled content/style features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IdentityMappingLoss(nn.Module):
    """
    Identity mapping loss for same-style pairs.
    
    Computes a transformation matrix from FST features and measures
    its distance from the identity matrix.
    """
    
    def __init__(
        self,
        matrix_size: int = 256,
        loss_type: str = "frobenius",
        regularization: str = "orthogonal",
        reg_weight: float = 0.01,
        extract_queries_only: bool = True,
    ):
        """
        Args:
            matrix_size: Size of transformation matrix (typically num_queries)
            loss_type: How to measure distance from identity
            regularization: Additional regularization type
            reg_weight: Weight for regularization term
            extract_queries_only: If True, extract only first matrix_size from features
        """
        super().__init__()
        self.matrix_size = matrix_size
        self.loss_type = loss_type.lower()
        self.regularization = regularization
        self.reg_weight = reg_weight
        self.extract_queries_only = extract_queries_only
        
        # Pre-compute identity matrix
        self.register_buffer(
            "identity_matrix",
            torch.eye(matrix_size, dtype=torch.float32)
        )
    
    def _extract_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract relevant features from FST output.
        
        FST outputs (B, N_L + H*W, D) where:
        - N_L = num_queries (learnable portion)
        - H*W = spatial portion (optional, often ignored)
        
        If extract_queries_only=True, extract only first matrix_size features.
        
        Args:
            features: (..., N, D) - Can have >3 dims; flatten to (B, N, D)
            
        Returns:
            features: (B, N, D) - Exactly 3 dimensions
        """
        # Flatten batch dimensions if needed
        original_shape = features.shape
        if len(original_shape) > 3:
            # Reshape to (B, N, D) by flattening all but last two dims
            features = features.reshape(-1, original_shape[-2], original_shape[-1])
        
        # Extract only queries if needed
        if self.extract_queries_only:
            features = features[:, :self.matrix_size, :]
        
        return features
        
    def compute_transformation_matrix(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute transformation matrix from source to target features.
        
        Args:
            source_features: (B, N, D) - Features from source image
                where N = num_queries (matrix_size), D = feature_dim
            target_features: (B, N, D) - Features from target image
            
        Returns:
            transformation_matrix: (B, N, N) - Transformation matrix
        """
        # Extract and normalize features
        source_features = self._extract_features(source_features)
        target_features = self._extract_features(target_features)
        
        B, N, D = source_features.shape
        
        # Normalize features for numerical stability
        source_norm = F.normalize(source_features, p=2, dim=-1)  # (B, N, D)
        target_norm = F.normalize(target_features, p=2, dim=-1)  # (B, N, D)
        
        # Compute transformation matrix: (B, N, D) @ (B, D, N) = (B, N, N)
        # This maps N query positions from source to target space
        transformation_matrix = torch.bmm(
            source_norm,                    # (B, N, D)
            target_norm.transpose(1, 2)     # (B, D, N)
        )  # Result: (B, N, N)
        
        return transformation_matrix
        
    def identity_distance_loss(
        self,
        transformation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance between transformation matrix and identity.
        
        Args:
            transformation_matrix: (B, N, N) where N = matrix_size
            
        Returns:
            Scalar loss measuring distance from identity
        """
        B, N, _ = transformation_matrix.shape
        
        # Create identity matrix with correct size
        identity = torch.eye(N, device=transformation_matrix.device, 
                            dtype=transformation_matrix.dtype).unsqueeze(0).expand(B, -1, -1)
        
        if self.loss_type == "frobenius":
            # Frobenius norm: sqrt(sum((T - I)^2))
            diff = transformation_matrix - identity
            loss = torch.norm(diff.reshape(B, -1), p='fro', dim=1).mean()
            
        elif self.loss_type == "mse":
            # Mean squared error
            loss = F.mse_loss(transformation_matrix, identity)
            
        elif self.loss_type == "cosine":
            # Cosine distance for each row
            T_flat = transformation_matrix.reshape(B * N, N)
            I_flat = identity.reshape(B * N, N)
            similarity = F.cosine_similarity(T_flat, I_flat, dim=1)
            loss = 1.0 - similarity.mean()
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
        
    def orthogonality_regularization(
        self,
        transformation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage transformation matrix to be orthogonal: T^T @ T ≈ I.
        
        Args:
            transformation_matrix: (B, N, N)
            
        Returns:
            Orthogonality regularization loss
        """
        B, N, _ = transformation_matrix.shape
        
        # Compute T^T @ T
        TtT = torch.bmm(
            transformation_matrix.transpose(1, 2),
            transformation_matrix
        )  # (B, N, N)
        
        # Create identity matrix with correct size
        identity = torch.eye(N, device=transformation_matrix.device,
                            dtype=transformation_matrix.dtype).unsqueeze(0).expand(B, -1, -1)
        
        # Frobenius norm of difference
        diff = TtT - identity
        loss = torch.norm(diff.reshape(B, -1), p='fro', dim=1).mean()
        
        return loss
    
    def spectral_regularization(
        self,
        transformation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize large singular values to prevent ill-conditioning.
        
        Args:
            transformation_matrix: (B, N, N)
            
        Returns:
            Spectral regularization loss
        """
        B = transformation_matrix.shape[0]
        
        # Compute singular values for each matrix in batch
        losses = []
        for i in range(B):
            U, S, V = torch.svd(transformation_matrix[i])
            # Penalize deviation from unit singular values
            loss = torch.mean((S - 1.0) ** 2)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute identity mapping loss for same-style pairs.
        
        Args:
            source_features: (..., N, D) - FST features from source image
                            Can be (B, N, D) or (B, N_L + H*W, D)
            target_features: (..., N, D) - FST features from target image
                            Both should have the SAME style
            
        Returns:
            loss: Scalar identity loss
            metrics: Dictionary with detailed loss components
        """
        # Extract and validate features
        source_features = self._extract_features(source_features)
        target_features = self._extract_features(target_features)
        
        # Compute transformation matrix
        T = self.compute_transformation_matrix(source_features, target_features)
        
        # Main identity loss
        identity_loss = self.identity_distance_loss(T)
        
        # Optional regularization
        reg_loss = torch.tensor(0.0, device=source_features.device)
        
        if self.regularization == "orthogonal":
            reg_loss = self.orthogonality_regularization(T)
        elif self.regularization == "spectral":
            reg_loss = self.spectral_regularization(T)
        
        # Total loss
        total_loss = identity_loss + self.reg_weight * reg_loss
        
        # Metrics for logging
        metrics = {
            "identity_loss": identity_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_identity_loss": total_loss.item(),
        }
        
        # Additional diagnostics
        with torch.no_grad():
            # Measure how close T is to identity (diagonal dominance)
            B, N, _ = T.shape
            diagonal = torch.diagonal(T, dim1=1, dim2=2)  # (B, N)
            off_diagonal = T - torch.eye(N, device=T.device).unsqueeze(0) * diagonal.unsqueeze(-1)
            
            metrics["diagonal_mean"] = diagonal.mean().item()
            metrics["diagonal_std"] = diagonal.std().item()
            metrics["off_diagonal_norm"] = torch.norm(off_diagonal.reshape(B, -1), dim=1).mean().item()
        
        return total_loss, metrics


class PooledIdentityMappingLoss(nn.Module):
    """
    Identity mapping loss computed over a pool of same-style pairs.
    
    For a batch containing multiple pairs, each pair shares the same style
    but has different content. This loss enforces that the transformation
    for all pairs should be identity.
    """
    
    def __init__(
        self,
        matrix_size: int = 256,
        loss_type: str = "frobenius",
        reduction: str = "mean",
        regularization: str = "orthogonal",
        reg_weight: float = 0.01,
    ):
        """
        Args:
            matrix_size: Size of transformation matrix
            loss_type: Distance metric from identity
            reduction: How to aggregate losses across pairs ("mean", "sum")
            regularization: Additional regularization type
            reg_weight: Weight for regularization
        """
        super().__init__()
        self.reduction = reduction
        
        self.base_loss = IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type=loss_type,
            regularization=regularization,
            reg_weight=reg_weight,
        )
    
    def forward(
        self,
        pair_features: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute identity loss over multiple same-style pairs.
        
        Args:
            pair_features: List of (source_features, target_features) tuples
                Each tuple contains:
                    source_features: (B, N, D)
                    target_features: (B, N, D)
                Both images in the pair have the same style, different content.
            
        Returns:
            loss: Aggregated identity loss
            metrics: Dictionary with loss statistics
        """
        if len(pair_features) == 0:
            return (
                torch.tensor(0.0, requires_grad=True),
                {"num_pairs": 0}
            )
        
        losses = []
        all_metrics = []
        
        for source_feat, target_feat in pair_features:
            pair_loss, pair_metrics = self.base_loss(source_feat, target_feat)
            losses.append(pair_loss)
            all_metrics.append(pair_metrics)
        
        # Aggregate losses
        if self.reduction == "mean":
            total_loss = torch.stack(losses).mean()
        elif self.reduction == "sum":
            total_loss = torch.stack(losses).sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        # Aggregate metrics
        aggregated_metrics = {
            "num_pairs": len(pair_features),
        }
        
        if all_metrics:
            # Average each metric across pairs
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return total_loss, aggregated_metrics


class AdaptiveIdentityMappingLoss(nn.Module):
    """
    Adaptive identity mapping loss that dynamically adjusts based on
    the actual similarity between source and target styles.
    
    If styles are very similar → strong identity constraint
    If styles are different → weak identity constraint (avoid false penalties)
    """
    
    def __init__(
        self,
        matrix_size: int = 256,
        loss_type: str = "frobenius",
        similarity_threshold: float = 0.8,
        max_weight: float = 1.0,
        min_weight: float = 0.1,
    ):
        """
        Args:
            matrix_size: Size of transformation matrix
            loss_type: Distance metric from identity
            similarity_threshold: Cosine similarity threshold for "same style"
            max_weight: Weight when styles are identical
            min_weight: Weight when styles are different
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        self.base_loss = IdentityMappingLoss(
            matrix_size=matrix_size,
            loss_type=loss_type,
        )
    
    def compute_style_similarity(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global style similarity between source and target.
        
        Args:
            source_features: (B, N, D)
            target_features: (B, N, D)
            
        Returns:
            similarity: (B,) - Cosine similarity for each pair
        """
        # Global average pooling
        source_global = source_features.mean(dim=1)  # (B, D)
        target_global = target_features.mean(dim=1)  # (B, D)
        
        # Cosine similarity
        similarity = F.cosine_similarity(source_global, target_global, dim=-1)
        
        return similarity
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive identity loss.
        
        Args:
            source_features: (B, N, D)
            target_features: (B, N, D)
            
        Returns:
            loss: Weighted identity loss
            metrics: Dictionary with statistics
        """
        # Compute base identity loss
        base_loss, base_metrics = self.base_loss(source_features, target_features)
        
        # Compute style similarity
        similarity = self.compute_style_similarity(source_features, target_features)
        
        # Adaptive weighting
        # If similarity > threshold: weight = max_weight
        # If similarity < threshold: weight linearly decreases to min_weight
        weight = torch.where(
            similarity > self.similarity_threshold,
            torch.tensor(self.max_weight, device=similarity.device),
            self.min_weight + (self.max_weight - self.min_weight) * (
                similarity / self.similarity_threshold
            )
        )
        
        # Weighted loss
        weighted_loss = base_loss * weight.mean()
        
        # Metrics
        metrics = {
            **base_metrics,
            "style_similarity": similarity.mean().item(),
            "adaptive_weight": weight.mean().item(),
            "weighted_identity_loss": weighted_loss.item(),
        }
        
        return weighted_loss, metrics


# ============================================================================
# Integration Example
# ============================================================================

def example_usage():
    """Example showing how to use the identity mapping loss."""
    
    # Assume we have FST module that outputs transformation features
    batch_size = 4
    num_queries = 256
    feature_dim = 1024
    
    # Create loss module
    identity_loss = IdentityMappingLoss(
        matrix_size=num_queries,
        loss_type="frobenius",
        regularization="orthogonal",
        reg_weight=0.01,
    )
    
    # Simulate same-style pairs (in practice, these come from FST module)
    # Pair 1: content A style 1 → content B style 1
    source_feat_1 = torch.randn(batch_size, num_queries, feature_dim)
    target_feat_1 = torch.randn(batch_size, num_queries, feature_dim)
    
    # Compute loss
    loss, metrics = identity_loss(source_feat_1, target_feat_1)
    
    print(f"Identity Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # For multiple pairs
    pooled_loss = PooledIdentityMappingLoss(
        matrix_size=num_queries,
        loss_type="frobenius",
    )
    
    pairs = [
        (torch.randn(2, num_queries, feature_dim), torch.randn(2, num_queries, feature_dim)),
        (torch.randn(2, num_queries, feature_dim), torch.randn(2, num_queries, feature_dim)),
        (torch.randn(2, num_queries, feature_dim), torch.randn(2, num_queries, feature_dim)),
    ]
    
    pool_loss, pool_metrics = pooled_loss(pairs)
    print(f"\nPooled Identity Loss: {pool_loss.item():.4f}")
    print(f"Pool Metrics: {pool_metrics}")


if __name__ == "__main__":
    example_usage()