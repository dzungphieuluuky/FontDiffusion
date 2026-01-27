"""
Consistency loss for Font Style Transformation (FST) module.

Ensures that style transformation features are consistent across different
content characters when the source and target styles remain the same.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging
import torchvision

logger = logging.getLogger(__name__)


class FSTConsistencyLoss(nn.Module):
    """
    Consistency loss for Font Style Transformation.

    Compares style transformation features across multiple content characters
    with the same source→target style pair to enforce style consistency.

    The key idea: L_{x→y}^r should be similar regardless of which reference
    character 'r' is used, since the style transformation x→y is content-independent.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
        normalize: bool = True,
    ):
        """
        Args:
            loss_type: Type of loss to use ("mse", "cosine", "l1")
            reduction: How to reduce the loss ("mean", "sum", "none")
            normalize: Whether to normalize features before comparison
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.normalize = normalize

        if self.loss_type not in ["mse", "cosine", "l1"]:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(
        self,
        transformation_features_list: list[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute consistency loss across multiple transformation features.

        Args:
            transformation_features_list: List of transformation features from FST module.
                Each tensor has shape (B, N_L + H*W, C) where:
                - B: batch size
                - N_L: number of learnable queries (e.g., 256)
                - H*W: spatial tokens from last scale (e.g., 36)
                - C: feature dimension (e.g., 1024)
                Length of list should be >= 2 for meaningful comparison.
            mask: Optional mask to ignore certain tokens, shape (B, N_L + H*W)

        Returns:
            Scalar consistency loss
        """
        if len(transformation_features_list) < 2:
            # Need at least 2 pairs to compute consistency
            return torch.tensor(0.0, device=transformation_features_list[0].device)

        # Normalize features if requested
        if self.normalize:
            transformation_features_list = [
                self._normalize_features(feat) for feat in transformation_features_list
            ]

        # Compute pairwise consistency loss
        if self.loss_type == "cosine":
            loss = self._cosine_consistency(transformation_features_list, mask)
        elif self.loss_type == "l1":
            loss = self._l1_consistency(transformation_features_list, mask)
        else:  # mse
            loss = self._mse_consistency(transformation_features_list, mask)

        return loss

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features along the channel dimension."""
        # features: (B, N, C)
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def _apply_mask(
        self, features: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply mask to features if provided."""
        if mask is not None:
            # mask: (B, N), features: (B, N, C)
            features = features * mask.unsqueeze(-1)
        return features

    def _mse_consistency(
        self,
        features_list: list[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute MSE-based consistency loss."""
        total_loss = 0.0
        num_pairs = 0

        # Compare all pairs
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat_i = self._apply_mask(features_list[i], mask)
                feat_j = self._apply_mask(features_list[j], mask)

                # MSE loss between pairs
                pair_loss = torch.nn.functional.mse_loss(
                    feat_i, feat_j, reduction=self.reduction
                )
                total_loss = total_loss + pair_loss
                num_pairs += 1

        # Average over all pairs
        return total_loss / num_pairs if num_pairs > 0 else total_loss

    def _l1_consistency(
        self,
        features_list: list[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute L1-based consistency loss."""
        total_loss = 0.0
        num_pairs = 0

        # Compare all pairs
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat_i = self._apply_mask(features_list[i], mask)
                feat_j = self._apply_mask(features_list[j], mask)

                # L1 loss between pairs
                pair_loss = torch.nn.functional.l1_loss(
                    feat_i, feat_j, reduction=self.reduction
                )
                total_loss = total_loss + pair_loss
                num_pairs += 1

        # Average over all pairs
        return total_loss / num_pairs if num_pairs > 0 else total_loss

    def _cosine_consistency(
        self,
        features_list: list[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cosine similarity-based consistency loss."""
        total_similarity = 0.0
        num_pairs = 0

        # Compare all pairs
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat_i = self._apply_mask(features_list[i], mask)
                feat_j = self._apply_mask(features_list[j], mask)

                # Cosine similarity (higher is better, so we minimize 1 - similarity)
                similarity = torch.nn.functional.cosine_similarity(
                    feat_i, feat_j, dim=-1
                )

                if mask is not None:
                    # Average only over unmasked tokens
                    similarity = (similarity * mask).sum() / (mask.sum() + 1e-8)
                else:
                    similarity = similarity.mean()

                # Convert to loss (1 - similarity, so 0 when identical)
                total_similarity = total_similarity + (1.0 - similarity)
                num_pairs += 1

        # Average over all pairs
        return total_similarity / num_pairs if num_pairs > 0 else total_similarity


class FSTStyleConsistencyLoss(nn.Module):
    """
    Style consistency loss that focuses on the learnable query portion
    of the transformation features (L_{x→y} part).

    This variant only compares the first N_L tokens (learnable queries),
    ignoring the spatial tokens from the last scale.
    """

    def __init__(
        self,
        num_queries: int = 256,
        loss_type: str = "mse",
        reduction: str = "mean",
        normalize: bool = True,
    ):
        """
        Args:
            num_queries: Number of learnable queries (N_L)
            loss_type: Type of loss ("mse", "cosine", "l1")
            reduction: How to reduce the loss
            normalize: Whether to normalize features
        """
        super().__init__()
        self.num_queries = num_queries
        self.base_loss = FSTConsistencyLoss(
            loss_type=loss_type,
            reduction=reduction,
            normalize=normalize,
        )

    def forward(
        self,
        transformation_features_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute consistency loss on learnable query portion only.

        Args:
            transformation_features_list: List of (B, N_L + H*W, C) tensors

        Returns:
            Scalar consistency loss
        """
        # Extract only the learnable query portion (first N_L tokens)
        query_features_list = [
            feat[:, : self.num_queries, :] for feat in transformation_features_list
        ]

        return self.base_loss(query_features_list)


class CombinedFSTLoss(nn.Module):
    """
    Combined loss for FST training that includes:
    1. Reconstruction loss (noise prediction)
    2. Consistency loss (style transformation)
    3. Optional perceptual loss
    """

    def __init__(
        self,
        consistency_weight: float = 0.1,
        consistency_loss_type: str = "mse",
        use_query_only: bool = True,
        num_queries: int = 256,
    ):
        """
        Args:
            consistency_weight: Weight for consistency loss
            consistency_loss_type: Type of consistency loss
            use_query_only: Whether to use query-only consistency
            num_queries: Number of learnable queries
        """
        super().__init__()
        self.consistency_weight = consistency_weight

        if use_query_only:
            self.consistency_loss = FSTStyleConsistencyLoss(
                num_queries=num_queries,
                loss_type=consistency_loss_type,
            )
        else:
            self.consistency_loss = FSTConsistencyLoss(
                loss_type=consistency_loss_type,
            )

    def forward(
        self,
        noise_pred: torch.Tensor,
        target_noise: torch.Tensor,
        transformation_features_list: list[torch.Tensor],
        offset_out_sum: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            noise_pred: Predicted noise from U-Net
            target_noise: Ground truth noise
            transformation_features_list: List of transformation features
            offset_out_sum: Optional offset loss term

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Main reconstruction loss
        losses["noise_loss"] = torch.nn.functional.mse_loss(noise_pred, target_noise)

        # Consistency loss
        if len(transformation_features_list) >= 2:
            losses["consistency_loss"] = self.consistency_loss(
                transformation_features_list
            )
        else:
            losses["consistency_loss"] = torch.tensor(0.0, device=noise_pred.device)

        # Offset loss (if applicable)
        if offset_out_sum is not None and isinstance(offset_out_sum, torch.Tensor):
            losses["offset_loss"] = offset_out_sum.mean()
        else:
            losses["offset_loss"] = torch.tensor(0.0, device=noise_pred.device)

        # Total loss
        losses["total_loss"] = (
            losses["noise_loss"]
            + self.consistency_weight * losses["consistency_loss"]
            + 0.01 * losses["offset_loss"]
        )

        return losses


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f"enc_{i + 1:d}").parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f"enc_{i + 1:d}")
            results.append(func(results[-1]))
        return results[1:]


class ContentPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss
