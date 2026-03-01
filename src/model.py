import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import logging

from src.tools.utilities import setup_logger

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.modules.content_encoder import ContentEncoder
from src.modules.style_encoder import StyleEncoder
from src.modules.scr import SCR
from src.modules.unet import UNet
from src.modules.skeleton_distance_transform import (
    SkeletonDistanceTransform,
    DualChannelContentEncoder,
)
from src.modules.frequency_decomposition import FrequencyDecomposition


logger = setup_logger(__name__)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def log_model_parameters(model: nn.Module, model_name: str = "Model") -> None:
    """
    Log parameter counts for a model and its submodules.

    Args:
        model: PyTorch model to analyze
        model_name: Name for logging
    """
    total, trainable = count_parameters(model)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{model_name} Parameter Summary")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Non-trainable parameters: {total - trainable:,}")

    # Log submodule details
    if hasattr(model, "named_children"):
        logger.info(f"\nSubmodule breakdown:")
        logger.info(f"{'-' * 80}")
        logger.info(f"{'Module Name':<40} {'Total Params':>15} {'Trainable':>15}")
        logger.info(f"{'-' * 80}")

        for name, module in model.named_children():
            mod_total, mod_trainable = count_parameters(module)
            logger.info(f"{name:<40} {mod_total:>15,} {mod_trainable:>15,}")

        logger.info(f"{'-' * 80}")

    logger.info(f"{'=' * 80}\n")


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder style encoder and unet."""

    @register_to_config
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet: UNet = unet
        self.style_encoder: StyleEncoder = style_encoder
        self.content_encoder: ContentEncoder = content_encoder

    def log_model_info(self) -> None:
        """Log parameter information for the base FontDiffuser model."""
        log_model_parameters(self, "FontDiffuserModel")

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        style_img_feature, _, _ = self.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)

        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer."""

    @register_to_config
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet: UNet = unet
        self.style_encoder: StyleEncoder = style_encoder
        self.content_encoder: ContentEncoder = content_encoder

    def log_model_info(self) -> None:
        """Log parameter information for the DPM FontDiffuser model."""
        log_model_parameters(self, "FontDiffuserModelDPM")

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        content_img_feture, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feture)

        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred


class FontDiffuserWithFST(ModelMixin, ConfigMixin):
    """
    FontDiffuser with FST enhancement, optional skeleton-distance transform,
    and optional frequency decomposition.
    """

    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder | DualChannelContentEncoder,
        mss_encoder: MultiScaleStyleEncoder,
        fst_module: FontStyleTransformationModule,
        fst_projection: nn.Linear,
        original_style_projection: nn.Linear,
        skeleton_transform: Optional[SkeletonDistanceTransform] = None,
        frequency_decomp: Optional[FrequencyDecomposition] = None,
    ):
        """
        Initialize FontDiffuserWithFST with optional frequency decomposition.

        Args:
            unet: U-Net for diffusion
            style_encoder: Style encoder
            content_encoder: Content encoder (may be wrapped with DualChannelContentEncoder)
            mss_encoder: Multi-scale style encoder
            fst_module: Font style transformation module
            fst_projection: Projection for FST features
            original_style_projection: Projection for original style features
            skeleton_transform: Optional skeleton-distance transform
            frequency_decomp: Optional frequency decomposition module
        """
        super().__init__()

        # Base modules
        self.content_encoder: ContentEncoder | DualChannelContentEncoder = (
            content_encoder
        )
        self.diffusion_unet: UNet = unet
        self.style_encoder: StyleEncoder = style_encoder

        # Additional modules for FST
        self.mss_encoder: MultiScaleStyleEncoder = mss_encoder
        self.fst_module: FontStyleTransformationModule = fst_module
        self.fst_projection: nn.Linear = fst_projection
        self.original_style_projection: nn.Linear = original_style_projection
        self.fst_num_queries: int = fst_module.num_queries

        # Optional transforms
        self.skeleton_transform: SkeletonDistanceTransform = skeleton_transform
        self.use_skeleton_content: bool = skeleton_transform is not None

        self.frequency_decomp: FrequencyDecomposition = frequency_decomp
        self.use_frequency_decomp: bool = frequency_decomp is not None

    def log_model_info(self):
        """Log model parameter information including skeleton and frequency decomposition."""
        logger.info("=" * 80)
        logger.info("FontDiffuserWithFST Model Information")
        logger.info("=" * 80)

        # Log skeleton configuration
        if self.use_skeleton_content:
            logger.info("✓ Skeleton-Distance Transform: ✅")
            logger.info(f"  Fusion method: {self.content_encoder.fusion_method}")
            if hasattr(self.content_encoder, "fusion_conv"):
                fusion_params = sum(
                    p.numel() for p in self.content_encoder.fusion_conv.parameters()
                )
                logger.info(f"  Fusion parameters: {fusion_params:,}")
        else:
            logger.info("Skeleton-Distance Transform: ❌")

        # Log frequency decomposition configuration
        if self.use_frequency_decomp:
            logger.info("✓ Frequency Decomposition: ✅")
            logger.info(f"  Low cutoff: {self.frequency_decomp.low_cutoff}")
            logger.info(f"  Mid cutoff: {self.frequency_decomp.mid_cutoff}")
            logger.info(
                f"  Filter type: {self.frequency_decomp.filter_builder.filter_type}"
            )
        else:
            logger.info("Frequency Decomposition: ❌")

        # Log component parameters
        components = {
            "U-Net": self.diffusion_unet,
            "Style Encoder": self.style_encoder,
            "Content Encoder (wrapped)"
            if self.use_skeleton_content
            else "Content Encoder": self.content_encoder
            if not self.use_skeleton_content
            else self.content_encoder.original_encoder,
            "MSS Encoder": self.mss_encoder,
            "FST Module": self.fst_module,
            "FST Projection": self.fst_projection,
            "Original Style Projection": self.original_style_projection,
        }

        total_params = 0
        trainable_params = 0

        for name, module in components.items():
            module_total, module_trainable = count_parameters(module)
            total_params += module_total
            trainable_params += module_trainable
            logger.info(
                f"{name:30s}: {module_total:12,} total | "
                f"{module_trainable:12,} trainable"
            )

        if self.use_skeleton_content and hasattr(self.content_encoder, "fusion_conv"):
            fusion_total, fusion_trainable = count_parameters(
                self.content_encoder.fusion_conv
            )
            total_params += fusion_total
            trainable_params += fusion_trainable
            logger.info(
                f"{'Skeleton Fusion Layer':30s}: {fusion_total:12,} total | "
                f"{fusion_trainable:12,} trainable"
            )

        logger.info("=" * 80)
        logger.info(
            f"{'TOTAL':30s}: {total_params:12,} total | {trainable_params:12,} trainable"
        )
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 80)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        content_images: torch.Tensor,
        style_source_images: torch.Tensor,
        style_target_images: torch.Tensor,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with FST, optional skeleton transform, and optional frequency decomposition.

        Args:
            noisy_latents: (B, 4, H, W) - noisy latent representations
            timestep: (B,) or scalar - diffusion timestep
            content_images: (B, 1, 96, 96) - batch of source font characters
            style_source_images: (B, 1, 96, 96) - batch of reference chars in source font
            style_target_images: (B, 1, 96, 96) - batch of reference chars in target font
            content_encoder_downsample_size: downsampling factor
            return_dict: whether to return dict or tuple

        Returns:
            Dictionary containing model outputs
        """
        batch_size = noisy_latents.shape[0]

        # Apply frequency decomposition if enabled
        if self.use_frequency_decomp:
            # Decompose content image into frequency bands
            content_bands = self.frequency_decomp(content_images)

            # Use low-frequency band for content (contains structure only)
            content_input = content_bands["low_freq"]

            logger.debug(
                f"Applied frequency decomposition: "
                f"original shape {content_images.shape} → "
                f"low-freq shape {content_input.shape}"
            )

            # Decompose style images as well
            style_source_bands = self.frequency_decomp(style_source_images)
            style_target_bands = self.frequency_decomp(style_target_images)

            # Use high-frequency bands for style (contains details/textures)
            style_source_input = style_source_bands["high_freq"]
            style_target_input = style_target_bands["high_freq"]
        else:
            content_input = content_images
            style_source_input = style_source_images
            style_target_input = style_target_images

        # Apply skeleton-distance transform if enabled
        if self.use_skeleton_content:
            content_input = self.skeleton_transform(content_input)
            logger.debug(
                f"Applied skeleton transform: "
                f"input shape → output shape {content_input.shape}"
            )

        # 1. Content encoding
        content_img_feature, content_residual_features = self.content_encoder(
            content_input
        )
        content_residual_features.append(content_img_feature)

        style_content_feature, style_content_res_features = self.content_encoder(
            style_target_input
        )
        style_content_res_features.append(style_content_feature)

        # 2. Original style encoding
        orig_style_feat, orig_style_vec, orig_style_residuals = self.style_encoder(
            style_target_input
        )

        # 3. Multi-scale style encoding
        source_style_features = self.mss_encoder(style_source_input)
        target_style_features = self.mss_encoder(style_target_input)

        # 4. Font style transformation
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )

        # 5. Prepare U-Net conditions
        fst_condition = self.fst_projection(transformation_features)
        orig_style_projected = self.original_style_projection(orig_style_vec)
        orig_style_projected = orig_style_projected.unsqueeze(1)

        # Combine FST and original style features
        combined_style_condition = torch.cat(
            [fst_condition, orig_style_projected], dim=1
        )

        # 6. Prepare encoder hidden states
        encoder_hidden_states = [
            orig_style_feat,
            content_residual_features,
            combined_style_condition,
            style_content_res_features,
        ]

        # 7. Diffusion U-Net forward
        noise_pred, offset_out_sum = self.diffusion_unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )

        if return_dict:
            return {
                "noise_pred": noise_pred,
                "offset_out_sum": offset_out_sum,
                "content_features": content_img_feature,
                "transformation_features": transformation_features,
                "fst_condition": fst_condition,
                "source_style_features": source_style_features,
                "target_style_features": target_style_features,
                "orig_style_feat": orig_style_feat,
                "orig_style_vec": orig_style_vec,
            }
        else:
            return noise_pred, offset_out_sum

    def compute_transformation_matrix(
        self,
        style_source_images: torch.Tensor,
        style_target_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the font style transformation features for a pair of images.

        This extracts the transformation learned by the FST module without
        going through the full diffusion process.

        Args:
            style_source_images: (B, 1, 96, 96) - source style reference
            style_target_images: (B, 1, 96, 96) - target style reference

        Returns:
            transformation_features: (B, N, D) - transformation matrix/features
        """
        # Extract multi-scale features from both images
        source_style_features = self.mss_encoder(style_source_images)
        target_style_features = self.mss_encoder(style_target_images)

        # Apply FST module to get transformation
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )

        return transformation_features

    def compute_consistency_loss(
        self,
        consistency_source_images: torch.Tensor,
        consistency_target_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss across multiple content pairs.

        The FST transformation should be similar for all pairs since they
        share the same source→target style transformation, regardless of content.

        Uses John Schulman's k3 estimator for KL divergence:
        KL(P||Q) ≈ E[(exp(log P(x) - log Q(x)) - 1) - (log P(x) - log Q(x))]
        Reference: http://joschu.net/blog/kl-approx.html

        Args:
            consistency_source_images: (B, k, 1, 96, 96) - k source images per batch
            consistency_target_images: (B, k, 1, 96, 96) - k target images per batch

        Returns:
            consistency_loss: scalar tensor (variance + KL divergence loss)
        """
        batch_size, num_pairs, C, H, W = consistency_source_images.shape

        # Flatten and extract features
        source_flat = consistency_source_images.view(-1, C, H, W)
        target_flat = consistency_target_images.view(-1, C, H, W)

        source_features = self.mss_encoder(source_flat)
        target_features = self.mss_encoder(target_flat)

        # Get transformation features
        transformation_features = self.fst_module(source_features, target_features)

        # Reshape: (B*k, N, D) → (B, k, N, D)
        T = transformation_features.view(
            batch_size,
            num_pairs,
            transformation_features.shape[1],
            transformation_features.shape[2],
        )

        # Compute statistics across pairs (dim=1)
        mean_T = T.mean(
            dim=1, keepdim=True
        )  # (B, 1, N, D) - target uniform distribution
        std_T = T.std(dim=1, keepdim=True)  # (B, 1, N, D)

        # Coefficient of Variation (scale-invariant variance measure)
        eps = 1e-6
        cv = std_T / (mean_T.abs() + eps)
        variance_loss = cv.mean()

        # KL divergence using Schulman's k3 estimator
        # P = uniform distribution (target), Q = empirical distribution (actual)
        # We want KL(P||Q) to be small → transformations are uniformly distributed

        # Compute log probability ratio: log P(x) - log Q(x)
        # For standardized Gaussian assumption:
        # log P(x) = -0.5 * x^2 (standard normal)
        # log Q(x) = -0.5 * ((x - μ) / σ)^2 - log(σ)

        # Standardize features: z = (T - mean) / std
        z = (T - mean_T) / (std_T + eps)  # (B, k, N, D)

        # Log probability under P (standard normal, mean=0, std=1)
        log_p = -0.5 * (z**2)

        # Log probability under Q (empirical distribution with learned mean/std)
        # Since z is already standardized, log Q(x) includes the normalization term
        log_q = -0.5 * (z**2) - torch.log(std_T + eps)

        # Log ratio
        logr = log_p - log_q  # (B, k, N, D)

        # Schulman's k3 estimator: E[(exp(logr) - 1) - logr]
        # This is an unbiased estimator with lower variance than k1 or k2
        kl_loss = ((logr.exp() - 1.0) - logr).mean()

        # Clip KL loss for numerical stability (optional but recommended)
        kl_loss = torch.clamp(kl_loss, min=0.0, max=10.0)

        # Combined loss: variance + KL divergence
        # Lower weight on KL since it's already normalized
        total_loss = variance_loss + 0.05 * kl_loss

        return total_loss

    def compute_identity_loss(
        self,
        identity_pair_sources: torch.Tensor,
        identity_pair_targets: torch.Tensor,
        num_queries: int = 220,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute identity mapping loss for same-style pairs.

        For same-style image pairs, FST should produce near-identity transformation.

        Args:
            identity_pair_sources: (B, 1, H, W) - Source style images
            identity_pair_targets: (B, 1, H, W) - Target style images (same style as source)
            num_queries: Number of learnable queries to use

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with diagnostics
        """
        # Extract multi-scale features from both
        source_style_features = self.mss_encoder(identity_pair_sources)
        target_style_features = self.mss_encoder(identity_pair_targets)

        # Apply FST to get transformation
        transformation_features = self.fst_module(
            source_style_features,
            target_style_features,
        )  # (B, N_L + H*W, D)

        # Extract learnable query portion only BEFORE computing correlation
        query_features = transformation_features[:, :num_queries, :]  # (B, N_L, D)

        B, N, D = query_features.shape

        # For identity mapping, source and target transformation should be same
        # Compute correlation matrix: should be identity
        source_norm = F.normalize(query_features, p=2, dim=-1)
        target_norm = F.normalize(query_features, p=2, dim=-1)

        # Self-similarity matrix (should be identity when same-style)
        # (B, D, N) @ (B, N, D) = (B, D, D)
        correlation = torch.bmm(
            source_norm.transpose(1, 2),  # (B, D, N)
            target_norm,  # (B, N, D)
        )  # (B, D, D)

        # Distance from identity matrix
        identity_matrix = (
            torch.eye(D, device=correlation.device).unsqueeze(0).expand(B, -1, -1)
        )  # (B, D, D)
        diff = correlation - identity_matrix

        # Frobenius norm: identity loss
        identity_loss = torch.norm(diff.reshape(B, -1), p="fro", dim=1).mean()

        # Orthogonality regularization: C^T @ C should also be identity
        CTC = torch.bmm(correlation.transpose(1, 2), correlation)
        ortho_loss = torch.norm(
            (CTC - identity_matrix).reshape(B, -1), p="fro", dim=1
        ).mean()

        total_loss = identity_loss + 0.01 * ortho_loss

        # Compute metrics
        with torch.inference_mode():
            diagonal = torch.diagonal(correlation, dim1=1, dim2=2)  # (B, D)
            metrics = {
                "identity_loss": identity_loss.item(),
                "ortho_loss": ortho_loss.item(),
                "diagonal_mean": diagonal.mean().item(),
                "diagonal_std": diagonal.std().item(),
            }

        return total_loss, metrics

    def compute_identity_loss_v2(
        self,
        identity_pair_sources: torch.Tensor,
        identity_pair_targets: torch.Tensor,
        identity_loss_module: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute identity mapping loss using the dedicated IdentityMappingLoss module.

        Args:
            identity_pair_sources: (B, 1, H, W) - Source style images (same style)
            identity_pair_targets: (B, 1, H, W) - Target style images (same style)
            identity_loss_module: IdentityMappingLoss instance

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with diagnostics from IdentityMappingLoss
        """
        # Extract multi-scale features from both
        source_style_features = self.mss_encoder(identity_pair_sources)
        target_style_features = self.mss_encoder(identity_pair_targets)

        # Apply FST to get transformation features
        transformation_features = self.fst_module(
            source_style_features,
            target_style_features,
        )  # (B, N_L + H*W, D)

        # Extract learnable query portion only
        query_features = transformation_features[:, : self.fst_num_queries, :]

        # Use IdentityMappingLoss module
        loss, metrics = identity_loss_module(query_features, query_features)

        return loss, metrics


class FontDiffuserModelDPMWithFST(ModelMixin, ConfigMixin):
    """
    DPM Forward function for FontDiffuser with FST enhancement, optional skeleton-distance transform,
    and optional frequency decomposition.
    """

    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder | DualChannelContentEncoder,
        mss_encoder: MultiScaleStyleEncoder,
        fst_module: FontStyleTransformationModule,
        fst_projection: nn.Linear,
        original_style_projection: nn.Linear,
        skeleton_transform: Optional[SkeletonDistanceTransform] = None,
        frequency_decomp: Optional[FrequencyDecomposition] = None,
    ):
        """
        Initialize FontDiffuserModelDPMWithFST.

        Args:
            unet: Pre-built U-Net module
            style_encoder: Pre-built style encoder
            content_encoder: Pre-built content encoder
            mss_encoder: Pre-built Multi-Scale Style Encoder
            fst_module: Pre-built Font Style Transformation module
            fst_projection: Pre-built projection layer (FST → cross-attn)
            original_style_projection: Pre-built projection layer (style vec → cross-attn)
            skeleton_transform: Optional skeleton-distance transform
            frequency_decomp: Optional frequency decomposition module
        """
        super().__init__()

        # Base modules
        self.content_encoder = content_encoder
        self.unet: UNet = unet
        self.style_encoder: StyleEncoder = style_encoder

        # Font style transformation modules
        self.mss_encoder: MultiScaleStyleEncoder = mss_encoder
        self.fst_module: FontStyleTransformationModule = fst_module
        self.fst_projection: nn.Linear = fst_projection
        self.original_style_projection: nn.Linear = original_style_projection

        # Skeleton transform
        self.use_skeleton_content: bool = skeleton_transform is not None
        self.skeleton_transform: SkeletonDistanceTransform = skeleton_transform

        # Frequency decomposition
        self.use_frequency_decomp: bool = frequency_decomp is not None
        self.frequency_decomp: FrequencyDecomposition = frequency_decomp

    def log_model_info(self) -> None:
        """Log parameter information for the DPM FST FontDiffuser model."""
        logger.info("\n" + "=" * 80)
        logger.info("FontDiffuserModelDPMWithFST Model Architecture")
        logger.info("=" * 80)

        # Log skeleton configuration
        if self.use_skeleton_content:
            logger.info("✓ Skeleton-Distance Transform: ✅")
            logger.info(f"  Fusion method: {self.content_encoder.fusion_method}")
            if hasattr(self.content_encoder, "fusion_conv"):
                fusion_params = sum(
                    p.numel() for p in self.content_encoder.fusion_conv.parameters()
                )
                logger.info(f"  Fusion parameters: {fusion_params:,}")
        else:
            logger.info("Skeleton-Distance Transform: ❌")

        # Log frequency decomposition configuration
        if self.use_frequency_decomp:
            logger.info("✓ Frequency Decomposition: ✅")
            logger.info(f"  Low cutoff: {self.frequency_decomp.low_cutoff}")
            logger.info(f"  Mid cutoff: {self.frequency_decomp.mid_cutoff}")
            logger.info(
                f"  Filter type: {self.frequency_decomp.filter_builder.filter_type}"
            )
        else:
            logger.info("Frequency Decomposition: ❌")

        # Get actual content encoder for parameter counting
        content_encoder_for_counting = (
            self.content_encoder.original_encoder
            if self.use_skeleton_content
            else self.content_encoder
        )

        components = [
            ("Content Encoder", content_encoder_for_counting),
            ("Style Encoder", self.style_encoder),
            ("Diffusion U-Net", self.unet),
            ("Multi-Scale Style Encoder (MSSE)", self.mss_encoder),
            ("Font Style Transformation (FST)", self.fst_module),
            ("FST Projection", self.fst_projection),
            ("Original Style Projection", self.original_style_projection),
        ]

        logger.info("\nComponent Parameters:")
        logger.info("-" * 80)
        logger.info(f"{'Component':<45} {'Total':>15} {'Trainable':>15}")
        logger.info("-" * 80)

        total_all = 0
        trainable_all = 0

        for name, component in components:
            total, trainable = count_parameters(component)
            total_all += total
            trainable_all += trainable
            frozen_marker = " [FROZEN]" if trainable == 0 and total > 0 else ""
            logger.info(f"{name:<45} {total:>15,} {trainable:>15,}{frozen_marker}")

        # Add skeleton fusion layer if enabled
        if self.use_skeleton_content and hasattr(self.content_encoder, "fusion_conv"):
            fusion_total, fusion_trainable = count_parameters(
                self.content_encoder.fusion_conv
            )
            total_all += fusion_total
            trainable_all += fusion_trainable
            logger.info(
                f"{'Skeleton Fusion Layer':<45} {fusion_total:>15,} {fusion_trainable:>15,}"
            )

        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<45} {total_all:>15,} {trainable_all:>15,}")
        logger.info(f"{'Non-trainable':<45} {'':<15} {total_all - trainable_all:>15,}")
        logger.info("=" * 80 + "\n")

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor],
        content_encoder_downsample_size: int,
        version: str = None,
    ) -> torch.Tensor:
        """
        DPM-compatible forward pass with FST, skeleton transform, and frequency decomposition.

        Args:
            x_t: (B, 4, H, W) - noisy latent representations
            timesteps: (B,) or scalar - diffusion timestep
            cond: tuple of (content_images, style_images)
            content_encoder_downsample_size: downsampling factor
            version: version string (unused, for compatibility)

        Returns:
            torch.Tensor: (B, 4, H, W) - predicted noise
        """
        content_images = cond[0]
        style_images = cond[1]

        # Apply frequency decomposition if enabled
        if self.use_frequency_decomp:
            # Decompose content image
            content_bands = self.frequency_decomp(content_images)
            content_input = content_bands["low_freq"]

            # Decompose style image
            style_bands = self.frequency_decomp(style_images)
            style_input = style_bands["high_freq"]

            logger.debug(
                f"Applied frequency decomposition: "
                f"content {content_images.shape} → low-freq {content_input.shape}, "
                f"style {style_images.shape} → high-freq {style_input.shape}"
            )
        else:
            content_input = content_images
            style_input = style_images

        # Apply skeleton-distance transform if enabled
        if self.use_skeleton_content:
            content_input = self.skeleton_transform(content_input)
            logger.debug(
                f"Applied skeleton transform: "
                f"input shape → output shape {content_input.shape}"
            )

        # 1. Original style encoding
        style_img_feature, style_vec, style_residual_features = self.style_encoder(
            style_input
        )

        # 2. Multi-scale style encoding
        target_style_features = self.mss_encoder(style_input)
        source_style_features = target_style_features  # Single-style mode

        # 3. Font style transformation
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )

        # 4. Prepare enhanced style condition
        fst_condition = self.fst_projection(transformation_features)
        orig_style_projected = self.original_style_projection(style_vec)
        orig_style_projected = orig_style_projected.unsqueeze(1)

        combined_style_condition = torch.cat(
            [fst_condition, orig_style_projected], dim=1
        )

        # 5. Content encoding
        content_img_feature, content_residual_features = self.content_encoder(
            content_input
        )
        content_residual_features.append(content_img_feature)

        style_content_feature, style_content_res_features = self.content_encoder(
            style_input
        )
        style_content_res_features.append(style_content_feature)

        # 6. Prepare encoder hidden states
        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            combined_style_condition,
            style_content_res_features,
        ]

        # 7. Diffusion U-Net forward
        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred
