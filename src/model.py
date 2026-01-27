import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import logging

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.modules.content_encoder import ContentEncoder
from src.modules.style_encoder import StyleEncoder
from src.modules.unet import UNet

logger = logging.getLogger(__name__)


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
    logger.info(f"\n{'='*80}")
    logger.info(f"{model_name} Parameter Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Non-trainable parameters: {total - trainable:,}")
    
    # Log submodule details
    if hasattr(model, 'named_children'):
        logger.info(f"\nSubmodule breakdown:")
        logger.info(f"{'-'*80}")
        logger.info(f"{'Module Name':<40} {'Total Params':>15} {'Trainable':>15}")
        logger.info(f"{'-'*80}")
        
        for name, module in model.named_children():
            mod_total, mod_trainable = count_parameters(module)
            logger.info(f"{name:<40} {mod_total:>15,} {mod_trainable:>15,}")
        
        logger.info(f"{'-'*80}")
    
    logger.info(f"{'='*80}\n")


class FontDiffuserWithFST(nn.Module):
    """Enhanced FontDiffuser with FSTDiff modules."""

    def __init__(
        self,
        original_fontdiffuser: nn.Module,
        feature_channels: list[int] = None,
        num_queries: int = 256,
        query_dim: int = 128,
        num_scales: int = 5,
    ):
        super().__init__()

        # Keep original FontDiffuser components
        self.content_encoder = original_fontdiffuser.content_encoder
        self.diffusion_unet = original_fontdiffuser.unet
        self.style_encoder = original_fontdiffuser.style_encoder

        # Add new FSTDiff modules
        self.mss_encoder = MultiScaleStyleEncoder(
            in_channels=3, base_channels=64, num_scales=num_scales
        )

        # Get actual feature channels from MSSE output if not provided
        if feature_channels is None:
            feature_channels = self.mss_encoder.get_output_channels()

        self.fst_module = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,
            query_dim=query_dim,
            num_scale_features=num_scales,
        )
        
        # Determine U-Net's cross-attention dimension
        cross_attn_dim = self._get_unet_cross_attention_dim()
        
        # Project FST output to U-Net cross-attention dimension
        self.fst_projection = nn.Linear(feature_channels[-1], cross_attn_dim)
        
        # Project original style vector (1024-dim) to cross-attention dimension
        self.original_style_projection = nn.Linear(1024, cross_attn_dim)

    def _get_unet_cross_attention_dim(self) -> int:
        """
        Infer the cross-attention dimension from the U-Net architecture.
        
        Returns:
            int: Cross-attention dimension (typically 1024 for FontDiffuser)
        """
        # Try to get from config if available
        if hasattr(self.diffusion_unet, 'config') and hasattr(self.diffusion_unet.config, 'cross_attention_dim'):
            return self.diffusion_unet.config.cross_attention_dim
        
        # Otherwise, inspect the first cross-attention layer
        for module in self.diffusion_unet.modules():
            if hasattr(module, 'to_k') and isinstance(module.to_k, nn.Linear):
                # The input features of to_k is the cross_attention_dim
                return module.to_k.in_features
        
        # Default fallback to 1024 (standard for FontDiffuser)
        return 1024

    def log_model_info(self) -> None:
        """Log detailed parameter information for the FST model and its components."""
        logger.info("\n" + "="*80)
        logger.info("FontDiffuserWithFST Model Architecture")
        logger.info("="*80)
        
        # Log individual component parameters
        components = [
            ("Content Encoder", self.content_encoder),
            ("Style Encoder", self.style_encoder),
            ("Diffusion U-Net", self.diffusion_unet),
            ("Multi-Scale Style Encoder (MSSE)", self.mss_encoder),
            ("Font Style Transformation (FST)", self.fst_module),
            ("FST Projection", self.fst_projection),
            ("Original Style Projection", self.original_style_projection),
        ]
        
        logger.info("\nComponent Parameters:")
        logger.info("-"*80)
        logger.info(f"{'Component':<45} {'Total':>15} {'Trainable':>15}")
        logger.info("-"*80)
        
        total_all = 0
        trainable_all = 0
        
        for name, component in components:
            total, trainable = count_parameters(component)
            total_all += total
            trainable_all += trainable
            frozen_marker = " [FROZEN]" if trainable == 0 and total > 0 else ""
            logger.info(f"{name:<45} {total:>15,} {trainable:>15,}{frozen_marker}")
        
        logger.info("-"*80)
        logger.info(f"{'TOTAL':<45} {total_all:>15,} {trainable_all:>15,}")
        logger.info(f"{'Non-trainable':<45} {'':<15} {total_all - trainable_all:>15,}")
        logger.info("="*80 + "\n")
        
        # Log FST-specific details
        logger.info("FST Module Details:")
        logger.info("-"*80)
        logger.info(f"  Feature channels: {self.fst_module.feature_channels}")
        logger.info(f"  Num queries: {self.fst_module.num_queries}")
        logger.info(f"  Query dim: {self.fst_module.query_dim}")
        logger.info(f"  Num scales: {self.fst_module.num_scale_features}")
        logger.info(f"  Cross-attention dim: {self._get_unet_cross_attention_dim()}")
        logger.info("="*80 + "\n")

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        content_img: torch.Tensor,
        style_source_img: torch.Tensor,
        style_target_img: torch.Tensor,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with tensor shape tracking.

        Tensor Shape Flow:
        1. content_img: (B, 1, 96, 96) → content_encoder → content_img_feature: (B, C, H, W)
        2. style_target_img: (B, 1, 96, 96) → style_encoder → orig_style_vec: (B, 1024)
        3. style_source/target_img → mss_encoder → [(B, 64, 48, 48), ..., (B, 1024, 6, 6)]
        4. FST module → transformation_features: (B, N_L + H*W, 1024) = (B, 292, 1024)
        5. Projection → fst_condition: (B, 292, cross_attn_dim)
        6. orig_style_vec → projection → (B, 1, cross_attn_dim)
        7. Combined → (B, 293, cross_attn_dim) for U-Net cross-attention

        Args:
            noisy_latents: (B, 4, H, W) - noisy latent representations
            timestep: (B,) or scalar - diffusion timestep
            content_img: (B, 1, 96, 96) - source font character to generate
            style_source_img: (B, 1, 96, 96) - reference char in source font
            style_target_img: (B, 1, 96, 96) - same reference char in target font
            content_encoder_downsample_size: downsampling factor for content encoder
            return_dict: whether to return dict or tuple

        Returns:
            Dictionary containing model outputs and intermediate features
        """
        batch_size = noisy_latents.shape[0]

        # ========== 1. CONTENT ENCODING ==========
        content_img_feature, content_residual_features = self.content_encoder(
            content_img
        )
        content_residual_features.append(content_img_feature)

        style_content_feature, style_content_res_features = self.content_encoder(
            style_target_img
        )
        style_content_res_features.append(style_content_feature)

        # ========== 2. ORIGINAL STYLE ENCODING ==========
        orig_style_feat, orig_style_vec, orig_style_residuals = self.style_encoder(
            style_target_img
        )
        # orig_style_feat: (B, C, H, W) spatial features
        # orig_style_vec: (B, 1024) global style vector

        # ========== 3. MULTI-SCALE STYLE ENCODING (MSSE) ==========
        source_style_features = self.mss_encoder(style_source_img)
        target_style_features = self.mss_encoder(style_target_img)
        # Each: List[(B, 64, 48, 48), (B, 128, 24, 24), (B, 256, 12, 12),
        #            (B, 512, 6, 6), (B, 1024, 6, 6)]

        # ========== 4. FONT STYLE TRANSFORMATION (FST) ==========
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )
        # Shape: (B, N_L + H*W, 1024) where N_L=256, H*W=36
        # Result: (B, 292, 1024)

        # ========== 5. PREPARE U-NET CONDITIONS ==========
        # Project FST features to U-Net cross-attention dimension
        fst_condition = self.fst_projection(transformation_features)
        # Shape: (B, 292, cross_attn_dim) e.g., (B, 292, 1024)

        # Project original style vector for compatibility
        orig_style_projected = self.original_style_projection(orig_style_vec)
        # Shape: (B, cross_attn_dim) → (B, 1024)
        orig_style_projected = orig_style_projected.unsqueeze(1)
        # Shape: (B, 1, cross_attn_dim) e.g., (B, 1, 1024)

        # Combine FST and original style features
        combined_style_condition = torch.cat(
            [fst_condition, orig_style_projected], dim=1
        )
        # Shape: (B, 293, cross_attn_dim) e.g., (B, 293, 1024)

        # ========== 6. PREPARE ENCODER HIDDEN STATES ==========
        # FontDiffuser U-Net expects a list:
        # [style_img_feature, content_residual_features, 
        #  style_hidden_states, style_content_res_features]
        encoder_hidden_states = [
            orig_style_feat,              # (B, C, H, W) spatial style features
            content_residual_features,    # List of content skip connections
            combined_style_condition,     # (B, 293, cross_attn_dim) for cross-attention
            style_content_res_features,   # List of style-content skip connections
        ]

        # ========== 7. DIFFUSION U-NET FORWARD ==========
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

    def get_loss_dict(
        self,
        outputs: dict[str, torch.Tensor],
        target_noise: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute loss components for training."""
        losses = {}

        noise_pred = outputs["noise_pred"]
        if reduction == "mean":
            losses["noise_loss"] = nn.functional.mse_loss(noise_pred, target_noise)
        else:
            losses["noise_loss"] = nn.functional.mse_loss(
                noise_pred, target_noise, reduction="sum"
            )

        offset_out_sum = outputs.get("offset_out_sum", 0)
        if isinstance(offset_out_sum, torch.Tensor):
            losses["offset_loss"] = (
                offset_out_sum.mean() if reduction == "mean" else offset_out_sum.sum()
            )
        else:
            losses["offset_loss"] = torch.tensor(0.0, device=noise_pred.device)

        losses["total_loss"] = losses["noise_loss"] + 0.01 * losses["offset_loss"]

        return losses


class FontDiffuserWithFSTWrapper(nn.Module):
    """
    Wrapper to maintain API compatibility with original FontDiffuserModel
    while using the enhanced FSTDiff architecture.
    """

    def __init__(self, fontdiffuser_with_fst):
        super().__init__()
        self.model = fontdiffuser_with_fst

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size=4,
    ):
        """
        API-compatible forward pass.

        Note: This assumes style_images contains both source and target.
        You may need to modify based on your data pipeline.
        """
        outputs = self.model(
            noisy_latents=x_t,
            timestep=timesteps,
            content_img=content_images,
            style_source_img=style_images,
            style_target_img=style_images,
            content_encoder_downsample_size=content_encoder_downsample_size,
            return_dict=True,
        )

        return outputs["noise_pred"], outputs["offset_out_sum"]


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

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
        style_img_feature, _, _ = self.config.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get the content feature
        content_img_feature, content_residual_features = self.config.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.config.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

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

        style_img_feature, _, style_residual_features = self.config.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content feature
        content_img_feture, content_residual_features = self.config.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.config.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred