import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

import torch
import torch.nn as nn
from typing import Dict

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.modules.content_encoder import ContentEncoder
from src.modules.style_encoder import StyleEncoder
from src.modules.unet import UNet


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

        # Determine feature channels from MSSE output shapes if not provided
        if feature_channels is None:
            feature_channels = [64, 128, 256, 512, 1024]

        self.fst_module = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,  # ← Use parameter instead of hardcoded
            query_dim=query_dim,
            num_scale_features=num_scales,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2,
        )

        # Projection layers to inject FST features into U-Net
        # FST output: (B, N_L + 36, 1024) where N_L=256, spatial=6x6=36
        cross_attn_dim = getattr(
            self.diffusion_unet.config, "cross_attention_dim", 1280
        )

        self.fst_projection = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, cross_attn_dim),
        )

        # Optional: Project original style features to same dimension for concatenation
        self.original_style_projection = nn.Sequential(
            nn.Linear(1024, cross_attn_dim), nn.LayerNorm(cross_attn_dim)
        )

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
        # Extract content features from the character to generate
        content_img_feature, content_residual_features = self.content_encoder(
            content_img
        )
        content_residual_features.append(content_img_feature)

        # Extract content features from style reference image
        style_content_feature, style_content_res_features = self.content_encoder(
            style_target_img
        )
        style_content_res_features.append(style_content_feature)

        # ========== 2. ORIGINAL STYLE ENCODING (for compatibility) ==========
        # Keep original style encoder output for SCR loss and baseline features
        orig_style_feat, orig_style_vec, orig_style_residuals = self.style_encoder(
            style_target_img
        )
        # Reshape for cross-attention: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = orig_style_feat.shape
        orig_style_hidden = orig_style_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ========== 3. MULTI-SCALE STYLE ENCODING (MSSE) ==========
        source_style_features = self.mss_encoder(style_source_img)
        target_style_features = self.mss_encoder(style_target_img)
        # Each: List of 5 tensors with shapes:
        # [(B, 64, 48, 48), (B, 128, 24, 24), (B, 256, 12, 12),
        #  (B, 512, 6, 6), (B, 1024, 6, 6)]

        # ========== 4. FONT STYLE TRANSFORMATION (FST) ==========
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )
        # Shape: (B, N_L + H*W, 1024) = (B, 256 + 36, 1024) = (B, 292, 1024)

        # ========== 5. PREPARE U-NET CONDITIONS ==========
        # Project FST features to U-Net cross-attention dimension
        fst_condition = self.fst_projection(transformation_features)
        # Shape: (B, 292, cross_attn_dim)

        # Project original style features for compatibility
        orig_style_projected = self.original_style_projection(orig_style_vec)
        # Shape: (B, cross_attn_dim)
        orig_style_projected = orig_style_projected.unsqueeze(
            1
        )  # (B, 1, cross_attn_dim)

        # Combine FST and original style features
        # You can choose to concatenate or use separately
        combined_style_condition = torch.cat(
            [fst_condition, orig_style_projected], dim=1
        )
        # Shape: (B, 293, cross_attn_dim)

        # ========== 6. PREPARE ENCODER HIDDEN STATES (FontDiffuser format) ==========
        # FontDiffuser expects: [style_img_feature, content_residual_features,
        #                        style_hidden_states, style_content_res_features]
        encoder_hidden_states = [
            orig_style_feat,  # For U-Net feature injection
            content_residual_features,  # Content skip connections
            combined_style_condition,  # Enhanced style condition with FST
            style_content_res_features,  # Style content skip connections
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
        """
        Compute loss components for training.

        Args:
            outputs: Dictionary from forward pass
            target_noise: Ground truth noise to predict
            reduction: 'mean' or 'sum'

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Main denoising loss (MSE between predicted and target noise)
        noise_pred = outputs["noise_pred"]
        if reduction == "mean":
            losses["noise_loss"] = nn.functional.mse_loss(noise_pred, target_noise)
        else:
            losses["noise_loss"] = nn.functional.mse_loss(
                noise_pred, target_noise, reduction="sum"
            )

        # Offset loss (if applicable)
        offset_out_sum = outputs.get("offset_out_sum", 0)
        if isinstance(offset_out_sum, torch.Tensor):
            losses["offset_loss"] = (
                offset_out_sum.mean() if reduction == "mean" else offset_out_sum.sum()
            )
        else:
            losses["offset_loss"] = torch.tensor(0.0, device=noise_pred.device)

        # Total loss
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
        # Split style_images if they contain both source and target
        # Or pass the same image twice if you only have target style
        outputs = self.model(
            noisy_latents=x_t,
            timestep=timesteps,
            content_img=content_images,
            style_source_img=style_images,  # May need adjustment
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
