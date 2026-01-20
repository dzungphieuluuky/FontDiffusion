import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict
from omegaconf import DictConfig

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule


class FontDiffuserWithFST(nn.Module):
    """Enhanced FontDiffuser with FSTDiff modules (MSSE + FST).
    
    Implements the FSTDiff architecture for font style transformation.
    Reference: FSTDiff paper (Section 3)
    """

    def __init__(
        self,
        original_fontdiffuser: nn.Module,
        feature_channels: list[int] = None,
        num_queries: int = 256,
        query_dim: int = 128,
        num_scales: int = 5,
        cfg: Optional[DictConfig] = None,
    ):
        """Initialize FontDiffuserWithFST.
        
        Args:
            original_fontdiffuser: Base FontDiffuser model with unet, encoders
            feature_channels: MSSE output channels per scale [64, 128, 256, 512, 1024]
            num_queries: Number of learnable queries in FST (N_L)
            query_dim: Dimension of each query vector (d)
            num_scales: Number of multi-scale levels (n_s)
            cfg: Optional Hydra DictConfig with FST parameters
        """
        super().__init__()
        self.cfg = cfg
        
        # Keep original FontDiffuser components
        self.content_encoder = original_fontdiffuser.content_encoder
        self.diffusion_unet = original_fontdiffuser.unet
        self.style_encoder = original_fontdiffuser.style_encoder

        # Extract config parameters if provided
        if cfg is not None:
            num_queries = cfg.get("fst_num_queries", num_queries)
            query_dim = cfg.get("fst_query_dim", query_dim)
            num_scales = cfg.get("fst_num_scales", num_scales)
            feature_channels = cfg.get(
                "fst_feature_channels", 
                [64, 128, 256, 512, 1024]
            )

        # ========== MSSE: Multi-Scale Style Encoder ==========
        # Extracts style features at multiple resolutions
        # Input: (B, 1, 512, 512) grayscale glyph image
        # Output: List of 5 feature tensors at decreasing resolutions
        self.mss_encoder = MultiScaleStyleEncoder(
            in_channels=1,  # Grayscale glyph input
            base_channels=64,
            num_scales=num_scales,
        )

        # Verify feature channels match MSSE output
        if feature_channels is None:
            feature_channels = self.mss_encoder.get_feature_channels()
        else:
            expected = self.mss_encoder.get_feature_channels()
            assert feature_channels == expected, (
                f"Feature channels mismatch: provided {feature_channels} "
                f"but MSSE outputs {expected}"
            )

        # ========== FST: Font Style Transformation Module ==========
        # Computes style transformation between source and target fonts
        # Input: Source features + Target features (5 scales each)
        # Output: (B, N_L + spatial_last, 1024) transformation features
        self.fst_module = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=num_queries,
            query_dim=query_dim,
            num_scale_features=num_scales,
            num_cross_attn_blocks=cfg.get("fst_cross_attn_blocks", 2) if cfg else 2,
            num_self_attn_blocks=cfg.get("fst_self_attn_blocks", 2) if cfg else 2,
        )

        # ========== FST Feature Projection to U-Net ==========
        # FST output: (B, N_L + 256, 1024) where 256 = 16x16 last scale spatial size
        # Project to U-Net cross-attention dimension
        cross_attn_dim = getattr(
            self.diffusion_unet.config, "cross_attention_dim", 1280
        )
        if cfg:
            cross_attn_dim = cfg.get("unet_cross_attn_dim", cross_attn_dim)

        self.fst_projection = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, cross_attn_dim),
        )

        # Optional: Project original style features for concatenation
        self.original_style_projection = nn.Sequential(
            nn.Linear(1024, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )

        # Store dimensions for debugging
        self.num_queries = num_queries
        self.feature_channels = feature_channels
        self.cross_attn_dim = cross_attn_dim

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        content_img: torch.Tensor,
        style_source_img: torch.Tensor,
        style_target_img: torch.Tensor,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with FSTDiff style transformation.

        Args:
            noisy_latents: (B, 4, H, W) - noisy latent representations
            timestep: (B,) - diffusion timestep indices
            content_img: (B, 1, 512, 512) - character to generate
            style_source_img: (B, 1, 512, 512) - reference in source font
            style_target_img: (B, 1, 512, 512) - reference in target font
            content_encoder_downsample_size: downsampling factor for content encoder
            return_dict: whether to return dict or tuple

        Returns:
            Dictionary with keys:
                - noise_pred: Predicted noise for denoising step
                - offset_out_sum: Offset loss from U-Net
                - transformation_features: FST output features (B, 512, 1024)
                - source_style_features: MSSE features from style_source
                - target_style_features: MSSE features from style_target
                - fst_condition: Projected FST features for U-Net
        """
        batch_size = noisy_latents.shape[0]

        # ========== 1. CONTENT ENCODING ==========
        # Extract content structure from character to generate
        content_img_feature, content_residual_features = self.content_encoder(
            content_img
        )
        content_residual_features.append(content_img_feature)

        # Extract content structure from target style reference
        style_content_feature, style_content_res_features = self.content_encoder(
            style_target_img
        )
        style_content_res_features.append(style_content_feature)

        # ========== 2. ORIGINAL STYLE ENCODING (FontDiffuser baseline) ==========
        # Keep original style encoder for compatibility and baseline features
        orig_style_feat, orig_style_vec, orig_style_residuals = self.style_encoder(
            style_target_img
        )
        # Reshape for cross-attention: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = orig_style_feat.shape
        orig_style_hidden = orig_style_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ========== 3. MULTI-SCALE STYLE ENCODING (MSSE) ==========
        # Extract style features at 5 scales from both source and target
        # MSSE input: 512x512 → initial conv (stride=2) → 256x256
        # Then 4 more downsamples: 128x128, 64x64, 32x32, 16x16
        source_style_features = self.mss_encoder(style_source_img)
        target_style_features = self.mss_encoder(style_target_img)
        
        # Each returns list of 5 tensors:
        # [(B, 64, 256, 256), (B, 128, 128, 128), (B, 256, 64, 64),
        #  (B, 512, 32, 32), (B, 1024, 16, 16)]
        assert len(source_style_features) == 5, "MSSE should output 5 scales"
        assert len(target_style_features) == 5, "MSSE should output 5 scales"

        # ========== 4. FONT STYLE TRANSFORMATION (FST) ==========
        # Compute style transformation L_{source→target}^r between fonts
        # Equation (3)-(9) in FSTDiff paper
        transformation_features = self.fst_module(
            source_style_features, target_style_features
        )
        # Shape: (B, N_L + last_spatial, 1024)
        # = (B, 256 + (16*16), 1024) = (B, 512, 1024)
        assert transformation_features.shape[0] == batch_size
        assert transformation_features.shape[2] == 1024, \
            f"FST output channels must be 1024, got {transformation_features.shape[2]}"

        # ========== 5. PROJECT FST FEATURES TO U-NET SPACE ==========
        # Project transformation features to U-Net cross-attention dimension
        fst_condition = self.fst_projection(transformation_features)
        # Shape: (B, 512, cross_attn_dim)

        # Project original style vector for compatibility
        orig_style_projected = self.original_style_projection(orig_style_vec)
        # Shape: (B, cross_attn_dim)
        orig_style_projected = orig_style_projected.unsqueeze(1)  # (B, 1, cross_attn_dim)

        # Combine FST and original style features
        # FST provides learned transformations, original provides baseline
        combined_style_condition = torch.cat(
            [fst_condition, orig_style_projected], dim=1
        )
        # Shape: (B, 513, cross_attn_dim)

        # ========== 6. PREPARE U-NET ENCODER HIDDEN STATES ==========
        # FontDiffuser expects specific format for cross-attention
        encoder_hidden_states = [
            orig_style_feat,  # Style feature map for U-Net feature injection
            content_residual_features,  # Content skip connections
            combined_style_condition,  # Enhanced style condition (FST + original)
            style_content_res_features,  # Style content skip connections
        ]

        # ========== 7. DIFFUSION U-NET FORWARD ==========
        # Run diffusion model with enhanced style conditioning
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
        outputs: Dict[str, torch.Tensor],
        target_noise: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Compute loss components for training.

        Args:
            outputs: Dictionary from forward pass
            target_noise: Ground truth noise to predict
            reduction: 'mean' or 'sum'

        Returns:
            Dictionary with keys:
                - noise_loss: Main denoising loss
                - offset_loss: Offset regularization loss
                - total_loss: Weighted sum of all losses
        """
        losses = {}

        # Main denoising loss: MSE between predicted and target noise
        noise_pred = outputs["noise_pred"]
        if reduction == "mean":
            losses["noise_loss"] = F.mse_loss(noise_pred, target_noise)
        else:
            losses["noise_loss"] = F.mse_loss(
                noise_pred, target_noise, reduction="sum"
            )

        # Offset loss (U-Net regularization, if applicable)
        offset_out_sum = outputs.get("offset_out_sum", 0)
        if isinstance(offset_out_sum, torch.Tensor):
            losses["offset_loss"] = (
                offset_out_sum.mean() if reduction == "mean" else offset_out_sum.sum()
            )
        else:
            losses["offset_loss"] = torch.tensor(
                0.0, device=noise_pred.device, dtype=noise_pred.dtype
            )

        # Total weighted loss
        # Get weight from config if available, default to 0.01
        offset_weight = 0.01
        if self.cfg:
            offset_weight = self.cfg.get("offset_loss_weight", 0.01)

        losses["total_loss"] = losses["noise_loss"] + offset_weight * losses["offset_loss"]

        return losses

    def get_trainable_parameters(self, freeze_original: bool = False):
        """Get trainable parameters with optional freezing.
        
        Args:
            freeze_original: If True, freeze original FontDiffuser encoders
                           and only train FST/MSSE modules
        
        Returns:
            Generator of trainable parameters
        """
        if freeze_original:
            # Only train new FST and MSSE modules
            for param in self.mss_encoder.parameters():
                yield param
            for param in self.fst_module.parameters():
                yield param
            for param in self.fst_projection.parameters():
                yield param
            for param in self.original_style_projection.parameters():
                yield param
        else:
            # Train everything
            for param in self.parameters():
                yield param

    def freeze_original_encoders(self):
        """Freeze original FontDiffuser encoders."""
        self.content_encoder.requires_grad_(False)
        self.style_encoder.requires_grad_(False)
        self.diffusion_unet.requires_grad_(False)

    def unfreeze_original_encoders(self):
        """Unfreeze original FontDiffuser encoders."""
        self.content_encoder.requires_grad_(True)
        self.style_encoder.requires_grad_(True)
        self.diffusion_unet.requires_grad_(True)


class FontDiffuserWithFSTWrapper(nn.Module):
    """Wrapper for API compatibility with original FontDiffuserModel.
    
    Maintains the original forward signature while using FSTDiff internally.
    """

    def __init__(self, fontdiffuser_with_fst: FontDiffuserWithFST):
        super().__init__()
        self.model = fontdiffuser_with_fst

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        style_images: torch.Tensor,
        content_images: torch.Tensor,
        content_encoder_downsample_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """API-compatible forward pass.

        Args:
            x_t: Noisy latents (B, 4, H, W)
            timesteps: Timestep indices (B,)
            style_images: Style reference images (B, 1, 512, 512)
            content_images: Content images (B, 1, 512, 512)
            content_encoder_downsample_size: Content encoder downsampling

        Returns:
            Tuple of (noise_pred, offset_out_sum)
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