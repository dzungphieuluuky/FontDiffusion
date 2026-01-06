import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Dict
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class MultiScaleTransformerBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.
    Used for extracting style features at each scale.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Self-attention for style features
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

        # Cross-attention (for fusing source and target)
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )

        # FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, feature_dim),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.ln3 = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, feature_dim)
            key: (batch_size, seq_len, feature_dim)
            value: (batch_size, seq_len, feature_dim)

        Returns:
            output: (batch_size, seq_len, feature_dim)
        """
        # Self-attention
        self_attn_out, _ = self.self_attn(query, query, query)
        query = self.ln1(query + self_attn_out)

        # Cross-attention
        cross_attn_out, _ = self.cross_attn(query, key, value)
        query = self.ln2(query + cross_attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.ln3(query + ffn_out)

        return query


class StyleTransformationModule(nn.Module):
    def __init__(
        self,
        num_scales: int = 4,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        style_image_size: int = 96,
        input_channels: int = 1,  # ✅ ADD THIS
    ):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim

        self.spatial_size = input_channels * style_image_size * style_image_size
        self.feature_projection = nn.Linear(self.spatial_size, feature_dim)

        # Multi-scale transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MultiScaleTransformerBlock(
                    feature_dim=feature_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_scales)
            ]
        )

        # Key/value weights for each scale
        self.key_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(feature_dim, feature_dim))
                for _ in range(num_scales)
            ]
        )
        self.value_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(feature_dim, feature_dim))
                for _ in range(num_scales)
            ]
        )

        # Initialize weights
        for weight in self.key_weights:
            nn.init.orthogonal_(weight)
        for weight in self.value_weights:
            nn.init.orthogonal_(weight)

    def extract_style_features(
        self, style_feature: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract multi-scale style features from style encoder output.

        Args:
            style_feature: Input style feature (B, C, H, W) from style encoder

        Returns:
            List of (key, value) tuples for each scale
        """
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
        batch_size = style_feature.shape[0]
        style_feature_flat = style_feature.view(batch_size, -1)  # (B, C*H*W)

        # ✅ PROJECT: (B, C*H*W) -> (B, feature_dim)
        projected = self.feature_projection(style_feature_flat)  # (B, feature_dim)

        features = []
        for scale_idx in range(self.num_scales):
            # ✅ NOW SHAPES MATCH: (B, feature_dim) @ (feature_dim, feature_dim)
            key = projected @ self.key_weights[scale_idx]  # (B, feature_dim)
            value = projected @ self.value_weights[scale_idx]  # (B, feature_dim)
            features.append((key, value))

        return features

    def compute_style_difference(
        self,
        source_features: List[Tuple[torch.Tensor, torch.Tensor]],
        target_features: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute style difference between source and target.

        Args:
            source_features: List of (key, value) from source
            target_features: List of (key, value) from target

        Returns:
            Aggregated style difference tensor
        """
        style_diffs = []

        for src_kv, tgt_kv in zip(source_features, target_features):
            src_key, src_val = src_kv
            tgt_key, tgt_val = tgt_kv

            # Contrastive loss: minimize difference between source and target
            diff = F.mse_loss(src_key, tgt_key) + F.mse_loss(src_val, tgt_val)
            style_diffs.append(diff)

        # Average across scales and expand for batch dimension
        style_diff_avg = torch.stack(style_diffs).mean()

        # Return expanded tensor matching batch size
        batch_size = source_features[0][0].shape[0]
        return (
            style_diff_avg.unsqueeze(0).expand(batch_size, -1).mean(dim=1, keepdim=True)
        )

    def forward(
        self,
        source_style_features: torch.Tensor,
        target_style_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for style transformation."""
        # Extract multi-scale features
        source_features = self.extract_style_features(source_style_features)
        target_features = self.extract_style_features(target_style_features)

        batch_size = source_style_features.shape[0]

        # ✅ Compute style differences across scales
        style_diffs = []
        for src_kv, tgt_kv in zip(source_features, target_features):
            src_key, src_val = src_kv
            tgt_key, tgt_val = tgt_kv

            # MSE between source and target
            key_diff = F.mse_loss(src_key, tgt_key, reduction="none").mean(
                dim=1
            )  # (B,)
            val_diff = F.mse_loss(src_val, tgt_val, reduction="none").mean(
                dim=1
            )  # (B,)
            style_diffs.append(key_diff + val_diff)

        # Average style difference: (B,)
        style_diff = torch.stack(style_diffs).mean(dim=0)

        # Return transformed feature (can be used for additional conditioning)
        # Return as (B, feature_dim) - simple vector representation
        transformed = source_features[0][0]  # (B, feature_dim)

        return transformed, style_diff

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention (Equation 5).
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def fuse_style_differences(
        self,
        style_differences: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse style differences across scales using self-attention (Equation 8).
        """
        batch_size = style_differences[0].shape[0]

        # Concatenate differences along channel dimension
        concatenated = torch.cat(
            style_differences, dim=-1
        )  # (batch_size, seq_len, feature_dim * num_scales)

        # MLP to adjust channel size
        if concatenated.dim() == 3:
            # (batch_size, seq_len, feature_dim * num_scales) -> (batch_size, feature_dim)
            fused = self.mlp_channel_adjust(concatenated.mean(dim=1))
        else:
            fused = self.mlp_channel_adjust(concatenated)

        return fused


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuser with content encoder,
    style encoder, style transformation module, and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_transform_module: Optional[StyleTransformationModule] = None,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_transform_module = style_transform_module

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
        source_style_images: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_t: Noisy latent
            timesteps: Diffusion timesteps
            style_images: Target style images
            content_images: Content images
            content_encoder_downsample_size: Downsampling size
            source_style_images: Source style images (optional) for style transformation

        Returns:
            noise_pred, offset_out_sum, style_transform_feature
        """
        # Extract target style features
        style_img_feature, _, style_residual_features = self.config.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.config.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)

        # Get reference content features from style image
        style_content_feature, style_content_res_features = self.config.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ FIXED: Compute style transformation with correct arguments
        style_transform_feature = None
        style_diff = None

        if (
            source_style_images is not None
            and self.config.style_transform_module is not None
        ):
            # Extract source style features
            source_style_img_feature, _, _ = self.config.style_encoder(
                source_style_images
            )

            # ✅ CORRECT: Pass only source and target style features
            style_transform_feature, style_diff = self.config.style_transform_module(
                source_style_features=source_style_img_feature,
                target_style_features=style_img_feature,
            )
        else:
            # Default style difference encoding
            style_diff = torch.zeros(
                batch_size,
                (
                    self.config.style_transform_module.feature_dim
                    if self.config.style_transform_module is not None
                    else 256
                ),
                device=style_img_feature.device,
            )

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_diff,
        ]

        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum, style_transform_feature


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuser with style transformation module."""

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_transform_module: Optional[StyleTransformationModule] = None,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_transform_module = style_transform_module

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
        source_cond: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        content_images = cond[0]
        style_images = cond[1]

        # Extract style features
        style_img_feature, _, style_residual_features = self.config.style_encoder(
            style_images
        )
        batch_size, channel, height, width = style_img_feature.shape

        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Extract content features
        content_img_feature, content_residual_features = self.config.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)

        style_content_feature, style_content_res_features = self.config.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ Compute style transformation if provided
        style_transform_feature = None
        if source_cond is not None and self.config.style_transform_module is not None:
            source_content_images, source_style_images = source_cond
            source_style_img_feature, _, _ = self.config.style_encoder(
                source_style_images
            )

            # Get style transformation output
            style_transform_feature, style_diff = self.config.style_transform_module(
                source_style_features=source_style_img_feature,
                target_style_features=style_img_feature,
            )
            # style_diff can be used for loss calculation or returned separately
            # but NOT included in encoder_hidden_states

        # ✅ FIXED: Only 4 elements - style_diff NOT included
        input_hidden_states = [
            style_img_feature,  # Index 0: (B, C, H, W)
            content_residual_features,  # Index 1: List of tensors
            style_hidden_states,  # Index 2: (B, H*W, C)
            style_content_res_features,  # Index 3: List of tensors
        ]

        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )

        noise_pred = out[0]
        return noise_pred
