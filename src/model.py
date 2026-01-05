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
        self.self_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
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
        input_feature_dim: int = 1024,  # ✅ ADD THIS for extracted features
    ):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.input_feature_dim = input_feature_dim

        # ✅ ADD: Projection layer to map extracted features to feature_dim
        self.feature_projection = nn.Linear(input_feature_dim, feature_dim)

        # Key/value weights for each scale
        self.key_weights = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim))
            for _ in range(num_scales)
        ])
        self.value_weights = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim))
            for _ in range(num_scales)
        ])

        # Initialize weights
        for weight in self.key_weights:
            nn.init.orthogonal_(weight)
        for weight in self.value_weights:
            nn.init.orthogonal_(weight)
    
    def extract_style_features(self, style_feature: torch.Tensor) -> list:
        """Extract multi-scale style features.
        
        Args:
            style_feature: Input style feature (B, C, H, W) or (B, D)
            
        Returns:
            List of projected features for each scale
        """
        # Flatten if needed
        if style_feature.dim() == 4:  # (B, C, H, W)
            style_feature = style_feature.view(style_feature.size(0), -1)  # (B, C*H*W)
        
        # ✅ PROJECT: (B, input_feature_dim) -> (B, feature_dim)
        projected = self.feature_projection(style_feature)  # (B, feature_dim)

        features = []
        for scale_idx in range(self.num_scales):
            # ✅ NOW SHAPES MATCH: (B, feature_dim) @ (feature_dim, feature_dim)
            key = projected @ self.key_weights[scale_idx]
            value = projected @ self.value_weights[scale_idx]
            features.append((key, value))

        return features
    
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
    
    def compute_style_difference(
        self,
        source_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute style differences at each scale (Equation 7).
        
        L_xy = L_y - L_x (difference between target and source)
        """
        style_differences = []
        for src_feat, tgt_feat in zip(source_features, target_features):
            # Element-wise difference
            diff = tgt_feat - src_feat
            style_differences.append(diff)
        
        return style_differences
    
    def fuse_style_differences(
        self,
        style_differences: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse style differences across scales using self-attention (Equation 8).
        """
        batch_size = style_differences[0].shape[0]
        
        # Concatenate differences along channel dimension
        concatenated = torch.cat(style_differences, dim=-1)  # (batch_size, seq_len, feature_dim * num_scales)
        
        # MLP to adjust channel size
        if concatenated.dim() == 3:
            # (batch_size, seq_len, feature_dim * num_scales) -> (batch_size, feature_dim)
            fused = self.mlp_channel_adjust(concatenated.mean(dim=1))
        else:
            fused = self.mlp_channel_adjust(concatenated)
        
        return fused
    
    def forward(
        self,
        source_style_features: torch.Tensor,
        target_style_features: torch.Tensor,
    ) -> tuple:
        """Forward pass for style transformation.
        
        Args:
            source_style_features: Source style features (B, 1024) or (B, C, H, W)
            target_style_features: Target style features (B, 1024) or (B, C, H, W)
            
        Returns:
            Tuple of (transformed_features, style_difference)
        """
        # Extract multi-scale features
        source_features = self.extract_style_features(source_style_features)
        target_features = self.extract_style_features(target_style_features)

        # Compute style difference
        style_diff = 0
        for src_kv, tgt_kv in zip(source_features, target_features):
            src_key, src_val = src_kv
            tgt_key, tgt_val = tgt_kv
            # Contrastive loss: minimize difference between source and target
            style_diff += F.mse_loss(src_key, tgt_key) + F.mse_loss(src_val, tgt_val)

        style_diff = style_diff / self.num_scales

        # Return transformed feature and difference
        transformed = source_features[0][0]  # Use first scale's key as output
        return transformed, style_diff

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
        style_img_feature, _, style_residual_features = self.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Get reference content features from style image
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ FIXED: Compute style transformation with correct arguments
        style_transform_feature = None
        style_diff = None
        
        if (
            source_style_images is not None 
            and self.style_transform_module is not None
        ):
            # Extract source style features
            source_style_img_feature, _, _ = self.style_encoder(
                source_style_images
            )

            # ✅ CORRECT: Pass only source and target style features
            style_transform_feature, style_diff = self.style_transform_module(
                source_style_features=source_style_img_feature,
                target_style_features=style_img_feature
            )
        else:
            # Default style difference encoding
            style_diff = torch.zeros(
                batch_size, 
                self.style_transform_module.feature_dim 
                if self.style_transform_module is not None 
                else 256,
                device=style_img_feature.device
            )

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_diff,
        ]

        out = self.unet(
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
        """
        Args:
            x_t: Noisy latent
            timesteps: Diffusion timesteps
            cond: Tuple of (content_images, target_style_images)
            content_encoder_downsample_size: Downsampling size
            version: Model version
            source_cond: Optional tuple (source_content, source_style) for style transformation
        
        Returns:
            noise_pred
        """
        content_images = cond[0]
        style_images = cond[1]

        # Extract target style features
        style_img_feature, _, style_residual_features = self.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Get reference content features
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ FIXED: Compute style transformation with correct arguments
        style_transform_feature = None
        style_diff = None
        
        if source_cond is not None and self.style_transform_module is not None:
            source_content_images, source_style_images = source_cond
            
            # Extract source style features
            source_style_img_feature, _, _ = self.style_encoder(
                source_style_images
            )
            
            # ✅ CORRECT: Pass only source and target style features
            style_transform_feature, style_diff = self.style_transform_module(
                source_style_features=source_style_img_feature,
                target_style_features=style_img_feature
            )
        else:
            style_diff = torch.zeros(
                batch_size,
                self.style_transform_module.feature_dim
                if self.style_transform_module is not None
                else 256,
                device=style_img_feature.device
            )

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_diff,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        return noise_pred