import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule

class FontDiffuserWithFST(nn.Module):
    """
    Enhanced FontDiffuser with FSTDiff modules.
    Architecture: ContentEncoder + MSSE + FST → Diffusion U-Net
    """
    def __init__(self, original_fontdiffuser):
        super().__init__()
        
        # Keep original FontDiffuser components
        self.content_encoder = original_fontdiffuser.content_encoder  # MCA blocks
        self.diffusion_unet = original_fontdiffuser.unet
        self.style_encoder = original_fontdiffuser.style_encoder  # Original for SCR loss
        
        # Add new FSTDiff modules
        self.mss_encoder = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5)
        
        # Determine feature channels from MSSE output shapes
        # Assuming input 96x96 → scales: [48, 24, 12, 6, 6] with channels [64, 128, 256, 512, 1024]
        feature_channels = [64, 128, 256, 512, 1024]
        self.fst_module = FontStyleTransformationModule(
            feature_channels=feature_channels,
            num_queries=256,
            query_dim=128,
            num_scale_features=5,
            num_cross_attn_blocks=2,
            num_self_attn_blocks=2
        )
        
        # Projection layers to inject FST features into U-Net
        self.fst_projection = nn.Sequential(
            nn.Linear(1024, 768),  # FST output is 1024-dim
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, self.diffusion_unet.config.cross_attention_dim)
        )
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        content_img: torch.Tensor,           # Source font character (to generate)
        style_source_img: torch.Tensor,      # Reference char in source font
        style_target_img: torch.Tensor,      # Same reference char in target font
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with tensor shape tracking.
        Expected shapes (B=batch):
        - All images: (B, 1, 96, 96)
        - noisy_latents: (B, 4, 24, 24) [assuming latent diffusion]
        - timestep: (B,) or scalar
        """
        # ========== 1. CONTENT ENCODING ==========
        content_features = self.content_encoder(content_img)
        # Shape: List[(B, C_i, H_i, W_i)] where i=1..n_c (typically 4 scales)
        
        # ========== 2. STYLE ENCODING (MSSE) ==========
        source_style_features = self.mss_encoder(style_source_img)  # List[5 tensors]
        target_style_features = self.mss_encoder(style_target_img)  # List[5 tensors]
        # Each list: [(B, 64, 48, 48), (B, 128, 24, 24), ..., (B, 1024, 6, 6)]
        
        # ========== 3. STYLE TRANSFORMATION (FST) ==========
        transformation_features = self.fst_module(source_style_features, target_style_features)
        # Shape: (B, N_L + 36, 1024) = (B, 256 + 36, 1024) = (B, 292, 1024)
        
        # ========== 4. PREPARE DIFFUSION CONDITIONS ==========
        # Project FST features to U-Net cross-attention dimension
        fst_condition = self.fst_projection(transformation_features)  # (B, 292, cross_attn_dim)
        
        # Original style features (for SCR loss compatibility)
        orig_style_feat = self.style_encoder(style_target_img)
        
        # ========== 5. DIFFUSION U-NET ==========
        # Modify the U-Net forward call to accept fst_condition
        model_output = self.custom_unet_forward(
            self.diffusion_unet,
            noisy_latents,
            timestep,
            content_conditions=content_features,
            style_conditions=fst_condition,
            original_style=orig_style_feat
        )
        
        return {
            'model_output': model_output,
            'content_features': content_features,
            'transformation_features': transformation_features,
            'source_style_features': source_style_features,
            'target_style_features': target_style_features
        }
    
    def custom_unet_forward(self, unet, x, t, **kwargs):
        """Adapted U-Net forward to handle new conditioning."""
        # This is where you modify FontDiffuser's U-Net
        # Typically involves adding cross-attention layers
        pass


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
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
        style_img_feature, _, _ = self.style_encoder(style_images)
    
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]

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
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
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

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred