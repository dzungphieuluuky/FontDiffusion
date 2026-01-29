from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
from torch import nn


from src import ContentEncoder, StyleEncoder, UNet, SCR
from src.model import FontStyleTransformationModule
from src.model import MultiScaleStyleEncoder

def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "DownBlock2D",
            "MCADownBlock2D",
            "MCADownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "StyleRSIUpBlock2D",
            "StyleRSIUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=args.unet_channels,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32,
    )

    return unet


def build_style_encoder(args):
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel, resolution=args.style_image_size[0]
    )
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel, resolution=args.content_image_size[0]
    )
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args):
    scr = SCR(
        temperature=args.temperature, mode=args.mode, image_size=args.scr_image_size
    )
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
    )
    return ddpm_scheduler

def build_fst(args):
    """Build Font Style Transformation module."""
    # Parse feature channels if string
    feature_channels = args.fst_feature_channels
    if isinstance(feature_channels, str):
        feature_channels = [int(x.strip()) for x in feature_channels.split(",")]
    
    fst_module = FontStyleTransformationModule(
        feature_channels=feature_channels,
        num_queries=args.fst_num_queries,
        query_dim=args.fst_query_dim,
        num_scale_features=args.fst_num_scales,
    )
    print(f"✓ Built FST module (queries={args.fst_num_queries}, dim={args.fst_query_dim})")
    return fst_module


def build_mss_encoder(args):
    """Build Multi-Scale Style Encoder."""
    num_scales = getattr(args, 'mss_num_scales', None) or getattr(args, 'fst_num_scales', 5)
    base_channels = getattr(args, 'mss_base_channels', 64)
    
    mss_encoder = MultiScaleStyleEncoder(
        in_channels=3,
        base_channels=base_channels,
        num_scales=num_scales,
    )
    print(f"✓ Built MSSE (scales={num_scales}, base_ch={base_channels})")
    return mss_encoder


def build_fst_projection(feature_dim: int, cross_attn_dim: int) -> nn.Linear:
    """Build FST projection layer."""
    projection = nn.Linear(feature_dim, cross_attn_dim)
    print(f"✓ Built FST projection ({feature_dim} → {cross_attn_dim})")
    return projection


def build_original_style_projection(style_dim: int, cross_attn_dim: int) -> nn.Linear:
    """Build original style projection layer."""
    projection = nn.Linear(style_dim, cross_attn_dim)
    print(f"✓ Built style projection ({style_dim} → {cross_attn_dim})")
    return projection


def get_unet_cross_attention_dim(unet: UNet) -> int:
    """
    Infer cross-attention dimension from U-Net.
    
    Args:
        unet: U-Net module
        
    Returns:
        Cross-attention dimension
    """
    # Try to get from config
    if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
        return unet.config.cross_attention_dim
    
    # Inspect first cross-attention layer
    for module in unet.modules():
        if hasattr(module, "to_k") and isinstance(module.to_k, nn.Linear):
            return module.to_k.in_features
    
    # Default fallback
    return 1024