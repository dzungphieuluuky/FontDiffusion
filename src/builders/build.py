import argparse
import os
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch import nn


from src import ContentEncoder, StyleEncoder, UNet, SCR
from src.model import FontStyleTransformationModule
from src.model import MultiScaleStyleEncoder
from src.modules.identity_mapping_loss import (
    IdentityMappingLoss,
    AdaptiveIdentityMappingLoss,
    PooledIdentityMappingLoss,
)
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{__name__}.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_state_dict_auto(path: str):
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors files.")
        return safe_load_file(path)
    else:
        return torch.load(path, map_location="cpu")


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


def build_style_encoder(args: argparse.Namespace) -> StyleEncoder:
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel, resolution=args.style_image_size[0]
    )
    print("Build CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args: argparse.Namespace) -> ContentEncoder:
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel, resolution=args.content_image_size[0]
    )
    print("Build CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args: argparse.Namespace) -> SCR:
    scr = SCR(
        temperature=args.temperature, mode=args.mode, image_size=args.scr_image_size
    )
    print("Build SCR module!")
    return scr


def build_ddpm_scheduler(args: argparse.Namespace) -> DDPMScheduler:
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
    )
    print("Build DDPM Scheduler!")
    return ddpm_scheduler


def build_fst(args: argparse.Namespace) -> FontStyleTransformationModule:
    """Build Font Style Transformation module."""
    # Parse feature channels if string
    feature_channels = args.fst_feature_channels
    if isinstance(feature_channels, str):
        feature_channels = [int(x.strip()) for x in feature_channels.split(",")]

    fst_module = FontStyleTransformationModule(
        num_queries=args.fst_num_queries,
        query_dim=args.fst_query_dim,
        msse_output_channels=feature_channels,
    )
    print(
        f"✓ Built FST module (queries={args.fst_num_queries}, dim={args.fst_query_dim})"
    )
    return fst_module


def build_mss_encoder(args: argparse.Namespace) -> MultiScaleStyleEncoder:
    """Build Multi-Scale Style Encoder."""
    num_scales = getattr(args, "mss_num_scales", None) or getattr(
        args, "fst_num_scales", 5
    )
    base_channels = getattr(args, "mss_base_channels", 64)

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


def build_identity_loss_module(args: argparse.Namespace) -> IdentityMappingLoss:
    """Build identity mapping loss module."""
    identity_loss = IdentityMappingLoss(
        matrix_size=getattr(args, "fst_num_queries", 256),
        loss_type=getattr(args, "identity_loss_type", "frobenius"),
        regularization=getattr(args, "identity_regularization", "orthogonal"),
        reg_weight=getattr(args, "identity_reg_weight", 0.01),
    )
    print(
        f"✓ Built IdentityMappingLoss "
        f"(matrix_size={args.fst_num_queries}, "
        f"loss_type={getattr(args, 'identity_loss_type', 'frobenius')})"
    )
    return identity_loss


def build_base_components(args: argparse.Namespace):
    unet: UNet = build_unet(args)
    style_encoder: StyleEncoder = build_style_encoder(args)
    content_encoder: ContentEncoder = build_content_encoder(args)
    scr: SCR = build_scr(args)
    ddpm_scheduler = build_ddpm_scheduler(args)
    components = {
        "unet": unet,
        "style_encoder": style_encoder,
        "content_encoder": content_encoder,
        "scr": scr,
        "ddpm_scheduler": ddpm_scheduler,
    }
    print("Built FontDiffuser base components.")
    return components


def build_fst_components(args: argparse.Namespace) -> dict:
    """
    Build all modules necessary for FontDiffuser FST model.
    Returns a dict of initialized components.
    """
    # Core encoders and modules
    content_encoder: ContentEncoder = build_content_encoder(args)
    style_encoder: StyleEncoder = build_style_encoder(args)
    mss_encoder: MultiScaleStyleEncoder = build_mss_encoder(args)
    fst_module: FontStyleTransformationModule = build_fst(args)
    unet: UNet = build_unet(args)
    scr: SCR = build_scr(args)
    ddpm_scheduler = build_ddpm_scheduler(args)

    # Projections
    cross_attn_dim = get_unet_cross_attention_dim(unet)
    fst_proj = build_fst_projection(args.fst_query_dim, cross_attn_dim)
    style_proj = build_original_style_projection(
        args.style_start_channel * 16, cross_attn_dim
    )

    # Loss module
    identity_loss: IdentityMappingLoss = build_identity_loss_module(args)

    components = {
        "content_encoder": content_encoder,
        "style_encoder": style_encoder,
        "mss_encoder": mss_encoder,
        "fst_module": fst_module,
        "unet": unet,
        "scr": scr,
        "ddpm_scheduler": ddpm_scheduler,
        "fst_projection": fst_proj,
        "style_projection": style_proj,
        "identity_loss_module": identity_loss,
    }
    print("Built FontDiffuser FST components.")
    return components


def load_components_from_ckpt(components: dict[str, nn.Module], ckpt_path: str):
    """Load components' state dicts from checkpoint."""
    for name, module in components.items():
        module_ckpt_path = f"{ckpt_path}/{name}.safetensors"
        if not os.path.exists(module_ckpt_path):
            print(
                f"Warning: Checkpoint for {name} not found at {module_ckpt_path}. Skipping."
            )
            module_ckpt_path = f"{ckpt_path}/{name}.pth"
        if not os.path.exists(module_ckpt_path):
            print(
                f"Warning: Checkpoint for {name} not found at {module_ckpt_path}. Skipping."
            )
            continue
        state_dict = load_state_dict_auto(module_ckpt_path)
        module.load_state_dict(state_dict)
        print(f"Loaded {name} from {module_ckpt_path}.")
