import argparse
import os
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch import nn

from src import ContentEncoder, StyleEncoder, UNet, SCR
from src.model import FontStyleTransformationModule, MultiScaleStyleEncoder
from src.modules.identity_mapping_loss import IdentityMappingLoss
from src.modules.skeleton_distance_transform import (
    SkeletonDistanceTransform,
    DualChannelContentEncoder,
)
from src.modules.adversarial import (
    StyleDiscriminator,
    AdversarialContentStyleLoss,
    StyleLabelExtractor,
)
from src.modules.frequency_decomposition import (
    FrequencyDecomposition,
    MultiScaleFrequencyEncoder,
)


import logging
from src.tools.utilities import setup_logger

logger = setup_logger(__name__, logging.INFO)


def load_state_dict_auto(path: str):
    """Load state dict from .safetensors or .pth file."""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors files.")
        return safe_load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def build_unet(args: argparse.Namespace) -> UNet:
    """Build U-Net model.
    Args:
        args (argparse.Namespace): Configuration arguments.

    Returns:
        UNet: U-Net model instance.
    """
    print("Building U-Net...")
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
    print("✓ U-Net built successfully.")
    return unet


def build_style_encoder(args: argparse.Namespace) -> StyleEncoder:
    """Build style encoder.
    Args:
        args (argparse.Namespace): Configuration arguments.
    Returns:
        StyleEncoder: Style encoder instance.
    """
    print("Building Style Encoder...")
    style_encoder = StyleEncoder(
        G_ch=args.style_start_channel, resolution=args.style_image_size[0]
    )
    print("✓ Style Encoder built successfully.")
    return style_encoder


def build_content_encoder(args: argparse.Namespace) -> ContentEncoder:
    """Build content encoder.
    Args:
        args (argparse.Namespace): Configuration arguments.
    Returns:
        ContentEncoder: Content encoder instance.
    """
    print("Building Content Encoder...")
    content_encoder = ContentEncoder(
        G_ch=args.content_start_channel, resolution=args.content_image_size[0]
    )
    print("✓ Content Encoder built successfully.")
    return content_encoder


def build_scr(args: argparse.Namespace) -> SCR:
    """Build SCR module.
    Args:
        args (argparse.Namespace): Configuration arguments.
    Returns:
        SCR: SCR module instance.
    """
    print("Building SCR module...")
    scr = SCR(
        temperature=args.temperature, mode=args.mode, image_size=args.scr_image_size
    )
    print("✓ SCR module built successfully.")
    return scr


def build_ddpm_scheduler(args: argparse.Namespace) -> DDPMScheduler:
    """Build DDPM scheduler.
    Args:
        args (argparse.Namespace): Configuration arguments.
    Returns:
        DDPMScheduler: DDPM scheduler instance.
    """
    print("Building DDPM Scheduler...")
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
    )
    print("✓ DDPM Scheduler built successfully.")
    return ddpm_scheduler


def build_fst(args: argparse.Namespace) -> FontStyleTransformationModule:
    """Build Font Style Transformation (FST) module.
    Args:
        args (argparse.Namespace): Configuration arguments.

    Returns:
        FontStyleTransformationModule: FST module instance.
    """
    print("Building Font Style Transformation (FST) module...")
    feature_channels = args.fst_feature_channels
    if isinstance(feature_channels, str):
        feature_channels = [int(x.strip()) for x in feature_channels.split(",")]

    fst_module = FontStyleTransformationModule(
        num_queries=args.fst_num_queries,
        query_dim=args.fst_query_dim,
        msse_output_channels=feature_channels,
    )
    print(
        f"✓ FST module built successfully (queries={args.fst_num_queries}, dim={args.fst_query_dim})."
    )
    return fst_module


def build_mss_encoder(args: argparse.Namespace) -> MultiScaleStyleEncoder:
    """Bulding Multi-Scale Style Encoder (MSSE).
    Args:
        args (argparse.Namespace): Configuration arguments.

    Returns:
        MultiScaleStyleEncoder: Multi-Scale Style Encoder instance.
    """
    print("Building Multi-Scale Style Encoder (MSSE)...")
    num_scales = getattr(args, "mss_num_scales", 5)
    base_channels = getattr(args, "mss_base_channels", 64)

    mss_encoder = MultiScaleStyleEncoder(
        in_channels=3,
        base_channels=base_channels,
        num_scales=num_scales,
    )
    print(
        f"✓ MSSE built successfully (scales={num_scales}, base_channels={base_channels})."
    )
    return mss_encoder


def build_fst_projection(feature_dim: int, cross_attn_dim: int) -> nn.Linear:
    """Build FST projection layer.
    Args:
        feature_dim (int): Dimension of FST features.
        cross_attn_dim (int): Dimension of U-Net cross-attention.
    """
    print(f"Building FST projection layer ({feature_dim} → {cross_attn_dim})...")
    projection = nn.Linear(feature_dim, cross_attn_dim)
    print("✓ FST projection layer built successfully.")
    return projection


def build_original_style_projection(style_dim: int, cross_attn_dim: int) -> nn.Linear:
    """Build original style projection layer."""
    print(
        f"Building original style projection layer ({style_dim} → {cross_attn_dim})..."
    )
    projection = nn.Linear(style_dim, cross_attn_dim)
    print("✓ Original style projection layer built successfully.")
    return projection


def build_skeleton_transform(args: argparse.Namespace) -> SkeletonDistanceTransform:
    """Build skeleton-distance transform module."""
    print("Building Skeleton-Distance Transform...")
    skeleton_config = {
        "method": getattr(args, "skeleton_method", "medial_axis"),
        "distance_method": getattr(args, "skeleton_distance_method", "hybrid"),
        "max_distance": getattr(args, "skeleton_max_distance", 12.0),
        "sigma": getattr(args, "skeleton_sigma", 1.5),
        "output_mode": getattr(args, "skeleton_output_mode", "dual_channel"),
        "normalize": True,
    }
    skeleton_transform = SkeletonDistanceTransform(**skeleton_config)
    print(f"✓ Skeleton-Distance Transform built successfully: {skeleton_config}")
    return skeleton_transform


def build_dual_channel_content_encoder(
    args: argparse.Namespace,
) -> DualChannelContentEncoder:
    """Build dual-channel content encoder for skeleton transform."""
    print(
        f"Building Dual-Channel Content Encoder (fusion method: {args.skeleton_fusion_method})..."
    )
    content_encoder = build_content_encoder(args)
    fusion_method = getattr(args, "skeleton_fusion_method", "concat")
    dual_channel_content_encoder = DualChannelContentEncoder(
        original_encoder=content_encoder,
        fusion_method=fusion_method,
        learnable_weights=False,
    )
    print("✓ Dual-Channel Content Encoder built successfully.")
    return dual_channel_content_encoder


def get_unet_cross_attention_dim(unet: UNet) -> int:
    """Infer cross-attention dimension from U-Net."""
    print("Inferring cross-attention dimension from U-Net...")
    if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
        return unet.config.cross_attention_dim

    for module in unet.modules():
        if hasattr(module, "to_k") and isinstance(module.to_k, nn.Linear):
            return module.to_k.in_features

    logger.warning(
        "Cross-attention dimension not found in U-Net. Using default value of 1024."
    )
    return 1024


def build_identity_loss_module(args: argparse.Namespace) -> IdentityMappingLoss:
    """Build identity mapping loss module."""
    print("Building Identity Mapping Loss module...")
    identity_loss = IdentityMappingLoss(
        matrix_size=getattr(args, "fst_num_queries", 256),
        loss_type=getattr(args, "identity_loss_type", "frobenius"),
        regularization=getattr(args, "identity_regularization", "orthogonal"),
        reg_weight=getattr(args, "identity_reg_weight", 0.01),
    )
    print("✓ Identity Mapping Loss module built successfully.")
    return identity_loss


def build_style_discriminator(args: argparse.Namespace) -> StyleDiscriminator:
    """Build style discriminator for adversarial content-style disentanglement.

    Args:
        args (argparse.Namespace): Configuration arguments with:
            - use_adversarial_disc: Whether to use adversarial training
            - num_styles: Number of style families
            - adversarial_hidden_dims: Hidden layer dimensions (optional)
            - adversarial_dropout: Dropout probability (optional)

    Returns:
        StyleDiscriminator: Style discriminator module instance.
    """
    if not getattr(args, "use_adversarial_disc", False):
        print("Adversarial discriminator disabled - skipping build")
        return None

    print("Building Style Discriminator for adversarial training...")

    num_styles = getattr(args, "num_styles", 10)
    input_channels = getattr(args, "adversarial_input_channels", 256)
    hidden_dims = getattr(args, "adversarial_hidden_dims", [512, 256, 128])
    dropout = getattr(args, "adversarial_dropout", 0.3)
    use_spectral_norm = getattr(args, "adversarial_spectral_norm", True)

    discriminator = StyleDiscriminator(
        input_channels=input_channels,
        num_styles=num_styles,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_spectral_norm=use_spectral_norm,
    )

    print(
        f"✓ Style Discriminator built successfully "
        f"({num_styles} styles, input_channels={input_channels})."
    )
    return discriminator


def build_adversarial_loss_module(
    args: argparse.Namespace,
) -> AdversarialContentStyleLoss:
    """Build adversarial loss module for content-style disentanglement.

    Args:
        args (argparse.Namespace): Configuration arguments with:
            - num_styles: Number of style families
            - adversarial_weight: Weight for adversarial loss
            - adversarial_entropy_weight: Weight for entropy regularization
            - adversarial_lambda: Gradient reversal strength

    Returns:
        AdversarialContentStyleLoss: Adversarial loss module instance.
    """
    if not getattr(args, "use_adversarial_disc", False):
        print("Adversarial discriminator disabled - skipping loss module build")
        return None

    print("Building Adversarial Content-Style Loss module...")

    num_styles = getattr(args, "num_styles", 10)
    adversarial_weight = getattr(args, "adversarial_weight", 0.5)
    entropy_weight = getattr(args, "adversarial_entropy_weight", 0.1)
    gradient_reversal_lambda = getattr(args, "adversarial_lambda", 1.0)

    loss_module = AdversarialContentStyleLoss(
        num_styles=num_styles,
        adversarial_weight=adversarial_weight,
        entropy_weight=entropy_weight,
        gradient_reversal_lambda=gradient_reversal_lambda,
    )

    print(
        f"✓ Adversarial Loss module built successfully "
        f"(weight={adversarial_weight}, entropy_weight={entropy_weight})."
    )
    return loss_module


def build_style_label_extractor(args: argparse.Namespace) -> StyleLabelExtractor:
    """Build style label extractor for filename-based style annotation.

    Args:
        args (argparse.Namespace): Configuration arguments (minimal deps)

    Returns:
        StyleLabelExtractor: Style label extractor instance.
    """
    if not getattr(args, "use_adversarial_disc", False):
        return None

    print("Building Style Label Extractor...")

    extractor = StyleLabelExtractor()

    print("✓ Style Label Extractor built successfully.")
    return extractor


def load_components(components: dict, args: argparse.Namespace) -> None:
    """
    Load state_dict for each module in components from its checkpoint in args.ckpt_dir.

    Args:
        components (dict): Mapping of module name (str) to module object.
        args: Namespace or object with 'ckpt_dir' attribute.
    """
    for name, module in components.items():
        ckpt_path = os.path.join(args.ckpt_dir, f"{name}.safetensors")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.ckpt_dir, f"{name}.pth")
        if os.path.exists(ckpt_path):
            state_dict = load_state_dict_auto(ckpt_path)
            module.load_state_dict(state_dict)
            logger.info(f"✓ Loaded weights for '{name}' from {ckpt_path}")
        else:
            logger.warning(f"⚠ Checkpoint for '{name}' not found in {args.ckpt_dir}")


def build_frequency_decomposition(args: argparse.Namespace) -> FrequencyDecomposition:
    """Build frequency decomposition module.

    Args:
        args (argparse.Namespace): Configuration arguments with:
            - frequency_image_size: Input image size (default: 96)
            - frequency_low_cutoff: Low/mid boundary (default: 0.10)
            - frequency_mid_cutoff: Mid/high boundary (default: 0.40)
            - frequency_filter_type: Filter type (default: "gaussian")
            - frequency_normalize_bands: Whether to normalize bands (default: True)
            - frequency_return_fft: Return FFT spectrums (default: False)

    Returns:
        FrequencyDecomposition: Frequency decomposition module instance.
    """
    print("Building Frequency Decomposition module...")

    image_size = getattr(args, "frequency_image_size", 96)
    low_cutoff = getattr(args, "frequency_low_cutoff", 0.10)
    mid_cutoff = getattr(args, "frequency_mid_cutoff", 0.40)
    filter_type = getattr(args, "frequency_filter_type", "gaussian")
    normalize_bands = getattr(args, "frequency_normalize_bands", True)
    return_fft = getattr(args, "frequency_return_fft", False)

    freq_decomp = FrequencyDecomposition(
        image_size=image_size,
        low_cutoff=low_cutoff,
        mid_cutoff=mid_cutoff,
        filter_type=filter_type,
        normalize_bands=normalize_bands,
        return_fft=return_fft,
    )

    print(
        f"✓ Frequency Decomposition built successfully "
        f"(low_cutoff={low_cutoff}, mid_cutoff={mid_cutoff}, filter={filter_type})."
    )
    return freq_decomp


def build_frequency_encoder_wrapper(
    args: argparse.Namespace,
    content_encoder: ContentEncoder,
    style_encoder: StyleEncoder,
) -> MultiScaleFrequencyEncoder:
    """Build multi-scale frequency encoder wrapper.

    Wraps content and style encoders to process frequency-decomposed inputs.

    Args:
        args (argparse.Namespace): Configuration arguments with:
            - frequency_image_size: Input image size (default: 96)
            - frequency_low_cutoff: Low/mid boundary (default: 0.10)
            - frequency_mid_cutoff: Mid/high boundary (default: 0.40)
            - frequency_use_mid_band: Use mid-frequency band (default: True)
            - frequency_mid_target: Where to send mid-band (default: "both")
        content_encoder: Content encoder to wrap
        style_encoder: Style encoder to wrap

    Returns:
        MultiScaleFrequencyEncoder: Wrapped encoder instance.
    """
    print("Building Multi-Scale Frequency Encoder wrapper...")

    image_size = getattr(args, "frequency_image_size", 96)
    low_cutoff = getattr(args, "frequency_low_cutoff", 0.10)
    mid_cutoff = getattr(args, "frequency_mid_cutoff", 0.40)
    use_mid_band = getattr(args, "frequency_use_mid_band", True)
    mid_band_target = getattr(args, "frequency_mid_target", "both")

    freq_encoder = MultiScaleFrequencyEncoder(
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        image_size=image_size,
        low_cutoff=low_cutoff,
        mid_cutoff=mid_cutoff,
        use_mid_band=use_mid_band,
        mid_band_target=mid_band_target,
    )

    print(
        f"✓ Frequency Encoder wrapper built successfully "
        f"(use_mid_band={use_mid_band}, mid_target={mid_band_target})."
    )
    return freq_encoder
