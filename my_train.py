"""
FontDiffuser Training Script
✅ Uses hash-based file naming with unicode characters
✅ Discovers images from filesystem (not from checkpoint paths)
✅ Compatible with results_checkpoint.json structure
"""

import os
import math
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import (
    RandomAffine,
    ColorJitter,
    RandomRotation,
    GaussianBlur,
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (
    FontDiffuserModel,
    ContentPerceptualLoss,
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
)
from utils import (
    save_args_to_yaml,
    x0_from_epsilon,
    reNormalize_img,
    normalize_mean_std,
)

try:
    from safetensors.torch import load_file as load_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors not installed. Only .pth files will be supported.")


logger = get_logger(__name__)


# ============================================================================
# Hash-based filename utilities
# ============================================================================


def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """Compute deterministic hash for a (character, style, font) combination"""
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def parse_content_filename(filename: str) -> Optional[str]:
    """
    Parse content filename to extract character
    Format: U+XXXX_[char]_hash.png or U+XXXX_hash.png

    Returns:
        Character string or None if parsing fails
    """
    parts = filename.replace(".png", "").split("_")

    if len(parts) < 2:
        return None

    # First part should be codepoint
    codepoint = parts[0]
    if not codepoint.startswith("U+"):
        return None

    try:
        # Decode character from codepoint
        char_code = int(codepoint.replace("U+", ""), 16)
        char = chr(char_code)
        return char
    except (ValueError, OverflowError):
        return None


def parse_target_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse target filename to extract character and style
    Format: U+XXXX_[char]_style_hash.png or U+XXXX_style_hash.png

    Returns:
        (character, style) tuple or None if parsing fails
    """
    parts = filename.replace(".png", "").split("_")

    if len(parts) < 3:
        return None

    # First part should be codepoint
    codepoint = parts[0]
    if not codepoint.startswith("U+"):
        return None

    try:
        # Decode character from codepoint
        char_code = int(codepoint.replace("U+", ""), 16)
        char = chr(char_code)

        # Hash is always last part
        hash_val = parts[-1]

        # Check if safe char is present (second part)
        safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""

        if len(parts) >= 4 and parts[1] == safe_char:
            # Format: U+XXXX_char_style_hash
            style = "_".join(parts[2:-1])
        else:
            # Format: U+XXXX_style_hash
            style = "_".join(parts[1:-1])

        return char, style

    except (ValueError, OverflowError, IndexError):
        return None


# ============================================================================
# Dynamic path discovery (filesystem-based)
# ============================================================================


def discover_content_images(content_dir: str) -> Dict[str, str]:
    """
    Discover content images from actual filesystem
    ✅ Uses hash-based naming

    Returns:
        Dict mapping character -> image_path
    """
    char_images = {}
    content_path = Path(content_dir)

    if not content_path.exists():
        raise FileNotFoundError(f"ContentImage directory not found: {content_dir}")

    print(f"\n🔍 Discovering content images from {content_dir}...")

    for img_file in tqdm(
        sorted(content_path.glob("*.png")),
        desc="Scanning content",
        ncols=100,
        unit="file",
    ):
        char = parse_content_filename(img_file.name)

        if char is None:
            continue

        # Store by character (not index)
        char_images[char] = str(img_file)

    if not char_images:
        raise ValueError(f"No content images found in {content_dir}")

    print(f"  ✓ Found {len(char_images)} content images")
    return char_images


def discover_target_images(target_dir: str) -> Dict[Tuple[str, str], str]:
    """
    Discover target images from actual filesystem
    ✅ Uses hash-based naming

    Returns:
        Dict mapping (character, style) -> image_path
    """
    target_images = {}
    target_path = Path(target_dir)

    if not target_path.exists():
        raise FileNotFoundError(f"TargetImage directory not found: {target_dir}")

    print(f"\n🔍 Discovering target images from {target_dir}...")

    # Iterate through style directories
    for style_dir in tqdm(
        sorted(target_path.iterdir()), desc="Scanning styles", ncols=100, unit="dir"
    ):
        if not style_dir.is_dir():
            continue

        style_name = style_dir.name

        # Find target images in this style directory
        for img_file in style_dir.glob("*.png"):
            parsed = parse_target_filename(img_file.name)

            if parsed is None:
                continue

            char, parsed_style = parsed

            # Validate style matches directory
            if parsed_style != style_name:
                tqdm.write(
                    f"  ⚠️  Style mismatch: {img_file.name} has style '{parsed_style}' "
                    f"but is in directory '{style_name}'"
                )
                continue

            target_images[(char, style_name)] = str(img_file)

    if not target_images:
        raise ValueError(f"No target images found in {target_dir}")

    print(f"  ✓ Found {len(target_images)} target images")
    return target_images


def validate_image_paths(
    content_images: Dict[str, str],
    target_images: Dict[Tuple[str, str], str],
) -> None:
    """
    ✅ Validate that all content images have corresponding targets
    """
    print(f"\n📋 Validating image paths...")

    # Find which characters have content images
    content_chars = set(content_images.keys())

    # Find which (char, style) pairs exist
    existing_pairs = set(target_images.keys())

    # Extract unique styles and characters from existing pairs
    existing_styles = set(style for char, style in existing_pairs)
    target_chars = set(char for char, style in existing_pairs)

    print(f"  Content images: {len(content_chars)} characters")
    print(f"  Target images: {len(target_images)} (char, style) pairs")
    print(f"  Unique styles: {len(existing_styles)}")
    print(f"  Unique characters in targets: {len(target_chars)}")

    # Find mismatches
    missing_content = target_chars - content_chars
    unused_content = content_chars - target_chars

    if missing_content:
        print(f"  ⚠️  {len(missing_content)} target chars missing content images")
        # Show sample
        sample = list(missing_content)[:5]
        print(f"      Examples: {sample}")

    if unused_content:
        print(f"  ⚠️  {len(unused_content)} content images have no targets")
        # Show sample
        sample = list(unused_content)[:5]
        print(f"      Examples: {sample}")


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def validate(model, val_dataloader, noise_scheduler, accelerator, args, global_step):
    """Run validation loop and return average validation loss"""
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for val_samples in val_dataloader:
            content_images = val_samples["content_image"]
            style_images = val_samples["style_image"]
            target_images = val_samples["target_image"]

            noise = torch.randn_like(target_images)
            bsz = target_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=target_images.device,
            ).long()
            noisy_target_images = noise_scheduler.add_noise(
                target_images, noise, timesteps
            )

            # Classifier-free validation (no masking)
            noise_pred, _ = model(
                x_t=noisy_target_images,
                timesteps=timesteps,
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )

            batch_val_loss = F.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )
            val_loss += batch_val_loss.item()
            num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss


def load_checkpoint_file(checkpoint_path: str, device="cpu"):
    """
    Load checkpoint from either .pth or .safetensors format

    Args:
        checkpoint_path: Path to checkpoint file (.pth or .safetensors)
        device: Device to load to

    Returns:
        Loaded state dict

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If unsupported file format
    """
    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    file_ext = os.path.splitext(checkpoint_path)[1].lower()

    try:
        if file_ext == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ValueError(
                    f"❌ safetensors format not supported.\n"
                    f"   Install: pip install safetensors\n"
                    f"   Or use .pth format instead"
                )

            print(f"  📂 Loading safetensors: {os.path.basename(checkpoint_path)}")
            state_dict = load_safetensors(checkpoint_path)
            return state_dict

        elif file_ext == ".pth":
            print(f"  📂 Loading .pth: {os.path.basename(checkpoint_path)}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            return state_dict

        else:
            raise ValueError(
                f"❌ Unsupported checkpoint format: {file_ext}\n"
                f"   Supported: .pth, .safetensors\n"
                f"   Got: {checkpoint_path}"
            )

    except Exception as e:
        raise RuntimeError(f"❌ Error loading checkpoint {checkpoint_path}: {e}")


def load_phase_1_checkpoint(args, unet, style_encoder, content_encoder):
    """
    Load Phase 1 checkpoint (unet, style_encoder, content_encoder)
    Handles both .pth and .safetensors formats

    Args:
        args: Training arguments
        unet: UNet model to load state into
        style_encoder: Style encoder model to load state into
        content_encoder: Content encoder model to load state into
    """
    if not args.phase_2:
        return

    ckpt_dir = Path(args.phase_1_ckpt_dir)

    if not ckpt_dir.exists():
        raise ValueError(f"❌ Phase 1 checkpoint directory not found: {ckpt_dir}")

    print(f"\n📥 Loading Phase 1 checkpoint from: {ckpt_dir}")
    print("=" * 60)

    # Define checkpoint files to load (try both formats)
    checkpoint_configs = [
        ("unet", unet),
        ("style_encoder", style_encoder),
        ("content_encoder", content_encoder),
    ]

    for model_name, model in checkpoint_configs:
        # Try .safetensors first, then .pth
        safetensors_path = ckpt_dir / f"{model_name}.safetensors"
        pth_path = ckpt_dir / f"{model_name}.pth"

        checkpoint_path = None

        if safetensors_path.exists():
            checkpoint_path = safetensors_path
        elif pth_path.exists():
            checkpoint_path = pth_path
        else:
            raise FileNotFoundError(
                f"❌ {model_name} checkpoint not found!\n"
                f"   Tried: {safetensors_path}\n"
                f"          {pth_path}"
            )

        try:
            state_dict = load_checkpoint_file(str(checkpoint_path))
            model.load_state_dict(state_dict)
            print(f"  ✓ {model_name}: {checkpoint_path.name}")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading {model_name}: {e}")

    print("=" * 60)
    print("✓ Phase 1 checkpoint loaded successfully\n")


def load_scr_checkpoint(scr_ckpt_path: str, scr_model):
    """
    Load SCR (Style-Content contrastive) module checkpoint
    Handles both .pth and .safetensors formats

    Args:
        scr_ckpt_path: Path to SCR checkpoint
        scr_model: SCR model to load state into
    """
    scr_ckpt_path = Path(scr_ckpt_path)

    if not scr_ckpt_path.exists():
        raise FileNotFoundError(f"❌ SCR checkpoint not found: {scr_ckpt_path}")

    print(f"📥 Loading SCR module from: {scr_ckpt_path}")

    try:
        state_dict = load_checkpoint_file(str(scr_ckpt_path))
        scr_model.load_state_dict(state_dict)
        print(f"  ✓ SCR module loaded: {scr_ckpt_path.name}\n")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading SCR module: {e}")


def save_checkpoint(model, accelerator, args, global_step, is_best=False):
    """Save model checkpoint in both .pth and .safetensors format"""
    if is_best:
        save_dir = f"{args.output_dir}/best_checkpoint"
    else:
        save_dir = f"{args.output_dir}/global_step_{global_step}"

    os.makedirs(save_dir, exist_ok=True)

    # Save as .pth (original format)
    torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
    torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
    torch.save(model, f"{save_dir}/total_model.pth")

    # ✅ ALSO save as .safetensors if available
    if SAFETENSORS_AVAILABLE:
        try:
            from safetensors.torch import save_file as save_safetensors

            save_safetensors(model.unet.state_dict(), f"{save_dir}/unet.safetensors")
            save_safetensors(
                model.style_encoder.state_dict(),
                f"{save_dir}/style_encoder.safetensors",
            )
            save_safetensors(
                model.content_encoder.state_dict(),
                f"{save_dir}/content_encoder.safetensors",
            )
        except Exception as e:
            print(f"⚠️  Could not save safetensors format: {e}")

    logging.info(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
        f"Saved checkpoint at global step {global_step}"
    )
    print(f"✓ Saved checkpoint at global step {global_step}")


def main():
    args = get_args()

    # Add missing arguments with defaults
    if not hasattr(args, "val_interval"):
        args.val_interval = 100  # Validate every 100 steps

    # ===== Validation: Check checkpoint interval vs max_train_steps =====
    if args.ckpt_interval > args.max_train_steps:
        raise ValueError(
            f"❌ ERROR: ckpt_interval ({args.ckpt_interval}) is larger than "
            f"max_train_steps ({args.max_train_steps})!\n"
            f"   Set ckpt_interval to a value < {args.max_train_steps}\n"
            f"   Suggested: ckpt_interval = {args.max_train_steps // 4}"
        )

    if args.max_train_steps % args.ckpt_interval != 0:
        recommended = (args.max_train_steps // args.ckpt_interval) * args.ckpt_interval
        print(
            f"⚠️  WARNING: max_train_steps ({args.max_train_steps}) is not divisible by "
            f"ckpt_interval ({args.ckpt_interval}).\n"
            f"   Last checkpoint will be at step {recommended}, final model will be saved separately."
        )
        logging.warning(
            f"max_train_steps ({args.max_train_steps}) not divisible by "
            f"ckpt_interval ({args.ckpt_interval})"
        )

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # ============================================================================
    # ✅ DISCOVER IMAGES FROM FILESYSTEM (hash-based naming)
    # ============================================================================
    print(f"\n📂 Discovering images from filesystem...")
    print("=" * 60)

    content_dir = os.path.join(args.data_root, "train", "ContentImage")
    target_dir = os.path.join(args.data_root, "train", "TargetImage")

    content_images = discover_content_images(content_dir)
    target_images = discover_target_images(target_dir)
    validate_image_paths(content_images, target_images)

    print("=" * 60)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    # ✅ Load Phase 1 checkpoint with proper error handling
    if args.phase_2:
        try:
            load_phase_1_checkpoint(args, unet, style_encoder, content_encoder)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"\n{e}")
            logging.error(str(e))
            raise

    model = FontDiffuserModel(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )

    # Build content perceptual loss
    perceptual_loss = ContentPerceptualLoss()

    scr = None
    if args.phase_2:
        scr = build_scr(args=args)
        try:
            load_scr_checkpoint(args.scr_ckpt_path, scr)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n{e}")
            logging.error(str(e))
            raise

        scr.requires_grad_(False)
        logging.info(f"Loaded SCR module from: {args.scr_ckpt_path}")

    # Load the training dataset with augmentation
    content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            RandomRotation(degrees=5),
            RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-5, 5)),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    style_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            RandomRotation(degrees=5),
            ColorJitter(brightness=0.15, contrast=0.15),
            GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            RandomRotation(degrees=3),
            ColorJitter(brightness=0.1, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_font_dataset = FontDataset(
        args=args,
        phase="train",
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Load validation dataset (without augmentation)
    val_content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_style_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_font_dataset = FontDataset(
        args=args,
        phase="val",
        transforms=[
            val_content_transforms,
            val_style_transforms,
            val_target_transforms,
        ],
        scr=args.phase_2,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_font_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Build optimizer and learning rate scheduler
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    # Move SCR module to target device
    if scr is not None:
        scr = scr.to(accelerator.device)

    # Initialize trackers and save config
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.experience_name,
            config={
                "max_train_steps": args.max_train_steps,
                "train_batch_size": args.train_batch_size,
                "learning_rate": args.learning_rate,
                "perceptual_coefficient": args.perceptual_coefficient,
                "offset_coefficient": args.offset_coefficient,
                "sc_coefficient": args.sc_coefficient if args.phase_2 else 0,
                "phase_2": args.phase_2,
                "seed": args.seed,
            },
        )
        save_args_to_yaml(
            args=args,
            output_file=f"{args.output_dir}/{args.experience_name}_config.yaml",
        )

    # Setup progress bar
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Training epoch setup
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    best_val_loss = float("inf")

    logging.info(
        f"Starting training with {num_train_epochs} epochs, "
        f"{num_update_steps_per_epoch} steps per epoch"
    )
    logging.info(
        f"Max training steps: {args.max_train_steps}, "
        f"Checkpoint interval: {args.ckpt_interval}"
    )

    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.0
        epoch_diff_loss = 0.0
        epoch_percep_loss = 0.0
        epoch_offset_loss = 0.0
        epoch_sc_loss = 0.0
        epoch_steps = 0

        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            with accelerator.accumulate(model):
                # Sample noise
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=target_images.device,
                ).long()

                # Add noise (forward diffusion)
                noisy_target_images = noise_scheduler.add_noise(
                    target_images, noise, timesteps
                )

                # Classifier-free guidance training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value == 1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict noise and offset
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                )

                # Calculate individual losses
                # ...existing code...

                # Calculate individual losses
                diff_loss = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )

                # Perceptual loss (content preservation)
                pred_original = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                )
                norm_pred_ori = normalize_mean_std(pred_original)

                percep_loss = perceptual_loss(
                    reNormalize_img(content_images), norm_pred_ori
                )

                # Offset loss (style consistency)
                offset_loss = torch.abs(offset_out_sum).mean()

                # Total loss (Phase 1)
                loss = (
                    diff_loss
                    + args.perceptual_coefficient * percep_loss
                    + args.offset_coefficient * offset_loss
                )

                # ✅ Phase 2: Add Style-Content Contrastive (SC) loss
                sc_loss = torch.tensor(0.0).to(accelerator.device)
                if args.phase_2 and scr is not None:
                    neg_images = samples.get("neg_images", None)
                    if neg_images is not None:
                        # Compute SC loss
                        sc_loss = scr(
                            content_images=content_images,
                            style_images=style_images,
                            neg_images=neg_images,
                            noise_pred=noise_pred,
                            timesteps=timesteps,
                        )
                        loss = loss + args.sc_coefficient * sc_loss

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Track losses
            epoch_train_loss += loss.item()
            epoch_diff_loss += diff_loss.item()
            epoch_percep_loss += percep_loss.item()
            epoch_offset_loss += offset_loss.item()
            if args.phase_2:
                epoch_sc_loss += sc_loss.item()
            epoch_steps += 1

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log training metrics
                logs = {
                    "train_loss": loss.item(),
                    "diff_loss": diff_loss.item(),
                    "percep_loss": percep_loss.item(),
                    "offset_loss": offset_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.phase_2:
                    logs["sc_loss"] = sc_loss.item()

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # ✅ Run validation
                if global_step % args.val_interval == 0:
                    avg_val_loss = validate(
                        model,
                        val_dataloader,
                        noise_scheduler,
                        accelerator,
                        args,
                        global_step,
                    )

                    accelerator.log({"val_loss": avg_val_loss}, step=global_step)

                    logging.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                        f"Step {global_step}: val_loss={avg_val_loss:.4f}"
                    )

                    # Save best checkpoint
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if accelerator.is_main_process:
                            save_checkpoint(
                                model, accelerator, args, global_step, is_best=True
                            )
                            print(
                                f"✓ New best model saved! val_loss={avg_val_loss:.4f}"
                            )
                            logging.info(
                                f"Saved best checkpoint at step {global_step} "
                                f"with val_loss={avg_val_loss:.4f}"
                            )

                # ✅ Save regular checkpoints
                if global_step % args.ckpt_interval == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(model, accelerator, args, global_step)

            # Stop if max steps reached
            if global_step >= args.max_train_steps:
                break

        # Epoch summary
        avg_epoch_loss = epoch_train_loss / max(epoch_steps, 1)
        avg_diff_loss = epoch_diff_loss / max(epoch_steps, 1)
        avg_percep_loss = epoch_percep_loss / max(epoch_steps, 1)
        avg_offset_loss = epoch_offset_loss / max(epoch_steps, 1)

        log_msg = (
            f"Epoch {epoch + 1}/{num_train_epochs} completed - "
            f"avg_loss: {avg_epoch_loss:.4f}, "
            f"diff: {avg_diff_loss:.4f}, "
            f"percep: {avg_percep_loss:.4f}, "
            f"offset: {avg_offset_loss:.4f}"
        )

        if args.phase_2:
            avg_sc_loss = epoch_sc_loss / max(epoch_steps, 1)
            log_msg += f", sc: {avg_sc_loss:.4f}"

        logging.info(log_msg)
        print(f"\n{log_msg}")

        if global_step >= args.max_train_steps:
            break

    # ✅ Save final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, args, global_step, is_best=False)
        print(f"\n✓ Training complete! Final checkpoint saved at step {global_step}")
        logging.info(f"Training completed at step {global_step}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
