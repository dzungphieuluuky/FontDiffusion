"""
Optimized sampling for FontDiffuser with SAFE optimizations
Uses hash-based file naming with unicode characters
Multi-character batch processing
Multi-font support
"""

import logging
import os
import time
from PIL import Image
from pathlib import Path
from typing import Optional, Any
from functools import lru_cache
from argparse import Namespace, ArgumentParser

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
    build_fst,
    build_mss_encoder,
    UNet,
    ContentEncoder,
    StyleEncoder,
)
from src.tools.utils import (
    ttf2im,
    load_ttf,
    is_char_in_font,
    save_args_to_yaml,
)

from src.tools.filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)

logger = logging.getLogger(__name__)


def arg_parse() -> Namespace:
    """Parse command line arguments"""
    from src.configs.fontdiffuser import get_parser

    parser: ArgumentParser = get_parser()

    # Original arguments
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--controlnet",
        type=bool,
        default=False,
        help="If in demo mode, the controlnet can be added.",
    )
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument(
        "--content_character",
        type=str,
        default=None,
        help="Single character, comma-separated list, or path to txt file",
    )
    parser.add_argument(
        "--characters_file",
        type=str,
        default=None,
        help="Path to text file with one character per line",
    )
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--save_image_dir", type=str, default=None, help="The saving directory."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--ttf_path",
        type=str,
        default="ttf/KaiXinSongA.ttf",
        help="Path to single TTF file or directory with multiple fonts",
    )

    # SAFE optimization arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 precision (SAFE - applied after loading weights)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing multiple characters",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        default=False,
        help="Use channels-last memory format (SAFE)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic algorithms for reproducibility",
    )
    
    # FST-specific arguments
    parser.add_argument(
        "--use_fst",
        action="store_true",
        default=False,
        help="Use FST-enhanced model for improved style transfer",
    )
    parser.add_argument(
        "--fst_ckpt_path",
        type=str,
        default=None,
        help="Path to FST module checkpoint (optional)",
    )
    parser.add_argument(
        "--fst_num_queries",
        type=int,
        default=256,
        help="Number of learnable queries in FST module",
    )
    parser.add_argument(
        "--fst_query_dim",
        type=int,
        default=128,
        help="Dimension of FST queries",
    )
    parser.add_argument(
        "--fst_num_scales",
        type=int,
        default=5,
        help="Number of scales in MSSE",
    )
    
    args: Namespace = parser.parse_args()

    style_image_size: int = getattr(args, "style_image_size", 96)
    content_image_size: int = getattr(args, "content_image_size", 96)
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


class FontManager:
    """Manages single or multiple font files"""

    def __init__(self, ttf_path: str) -> None:
        self.fonts: dict[str, dict[str, Any]] = {}
        self.font_paths: list[str] = []
        self._load_fonts(ttf_path)

    def _load_fonts(self, ttf_path: str) -> None:
        """Load font(s) from path"""
        if os.path.isfile(ttf_path):
            # Single font file
            self.font_paths = [ttf_path]
            font_name: str = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": None,  # Lazy load
                "name": font_name,
            }
            logger.info(f"✓ Font loaded: {font_name}")

        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions: set = {".ttf", ".otf", ".TTF", ".OTF"}
            font_files: list[str] = [
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_extensions
            ]

            if not font_files:
                raise ValueError(f"No font files found in directory: {ttf_path}")

            self.font_paths = sorted(font_files)

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Loading {len(font_files)} fonts from directory...")
            logger.info("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                self.fonts[font_name] = {
                    "path": font_path,
                    "font": None,  # Lazy load
                    "name": font_name,
                }
                logger.info(f"✓ {font_name}")

            logger.info("=" * 60)
            logger.info(f"Loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")

    def get_font_names(self) -> list[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())

    @lru_cache(maxsize=32)
    def get_font(self, font_name: str) -> Any:
        """Get font object by name (cached)"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")

        # Lazy load font
        if self.fonts[font_name]["font"] is None:
            self.fonts[font_name]["font"] = load_ttf(self.fonts[font_name]["path"])

        return self.fonts[font_name]["font"]

    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["path"]

    @lru_cache(maxsize=1024)
    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font (cached)"""
        font_path: str = self.get_font_path(font_name)
        return is_char_in_font(font_path=font_path, char=char)

    def get_available_chars_for_font(
        self, font_name: str, characters: list[str]
    ) -> list[str]:
        """Get list of characters available in specific font"""
        return [char for char in characters if self.is_char_in_font(font_name, char)]


def parse_characters(
    content_character: str = None, characters_file: str = None
) -> list[str]:
    """
    Parse character input from multiple sources

    Args:
        content_character: Single character, comma-separated list, or path to txt file
        characters_file: Path to text file with one character per line

    Returns:
        list of individual characters
    """
    chars: list[str] = []

    # Priority 1: characters_file argument
    if characters_file and os.path.isfile(characters_file):
        logger.info(f"Loading characters from file: {characters_file}")
        with open(characters_file, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped: str = line.strip()
                if line_stripped and not line_stripped.startswith("#"):
                    if len(line_stripped) == 1:
                        chars.append(line_stripped)
                    else:
                        chars.extend(list(line_stripped))
        logger.info(f"  Loaded {len(chars)} characters")
        return chars

    # Priority 2: content_character argument
    if content_character:
        # Check if it's a file path
        if os.path.isfile(content_character):
            logger.info(f"Loading characters from file: {content_character}")
            with open(content_character, "r", encoding="utf-8") as f:
                for line in f:
                    line_stripped: str = line.strip()
                    if line_stripped and not line_stripped.startswith("#"):
                        if len(line_stripped) == 1:
                            chars.append(line_stripped)
                        else:
                            chars.extend(list(line_stripped))
            logger.info(f"  Loaded {len(chars)} characters")
            return chars

        # Check if comma-separated
        if "," in content_character:
            chars = [c.strip() for c in content_character.split(",") if c.strip()]
        else:
            # Single character
            stripped: str = content_character.strip()
            chars = [stripped] if len(stripped) == 1 else list(stripped)

    return chars


def get_content_transform(content_image_size: tuple[int, int]) -> transforms.Compose:
    """Cached content transform"""
    return transforms.Compose(
        [
            transforms.Resize(
                content_image_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def get_style_transform(style_image_size: tuple[int, int]) -> transforms.Compose:
    """Cached style transform"""
    return transforms.Compose(
        [
            transforms.Resize(
                style_image_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_state_dict_auto(path: str):
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors files.")
        return safe_load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def load_fontdiffuser_pipeline(args: Namespace, use_fst: bool = False) -> FontDiffuserDPMPipeline:
    """Load Font Diffuser pipeline with optimizations"""
    logger.info(f"Loading FontDiffuser{'WithFST' if use_fst else ''} pipeline...")

    # Load base components
    unet: UNet = build_unet(args=args)
    unet_ckpt_path = (
        f"{args.ckpt_dir}/unet.safetensors"
        if os.path.exists(f"{args.ckpt_dir}/unet.safetensors")
        else f"{args.ckpt_dir}/unet.pth"
    )
    unet.load_state_dict(load_state_dict_auto(unet_ckpt_path))

    style_encoder: StyleEncoder = build_style_encoder(args=args)
    style_encoder_ckpt_path = (
        f"{args.ckpt_dir}/style_encoder.safetensors"
        if os.path.exists(f"{args.ckpt_dir}/style_encoder.safetensors")
        else f"{args.ckpt_dir}/style_encoder.pth"
    )
    style_encoder.load_state_dict(load_state_dict_auto(style_encoder_ckpt_path))

    content_encoder: ContentEncoder = build_content_encoder(args=args)
    content_encoder_ckpt_path = (
        f"{args.ckpt_dir}/content_encoder.safetensors"
        if os.path.exists(f"{args.ckpt_dir}/content_encoder.safetensors")
        else f"{args.ckpt_dir}/content_encoder.pth"
    )
    content_encoder.load_state_dict(load_state_dict_auto(content_encoder_ckpt_path))

    logger.info("✓ Loaded model state_dict successfully")

    if use_fst:
        from src.model import FontDiffuserModelDPMWithFST
        
        # Load FST-specific weights if available
        if hasattr(args, "fst_ckpt_path") and args.fst_ckpt_path:
            logger.info(f"Loading FST weights from {args.fst_ckpt_path}...")
            fst_state_dict = load_state_dict_auto(args.fst_ckpt_path)
        if hasattr(args, "mss_ckpt_path") and args.mss_ckpt_path:
            logger.info(f"Loading MSS weights from {args.mss_ckpt_path}...")
            mss_state_dict = load_state_dict_auto(args.mss_ckpt_path)

        fst_module = build_fst(args=args)
        mss_encoder = build_mss_encoder(args=args)

        model: FontDiffuserModelDPMWithFST = FontDiffuserModelDPMWithFST(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
            fst_module=fst_module,
            mss_encoder=mss_encoder,
            feature_channels=getattr(args, "fst_feature_channels", None),
            num_queries=getattr(args, "fst_num_queries", 256),
            query_dim=getattr(args, "fst_query_dim", 128),
            num_scales=getattr(args, "fst_num_scales", 5),
        )
    else:
        model: FontDiffuserModelDPM = FontDiffuserModelDPM(
            unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
        )

    # Apply FP16 conversion AFTER model creation
    if getattr(args, "fp16", False):
        logger.info("Converting to FP16 precision...")
        model = model.half()
        logger.info("✓ Converted to FP16")

    # SAFE: Apply channels-last memory format
    if getattr(args, "channels_last", False):
        logger.info("Converting to channels-last memory format...")
        model = model.to(memory_format=torch.channels_last)
        logger.info("✓ Converted to channels-last")

    # Apply torch.compile if requested
    if getattr(args, "compile", False):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        logger.info("✓ Model compiled")

    # Move to device with proper dtype
    dtype: torch.dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32
    model.to(args.device, dtype=dtype)
    model.eval()

    logger.info("✓ Model moved to device")
    
    # Log model info
    if hasattr(model, "log_model_info"):
        model.log_model_info()

    # Load the training ddpm_scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    logger.info("✓ Loaded training DDPM scheduler successfully")

    # Load the DPM_Solver to generate the sample
    pipe: FontDiffuserDPMPipeline = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=getattr(args, "model_type", "noise"),
        guidance_type=getattr(args, "guidance_type", "classifier-free"),
        guidance_scale=getattr(args, "guidance_scale", 7.5),
    )
    logger.info("✓ Loaded DPM-Solver pipeline successfully")
    return pipe

def sampling_batch(
    args: Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    font_manager: FontManager,
    font_name: str,
    style_image_path: str,
    style_name: str = "style0",
    save_content_images: bool = True,
) -> tuple[list[Image.Image] | list[str] | float]:
    """
    Batch sampling for multiple characters with single font and style
    Uses hash-based file naming
    """
    # Process images in batch
    content_batch, style_batch, content_pils, valid_chars = image_process_batch(
        args, characters, font_manager, font_name, style_image_path
    )

    if (
        content_batch is None
        or valid_chars is None
        or content_pils is None
        or style_batch is None
    ):
        return None, None, 0.0

    # set seed for reproducibility
    if hasattr(args, "seed") and args.seed:
        set_seed(seed=args.seed)

    # Save content images if requested
    if save_content_images and getattr(args, "save_image", False):
        content_dir: str = os.path.join(args.save_image_dir, "ContentImage")
        os.makedirs(content_dir, exist_ok=True)

        for char, pil_img in zip(valid_chars, content_pils):
            content_filename = get_content_filename(char)
            pil_img.save(os.path.join(content_dir, content_filename))

    with torch.no_grad():
        dtype: torch.dtype = (
            torch.float16 if getattr(args, "fp16", False) else torch.float32
        )
        content_batch = content_batch.to(args.device, dtype=dtype)
        style_batch = style_batch.to(args.device, dtype=dtype)

        if getattr(args, "channels_last", False):
            content_batch = content_batch.to(memory_format=torch.channels_last)
            style_batch = style_batch.to(memory_format=torch.channels_last)

        logger.info(f"  Sampling {len(valid_chars)} characters with DPM-Solver++ ...")
        start: float = time.time()

        # Process in batches
        all_images: list[Image.Image] = []
        batch_size: int = getattr(args, "batch_size", 1)

        for i in range(0, len(content_batch), batch_size):
            batch_content: torch.Tensor = content_batch[i : i + batch_size]
            batch_style: torch.Tensor = style_batch[i : i + batch_size]

            images: list[Image.Image] = pipe.generate(
                content_images=batch_content,
                style_images=batch_style,
                batch_size=len(batch_content),
                order=getattr(args, "order", None),
                num_inference_steps=getattr(args, "num_inference_steps", 20),
                content_encoder_downsample_size=getattr(
                    args, "content_encoder_downsample_size", None
                ),
                t_start=getattr(args, "t_start", None),
                t_end=getattr(args, "t_end", None),
                dm_size=getattr(args, "content_image_size", (96, 96)),
                algorithm_type=getattr(args, "algorithm_type", None),
                skip_type=getattr(args, "skip_type", None),
                method=getattr(args, "method", None),
                correcting_x0_fn=getattr(args, "correcting_x0_fn", None),
            )

            all_images.extend(images)

        end: float = time.time()
        inference_time: float = end - start

        # Save generated images with hash-based naming
        if getattr(args, "save_image", False):
            target_dir: str = os.path.join(
                args.save_image_dir, "TargetImage", style_name
            )
            os.makedirs(target_dir, exist_ok=True)

            for char, img in zip(valid_chars, all_images):
                target_filename = get_target_filename(char, style_name)
                img_path: str = os.path.join(target_dir, target_filename)
                img.save(img_path)

        logger.info(
            f"  ✓ Generated {len(all_images)} images in {inference_time:.2f}s ({inference_time / len(all_images):.3f}s/img)"
        )

        return all_images, valid_chars, inference_time


def image_process_batch(
    args: Namespace,
    characters: list[str],
    font_manager: FontManager,
    font_name: str,
    style_image_path: str,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[Image.Image] | None,
    list[str] | None,
]:
    """Process multiple characters in batch"""
    # Load style image
    style_image: Image.Image = Image.open(style_image_path).convert("RGB")
    style_transform: transforms.Compose = get_style_transform(args.style_image_size)

    # Get font
    font: Any = font_manager.get_font(font_name)
    content_transform: transforms.Compose = get_content_transform(
        args.content_image_size
    )

    # Get available characters
    available_chars: list[str] = font_manager.get_available_chars_for_font(
        font_name, characters
    )

    if not available_chars:
        logger.info(f"Warning: No characters available in font '{font_name}'")
        return None, None, None, None

    # Generate content images
    content_images: list[torch.Tensor] = []
    content_images_pil: list[Image.Image] = []

    for char in available_chars:
        try:
            content_image: Image.Image = ttf2im(font=font, char=char)
            if content_image is None:
                continue
            content_images_pil.append(content_image.copy())
            content_images.append(content_transform(content_image))
        except Exception as e:
            logger.info(f"Error processing character '{char}': {e}")
            continue

    if not content_images:
        return None, None, None, None

    # Stack into batch
    content_batch: torch.Tensor = torch.stack(content_images)
    style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(
        len(content_images), 1, 1, 1
    )

    return content_batch, style_batch, content_images_pil, available_chars


def main() -> None:
    """Main function"""
    args: Namespace = arg_parse()

    logger.info("\n" + "=" * 60)
    logger.info("FontDiffuser Optimized Sampling")
    logger.info("=" * 60)
    logger.info(f"Model: {args.ckpt_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"FST Mode: {getattr(args, 'use_fst', False)}")
    logger.info(f"FP16: {getattr(args, 'fp16', False)}")
    logger.info(f"Channels Last: {getattr(args, 'channels_last', False)}")
    logger.info(f"Compile: {getattr(args, 'compile', False)}")
    logger.info(f"Batch Size: {getattr(args, 'batch_size', 1)}")
    logger.info("=" * 60 + "\n")

    # Load pipeline with FST support
    pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(
        args=args, use_fst=args.use_fst
    )

    # Parse characters
    characters: list[str] = parse_characters(
        getattr(args, "content_character", None), getattr(args, "characters_file", None)
    )

    # Check if multi-character or multi-font mode
    if getattr(args, "character_input", False) and characters:
        if len(characters) > 1 or os.path.isdir(args.ttf_path):
            # Multi-character or multi-font mode
            logger.info(f"\n{'=' * 60}")
            logger.info("BATCH MODE ACTIVATED")
            logger.info(
                f"Characters: {len(characters)} - {characters[:10]}{'...' if len(characters) > 10 else ''}"
            )
            logger.info("=" * 60)

            # Load font manager
            font_manager: FontManager = FontManager(args.ttf_path)
            font_names: list[str] = font_manager.get_font_names()

            if not getattr(args, "demo", False):
                os.makedirs(args.save_image_dir, exist_ok=True)
                save_args_to_yaml(
                    args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml"
                )

            # Determine style name from path
            style_name: str = os.path.splitext(os.path.basename(args.style_image_path))[
                0
            ]

            # Process each font
            total_generated: int = 0
            for font_idx, font_name in enumerate(font_names):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"[Font {font_idx + 1}/{len(font_names)}] {font_name}")
                logger.info("=" * 60)

                # Get available characters
                available: list[str] = font_manager.get_available_chars_for_font(
                    font_name, characters
                )
                logger.info(
                    f"  Available characters: {len(available)}/{len(characters)}"
                )

                if not available:
                    logger.info("  ⚠ Skipping font (no characters available)")
                    continue

                # Sample in batch
                images, valid_chars, inf_time = sampling_batch(
                    args,
                    pipe,
                    characters,
                    font_manager,
                    font_name,
                    args.style_image_path,
                    style_name,
                    save_content_images=True,
                )

                if images is None:
                    continue

                total_generated += len(images)

            logger.info("\n" + "=" * 60)
            logger.info("✓ BATCH PROCESSING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total images generated: {total_generated}")
            logger.info(f"\nOutput structure:")
            logger.info(f"  {args.save_image_dir}/")
            logger.info(f"    ├── ContentImage/")
            logger.info(f"    │   └── U+XXXX_char_hash.png")
            logger.info(f"    └── TargetImage/")
            logger.info(f"        └── {style_name}/")
            logger.info(f"            └── U+XXXX_char_{style_name}_hash.png")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()
