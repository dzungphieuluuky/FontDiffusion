"""
Optimized sampling for FontDiffuser with Hydra configuration
Uses hash-based file naming with unicode characters
Multi-character batch processing
Multi-font support
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision.transforms as transforms
from PIL import Image
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
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

logger = logging.getLogger("OptimizedSampler")


class FontManager:
    """Manages single or multiple font files"""

    def __init__(self, ttf_path: str) -> None:
        self.fonts: dict[str, dict[str, Any]] = {}
        self.font_paths: list[str] = []
        self._load_fonts(ttf_path)

    def _load_fonts(self, ttf_path: str) -> None:
        """Load font(s) from path"""
        if os.path.isfile(ttf_path):
            self.font_paths = [ttf_path]
            font_name: str = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": None,
                "name": font_name,
            }
            logger.info(f"✓ Font loaded: {font_name}")

        elif os.path.isdir(ttf_path):
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
                    "font": None,
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


def parse_characters(content_character: str = None) -> list[str]:
    """Parse character input from various sources"""
    chars: list[str] = []

    if content_character:
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

        if "," in content_character:
            chars = [c.strip() for c in content_character.split(",") if c.strip()]
        else:
            stripped: str = content_character.strip()
            chars = [stripped] if len(stripped) == 1 else list(stripped)

    return chars


def get_content_transform(content_image_size: tuple[int, int]) -> transforms.Compose:
    """Content transform"""
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
    """Style transform"""
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
    """Load state_dict from .pth or .safetensors"""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors files.")
        return safe_load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def load_fontdiffuser_pipeline(cfg: DictConfig) -> FontDiffuserDPMPipeline:
    """Load FontDiffuser pipeline with optimizations"""
    logger.info("Loading FontDiffuser pipeline...")

    # Build components
    unet: UNet = build_unet(cfg=cfg)
    unet_ckpt_path = (
        f"{cfg.ckpt_dir}/unet.safetensors"
        if os.path.exists(f"{cfg.ckpt_dir}/unet.safetensors")
        else f"{cfg.ckpt_dir}/unet.pth"
    )
    unet.load_state_dict(load_state_dict_auto(unet_ckpt_path))

    style_encoder: StyleEncoder = build_style_encoder(cfg=cfg)
    style_encoder_ckpt_path = (
        f"{cfg.ckpt_dir}/style_encoder.safetensors"
        if os.path.exists(f"{cfg.ckpt_dir}/style_encoder.safetensors")
        else f"{cfg.ckpt_dir}/style_encoder.pth"
    )
    style_encoder.load_state_dict(load_state_dict_auto(style_encoder_ckpt_path))

    content_encoder: ContentEncoder = build_content_encoder(cfg=cfg)
    content_encoder_ckpt_path = (
        f"{cfg.ckpt_dir}/content_encoder.safetensors"
        if os.path.exists(f"{cfg.ckpt_dir}/content_encoder.safetensors")
        else f"{cfg.ckpt_dir}/content_encoder.pth"
    )
    content_encoder.load_state_dict(load_state_dict_auto(content_encoder_ckpt_path))

    logger.info("✓ Loaded model state_dict successfully")

    if cfg.fp16:
        logger.info("Converting to FP16 precision...")
        unet = unet.half()
        style_encoder = style_encoder.half()
        content_encoder = content_encoder.half()
        logger.info("✓ Converted to FP16")

    if cfg.channels_last:
        logger.info("Converting to channels-last memory format...")
        unet = unet.to(memory_format=torch.channels_last)
        style_encoder = style_encoder.to(memory_format=torch.channels_last)
        content_encoder = content_encoder.to(memory_format=torch.channels_last)
        logger.info("✓ Converted to channels-last")

    if cfg.compile:
        logger.info("Compiling model with torch.compile...")
        unet = torch.compile(unet)
        style_encoder = torch.compile(style_encoder)
        content_encoder = torch.compile(content_encoder)
        logger.info("✓ Model compiled")

    model: FontDiffuserModelDPM = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )

    dtype: torch.dtype = torch.float16 if cfg.fp16 else torch.float32
    model.to(cfg.device, dtype=dtype)
    model.eval()

    logger.info("✓ Model moved to device")

    train_scheduler: Any = build_ddpm_scheduler(cfg=cfg)
    logger.info("✓ Loaded training DDPM scheduler successfully")

    pipe: FontDiffuserDPMPipeline = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=cfg.model_type,
        guidance_type=cfg.guidance_type,
        guidance_scale=cfg.guidance_scale,
    )
    logger.info("✓ Loaded DPM-Solver pipeline successfully")
    return pipe


def sampling_batch(
    cfg: DictConfig,
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    font_manager: FontManager,
    font_name: str,
    style_image_path: str,
    style_name: str = "style0",
    save_content_images: bool = True,
) -> tuple[list[Image.Image] | None, list[str] | None, float]:
    """Batch sampling for multiple characters"""
    content_batch, style_batch, content_pils, valid_chars = image_process_batch(
        cfg, characters, font_manager, font_name, style_image_path
    )

    if (
        content_batch is None
        or valid_chars is None
        or content_pils is None
        or style_batch is None
    ):
        return None, None, 0.0

    if cfg.seed:
        set_seed(seed=cfg.seed)

    if save_content_images and cfg.save_image:
        content_dir: str = os.path.join(cfg.save_image_dir, "ContentImage")
        os.makedirs(content_dir, exist_ok=True)

        for char, pil_img in zip(valid_chars, content_pils):
            content_filename = get_content_filename(char)
            pil_img.save(os.path.join(content_dir, content_filename))

    with torch.no_grad():
        dtype: torch.dtype = (
            torch.float16 if cfg.fp16 else torch.float32
        )
        content_batch = content_batch.to(cfg.device, dtype=dtype)
        style_batch = style_batch.to(cfg.device, dtype=dtype)

        if cfg.channels_last:
            content_batch = content_batch.to(memory_format=torch.channels_last)
            style_batch = style_batch.to(memory_format=torch.channels_last)

        logger.info(f"  Sampling {len(valid_chars)} characters with DPM-Solver++ ...")
        start: float = time.time()

        all_images: list[Image.Image] = []
        batch_size: int = cfg.batch_size

        for i in range(0, len(content_batch), batch_size):
            batch_content: torch.Tensor = content_batch[i : i + batch_size]
            batch_style: torch.Tensor = style_batch[i : i + batch_size]

            images: list[Image.Image] = pipe.generate(
                content_images=batch_content,
                style_images=batch_style,
                batch_size=len(batch_content),
                order=cfg.order,
                num_inference_steps=cfg.num_inference_steps,
                content_encoder_downsample_size=cfg.content_encoder_downsample_size,
                t_start=cfg.t_start,
                t_end=cfg.t_end,
                dm_size=(cfg.content_image_size, cfg.content_image_size),
                algorithm_type=cfg.algorithm_type,
                skip_type=cfg.skip_type,
                method=cfg.method,
                correcting_x0_fn=cfg.correcting_x0_fn,
            )

            all_images.extend(images)

        end: float = time.time()
        inference_time: float = end - start

        if cfg.save_image:
            target_dir: str = os.path.join(
                cfg.save_image_dir, "TargetImage", style_name
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
    cfg: DictConfig,
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
    style_image: Image.Image = Image.open(style_image_path).convert("RGB")
    style_transform: transforms.Compose = get_style_transform(
        (cfg.style_image_size, cfg.style_image_size)
    )

    font: Any = font_manager.get_font(font_name)
    content_transform: transforms.Compose = get_content_transform(
        (cfg.content_image_size, cfg.content_image_size)
    )

    available_chars: list[str] = font_manager.get_available_chars_for_font(
        font_name, characters
    )

    if not available_chars:
        logger.info(f"Warning: No characters available in font '{font_name}'")
        return None, None, None, None

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

    content_batch: torch.Tensor = torch.stack(content_images)
    style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(
        len(content_images), 1, 1, 1
    )

    return content_batch, style_batch, content_images_pil, available_chars


@hydra.main(version_base=None, config_path="configs/inference", config_name="optimized")
def main(cfg: DictConfig) -> None:
    """Main function for optimized sampling"""
    logger.info("\n" + "=" * 60)
    logger.info("FONTDIFFUSER - OPTIMIZED SAMPLING")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 60 + "\n")

    if not cfg.ckpt_dir:
        raise ValueError("ckpt_dir must be specified")

    pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(cfg=cfg)

    characters: list[str] = parse_characters(cfg.content_character)

    if cfg.character_input and characters:
        if len(characters) > 1 or os.path.isdir(cfg.ttf_path):
            logger.info(f"\n{'=' * 60}")
            logger.info("BATCH MODE ACTIVATED")
            logger.info(
                f"Characters: {len(characters)} - {characters[:10]}{'...' if len(characters) > 10 else ''}"
            )
            logger.info("=" * 60)

            font_manager: FontManager = FontManager(cfg.ttf_path)
            font_names: list[str] = font_manager.get_font_names()

            if not cfg.demo:
                os.makedirs(cfg.save_image_dir, exist_ok=True)
                save_args_to_yaml(
                    args=OmegaConf.to_container(cfg, resolve=True),
                    output_file=f"{cfg.save_image_dir}/sampling_config.yaml"
                )

            style_name: str = os.path.splitext(os.path.basename(cfg.style_image_path))[0]

            total_generated: int = 0
            for font_idx, font_name in enumerate(font_names):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"[Font {font_idx + 1}/{len(font_names)}] {font_name}")
                logger.info("=" * 60)

                available: list[str] = font_manager.get_available_chars_for_font(
                    font_name, characters
                )
                logger.info(
                    f"  Available characters: {len(available)}/{len(characters)}"
                )

                if not available:
                    logger.info("  ⚠ Skipping font (no characters available)")
                    continue

                images, valid_chars, inf_time = sampling_batch(
                    cfg,
                    pipe,
                    characters,
                    font_manager,
                    font_name,
                    cfg.style_image_path,
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
            logger.info("=" * 60)


if __name__ == "__main__":
    main()