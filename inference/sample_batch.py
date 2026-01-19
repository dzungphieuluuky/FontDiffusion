"""
Batch sampling and evaluation for FontDiffuser using Hydra configuration
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from src.tools.utilities import HFTqdm
from src.tools.utils import load_ttf, ttf2im, is_char_in_font
from src.tools.filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)

from inference.sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
    FontManager,
)

logger = logging.getLogger("BatchSampler")

# Optional dependencies
try:
    import lpips
    LPIPS_AVAILABLE: bool = True
except ImportError:
    LPIPS_AVAILABLE: bool = False
    logger.warning("lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE: bool = True
except ImportError:
    FID_AVAILABLE: bool = False
    logger.warning("pytorch-fid not available. Install with: pip install pytorch-fid")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE: bool = True
except ImportError:
    SSIM_AVAILABLE: bool = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")

try:
    import wandb
    WANDB_AVAILABLE: bool = True
except ImportError:
    WANDB_AVAILABLE: bool = False
    logger.warning("wandb not available. Install with: pip install wandb")


class GenerationTracker:
    """Tracks generated (character, style, font) combinations"""

    def __init__(self, checkpoint_path: str | None):
        self.generated_hashes: set[str] = set()
        self.generations: list[dict[str, str]] = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint"""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results: dict = json.load(f)

            raw_generations: list[dict] = results.get("generations", [])
            seen_hashes: set[str] = set()
            unique_generations: list[dict] = []

            for gen in raw_generations:
                target_hash: str = gen.get("target_hash")
                if not target_hash:
                    char = gen.get("character", "")
                    style = gen.get("style", "")
                    font = gen.get("font", "")
                    if not char or not style:
                        continue
                    target_hash = compute_file_hash(char, style, font)

                if target_hash in seen_hashes:
                    continue

                seen_hashes.add(target_hash)
                self.generated_hashes.add(target_hash)
                unique_generations.append(gen)

            self.generations = unique_generations
            logger.info(f"✓ Loaded checkpoint: {len(self.generations)} unique generations")

        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")

    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if combination has been generated"""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes

    def mark_generated(self, char: str, style: str, font: str = "") -> None:
        """Mark combination as generated"""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)

    def add_generation(self, generation: dict) -> None:
        """Add generation record"""
        self.generations.append(generation)
        char = generation.get("character", "")
        style = generation.get("style", "")
        font = generation.get("font", "")
        self.mark_generated(char, style, font)


class QualityEvaluator:
    """Evaluates generated images"""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device: str = device

        if LPIPS_AVAILABLE:
            self.lpips_fn: lpips.LPIPS = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

        self.transform_to_tensor = transforms.ToTensor()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0

        try:
            img1_tensor: torch.Tensor = (
                self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
            )
            img2_tensor: torch.Tensor = (
                self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
            )

            with torch.inference_mode():
                lpips_value: float = self.lpips_fn(img1_tensor, img2_tensor).item()

            return lpips_value
        except Exception as e:
            logger.warning(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0

        try:
            img1_gray: np.ndarray = np.array(img1.convert("L"))
            img2_gray: np.ndarray = np.array(img2.convert("L"))
            ssim_value: float = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            logger.warning(f"Error computing SSIM: {e}")
            return -1.0

    def save_image(self, image: Image.Image, path: str) -> None:
        """Save PIL image to path"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)
        except Exception as e:
            logger.warning(f"Error saving image to {path}: {e}")


def load_characters(characters_arg: str, start_line: int = 1, end_line: int = None) -> list[str]:
    """Load characters from file or comma-separated string"""
    chars: list[str] = []
    
    if os.path.isfile(characters_arg):
        with open(characters_arg, "r", encoding="utf-8") as f:
            all_lines: list[str] = f.readlines()

        start_idx: int = max(0, start_line - 1)
        end_idx: int = len(all_lines) if end_line is None else min(len(all_lines), end_line)

        if start_idx >= len(all_lines):
            raise ValueError(f"start_line ({start_line}) exceeds file length ({len(all_lines)})")

        logger.info(f"📖 Loading characters from file: {characters_arg}")
        logger.info(f"   Lines {start_line} to {end_idx} (total: {len(all_lines)})")

        for line in all_lines[start_idx:end_idx]:
            char: str = line.strip()
            if char and len(char) == 1:
                chars.append(char)
    else:
        for c in [x.strip() for x in characters_arg.split(",") if x.strip()]:
            if len(c) == 1:
                chars.append(c)

    if not chars:
        raise ValueError("No valid characters loaded!")

    logger.info(f"✅ Successfully loaded {len(chars)} characters")
    return chars


def load_style_images(style_images_arg: str) -> list[tuple[str, str]]:
    """Load style image paths"""
    import glob

    image_exts: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
    style_paths: list[str] = []

    if os.path.isdir(style_images_arg):
        logger.info(f"📂 Loading style images from directory: {style_images_arg}")
        style_paths: list[str] = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()
        logger.info(f"   Found {len(style_paths)} image files")

    elif "*" in style_images_arg or "?" in style_images_arg:
        logger.info(f"🔍 Loading style images using glob: {style_images_arg}")
        style_paths = glob.glob(style_images_arg, recursive=True)
        style_paths = [
            p for p in style_paths
            if os.path.splitext(p)[1].lower() in image_exts and os.path.isfile(p)
        ]
        style_paths.sort()
        logger.info(f"   Found {len(style_paths)} images")

    else:
        raw_paths = [p.strip() for p in style_images_arg.split(",")]
        for path in raw_paths:
            if path and os.path.isfile(path):
                if os.path.splitext(path)[1].lower() in image_exts:
                    style_paths.append(path)

    if not style_paths:
        raise ValueError("No valid style images found!")

    verified_paths: list[tuple[str, str]] = []
    for path in style_paths:
        if os.path.isfile(path):
            style_name = os.path.splitext(os.path.basename(path))[0]
            verified_paths.append((path, style_name))

    logger.info(f"✅ Successfully loaded {len(verified_paths)} style images\n")
    return verified_paths


def save_checkpoint(results: dict, output_dir: str) -> None:
    """Save results_checkpoint.json"""
    try:
        checkpoint_path: str = os.path.join(output_dir, "results_checkpoint.json")
        if "metrics" not in results:
            results["metrics"] = {}

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ Saved checkpoint ({len(results.get('generations', []))} generations)")

    except Exception as e:
        logger.warning(f"Error saving checkpoint: {e}")


def generate_content_images(
    characters: list[str],
    font_manager: FontManager,
    output_dir: str,
) -> dict[str, str]:
    """Generate and save content character images"""
    content_dir: str = os.path.join(output_dir, "ContentImage")
    os.makedirs(content_dir, exist_ok=True)

    font_names: list[str] = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")

    char_paths: dict[str, str] = {}
    generated_new: int = 0

    for char in HFTqdm(
        characters,
        desc="📸 Generating content images",
        colour="magenta",
    ):
        found_font = None
        for font_name in font_names:
            if font_manager.is_char_in_font(font_name, char):
                found_font = font_name
                break

        if not found_font:
            continue

        try:
            content_filename = get_content_filename(char)
            char_path: str = os.path.join(content_dir, content_filename)

            if os.path.exists(char_path):
                char_paths[char] = char_path
                continue

            font = font_manager.get_font(found_font)
            content_img: Image.Image = ttf2im(font=font, char=char)
            content_img.save(char_path)
            char_paths[char] = char_path
            generated_new += 1

        except Exception as e:
            logger.warning(f"Error generating '{char}': {e}")

    logger.info(f"Content Image Generation: Generated {generated_new} new, Reused {len(char_paths) - generated_new}")
    return char_paths


def sampling_batch_optimized(
    cfg: DictConfig,
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_image_path: str,
    font_manager: FontManager,
    font_name: str,
) -> tuple[list[Image.Image] | None, list[str] | None, float]:
    """Batch sampling for multiple characters"""

    available_chars: list[str] = font_manager.get_available_chars_for_font(
        font_name, characters
    )

    if not available_chars:
        return None, None, 0.0

    try:
        style_image: Image.Image = Image.open(style_image_path).convert("RGB")
        style_transform: transforms.Compose = get_style_transform(
            (cfg.style_image_size, cfg.style_image_size)
        )

        font = font_manager.get_font(font_name)
        content_transform: transforms.Compose = get_content_transform(
            (cfg.content_image_size, cfg.content_image_size)
        )

        content_images: list[torch.Tensor] = []
        content_images_pil: list[Image.Image] = []

        for char in available_chars:
            try:
                content_image: Image.Image = ttf2im(font=font, char=char)
                content_images_pil.append(content_image.copy())
                content_images.append(content_transform(content_image))
            except Exception as e:
                logger.warning(f"Error processing '{char}': {e}")
                continue

        if not content_images:
            return None, None, 0.0

        content_batch: torch.Tensor = torch.stack(content_images)
        style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(
            len(content_images), 1, 1, 1
        )

        with torch.inference_mode():
            dtype: torch.dtype = torch.float16 if cfg.fp16 else torch.float32
            content_batch = content_batch.to(cfg.device, dtype=dtype)
            style_batch = style_batch.to(cfg.device, dtype=dtype)

            start: float = time.perf_counter()

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

            end: float = time.perf_counter()
            total_time: float = end - start

            return all_images, available_chars, total_time

    except Exception as e:
        logger.warning(f"Error in batch sampling: {e}")
        return None, None, 0.0


def batch_generate_images(
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_paths_with_names: list[tuple[str, str]],
    output_dir: str,
    cfg: DictConfig,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
) -> dict:
    """Main batch generation pipeline"""

    char_paths = generate_content_images(characters, font_manager, output_dir)

    if not char_paths:
        raise ValueError("No content images generated!")

    results = {
        "generations": generation_tracker.generations.copy(),
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": cfg.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": sorted(list(char_paths.keys())),
        "styles": [name for _, name in style_paths_with_names],
        "total_chars": len(char_paths),
        "total_styles": len(style_paths_with_names),
    }

    target_base_dir = os.path.join(output_dir, "TargetImage")
    os.makedirs(target_base_dir, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"{'BATCH IMAGE GENERATION':^60}")
    logger.info("=" * 60)
    logger.info(f"Fonts:                {len(font_manager.get_font_names())}")
    logger.info(f"Styles:               {len(style_paths_with_names)}")
    logger.info(f"Characters:           {len(char_paths)}")
    logger.info(f"Batch size:           {cfg.batch_size}")
    logger.info("=" * 60 + "\n")

    font_names = font_manager.get_font_names()
    primary_font = font_names[0]

    generated_count = 0
    skipped_count = 0

    for style_idx, (style_path, style_name) in HFTqdm(
        enumerate(style_paths_with_names),
        total=len(style_paths_with_names),
        desc="🎨 Generating styles",
    ):
        style_dir = os.path.join(target_base_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)

        try:
            chars_to_generate = [
                char
                for char in characters
                if not generation_tracker.is_generated(char, style_name, primary_font)
            ]

            if not chars_to_generate:
                skipped_count += len(characters)
                continue

            images, valid_chars, batch_time = sampling_batch_optimized(
                cfg,
                pipe,
                chars_to_generate,
                style_path,
                font_manager,
                primary_font,
            )

            if images is None:
                skipped_count += len(chars_to_generate)
                continue

            for char, img in zip(valid_chars, images):
                try:
                    target_filename = get_target_filename(char, style_name)
                    img_path = os.path.join(style_dir, target_filename)
                    content_filename = get_content_filename(char)

                    evaluator.save_image(img, img_path)

                    generation_record = {
                        "character": char,
                        "char_code": f"U+{ord(char):04X}",
                        "style": style_name,
                        "font": primary_font,
                        "content_image_path": f"ContentImage/{content_filename}",
                        "target_image_path": f"TargetImage/{style_name}/{target_filename}",
                        "target_hash": compute_file_hash(char, style_name, primary_font),
                    }

                    results["generations"].append(generation_record)
                    generation_tracker.add_generation(generation_record)
                    generated_count += 1

                except Exception as e:
                    logger.warning(f"Error saving '{char}': {e}")

            results["metrics"]["inference_times"].append({
                "style": style_name,
                "font": primary_font,
                "total_time": batch_time,
                "num_images": len(images),
            })

            if cfg.save_interval > 0 and (style_idx + 1) % cfg.save_interval == 0:
                save_checkpoint(results, output_dir)

        except Exception as e:
            logger.warning(f"Error generating {style_name}: {e}")

    return results


def evaluate_results(
    results: dict,
    evaluator: QualityEvaluator,
    ground_truth_dir: str = None,
) -> dict:
    """Evaluate generated images"""

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        logger.info("No ground truth directory provided, skipping evaluation")
        return results

    logger.info("=" * 60)
    logger.info(f"{'EVALUATING GENERATED IMAGES':^60}")
    logger.info("=" * 60)

    lpips_scores: list[float] = []
    ssim_scores: list[float] = []

    for gen in HFTqdm(
        results["generations"],
        desc="📊 Evaluating",
        colour="green",
    ):
        char: str = gen["character"]
        style: str = gen["style"]
        target_path: str = gen["target_image_path"]

        gt_filename = get_target_filename(char, style)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            continue

        try:
            generated_img: Image.Image = Image.open(target_path).convert("RGB")
            gt_img: Image.Image = Image.open(gt_path).convert("RGB")

            if LPIPS_AVAILABLE:
                lpips_score = evaluator.compute_lpips(generated_img, gt_img)
                if lpips_score >= 0:
                    lpips_scores.append(lpips_score)
                    gen["lpips"] = lpips_score

            if SSIM_AVAILABLE:
                ssim_score = evaluator.compute_ssim(generated_img, gt_img)
                if ssim_score >= 0:
                    ssim_scores.append(ssim_score)
                    gen["ssim"] = ssim_score

        except Exception as e:
            logger.warning(f"Error evaluating {char}/{style}: {e}")

    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }

    logger.info("=" * 60)
    return results


def log_to_wandb(results: dict, cfg: DictConfig) -> None:
    """Log results to Weights & Biases"""

    if not WANDB_AVAILABLE or not cfg.use_wandb:
        return

    try:
        logger.info("=" * 60)
        logger.info(f"{'LOGGING TO WEIGHTS & BIASES':^60}")
        logger.info("=" * 60)

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        wandb.log({
            "total_generations": len(results.get("generations", [])),
            "num_characters": results.get("total_chars", 0),
            "num_styles": results.get("total_styles", 0),
        })

        metrics = results.get("metrics", {})
        if "lpips" in metrics and isinstance(metrics["lpips"], dict):
            wandb.log({f"lpips/{k}": v for k, v in metrics["lpips"].items()})

        if "ssim" in metrics and isinstance(metrics["ssim"], dict):
            wandb.log({f"ssim/{k}": v for k, v in metrics["ssim"].items()})

        wandb.finish()
        logger.info("✓ Successfully logged to Weights & Biases")
        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"Error logging to wandb: {e}")


@hydra.main(version_base=None, config_path="configs/inference", config_name="batch")
def main(cfg: DictConfig) -> None:
    """Main function"""
    logger.info("=" * 60)
    logger.info("FontDiffusion Batch Inference")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 60 + "\n")

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing mandatory keys: {missing_keys}")

    try:
        characters: list[str] = load_characters(
            cfg.characters, cfg.start_line, cfg.end_line
        )
        style_paths_with_names: list[tuple[str, str]] = load_style_images(cfg.style_images)

        font_manager: FontManager = FontManager(cfg.ttf_path)
        logger.info(f"✓ Loaded {len(font_manager.get_font_names())} fonts\n")

        os.makedirs(cfg.output_dir, exist_ok=True)

        checkpoint_path = os.path.join(cfg.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(cfg=cfg)
        evaluator: QualityEvaluator = QualityEvaluator(device=cfg.device)

        results: dict = batch_generate_images(
            pipe,
            characters,
            style_paths_with_names,
            cfg.output_dir,
            cfg,
            evaluator,
            font_manager,
            generation_tracker,
        )

        if cfg.evaluate and cfg.ground_truth_dir:
            results = evaluate_results(results, evaluator, cfg.ground_truth_dir)

        logger.info("\n💾 Saving final checkpoint...")
        save_checkpoint(results, cfg.output_dir)

        if cfg.use_wandb:
            log_to_wandb(results, cfg)

        logger.info("=" * 60)
        logger.info("✅ Generation complete!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\n⚠ Generation interrupted by user!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()