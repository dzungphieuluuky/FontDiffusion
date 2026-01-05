import os
import sys
import time
import json
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from huggingface_hub.utils import tqdm, enable_progress_bars
import logging
import wandb

import numpy as np
from PIL import Image
from argparse import Namespace, ArgumentParser

from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from utilities import (
    get_hf_bar,
)
from font_manager import FontManager
from generation_tracker import GenerationTracker
from quality_evaluator import (
    QualityEvaluator,
    LPIPS_AVAILABLE,
    FID_AVAILABLE,
    SSIM_AVAILABLE,
    WANDB_AVAILABLE,
)

enable_progress_bars()

# Import FontDiffuser modules
from sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
)
from utils import (
    load_ttf,
    ttf2im,
    is_char_in_font,
)

from filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)

def parse_args() -> Namespace:
    """
    Parse command‑line arguments for the batch‑sampling and evaluation pipeline.
    Arguments are grouped by functional domain for readability.
    """
    parser = ArgumentParser(description="Batch sampling and evaluation")

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------
    io = parser.add_argument_group("Input / Output")
    io.add_argument(
        "--characters",
        type=str,
        required=True,
        help="Comma‑separated list of characters or path to text file",
    )
    io.add_argument(
        "--start_line",
        type=int,
        default=1,
        help="Start line number for character file (1‑indexed)",
    )
    io.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="End line number for character file (inclusive, None = EOF)",
    )
    io.add_argument(
        "--style_images",
        type=str,
        required=True,
        help="Comma‑separated paths to style images or directory",
    )
    io.add_argument(
        "--output_dir",
        type=str,
        default="my_dataset/train_original",
        help="Output directory (creates ContentImage/ and TargetImage/ subdirs)",
    )
    io.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory with ground‑truth images for evaluation",
    )

    # -----------------------------------------------------------------
    # Model configuration
    # -----------------------------------------------------------------
    model = parser.add_argument_group("Model configuration")
    model.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory",
    )
    model.add_argument(
        "--ttf_path",
        type=str,
        required=True,
        help="Path to TTF font file or directory with multiple fonts",
    )
    model.add_argument("--device", type=str, default="cuda", help="Device to use")
    model.add_argument(
        "--num_scales",
        type=int,
        default=4,
        help="Number of scales in style transformation",
    )
    model.add_argument(
        "--feature_dim",
        type=int,
        default=512,
        help="Feature dimension for style transformation",
    )
    model.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for style transformation",
    )
    model.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    model.add_argument(
        "--ffn_dim",
        type=int,
        default=2048,
        help="Feed‑forward network dimension",
    )
    model.add_argument(
        "--style_transform_coefficient",
        type=float,
        default=0.1,
        help="Loss coefficient for style transformation",
    )

    # -----------------------------------------------------------------
    # Generation parameters
    # -----------------------------------------------------------------
    gen = parser.add_argument_group("Generation parameters")
    gen.add_argument(
        "--num_inference_steps",
        type=int,
        default=15,
        help="Number of inference steps",
    )
    gen.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    gen.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    gen.add_argument("--seed", type=int, default=42, help="Random seed")

    # -----------------------------------------------------------------
    # Optimization flags
    # -----------------------------------------------------------------
    opt = parser.add_argument_group("Optimization flags")
    opt.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 precision",
    )
    opt.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile",
    )
    opt.add_argument(
        "--channels_last",
        action="store_true",
        default=True,
        help="Use channels‑last memory format",
    )
    opt.add_argument(
        "--enable_xformers",
        action="store_true",
        default=False,
        help="Enable xformers",
    )
    opt.add_argument(
        "--fast_sampling",
        action="store_true",
        default=False,
        help="Use fast sampling mode",
    )

    # -----------------------------------------------------------------
    # Checkpoint & resume
    # -----------------------------------------------------------------
    ckpt = parser.add_argument_group("Checkpoint & resume")
    ckpt.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save results every N styles (0 = only save at end)",
    )

    # -----------------------------------------------------------------
    # Evaluation flags
    # -----------------------------------------------------------------
    eval = parser.add_argument_group("Evaluation flags")
    eval.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate generated images",
    )
    eval.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="Compute FID (requires ground truth)",
    )
    eval.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        default=False,
        help="Enable attention slicing for memory efficiency",
    )

    # -----------------------------------------------------------------
    # WandB configuration
    # -----------------------------------------------------------------
    wandb = parser.add_argument_group("WandB configuration")
    wandb.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Log results to Weights & Biases",
    )
    wandb.add_argument(
        "--wandb_project",
        type=str,
        default="fontdiffuser-eval",
        help="WandB project name",
    )
    wandb.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    wandb.add_argument(
        "--dataset_split",
        type=str,
        default="train_original",
        help="Dataset split name (e.g., train_original, val)",
    )

    return parser.parse_args()

def load_characters(
    characters_arg: str, start_line: int = 1, end_line: Optional[int] = None
) -> List[str]:
    """Load characters from file or comma-separated string with line range support"""
    chars: List[str] = []
    if os.path.isfile(characters_arg):
        with open(characters_arg, "r", encoding="utf-8") as f:
            all_lines: List[str] = f.readlines()

        # Adjust for 1-indexed input
        start_idx: int = max(0, start_line - 1)
        end_idx: int = (
            len(all_lines) if end_line is None else min(len(all_lines), end_line)
        )

        if start_idx >= len(all_lines):
            raise ValueError(
                f"❌ start_line ({start_line}) exceeds file length ({len(all_lines)} lines)\n"
                f"   Your file only has {len(all_lines)} lines, but you're trying to start at line {start_line}."
            )

        if start_idx >= end_idx:
            raise ValueError(
                f"❌ Invalid line range: start_line={start_line}, end_line={end_line}\n"
                f"   File has {len(all_lines)} lines.\n"
                f"   Computed range [{start_idx}:{end_idx}] is empty.\n"
                f"   Make sure start_line <= end_line and both are within file bounds."
            )

        logging.info(f"📖 Loading characters from file: {characters_arg}")
        logging.info(
            f"   Lines {start_line} to {end_idx} (total file: {len(all_lines)} lines)"
        )
        logging.info(f"   Processing {end_idx - start_idx} lines...")

        for line_num, line in get_hf_bar(
            enumerate(all_lines[start_idx:end_idx], start=start_line),
            total=(end_idx - start_idx),
            desc="📖 Reading character file",
            colour="green",
        ):
            char: str = line.strip()
            if not char:
                continue
            if len(char) != 1:
                logging.info(
                    f"Warning: Skipping line {line_num}: expected 1 char, got {len(char)}: '{char}'"
                )
                continue
            chars.append(char)

    else:
        for c in [x.strip() for x in characters_arg.split(",") if x.strip()]:
            if len(c) != 1:
                raise ValueError(
                    f"Invalid character in argument: '{c}' (must be single char)"
                )
            chars.append(c)

    if not chars:
        raise ValueError(
            f"❌ No valid characters loaded!\n"
            f"   Check your character file or line range (start={start_line}, end={end_line})"
        )

    logging.info(f"Successfully loaded {len(chars)} single characters.")
    return chars


def load_style_images(style_images_arg: str) -> List[Tuple[str, str]]:
    """
    Load style image paths and extract style names
    Returns: List of (style_path, style_name) tuples
    """
    if os.path.isdir(style_images_arg):
        # Load all images from directory
        image_exts: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
        style_paths: List[str] = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()

        logging.info(f"Loading {len(style_paths)} style images from directory...")
        verified_paths = []
        for path in get_hf_bar(
            style_paths,
            desc="✓ Verifying style images",
            colour="green",
        ):
            if os.path.isfile(path):
                # Extract style name from filename (without extension)
                style_name = os.path.splitext(os.path.basename(path))[0]
                verified_paths.append((path, style_name))

        return verified_paths
    else:
        style_paths: List[str] = [p.strip() for p in style_images_arg.split(",")]
        result = []
        for path in style_paths:
            style_name = os.path.splitext(os.path.basename(path))[0]
            result.append((path, style_name))
        return result


def create_args_namespace(args: Namespace) -> Namespace:
    """Create args namespace for FontDiffuser pipeline"""

    try:
        from configs.fontdiffuser import get_parser

        parser: ArgumentParser = get_parser()
        default_args: Namespace = parser.parse_args([])
    except Exception:
        default_args: Namespace = Namespace()

    # Override with user arguments
    for key, value in vars(args).items():
        setattr(default_args, key, value)

    # Ensure image sizes are tuples
    if not hasattr(default_args, "style_image_size"):
        default_args.style_image_size = (96, 96)
    elif isinstance(default_args.style_image_size, int):
        default_args.style_image_size = (
            default_args.style_image_size,
            default_args.style_image_size,
        )

    if not hasattr(default_args, "content_image_size"):
        default_args.content_image_size = (96, 96)
    elif isinstance(default_args.content_image_size, int):
        default_args.content_image_size = (
            default_args.content_image_size,
            default_args.content_image_size,
        )

    # Set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    default_args.resolution = 96

    # Generation parameters
    default_args.algorithm_type = getattr(default_args, "algorithm_type", "dpmsolver++")
    default_args.guidance_type = getattr(
        default_args, "guidance_type", "classifier-free"
    )
    default_args.method = getattr(default_args, "method", "multistep")
    default_args.order = getattr(default_args, "order", 2)
    default_args.model_type = getattr(default_args, "model_type", "noise")
    default_args.t_start = getattr(default_args, "t_start", 1.0)
    default_args.t_end = getattr(default_args, "t_end", 1e-3)
    default_args.skip_type = getattr(default_args, "skip_type", "time_uniform")
    default_args.correcting_x0_fn = getattr(default_args, "correcting_x0_fn", None)
    default_args.content_encoder_downsample_size = getattr(
        default_args, "content_encoder_downsample_size", 3
    )

    return default_args


def save_checkpoint(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save results_checkpoint.json (single source of truth)
    """
    try:
        checkpoint_path: str = os.path.join(output_dir, "results_checkpoint.json")

        # Ensure metrics exist
        if "metrics" not in results:
            results["metrics"] = {"lpips": [], "ssim": [], "inference_times": []}

        # Save checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        num_gens = len(results.get("generations", []))
        logging.info(f" Saved results_checkpoint.json ({num_gens} generations)")

    except Exception as e:
        logging.info(f"  ⚠ Error saving checkpoint: {e}")


def generate_content_images(
    characters: List[str],
    font_manager: FontManager,
    output_dir: str,
    generation_tracker: GenerationTracker,
) -> Dict[str, str]:
    """
    Generate and save content character images
    ✅ CORRECTED: Only generates if content image doesn't already exist
    Returns: char_paths dict mapping character to file path
    """
    content_dir: str = os.path.join(output_dir, "ContentImage")
    os.makedirs(content_dir, exist_ok=True)

    font_names: List[str] = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")

    logging.info(f"{'=' * 60}")
    logging.info(f"Generating Content Images")
    logging.info(f"Using {len(font_names)} fonts")
    logging.info(f"Characters: {len(characters)}")
    logging.info("=" * 60)

    char_paths: Dict[str, str] = {}
    chars_without_fonts: List[str] = []
    chars_already_exist: List[str] = []
    generated_new: int = 0

    for char in get_hf_bar(
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
            logging.info(f"  ⚠ Warning: '{char}' not in any font, skipping...")
            chars_without_fonts.append(char)
            continue

        try:
            # ✅ Generate expected filename
            content_filename = get_content_filename(char)
            char_path: str = os.path.join(content_dir, content_filename)

            # ✅ Check if content image already exists
            if os.path.exists(char_path):
                logging.info(
                    f"  ✓ Content image already exists for '{char}' at {char_path}"
                )
                char_paths[char] = char_path
                chars_already_exist.append(char)
                continue

            # Generate new content image only if it doesn't exist
            font = font_manager.get_font(found_font)
            content_img: Image.Image = ttf2im(font=font, char=char)

            content_img.save(char_path)
            logging.info(
                f"  ✓ Generated new content image for '{char}' at {char_path}."
            )
            char_paths[char] = char_path
            generated_new += 1

        except Exception as e:
            logging.info(f"  ✗ Error generating '{char}': {e}")

    logging.info(f"{'=' * 60}")
    logging.info(f"Content Image Generation Summary:")
    logging.info(f"  Total characters:       {len(characters)}")
    logging.info(f"  Generated (new):        {generated_new}")
    logging.info(f"  Already exist (reused): {len(chars_already_exist)}")
    logging.info(f"  Not in any font:        {len(chars_without_fonts)}")
    logging.info(f"  Total usable:           {len(char_paths)}")
    logging.info("=" * 60)

    return char_paths

def batch_generate_images(
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_paths_with_names: List[Tuple[str, str]],
    output_dir: str,
    args: Namespace,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
) -> Dict[str, Any]:
    """
    ✅ Main batch generation with hash-based file naming
    """

    # Generate ALL content images first
    logging.info(f"{'=' * 60}")
    logging.info(f"{'GENERATING CONTENT IMAGES':^60}")
    logging.info("=" * 60)

    char_paths = generate_content_images(
        characters, font_manager, output_dir, generation_tracker
    )

    if not char_paths:
        raise ValueError("No content images generated!")

    # Extract ALL unique characters and styles from checkpoint
    all_chars_in_checkpoint: Set[str] = set()
    all_styles_in_checkpoint: Set[str] = set()

    for gen in generation_tracker.generations:
        all_chars_in_checkpoint.add(gen.get("character", ""))
        all_styles_in_checkpoint.add(gen.get("style", ""))

    # Add current session's chars
    all_chars_in_checkpoint.update(char_paths.keys())

    # Add current session's styles
    for style_path, style_name in style_paths_with_names:
        if any(
            gen.get("style") == style_name for gen in generation_tracker.generations
        ):
            all_styles_in_checkpoint.add(style_name)

    # Initialize results from tracker
    results = {
        "generations": generation_tracker.generations.copy(),
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": args.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": sorted(list(all_chars_in_checkpoint)),
        "styles": sorted(list(all_styles_in_checkpoint)),
        "total_chars": len(all_chars_in_checkpoint),
        "total_styles": len(all_styles_in_checkpoint),
    }

    # Setup directories
    target_base_dir = os.path.join(output_dir, "TargetImage")
    os.makedirs(target_base_dir, exist_ok=True)

    # Print configuration
    logging.info(f"{'=' * 60}")
    logging.info(f"{'BATCH IMAGE GENERATION':^60}")
    logging.info("=" * 60)
    logging.info(f"Fonts:                {len(font_manager.get_font_names())}")
    logging.info(f"Styles:               {len(style_paths_with_names)}")
    logging.info(f"Characters (input):   {len(characters)}")
    logging.info(f"Characters (content): {len(char_paths)}")
    logging.info(f"Batch size:           {args.batch_size}")
    logging.info(
        f"Previously generated: {len(generation_tracker.generations)} unique pairs"
    )
    logging.info(f"Unique chars seen:    {len(all_chars_in_checkpoint)}")
    logging.info(f"Unique styles used:   {len(all_styles_in_checkpoint)}")
    logging.info("=" * 60 + "\n")

    # Use first font for all characters
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded!")

    primary_font = font_names[0]
    logging.info(f"Using font: {primary_font}")
    logging.info("=" * 60 + "\n")

    # Initialize counters
    generated_count = 0
    skipped_count = 0
    failed_count = 0
    generation_start_time = time.time()

    # Main generation loop
    for style_idx, (style_path, style_name) in get_hf_bar(
        enumerate(style_paths_with_names),
        total=len(style_paths_with_names),
        desc="🎨 Generating styles",
    ):
        style_dir = os.path.join(target_base_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)

        try:
            # Filter characters that haven't been generated yet
            chars_to_generate = [
                char
                for char in characters
                if not generation_tracker.is_generated(char, style_name, primary_font)
            ]

            if not chars_to_generate:
                logging.info(
                    f"  ⊘ {style_name}: All characters already generated, skipping"
                )
                skipped_count += len(characters)
                continue

            logging.info(
                f"  🔄 {style_name}: Generating {len(chars_to_generate)}/{len(characters)} new images"
            )

            # ✅ PASS STYLE TRANSFORM FLAG TO SAMPLING
            images, valid_chars, batch_time = sampling_batch_optimized(
                args,
                pipe,
                chars_to_generate,
                style_path,
                font_manager,
                primary_font,
            )

            if images is None:
                logging.info(f"  ⚠️ {style_name}: No images generated")
                skipped_count += len(chars_to_generate)
                continue

            logging.info(f"  ✓ {style_name}: {len(images)} images in {batch_time:.2f}s")

            # Save images and metadata
            for char, img in zip(valid_chars, images):
                try:
                    if not font_manager.is_char_in_font(primary_font, char):
                        logging.error(
                            f"    ✗ Character '{char}' (U+{ord(char):04X}) not in font {primary_font}, skipping"
                        )
                        failed_count += 1
                        continue

                    target_filename = get_target_filename(char, style_name)

                    import re
                    expected_pattern = r".+\+.\.png"
                    if not re.match(expected_pattern, target_filename):
                        raise ValueError(
                            f"Invalid filename format: {target_filename}\n"
                            f"  Expected: U+XXXX_[optional_char]_style_hash.png"
                        )

                    img_path = os.path.join(style_dir, target_filename)
                    content_filename = get_content_filename(char)
                    content_path_rel = f"ContentImage/{content_filename}"
                    target_path_rel = f"TargetImage/{style_name}/{target_filename}"

                    evaluator.save_image(img, img_path)
                    logging.info(
                        f"    ✓ Saved generated image for '{char}' (U+{ord(char):04X}) at {img_path}."
                    )

                    generation_record = {
                        "character": char,
                        "char_code": f"U+{ord(char):04X}",
                        "style": style_name,
                        "font": primary_font,
                        "content_image_path": content_path_rel,
                        "target_image_path": target_path_rel,
                        "content_hash": compute_file_hash(char, "", primary_font),
                        "target_hash": compute_file_hash(char, style_name, primary_font),
                        "content_filename": content_filename,
                        "target_filename": target_filename,
                    }

                    results["generations"].append(generation_record)
                    generation_tracker.add_generation(generation_record)

                    all_chars_in_checkpoint.add(char)
                    all_styles_in_checkpoint.add(style_name)
                    results["characters"] = sorted(list(all_chars_in_checkpoint))
                    results["styles"] = sorted(list(all_styles_in_checkpoint))
                    results["total_chars"] = len(all_chars_in_checkpoint)
                    results["total_styles"] = len(all_styles_in_checkpoint)

                    generated_count += 1

                except ValueError as e:
                    logging.error(f"    ✗ Invalid filename for '{char}': {e}")
                    failed_count += 1
                except Exception as e:
                    logging.error(f"    ✗ Error saving '{char}': {e}")
                    failed_count += 1

            # Track inference time
            results["metrics"]["inference_times"].append(
                {
                    "style": style_name,
                    "font": primary_font,
                    "total_time": batch_time,
                    "num_images": len(images),
                    "time_per_image": batch_time / len(images) if images else 0,
                }
            )

            # Save checkpoint periodically
            if args.save_interval > 0 and (style_idx + 1) % args.save_interval == 0:
                _print_checkpoint_status(
                    style_idx + 1,
                    len(style_paths_with_names),
                    generated_count,
                    skipped_count,
                    generation_start_time,
                )
                save_checkpoint(results, output_dir)

        except Exception as e:
            logging.info(f"  ✗ {style_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += len(chars_to_generate)

    # Final statistics
    _print_generation_summary(
        generated_count,
        skipped_count,
        failed_count,
        len(characters) * len(style_paths_with_names),
        generation_start_time,
    )

    return results


def sampling_batch_optimized(
    args: Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_image_path: Union[str, Image.Image],
    font_manager: FontManager,
    font_name: str,
) -> Tuple[Optional[List[Image.Image]], Optional[List[str]], Optional[float]]:
    """
    Batch sampling for multiple characters.
    
    Style transformation is NOT applied during inference.
    It's only used during training (Phase 2).
    """
    try:
        if isinstance(style_image_path, str):
            style_image = Image.open(style_image_path).convert("RGB")
        else:
            style_image = style_image_path

        style_transform = get_style_transform(args.style_image_size)
        style_image = style_transform(style_image)

        content_transform = get_content_transform(args.content_image_size)
        
        images = []
        valid_chars = []
        generation_times = []

        with get_hf_bar(
            total=len(characters),
            desc=f"Generating {font_name}",
            unit="char",
        ) as pbar:
            for char in characters:
                try:
                    if not is_char_in_font(font_manager.get_font_path(font_name), char):
                        continue

                    content_image = ttf2im(
                        load_ttf(font_manager.get_font_path(font_name)),
                        char,
                        args.content_image_size,
                    )
                    content_image = content_transform(content_image)

                    start_time = time.time()
                    gen_images = pipe.generate(
                        content_images=content_image.unsqueeze(0),
                        style_images=style_image.unsqueeze(0),
                        batch_size=1,
                        order=args.order,
                        num_inference_step=args.num_inference_steps,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        t_start=args.t_start,
                        t_end=args.t_end,
                        dm_size=args.content_image_size,
                        algorithm_type=args.algorithm_type,
                        skip_type=args.skip_type,
                        method=args.method,
                        correcting_x0_fn=args.correcting_x0_fn,
                    )
                    generation_times.append(time.time() - start_time)

                    images.extend(gen_images)
                    valid_chars.append(char)

                except Exception as e:
                    logging.warning(f"Failed to generate character '{char}': {e}")
                    continue
                finally:
                    pbar.update(1)

        avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
        return images, valid_chars, avg_time

    except Exception as e:
        logging.error(f"Batch generation failed: {e}")
        return None, None, None
        
def _print_checkpoint_status(
    current_style: int,
    total_styles: int,
    generated: int,
    skipped: int,
    start_time: float,
) -> None:
    """Print periodic checkpoint status"""
    elapsed = time.time() - start_time
    remaining = (
        elapsed * (total_styles - current_style) / current_style
        if current_style > 0
        else 0
    )

    logging.info(f"{'=' * 60}")
    logging.info(f"{'CHECKPOINT':^60}")
    logging.info("=" * 60)
    logging.info(f"Progress:           {current_style}/{total_styles} styles")
    logging.info(f"Generated:          {generated} pairs")
    logging.info(f"Skipped:            {skipped} pairs")
    logging.info(f"Elapsed time:       {elapsed / 60:.1f} minutes")
    logging.info(f"Est. remaining:     {remaining / 60:.1f} minutes")
    logging.info("=" * 60)


def _print_generation_summary(
    generated: int, skipped: int, failed: int, total: int, start_time: float
) -> None:
    """Print final generation summary"""
    elapsed = time.time() - start_time

    logging.info("=" * 60)
    logging.info(f"{'GENERATION COMPLETE':^60}")
    logging.info("=" * 60)
    logging.info(f"Pair Statistics:")
    logging.info(f"  Total possible:     {total}")
    logging.info(f"  Generated (new):    {generated}")
    logging.info(f"  Skipped (exist):    {skipped}")
    logging.info(f"  Failed (no font):   {failed}")
    logging.info(f"Timing:")
    logging.info(f"  Total time:         {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    logging.info(
        f"  Avg per pair:       {elapsed / generated * 1000:.1f}ms"
        if generated > 0
        else "  Avg per pair:       N/A"
    )
    logging.info("=" * 60)


def evaluate_results(
    results: Dict[str, Any],
    evaluator: QualityEvaluator,
    ground_truth_dir: Optional[str] = None,
    compute_fid: bool = False,
) -> Dict[str, Any]:
    """Evaluate generated images against ground truth"""

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        logging.info(
            "\n⚠ No ground truth directory provided or not found, skipping evaluation"
        )
        return results

    logging.info("=" * 60)
    logging.info(f"{'EVALUATING GENERATED IMAGES':^60}")
    logging.info("=" * 60)

    lpips_scores: List[float] = []
    ssim_scores: List[float] = []
    evaluated_pairs: int = 0
    missing_gt: int = 0

    # Evaluate each generation
    for gen in get_hf_bar(
        results["generations"],
        desc="📊 Evaluating",
        colour="green",
    ):
        char: str = gen["character"]
        style: str = gen["style"]
        font: str = gen.get("font", "")

        # Get generated image path
        target_path: str = gen["target_image_path"]
        generated_path: str = os.path.join(
            os.path.dirname(os.path.dirname(target_path)), target_path
        )

        if not os.path.exists(generated_path):
            continue

        # Find ground truth image
        gt_filename = get_target_filename(char, style)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            # Try alternative naming
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        try:
            # Load images
            generated_img: Image.Image = Image.open(generated_path).convert("RGB")
            gt_img: Image.Image = Image.open(gt_path).convert("RGB")

            # Compute metrics
            if LPIPS_AVAILABLE:
                lpips_score: float = evaluator.compute_lpips(generated_img, gt_img)
                if lpips_score >= 0:
                    lpips_scores.append(lpips_score)
                    gen["lpips"] = lpips_score

            if SSIM_AVAILABLE:
                ssim_score: float = evaluator.compute_ssim(generated_img, gt_img)
                if ssim_score >= 0:
                    ssim_scores.append(ssim_score)
                    gen["ssim"] = ssim_score

            evaluated_pairs += 1

        except Exception as e:
            logging.info(f"  ⚠ Error evaluating {char}/{style}: {e}")
            continue

    # Compute aggregate metrics
    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "min": float(np.min(lpips_scores)),
            "max": float(np.max(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }
        logging.info(f"📊 LPIPS Statistics:")
        logging.info(f"  Mean:   {results['metrics']['lpips']['mean']:.4f}")
        logging.info(f"  Std:    {results['metrics']['lpips']['std']:.4f}")
        logging.info(f"  Median: {results['metrics']['lpips']['median']:.4f}")
        logging.info(
            f"  Range:  [{results['metrics']['lpips']['min']:.4f}, {results['metrics']['lpips']['max']:.4f}]"
        )

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "min": float(np.min(ssim_scores)),
            "max": float(np.max(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }
        logging.info(f"📊 SSIM Statistics:")
        logging.info(f"  Mean:   {results['metrics']['ssim']['mean']:.4f}")
        logging.info(f"  Std:    {results['metrics']['ssim']['std']:.4f}")
        logging.info(f"  Median: {results['metrics']['ssim']['median']:.4f}")
        logging.info(
            f"  Range:  [{results['metrics']['ssim']['min']:.4f}, {results['metrics']['ssim']['max']:.4f}]"
        )

    # Compute FID if requested
    if compute_fid and FID_AVAILABLE:
        logging.info("\n📊 Computing FID score...")
        try:
            # Create temporary directories for FID computation
            fake_dir = os.path.join(
                os.path.dirname(generated_path), "..", "TargetImage"
            )
            real_dir = os.path.join(ground_truth_dir, "TargetImage")

            if os.path.exists(fake_dir) and os.path.exists(real_dir):
                fid_value: float = evaluator.compute_fid(real_dir, fake_dir)
                if fid_value >= 0:
                    results["metrics"]["fid"] = fid_value
                    logging.info(f"  FID Score: {fid_value:.2f}")
            else:
                logging.info("  ⚠ Cannot compute FID: directories not found")
        except Exception as e:
            logging.info(f"  ⚠ Error computing FID: {e}")

    logging.info("=" * 60)
    logging.info(f"{'EVALUATION SUMMARY':^60}")
    logging.info("=" * 60)
    logging.info(f"Evaluated pairs:    {evaluated_pairs}")
    logging.info(f"Missing GT images:  {missing_gt}")
    logging.info(f"LPIPS samples:      {len(lpips_scores)}")
    logging.info(f"SSIM samples:       {len(ssim_scores)}")
    logging.info("=" * 60)

    return results


def log_to_wandb(results: Dict[str, Any], args: Namespace) -> None:
    """Log results to Weights & Biases"""

    if not WANDB_AVAILABLE:
        logging.info("\n⚠ Wandb not available, skipping logging")
        return

    try:
        logging.info("=" * 60)
        logging.info(f"{'LOGGING TO WEIGHTS & BIASES':^60}")
        logging.info("=" * 60)

        # Initialize wandb
        run_name = (
            args.wandb_run_name
        )

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset_split": args.dataset_split,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
                "batch_size": args.batch_size,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "fp16": args.fp16,
                "compile": args.compile,
                "xformers": args.enable_xformers,
            },
        )

        # Log generation statistics
        num_generations = len(results.get("generations", []))
        wandb.log(
            {
                "total_generations": num_generations,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
            }
        )

        # Log metrics if available
        metrics = results.get("metrics", {})

        if "lpips" in metrics and isinstance(metrics["lpips"], dict):
            wandb.log(
                {
                    "lpips/mean": metrics["lpips"]["mean"],
                    "lpips/std": metrics["lpips"]["std"],
                    "lpips/median": metrics["lpips"]["median"],
                    "lpips/min": metrics["lpips"]["min"],
                    "lpips/max": metrics["lpips"]["max"],
                }
            )

        if "ssim" in metrics and isinstance(metrics["ssim"], dict):
            wandb.log(
                {
                    "ssim/mean": metrics["ssim"]["mean"],
                    "ssim/std": metrics["ssim"]["std"],
                    "ssim/median": metrics["ssim"]["median"],
                    "ssim/min": metrics["ssim"]["min"],
                    "ssim/max": metrics["ssim"]["max"],
                }
            )

        if "fid" in metrics:
            wandb.log({"fid": metrics["fid"]})

        # Log inference timing
        if "inference_times" in metrics and metrics["inference_times"]:
            timing_data = metrics["inference_times"]

            total_times = [t["total_time"] for t in timing_data if "total_time" in t]
            times_per_image = [
                t["time_per_image"] for t in timing_data if "time_per_image" in t
            ]

            if total_times:
                wandb.log(
                    {
                        "timing/mean_batch_time": np.mean(total_times),
                        "timing/total_time": np.sum(total_times),
                    }
                )

            if times_per_image:
                wandb.log(
                    {
                        "timing/mean_time_per_image": np.mean(times_per_image),
                        "timing/median_time_per_image": np.median(times_per_image),
                    }
                )

        # Log sample images
        logging.info("\n📸 Logging sample images...")
        sample_generations = results.get("generations", [])[:20]  # Log first 20

        sample_images = []
        for gen in sample_generations:
            target_path = gen.get("target_image_path", "")
            if target_path:
                full_path = os.path.join(args.output_dir, target_path)
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path)
                        sample_images.append(
                            wandb.Image(
                                img,
                                caption=f"{gen['character']} - {gen['style']} ({gen.get('font', '')})",
                            )
                        )
                    except Exception as e:
                        logging.info(f"  ⚠ Error loading image {full_path}: {e}")

        if sample_images:
            wandb.log({"sample_images": sample_images})
            logging.info(f"✓ Logged {len(sample_images)} sample images")

        # Create summary table
        generation_table = wandb.Table(
            columns=[
                "Character",
                "Style",
                "Font",
                "LPIPS",
                "SSIM",
                "Content Path",
                "Target Path",
            ]
        )

        for gen in results.get("generations", [])[:100]:  # Log first 100
            generation_table.add_data(
                gen.get("character", ""),
                gen.get("style", ""),
                gen.get("font", ""),
                gen.get("lpips", -1),
                gen.get("ssim", -1),
                gen.get("content_image_path", ""),
                gen.get("target_image_path", ""),
            )

        wandb.log({"generations": generation_table})

        # Finish run
        wandb.finish()

        logging.info("\n✓ Successfully logged to Weights & Biases")
        logging.info(f"  Project: {args.wandb_project}")
        logging.info(f"  Run: {run_name}")
        logging.info("=" * 60)

    except Exception as e:
        logging.info(f"⚠ Error logging to wandb: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Main function"""
    args: Namespace = parse_args()
    results: Dict[str, Any] = {}

    logging.info("=" * 60)
    logging.info("FONTDIFFUSER SYNTHESIS DATA GENERATION MAGIC")
    logging.info("=" * 60)

    try:
        # Load characters
        characters: List[str] = load_characters(
            args.characters, args.start_line, args.end_line
        )

        # Load style images with names
        style_paths_with_names: List[Tuple[str, str]] = load_style_images(
            args.style_images
        )

        logging.info(f"Initializing font manager...")
        font_manager: FontManager = FontManager(args.ttf_path)
        logging.info(f"✓ Loaded {len(font_manager.get_font_names())} fonts.")

        logging.info(f"📊 Configuration:")
        logging.info(f"  Dataset split: {args.dataset_split}")
        logging.info(
            f"  Characters: {len(characters)} (lines {args.start_line}-{args.end_line or 'end'})"
        )
        logging.info(f"  Styles: {len(style_paths_with_names)}")
        logging.info(f"  Output Directory: {args.output_dir}")
        logging.info(f"  Checkpoint Directory: {args.ckpt_dir}")
        logging.info(f"  Device: {args.device}")
        logging.info(f"  Batch Size: {args.batch_size}")
        logging.info(
            f"  Results checkpoint path: {os.path.join(args.output_dir, 'results_checkpoint.json')}"
        )

        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize generation tracker
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        # Create args namespace for pipeline
        pipeline_args: Namespace = create_args_namespace(args)

        logging.info("\nLoading FontDiffuser pipeline...")
        pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(pipeline_args)

        # Add this block to enable torch.compile if requested
        if getattr(args, "compile", False):
            import torch

            logging.info("🔧 Compiling model components with torch.compile...")
            try:
                if hasattr(pipe.model.config, "unet"):
                    pipe.model.config.unet = torch.compile(pipe.model.config.unet)
                if hasattr(pipe.model.config, "style_encoder"):
                    pipe.model.config.style_encoder = torch.compile(
                        pipe.model.config.style_encoder
                    )
                if hasattr(pipe.model.config, "content_encoder"):
                    pipe.model.config.content_encoder = torch.compile(
                        pipe.model.config.content_encoder
                    )
                logging.info("✓ Compilation complete.")
            except Exception as e:
                logging.info(f"⚠ Compilation failed: {e}")

        evaluator: QualityEvaluator = QualityEvaluator(device=args.device)

        # Generate images
        results: Dict[str, Any] = batch_generate_images(
            pipe,
            characters,
            style_paths_with_names,
            args.output_dir,
            pipeline_args,
            evaluator,
            font_manager,
            generation_tracker,
        )

        # Evaluate if requested
        if args.evaluate and args.ground_truth_dir:
            results = evaluate_results(
                results, evaluator, args.ground_truth_dir, args.compute_fid
            )

        # Save final checkpoint
        logging.info("\n💾 Saving final checkpoint...")
        save_checkpoint(results, args.output_dir)

        if args.use_wandb:
            log_to_wandb(results, args)

        logging.info("=" * 60)
        logging.info("✅ GENERATION COMPLETE!")
        logging.info("=" * 60)
        logging.info(f"Output structure:")
        logging.info(f"  {args.output_dir}/")
        logging.info(f"    ├── ContentImage/")
        logging.info(f"    │   ├── U+XXXX_char_hash.png")
        logging.info(f"    │   └── ...")
        logging.info(f"    ├── TargetImage/")
        logging.info(f"    │   ├── style0/")
        logging.info(f"    │   │   ├── U+XXXX_char_style0_hash.png")
        logging.info(f"    │   │   └── ...")
        logging.info(f"    │   └── ...")
        logging.info(f"    └── results_checkpoint.json ✅ (single source of truth)")

    except KeyboardInterrupt:
        logging.info("\n\n⚠ Generation interrupted by user!")
        logging.info("💾 Saving emergency checkpoint...")
        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
            logging.info("✓ Latest state saved to results_checkpoint.json")
        sys.exit(1)

    except Exception as e:
        logging.info(f"✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()

        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Example usage
python sample_batch.py \
    --characters chars.txt \
    --style_images styles/ \
    --enable_style_transform \
    --output_dir output/
"""