import sys
from pathlib import Path
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image

from src.tools.filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)
from src.tools.utilities import HFTqdm
from src.tools.utils import ttf2im

from inference.sample_optimized import (
    get_content_transform,
    get_style_transform,
    load_fontdiffuser_pipeline,
)
from inference.sample_batch import (
    FontManager,
    QualityEvaluator,
    GenerationTracker,
    load_characters,
    load_style_images,
    save_checkpoint,
    log_to_wandb,
)

from src.configs.fontdiffuser import get_parser
from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline

logger = logging.getLogger(__name__)
# Optional dependencies
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logger.warning("pytorch-fid not available. Install with: pip install pytorch-fid")

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


def generate_content_images_with_accelerator(
    characters: list[str],
    font_manager: FontManager,
    output_dir: str,
    accelerator: Accelerator,
) -> dict[str, str]:
    """Generate content images distributed across GPUs.

    Args:
        characters: list of characters to generate
        font_manager: Font manager instance
        output_dir: Output directory
        accelerator: Accelerator instance

    Returns:
        Dictionary mapping character to image path
    """
    output_dir: Path = Path(output_dir)
    content_dir: Path = output_dir / "ContentImage"

    # Main process creates directory
    if accelerator.is_main_process:
        content_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")

    # Split characters across GPUs
    local_char_paths = {}
    with accelerator.split_between_processes(characters) as local_chars:
        # FIX 1: All processes must iterate (don't use conditional progress bar disable)
        for char in local_chars:
            # Find font containing character
            found_font = None
            for font_name in font_names:
                if font_manager.is_char_in_font(font_name, char):
                    found_font = font_name
                    break

            if not found_font:
                continue

            try:
                # Generate content image
                font = font_manager.get_font(found_font)
                content_filename = get_content_filename(char)
                char_path = content_dir / content_filename

                # Skip if already exists
                if char_path.exists():
                    local_char_paths[char] = str(char_path)
                    continue

                # Generate new content image
                content_img = ttf2im(font=font, char=char)
                content_img.save(str(char_path))
                local_char_paths[char] = str(char_path)

            except Exception as e:
                logger.warning(
                    f"GPU {accelerator.process_index}: Error generating '{char}': {e}"
                )

    # FIX 2: Gather results from all GPUs properly
    accelerator.wait_for_everyone()
    all_char_paths_list = gather_object([local_char_paths])

    # FIX 3: All processes merge results (not just main process)
    merged_char_paths = {}
    for paths in all_char_paths_list:
        merged_char_paths.update(paths)

    if accelerator.is_main_process:
        logger.info(f"Generated {len(merged_char_paths)} content images")

    # FIX 4: Return on ALL processes, not just main
    return merged_char_paths


def sampling_batch_with_accelerator(
    args: argparse.Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_image_path: str | Image.Image,
    font_manager: FontManager,
    font_name: str,
) -> tuple[list[Image.Image] | None, list[str] | None, float | None]:
    """Batch sampling for multiple characters.

    Args:
        args: Arguments
        pipe: Pipeline
        characters: list of characters
        style_image_path: Style image path or PIL image
        font_manager: Font manager
        font_name: Font name to use

    Returns:
        tuple of (images, valid_chars, batch_time)
    """
    # Get available characters for this font
    available_chars = font_manager.get_available_chars_for_font(font_name, characters)
    if not available_chars:
        return None, None, None

    try:
        # Load style image
        if isinstance(style_image_path, str):
            style_image = Image.open(style_image_path).convert("RGB")
        else:
            style_image = style_image_path.convert("RGB")

        style_transform = get_style_transform(args.style_image_size)
        font = font_manager.get_font(font_name)
        content_transform = get_content_transform(args.content_image_size)

        # Generate content images
        content_images = []
        for char in available_chars:
            try:
                content_image = ttf2im(font=font, char=char)
                content_images.append(content_transform(content_image))
            except Exception as e:
                logger.warning(f"Error processing '{char}': {e}")

        if not content_images:
            return None, None, None

        # Prepare batches
        content_batch = torch.stack(content_images)
        style_batch = style_transform(style_image)[None, :].repeat(
            len(content_images), 1, 1, 1
        )

        with torch.inference_mode():
            dtype = torch.float16 if args.fp16 else torch.float32
            content_batch = content_batch.to(args.device, dtype=dtype)
            style_batch = style_batch.to(args.device, dtype=dtype)

            start = time.perf_counter()

            # Process in batches
            all_images = []
            for i in HFTqdm(range(0, len(content_batch), args.batch_size)):
                batch_content = content_batch[i : i + args.batch_size]
                batch_style = style_batch[i : i + args.batch_size]

                images = pipe.generate(
                    content_images=batch_content,
                    style_images=batch_style,
                    batch_size=len(batch_content),
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
                all_images.extend(images)

            end = time.perf_counter()
            total_time = end - start

            return all_images, available_chars, total_time

    except Exception as e:
        logger.error(f"Batch sampling failed: {e}")
        return None, None, None


def batch_generate_images_with_accelerator(
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_paths_with_names: list[tuple[str, str]],
    output_dir: str,
    args: argparse.Namespace,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
    accelerator: Accelerator,
) -> dict[str, list[str] | dict[str, dict[str, float]]]:
    """Main batch generation with multi-GPU support."""

    # FIX 5: Generate content images on all processes
    char_paths = generate_content_images_with_accelerator(
        characters, font_manager, output_dir, accelerator
    )

    # FIX 6: Check on all processes, not just main
    if not char_paths:
        raise ValueError("No content images generated")

    # FIX 7: Initialize results on ALL processes
    all_chars_in_checkpoint = set(
        gen.get("character", "") for gen in generation_tracker.generations
    )
    all_styles_in_checkpoint = set(
        gen.get("style", "") for gen in generation_tracker.generations
    )
    all_chars_in_checkpoint.update(char_paths.keys())

    results = {
        "generations": [],
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": args.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": sorted(list(all_chars_in_checkpoint)),
        "styles": sorted(list(all_styles_in_checkpoint)),
        "total_chars": len(all_chars_in_checkpoint),
        "total_styles": len(all_styles_in_checkpoint),
    }

    # Setup directories (all processes can do this safely)
    target_base_dir = os.path.join(output_dir, "TargetImage")
    os.makedirs(target_base_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get primary font
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")
    primary_font = font_names[0]

    if accelerator.is_main_process:
        logger.info(
            f"Generating images: {len(characters)} chars × {len(style_paths_with_names)} styles"
        )
        logger.info(f"Using {accelerator.num_processes} GPUs")
        logger.info(f"Primary font: {primary_font}")

    # Counters (local to each process)
    local_generated_count = 0
    local_skipped_count = 0
    local_failed_count = 0

    # FIX 8: Distribute styles across GPUs
    with accelerator.split_between_processes(style_paths_with_names) as local_styles:
        for style_idx, (style_path, style_name) in enumerate(local_styles):
            try:
                style_dir = os.path.join(target_base_dir, style_name)
                os.makedirs(style_dir, exist_ok=True)

                # Filter characters not yet generated
                chars_to_generate = [
                    char
                    for char in characters
                    if not generation_tracker.is_generated(
                        char, style_name, primary_font
                    )
                ]

                if not chars_to_generate:
                    local_skipped_count += len(characters)
                    continue

                # Generate batch
                images, valid_chars, batch_time = sampling_batch_with_accelerator(
                    args,
                    pipe,
                    chars_to_generate,
                    style_path,
                    font_manager,
                    primary_font,
                )

                if images is None:
                    local_skipped_count += len(chars_to_generate)
                    continue

                # Save images (each GPU saves its own)
                for char, img in zip(valid_chars, images):
                    try:
                        target_filename = get_target_filename(char, style_name)
                        img_path = os.path.join(style_dir, target_filename)
                        content_filename = get_content_filename(char)
                        content_path_rel = f"ContentImage/{content_filename}"
                        target_path_rel = f"TargetImage/{style_name}/{target_filename}"

                        evaluator.save_image(img, img_path)

                        # Add generation record
                        generation_record = {
                            "character": char,
                            "char_code": f"U+{ord(char):04X}",
                            "style": style_name,
                            "font": primary_font,
                            "content_image_path": content_path_rel,
                            "target_image_path": target_path_rel,
                            "content_hash": compute_file_hash(char, "", primary_font),
                            "target_hash": compute_file_hash(
                                char, style_name, primary_font
                            ),
                            "content_filename": content_filename,
                            "target_filename": target_filename,
                        }

                        results["generations"].append(generation_record)
                        generation_tracker.add_generation(generation_record)
                        local_generated_count += 1

                    except Exception as e:
                        logger.warning(
                            f"GPU {accelerator.process_index}: Error saving '{char}': {e}"
                        )
                        local_failed_count += 1

                # Record inference time
                if batch_time is not None:
                    results["metrics"]["inference_times"].append(
                        {
                            "style": style_name,
                            "font": primary_font,
                            "total_time": batch_time,
                            "num_images": len(images),
                            "time_per_image": batch_time / len(images),
                        }
                    )

                # FIX 9: Checkpoint synchronization - all processes wait
                if args.save_interval > 0 and (style_idx + 1) % args.save_interval == 0:
                    accelerator.wait_for_everyone()

                    # FIX 10: Gather results from all GPUs before saving
                    all_generations = gather_object(results["generations"])

                    if accelerator.is_main_process:
                        # Merge all generations
                        merged_generations = []
                        for gen_list in all_generations:
                            merged_generations.extend(gen_list)

                        checkpoint_results = results.copy()
                        checkpoint_results["generations"] = merged_generations
                        save_checkpoint(checkpoint_results, args.output_dir)
                        logger.info(
                            f"Checkpoint saved at style {style_idx + 1}/{len(local_styles)}"
                        )

                    accelerator.wait_for_everyone()

            except Exception as e:
                logger.error(
                    f"GPU {accelerator.process_index}: Error processing {style_name}: {e}"
                )
                local_failed_count += (
                    len(chars_to_generate)
                    if "chars_to_generate" in locals()
                    else len(characters)
                )

    # FIX 11: Gather final results from all GPUs
    accelerator.wait_for_everyone()

    all_generations = gather_object(results["generations"])
    all_inference_times = gather_object(results["metrics"]["inference_times"])

    # FIX 12: Gather counters from all processes
    all_generated_counts = gather_object([local_generated_count])
    all_skipped_counts = gather_object([local_skipped_count])
    all_failed_counts = gather_object([local_failed_count])

    if accelerator.is_main_process:
        # Merge results
        merged_generations = []
        for gen_list in all_generations:
            merged_generations.extend(gen_list)

        merged_inference_times = []
        for time_list in all_inference_times:
            merged_inference_times.extend(time_list)

        results["generations"] = merged_generations
        results["metrics"]["inference_times"] = merged_inference_times

        # Sum counters
        total_generated = sum(all_generated_counts)
        total_skipped = sum(all_skipped_counts)
        total_failed = sum(all_failed_counts)

        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generated: {total_generated} images")
        logger.info(f"Skipped: {total_skipped} images")
        logger.info(f"Failed: {total_failed} images")
        logger.info(f"Total characters: {len(all_chars_in_checkpoint)}")
        logger.info(f"Total styles: {len(all_styles_in_checkpoint)}")
        logger.info("=" * 60)

        # Update results metadata
        results["characters"] = sorted(list(all_chars_in_checkpoint))
        results["styles"] = sorted(list(all_styles_in_checkpoint))
        results["total_chars"] = len(all_chars_in_checkpoint)
        results["total_styles"] = len(all_styles_in_checkpoint)

        save_checkpoint(results, args.output_dir)

    # FIX 13: Wait before returning
    accelerator.wait_for_everyone()
    return results


def evaluate_results_with_accelerator(
    results: dict[str, list[dict[str, str]]],
    evaluator: QualityEvaluator,
    output_dir: str,
    ground_truth_dir: str | None,
    compute_fid: bool = False,
    accelerator: Accelerator | None = None,
) -> dict:
    """Evaluate generated images on main process."""

    if not accelerator or not accelerator.is_main_process:
        return results

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        logger.info("No ground truth directory provided, skipping evaluation")
        return results

    logger.info("=" * 60)
    logger.info("EVALUATING GENERATED IMAGES")
    logger.info("=" * 60)

    lpips_scores = []
    ssim_scores = []
    evaluated = 0

    target_base_dir = os.path.join(output_dir, "TargetImage")

    for gen in results["generations"]:
        char = gen["character"]
        style = gen["style"]
        target_path = os.path.join(target_base_dir, style, gen["target_filename"])

        if not os.path.exists(target_path):
            continue

        # Try to find ground truth
        gt_filename = get_target_filename(char, style)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            continue

        try:
            generated_img = Image.open(target_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            lpips_score = evaluator.compute_lpips(generated_img, gt_img)
            if lpips_score >= 0:
                lpips_scores.append(lpips_score)
                gen["lpips"] = lpips_score

            ssim_score = evaluator.compute_ssim(generated_img, gt_img)
            if ssim_score >= 0:
                ssim_scores.append(ssim_score)
                gen["ssim"] = ssim_score

            evaluated += 1
        except Exception as e:
            logger.warning(f"Error evaluating {char}/{style}: {e}")

    # Log metrics
    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }
        logger.info(f"LPIPS: mean={results['metrics']['lpips']['mean']:.4f}")

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }
        logger.info(f"SSIM: mean={results['metrics']['ssim']['mean']:.4f}")

    logger.info(f"Evaluated {evaluated} image pairs")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()

    # Convert image sizes to tuples
    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else "no",
    )

    # Override device to use accelerator's device
    args.device = accelerator.device

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("FontDiffuser Multi-GPU Batch Sampler")
        logger.info("=" * 60)
        logger.info(f"Using {accelerator.num_processes} GPUs")
        logger.info(f"FST Enhancement: {'ENABLED' if args.use_fst else 'DISABLED'}")
        logger.info(f"Mixed Precision: {'fp16' if args.fp16 else 'none'}")

    try:
        # Load data on all processes
        characters = load_characters(args.characters, args.start_line, args.end_line)
        style_paths_with_names = load_style_images(args.style_images)

        # Validate after loading
        if not characters:
            raise ValueError("No characters loaded")
        if not style_paths_with_names:
            raise ValueError("No style images loaded")

        # Initialize font manager on all processes
        font_manager = FontManager(args.ttf_path)
        if not font_manager.get_font_names():
            raise ValueError(f"No fonts loaded from {args.ttf_path}")

        if accelerator.is_main_process:
            logger.info(f"✓ Loaded {len(font_manager.get_font_names())} fonts")
            logger.info(f"📊 Configuration:")
            logger.info(f"  Dataset split: {args.dataset_split}")
            logger.info(
                f"  Characters: {len(characters)} (lines {args.start_line}-{args.end_line or 'end'})"
            )
            logger.info(f"  Styles: {len(style_paths_with_names)}")
            logger.info(f"  Output Directory: {args.output_dir}")
            logger.info(f"  Checkpoint Directory: {args.ckpt_dir}")
            logger.info(f"  Device per process: {args.device}")
            logger.info(f"  Batch Size: {args.batch_size}")
            logger.info(
                f"  Results checkpoint path: {os.path.join(args.output_dir, 'results_checkpoint.json')}"
            )

        # Create output directory on all processes
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # Initialize generation tracker on all processes
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        # FIX 15: Load pipeline on all processes
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("Loading FontDiffuser pipeline...")
            logger.info("=" * 60)

        pipe = load_fontdiffuser_pipeline(args, use_fst=args.use_fst)
        if accelerator.is_main_process:
            logger.info("✓ Pipeline loaded successfully.")

        # FIX 16: Prepare pipeline BEFORE wait_for_everyone
        pipe = accelerator.prepare(pipe)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logger.info("✓ Pipeline prepared with Accelerator.")

        # Initialize evaluator on all processes
        evaluator = QualityEvaluator(device=args.device)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logger.info("✓ Quality evaluator initialized.")
            logger.info(
                f"Generating {len(characters)} × {len(style_paths_with_names)} images"
            )

        # FIX 17: All processes run generation
        results = batch_generate_images_with_accelerator(
            pipe,
            characters,
            style_paths_with_names,
            args.output_dir,
            args,
            evaluator,
            font_manager,
            generation_tracker,
            accelerator,
        )

        accelerator.wait_for_everyone()

        # Evaluate on main process only
        if accelerator.is_main_process:
            if args.evaluate and args.ground_truth_dir:
                results = evaluate_results_with_accelerator(
                    results,
                    evaluator,
                    args.output_dir,
                    args.ground_truth_dir,
                    args.compute_fid,
                    accelerator,
                )

            # Log to wandb
            if args.use_wandb:
                log_to_wandb(results, args)

            logger.info("=" * 60)
            logger.info("✅ NomGenie dataset generation complete!")
            logger.info("=" * 60)

        # FIX 18: Final synchronization
        accelerator.wait_for_everyone()

    except KeyboardInterrupt:
        logger.warning(
            f"GPU {accelerator.process_index}: Generation interrupted by user"
        )
        accelerator.wait_for_everyone()
        sys.exit(130)

    except Exception as e:
        logger.error(
            f"GPU {accelerator.process_index}: Fatal error: {e}", exc_info=True
        )
        accelerator.wait_for_everyone()
        sys.exit(1)

    finally:
        if accelerator.is_main_process:
            logger.info("Cleaning up resources...")
        try:
            # FIX 19: Proper cleanup
            accelerator.wait_for_everyone()
            accelerator.free_memory()

            # Only destroy process group if initialized
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            if accelerator.is_main_process:
                logger.info("✓ Cleanup complete")

        except Exception as e:
            logger.warning(
                f"GPU {accelerator.process_index}: Error during cleanup: {e}"
            )

if __name__ == "__main__":
    main()