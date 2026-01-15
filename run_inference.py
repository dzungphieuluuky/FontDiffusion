"""
FontDiffuser Inference Entry Point

This module provides a unified entry point for FontDiffuser inference tasks.
Supports single-image, batch, and distributed multi-GPU generation.

Usage:
    # Single image inference
    python run_inference.py --mode sample_optimized \
        --ckpt_dir ckpt/ \
        --content_character "A" \
        --style_image_path style.png

    # Batch inference
    python run_inference.py --mode sample_batch \
        --characters chars.txt \
        --style_images styles/ \
        --output_dir results/

    # Multi-GPU distributed inference
    accelerate launch run_inference.py --mode sample_distributed \
        --characters chars.txt \
        --style_images styles/ \
        --output_dir results/
"""

import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for inference entry point."""
    parser = argparse.ArgumentParser(
        prog="run_inference.py",
        description="FontDiffuser Inference - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

    Single character generation:
        python run_inference.py --mode sample_optimized \\
            --ckpt_dir ckpt/ \\
            --character_input \\
            --content_character "你" \\
            --style_image_path style.png \\
            --save_image

    Batch generation (single GPU):
        python run_inference.py --mode sample_batch \\
            --characters characters.txt \\
            --style_images "styles/*.png" \\
            --ttf_path "fonts/default.ttf" \\
            --output_dir my_dataset/ \\
            --batch_size 8 \\
            --fp16 \\
            --compile

    Distributed generation (multi-GPU with checkpoint):
        accelerate launch run_inference.py --mode sample_distributed \\
            --characters characters.txt \\
            --style_images styles/ \\
            --ttf_path "fonts/*.ttf" \\
            --output_dir my_dataset/ \\
            --batch_size 4 \\
            --save_interval 10 \\
            --use_wandb

    Resume interrupted generation:
        python run_inference.py --mode sample_batch \\
            --characters characters.txt \\
            --style_images styles/ \\
            --output_dir my_dataset/
        """,
    )

    # Inference mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample_optimized", "sample_batch", "sample_distributed"],
        default="sample_optimized",
        help="Inference mode to use (default: sample_optimized)",
    )

    # Common arguments for all modes
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpt/",
        help="Directory containing model checkpoints (default: ckpt/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision for faster inference and lower memory",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for optimized inference (requires PyTorch 2.0+)",
    )

    # Sample optimized arguments
    parser.add_argument(
        "--character_input",
        action="store_true",
        help="Use character input instead of image input",
    )
    parser.add_argument(
        "--content_character",
        type=str,
        default=None,
        help="Character to generate (for --character_input mode)",
    )
    parser.add_argument(
        "--content_image_path",
        type=str,
        default=None,
        help="Path to content image (for image input mode)",
    )
    parser.add_argument(
        "--style_image_path",
        type=str,
        default=None,
        help="Path to style image (required for all modes)",
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
        help="Save generated images to disk",
    )
    parser.add_argument(
        "--save_image_dir",
        type=str,
        default="results/",
        help="Directory to save generated images (default: results/)",
    )

    # Batch and distributed arguments
    parser.add_argument(
        "--characters",
        type=str,
        default=None,
        help="Path to characters file (one per line) or comma-separated characters",
    )
    parser.add_argument(
        "--style_images",
        type=str,
        default=None,
        help="Path to style images directory, glob pattern, or comma-separated paths",
    )
    parser.add_argument(
        "--ttf_path",
        type=str,
        default=None,
        help="Path to TTF font file, directory, or glob pattern",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for generated dataset (default: results/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for conditional generation (default: 7.5)",
    )

    # Distributed arguments
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save checkpoint every N styles (default: 1)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fontdiffuser",
        help="Weights & Biases project name (default: fontdiffuser)",
    )

    # Batch-specific arguments
    parser.add_argument(
        "--start_line",
        type=int,
        default=0,
        help="Start line for character input (default: 0)",
    )
    parser.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="End line for character input (default: None, process all)",
    )

    # Model arguments
    parser.add_argument(
        "--content_image_size",
        type=int,
        default=64,
        help="Content image size (default: 64)",
    )
    parser.add_argument(
        "--style_image_size",
        type=int,
        default=256,
        help="Style image size (default: 256)",
    )

    return parser


def run_sample_optimized(args: argparse.Namespace) -> int:
    """Run single-image optimized inference.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_optimized import main as sample_optimized_main

        logger.info("Starting sample_optimized inference...")
        sample_optimized_main(args)
        logger.info("✓ Sample optimized inference complete")
        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in sample_optimized: {e}", exc_info=True)
        return 1


def run_sample_batch(args: argparse.Namespace) -> int:
    """Run batch inference.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_batch import main as sample_batch_main

        logger.info("Starting sample_batch inference...")
        sample_batch_main(args)
        logger.info("✓ Sample batch inference complete")
        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in sample_batch: {e}", exc_info=True)
        return 1


def run_sample_distributed(args: argparse.Namespace) -> int:
    """Run distributed multi-GPU inference.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_distributed import main as sample_distributed_main

        logger.info("Starting sample_distributed inference...")
        sample_distributed_main(args)
        logger.info("✓ Sample distributed inference complete")
        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in sample_distributed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point for inference operations.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate mode
    mode = args.mode

    logger.info("=" * 70)
    logger.info(f"FontDiffuser Inference - Mode: {mode}")
    logger.info("=" * 70)

    # Route to appropriate inference mode
    if mode == "sample_optimized":
        return run_sample_optimized(args)
    elif mode == "sample_batch":
        return run_sample_batch(args)
    elif mode == "sample_distributed":
        return run_sample_distributed(args)
    else:
        logger.error(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())