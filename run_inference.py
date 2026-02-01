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
from argparse import Namespace
from src.tools.utils import save_args_to_yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for inference entry point.

    Only defines --mode. All other arguments are passed through to submodules.
    """
    parser = argparse.ArgumentParser(
        prog="run_inference.py",
        description="FontDiffuser Inference - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:

    sample_optimized
        Single image optimized inference for one character and style.
        Best for interactive use and quick generation.

    sample_batch
        Batch inference for multiple characters and styles.
        Supports single GPU with optimizations.

    sample_distributed
        Multi-GPU distributed inference using Accelerate.
        For large-scale dataset generation with resumable checkpoints.

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

SUBCOMMAND HELP:

    Get help for specific mode (shows all available arguments):
        python run_inference.py --mode sample_optimized --help
        python run_inference.py --mode sample_batch --help
        accelerate launch run_inference.py --mode sample_distributed --help
        """,
    )

    # Only --mode parameter at this level
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample_optimized", "sample_batch", "sample_distributed"],
        default="sample_optimized",
        help="Inference mode to use (default: sample_optimized)",
    )

    return parser


def run_sample_optimized(remaining_args: list[str]) -> int:
    """Run single-image optimized inference.

    Args:
        remaining_args: All arguments after --mode (passed to sample_optimized)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_optimized import main as sample_optimized_main

        logger.info("Starting sample_optimized inference...")

        # Reconstruct sys.argv for submodule
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + remaining_args

        try:
            sample_optimized_main()
            logger.info("✓ Sample optimized inference complete")
            return 0
        finally:
            sys.argv = old_argv

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


def run_sample_batch(remaining_args: list[str]) -> int:
    """Run batch inference.

    Args:
        remaining_args: All arguments after --mode (passed to sample_batch)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_batch import main as sample_batch_main

        logger.info("Starting sample_batch inference...")

        # Reconstruct sys.argv for submodule
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + remaining_args

        try:
            sample_batch_main()
            logger.info("✓ Sample batch inference complete")
            return 0
        finally:
            sys.argv = old_argv

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


def run_sample_distributed(remaining_args: list[str]) -> int:
    """Run distributed multi-GPU inference.

    Args:
        remaining_args: All arguments after --mode (passed to sample_distributed)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from inference.sample_distributed import main as sample_distributed_main

        logger.info("Starting sample_distributed inference...")

        # Reconstruct sys.argv for submodule
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + remaining_args

        try:
            sample_distributed_main()
            logger.info("✓ Sample distributed inference complete")
            return 0
        finally:
            sys.argv = old_argv

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


def save_sampling_config(args: Namespace, output_path: str) -> None:
    """Save sampling configuration to a text file.

    Args:
        args: Argument namespace containing configuration
        output_path: Path to save the configuration file
    """
    save_args_to_yaml(args, output_path)


def main() -> int:
    """Main entry point for inference operations.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse only --mode at this level
    parser = create_parser()
    args, remaining_args = parser.parse_known_args()

    # Import centralized parser for summary flag
    from src.configs.fontdiffuser import get_parser as get_fd_parser
    fd_parser = get_fd_parser()
    fd_args, _ = fd_parser.parse_known_args(remaining_args)

    mode: str = args.mode

    logger.info("=" * 70)
    logger.info(f"FontDiffuser Inference - Mode: {mode}")
    logger.info("=" * 70)
    save_sampling_config(args, "sampling_config.yaml")

    # Torchinfo summary (if enabled)
    if getattr(fd_args, "summary", False):
        try:
            import torch
            from torchinfo import summary
            from src import build_unet, build_style_encoder, build_content_encoder

            logger.info("Generating model summary (torchinfo)...")
            # Build models with parsed args
            unet = build_unet(args=fd_args)
            style_encoder = build_style_encoder(args=fd_args)
            content_encoder = build_content_encoder(args=fd_args)

            # Example input shapes (batch_size=1, C=1, H, W)
            img_size = fd_args.resolution if hasattr(fd_args, "resolution") else 96
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet = unet.to(device)
            style_encoder = style_encoder.to(device)
            content_encoder = content_encoder.to(device)

            with open("model_summary.txt", "w", encoding="utf-8") as f:
                f.write("UNet:\n")
                f.write(str(summary(unet, input_size=(1, 1, img_size, img_size), device=device, verbose=0)))
                f.write("\n\nStyleEncoder:\n")
                f.write(str(summary(style_encoder, input_size=(1, 1, img_size, img_size), device=device, verbose=0)))
                f.write("\n\nContentEncoder:\n")
                f.write(str(summary(content_encoder, input_size=(1, 1, img_size, img_size), device=device, verbose=0)))
            logger.info("✓ Model summary saved to model_summary.txt")
        except ImportError:
            logger.warning("torchinfo not installed; skipping model summary.")
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")

    # Route to appropriate inference mode with remaining arguments
    if mode == "sample_optimized":
        logger.info("Running single-image optimized inference...")
        return run_sample_optimized(remaining_args)
    elif mode == "sample_batch":
        logger.info("Running batch inference...")
        return run_sample_batch(remaining_args)
    elif mode == "sample_distributed":
        logger.info("Running distributed multi-GPU inference...")
        return run_sample_distributed(remaining_args)
    else:
        logger.error(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
