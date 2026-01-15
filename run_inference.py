"""
FontDiffuser Inference Entry Point

This module provides a unified entry point for FontDiffuser inference tasks.
Supports single-image, batch, and distributed multi-GPU generation.

Usage:
    # Single image inference
    python inference.py sample_optimized \
        --ckpt_dir ckpt/ \
        --content_character "A" \
        --style_image_path style.png

    # Batch inference
    python inference.py sample_batch \
        --characters chars.txt \
        --style_images styles/ \
        --output_dir results/

    # Multi-GPU distributed inference
    accelerate launch inference.py sample_distributed \
        --characters chars.txt \
        --style_images styles/ \
        --output_dir results/
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_sample_optimized(argv: list[str]) -> int:
    """Run single-image optimized inference."""
    try:
        from inference.sample_optimized import main as sample_optimized_main

        # Update sys.argv temporarily
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + argv

        try:
            sample_optimized_main()
            return 0
        finally:
            sys.argv = old_argv

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def run_sample_batch(argv: list[str]) -> int:
    """Run batch inference."""
    try:
        from inference.sample_batch import main as sample_batch_main

        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + argv

        try:
            sample_batch_main()
            return 0
        finally:
            sys.argv = old_argv

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def run_sample_distributed(argv: list[str]) -> int:
    """Run distributed multi-GPU inference."""
    try:
        from inference.sample_distributed import main as sample_distributed_main

        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + argv

        try:
            sample_distributed_main()
            return 0
        finally:
            sys.argv = old_argv

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def print_usage() -> None:
    """Print usage information."""
    usage_text = """
    ╔════════════════════════════════════════════════════════════════╗
    ║          FontDiffuser Inference - Unified Entry Point          ║
    ╚════════════════════════════════════════════════════════════════╝

    USAGE:
        python inference.py <command> [options]

    COMMANDS:
        sample_optimized, single
            Single image optimized inference for one character and style.
            Best for interactive use and quick generation.

            Example:
                python inference.py sample_optimized \\
                    --ckpt_dir ckpt/ \\
                    --content_character "A" \\
                    --style_image_path style.png \\
                    --save_image \\
                    --save_image_dir results/

        sample_batch, batch
            Batch inference for multiple characters and styles.
            Supports single GPU with optimizations.

            Example:
                python inference.py sample_batch \\
                    --characters chars.txt \\
                    --style_images styles/ \\
                    --output_dir results/ \\
                    --batch_size 4 \\
                    --num_inference_steps 20

        sample_distributed, distributed
            Multi-GPU distributed inference using Accelerate.
            For large-scale dataset generation with resumable checkpoints.

            Example:
                accelerate launch inference.py sample_distributed \\
                    --characters chars.txt \\
                    --style_images styles/ \\
                    --output_dir results/ \\
                    --batch_size 4

        help, --help, -h
            Show this help message.

        --version, -v
            Show FontDiffuser version.

    GLOBAL OPTIONS:
        --help, -h          Show help for specific command
        --version, -v       Show FontDiffuser version

    FEATURES:
        ✓ Hash-based file naming for reproducibility
        ✓ Checkpoint-based resumable generation
        ✓ Multi-GPU distributed support
        ✓ Mixed precision (FP16) optimization
        ✓ Memory-efficient batch processing
        ✓ Quality evaluation (LPIPS, SSIM, FID)
        ✓ Weights & Biases (wandb) integration

    OUTPUT STRUCTURE:
        results/
        ├── ContentImage/
        │   ├── U+XXXX_char_hash.png
        │   └── ...
        ├── TargetImage/
        │   ├── style_name/
        │   │   ├── U+XXXX_char_style_hash.png
        │   │   └── ...
        │   └── ...
        └── results_checkpoint.json        (resumable checkpoint)

    EXAMPLES:

        1. Single character generation:
            python inference.py sample_optimized \\
                --ckpt_dir ckpt/ \\
                --character_input \\
                --content_character "你" \\
                --style_image_path style.png \\
                --save_image

        2. Batch generation (single GPU):
            python inference.py sample_batch \\
                --characters characters.txt \\
                --style_images "styles/*.png" \\
                --ttf_path "fonts/default.ttf" \\
                --output_dir my_dataset/ \\
                --batch_size 8 \\
                --fp16 \\
                --compile

        3. Distributed generation (multi-GPU with checkpoint):
            accelerate launch inference.py sample_distributed \\
                --characters characters.txt \\
                --style_images styles/ \\
                --ttf_path "fonts/*.ttf" \\
                --output_dir my_dataset/ \\
                --batch_size 4 \\
                --save_interval 10 \\
                --use_wandb

        4. Resume interrupted generation:
            # results_checkpoint.json is automatically loaded if exists
            python inference.py sample_batch \\
                --characters characters.txt \\
                --style_images styles/ \\
                --output_dir my_dataset/

    TROUBLESHOOTING:

        Q: ImportError: No module named 'inference'
        A: Run from project root: cd d:\\School\\FontDiffusion

        Q: CUDA out of memory
        A: Reduce --batch_size, enable --fp16, or use smaller model

        Q: Generation interrupted
        A: Run same command again - checkpoint auto-loads and resumes

    FOR MORE INFO:
        See README.md for detailed documentation
    """
    print(usage_text)


def main() -> int:
    """Main entry point for inference operations."""

    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1]
    args = sys.argv[2:]

    if command in ["sample_optimized", "single"]:
        return run_sample_optimized(args)

    elif command in ["sample_batch", "batch"]:
        return run_sample_batch(args)

    elif command in ["sample_distributed", "distributed"]:
        return run_sample_distributed(args)

    elif command in ["--help", "-h", "help"]:
        print_usage()
        return 0

    else:
        logger.error(f"Unknown command: {command}")
        print_usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
