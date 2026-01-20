"""
FontDiffuser Inference Entry Point with Hydra Configuration

Supports single-image, batch, and distributed multi-GPU generation.

Usage:
    # Single image inference
    python run_inference.py --config-name=optimized \
        ckpt_dir=ckpt/ \
        content_character="A" \
        style_image_path=style.png \
        save_image=true

    # Batch inference
    python run_inference.py --config-name=batch \
        characters=chars.txt \
        style_images=styles/ \
        output_dir=results/

    # Multi-GPU distributed
    accelerate launch run_inference.py --config-name=distributed \
        characters=chars.txt \
        style_images=styles/ \
        output_dir=results/
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for inference operations"""
    
    if len(sys.argv) < 2:
        logger.error("Missing required argument: --config-name")
        logger.info("\nUsage:")
        logger.info("  python run_inference.py --config-name optimized [other args...]")
        logger.info("  python run_inference.py --config-name batch [other args...]")
        logger.info("  python run_inference.py --config-name distributed [other args...]")
        return 1

    config_name = None
    for arg in sys.argv[1:]:
        if arg.startswith("--config-name="):
            config_name = arg.split("=")[1]
            break
        elif arg == "--config-name" and sys.argv.index(arg) + 1 < len(sys.argv):
            config_name = sys.argv[sys.argv.index(arg) + 1]
            break

    if config_name == "optimized":
        logger.info("Running optimized (single-image) inference...")
        from inference.sample_optimized import main as sample_optimized_main
        try:
            sample_optimized_main()
            return 0
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1

    elif config_name == "batch":
        logger.info("Running batch inference...")
        from inference.sample_batch import main as sample_batch_main
        try:
            sample_batch_main()
            return 0
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1

    elif config_name == "distributed":
        logger.info("Running distributed inference...")
        from inference.sample_distributed import main as sample_distributed_main
        try:
            sample_distributed_main()
            return 0
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1

    else:
        logger.error(f"Unknown config: {config_name}")
        logger.info("Available configs: optimized, batch, distributed")
        return 1


if __name__ == "__main__":
    sys.exit(main())