"""
Training runner for FontDiffuserUnicalliTrainer.
Entry point that initializes and runs the UniCalli-enhanced FST trainer.
"""

import argparse
import os
import sys

from src.configs.fontdiffuser import get_parser
from src.trainers.trainer_unicalli import FontDiffuserUnicalliTrainer

# Setup logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_args():
    """Parse command line arguments with FST and UniCalli specific options."""
    parser: argparse.ArgumentParser = get_parser()
    
    # UniCalli Arguments Integration
    unicalli_group = parser.add_argument_group("UniCalli Improvements")
    unicalli_group.add_argument(
        "--style_noise_fraction", type=float, default=0.0,
        help="Maximum fraction of full noise applied to style images. 0.0 = completely clean."
    )
    unicalli_group.add_argument(
        "--p_drop_content", type=float, default=0.1,
        help="Probability of dropping content condition."
    )
    unicalli_group.add_argument(
        "--p_drop_style", type=float, default=0.05,
        help="Probability of dropping style condition."
    )
    unicalli_group.add_argument(
        "--use_hard_negative", action="store_true", default=True,
        help="If True, replace with a shuffled batch sample instead of pure noise."
    )
    unicalli_group.add_argument(
        "--curriculum_steps", type=int, default=1000,
        help="If > 0, p_drop ramps from 0 to target over this many steps."
    )

    args: argparse.Namespace = parser.parse_args()

    # Handle local rank for distributed training
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Convert image sizes to tuples
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():
    """Main training entry point."""
    try:
        # Parse arguments
        args: argparse.Namespace = get_args()

        # Log configuration
        logger.info("=" * 80)
        logger.info("FontDiffusion with UniCalli Improvements - Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Experiment: {args.experience_name}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"FST enabled: {args.use_fst}")
        if args.use_fst:
            logger.info(f"  Feature channels: {args.fst_feature_channels}")
            logger.info(f"  Num queries: {args.fst_num_queries}")
            logger.info(f"  Query dim: {args.fst_query_dim}")
            logger.info(f"  Num scales: {args.fst_num_scales}")
            logger.info(f"  Style source same prob: {args.style_source_same_prob}")
            
        logger.info("UniCalli Improvements:")
        logger.info(f"  Style noise fraction: {args.style_noise_fraction}")
        logger.info(f"  P_drop Content: {args.p_drop_content}")
        logger.info(f"  P_drop Style: {args.p_drop_style}")
        logger.info(f"  Use hard negative: {args.use_hard_negative}")
        logger.info(f"  Curriculum steps: {args.curriculum_steps}")
            
        logger.info(f"Phase 2 (SCR): {args.phase_2}")
        logger.info(f"Batch size: {args.train_batch_size}")
        logger.info(f"Max steps: {args.max_train_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)

        # Initialize trainer
        trainer = FontDiffuserUnicalliTrainer(args)

        # Setup components
        logger.info("Setting up training components...")
        trainer.setup()
        logger.info("✓ Setup complete")

        # Run training
        logger.info("Starting training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)

        # Export to ONNX if flag is set
        if args.export_onnx:
            logger.info("\nStarting ONNX export...")
            trainer.export_to_onnx()
            logger.info("✓ ONNX export finished!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
