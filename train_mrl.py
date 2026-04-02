"""
Training runner for FontDiffuserWithMRL.
Entry point that initializes and runs the MRL trainer.
"""

import argparse
import os
import sys

from src.configs.fontdiffuser import get_parser
from src.trainers.trainer_mrl import FontDiffuserMRLTrainer

# Setup logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments with MRL-specific options."""
    parser: argparse.ArgumentParser = get_parser()

    # Add MRL-specific arguments
    parser.add_argument(
        "--use_mrl",
        action="store_true",
        default=True,
        help="Enable Matryoshka Representation Learning (MRL)",
    )
    parser.add_argument(
        "--mrl_nesting_dims",
        type=str,
        default="64,128,256,512",
        help="MRL nesting dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mrl_freq_radii",
        type=str,
        default="0.1,0.3,0.5",
        help="MRL frequency band radii (comma-separated); must have len(nesting_dims)-1 values",
    )
    parser.add_argument(
        "--mrl_content_weight",
        type=float,
        default=1.0,
        help="Weight for MRL content loss",
    )
    parser.add_argument(
        "--mrl_fourier_weight",
        type=float,
        default=0.3,
        help="Weight for MRL Fourier alignment loss",
    )
    parser.add_argument(
        "--mrl_temperature",
        type=float,
        default=0.07,
        help="Temperature for MRL InfoNCE loss",
    )
    parser.add_argument(
        "--use_mrl_fourier_alignment",
        action="store_true",
        default=True,
        help="Use MRL Fourier alignment",
    )
    parser.add_argument(
        "--mrl_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for MRL training (MRL only phase)",
    )
    parser.add_argument(
        "--mrl_rampdown_steps",
        type=int,
        default=1000,
        help="Steps over which to ramp down MRL weight",
    )
    parser.add_argument(
        "--mrl_start_weight",
        type=float,
        default=1.0,
        help="MRL loss weight at start of phase 2 (after warmup)",
    )
    parser.add_argument(
        "--mrl_final_weight",
        type=float,
        default=0.3,
        help="Final MRL loss weight (phase 3 and beyond)",
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
        logger.info("FontDiffuserWithMRL Training Configuration")
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
        logger.info(f"MRL enabled: {args.use_mrl}")
        if args.use_mrl:
            logger.info(f"  Nesting dims: {args.mrl_nesting_dims}")
            logger.info(f"  Freq radii: {args.mrl_freq_radii}")
            logger.info(f"  Content weight: {args.mrl_content_weight}")
            logger.info(f"  Fourier weight: {args.mrl_fourier_weight}")
            logger.info(f"  Temperature: {args.mrl_temperature}")
            logger.info(f"  Warmup steps: {args.mrl_warmup_steps}")
            logger.info(f"  Rampdown steps: {args.mrl_rampdown_steps}")
            logger.info(f"  Start weight: {args.mrl_start_weight}")
            logger.info(f"  Final weight: {args.mrl_final_weight}")
        logger.info(f"Phase 2 (SCR): {args.phase_2}")
        logger.info(f"Batch size: {args.train_batch_size}")
        logger.info(f"Max steps: {args.max_train_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)

        # Initialize trainer
        trainer = FontDiffuserMRLTrainer(args)

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


"""
Example training commands for FontDiffuserWithMRL:

# ============================================================================
# PHASE 1: Train with FST + MRL from scratch
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_phase1" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --ckpt_interval=5000 \
    --log_interval=100 \
    --output_dir="outputs/mrl_training" \
    --style_source_same_prob=0.5 \
    --mixed_precision="fp16" \
    --fst_num_queries=220 \
    --fst_query_dim=128 \
    --fst_num_scales=5 \
    --mrl_nesting_dims="64,128,256,512" \
    --mrl_freq_radii="0.1,0.3,0.5" \
    --mrl_content_weight=1.0 \
    --mrl_fourier_weight=0.3 \
    --mrl_warmup_steps=500 \
    --mrl_rampdown_steps=1000

# ============================================================================
# PHASE 2: Fine-tune with SCR loss (from Phase 1 checkpoint)
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --phase_2 \
    --phase_1_ckpt_dir="outputs/mrl_training/checkpoint_step_100000" \
    --scr_ckpt_path="ckpt/scr_210000.pth" \
    --experience_name="fontdiffuser_mrl_phase2" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --output_dir="outputs/mrl_training_phase2" \
    --mixed_precision="fp16" \
    --mrl_start_weight=0.5 \
    --mrl_final_weight=0.1

# ============================================================================
# CUSTOM MRL CONFIGURATION: Experiment with different MRL settings
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_custom" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/mrl_custom" \
    --fst_feature_channels="128,256,512,1024" \
    --fst_num_queries=128 \
    --mrl_nesting_dims="128,256,512" \
    --mrl_freq_radii="0.2,0.6" \
    --mrl_content_weight=0.8 \
    --mrl_fourier_weight=0.2 \
    --mrl_temperature=0.05 \
    --style_source_same_prob=0.7

# ============================================================================
# MULTI-GPU: Train on multiple GPUs
# ============================================================================
accelerate launch --multi_gpu --num_processes=4 train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_multigpu" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --output_dir="outputs/mrl_multigpu" \
    --mixed_precision="fp16" \
    --mrl_nesting_dims="64,128,256,512" \
    --mrl_freq_radii="0.1,0.3,0.5"

# ============================================================================
# DEBUGGING: Quick test run
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_debug" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --max_train_steps=100 \
    --ckpt_interval=50 \
    --log_interval=10 \
    --output_dir="outputs/debug_mrl"

# ============================================================================
# WITHOUT FOURIER ALIGNMENT: Reduce memory usage
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --use_mrl_fourier_alignment=False \
    --experience_name="fontdiffuser_mrl_no_fourier" \
    --data_root="my_dataset" \
    --train_batch_size=8 \
    --max_train_steps=100000 \
    --output_dir="outputs/mrl_no_fourier" \
    --mrl_freq_radii="0.1,0.3,0.5" \
    --learning_rate=5e-5

# ============================================================================
# RESUME: Resume from checkpoint
# ============================================================================
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_resumed" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=150000 \
    --resume_from_checkpoint="outputs/mrl_training/checkpoint_step_100000/training_state.pt" \
    --output_dir="outputs/mrl_training"
"""
