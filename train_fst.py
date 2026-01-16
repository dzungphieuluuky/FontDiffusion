"""
Training runner for FontDiffuserWithFST.
Simple entry point that initializes and runs the FST trainer.
"""

import logging
import os
import sys

from configs.fontdiffuser import get_parser
from training.trainer_fst import FontDiffuserFSTTrainer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments with FST-specific options."""
    parser = get_parser()

    # Add FSTDiff-specific arguments
    parser.add_argument(
        "--use_fst",
        action="store_true",
        help="Use FSTDiff enhancement with MSSE and FST modules",
    )
    parser.add_argument(
        "--fst_feature_channels",
        type=str,
        default="64,128,256,512,1024",
        help="Feature channels for FST module (comma-separated)",
    )
    parser.add_argument(
        "--fst_num_queries",
        type=int,
        default=220,
        help="Number of learnable queries in FST (default 220 for 256 total)",
    )
    parser.add_argument(
        "--fst_query_dim",
        type=int,
        default=128,
        help="Dimension of query vectors in FST",
    )
    parser.add_argument(
        "--fst_num_scales",
        type=int,
        default=5,
        help="Number of multi-scale features in MSSE",
    )
    parser.add_argument(
        "--style_source_same_prob",
        type=float,
        default=0.5,
        help="Probability that source and target style use same font style",
    )
    parser.add_argument(
        "--freeze_original_encoders",
        action="store_true",
        help="Freeze original style and content encoders during training",
    )

    args = parser.parse_args()

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
        args = get_args()

        # Log configuration
        logger.info("=" * 80)
        logger.info("FONTDIFFUSER FST TRAINING")
        logger.info("=" * 80)
        logger.info(f"Experiment: {args.experience_name}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"FST enabled: {args.use_fst}")
        if args.use_fst:
            logger.info(f"  Feature channels: {args.fst_feature_channels}")
            logger.info(f"  Num queries: {args.fst_num_queries}")
            logger.info(f"  Query dim: {args.fst_query_dim}")
            logger.info(f"  Num scales: {args.fst_num_scales}")
            logger.info(f"  Freeze encoders: {args.freeze_original_encoders}")
            logger.info(f"  Style source same prob: {args.style_source_same_prob}")
        logger.info(f"Phase 2 (SCR): {args.phase_2}")
        logger.info(f"Batch size: {args.train_batch_size}")
        logger.info(f"Max steps: {args.max_train_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)

        # Initialize trainer
        trainer = FontDiffuserFSTTrainer(args)

        # Setup components
        logger.info("Setting up training components...")
        trainer.setup()
        logger.info("✓ Setup complete")

        # Run training
        logger.info("Starting training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Example training commands for FontDiffuserWithFST:

# ============================================================================
# PHASE 1: Train with FST modules from scratch
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_phase1" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --ckpt_interval=5000 \
    --log_interval=100 \
    --output_dir="outputs/fst_training" \
    --style_source_same_prob=0.5 \
    --mixed_precision="fp16" \
    --fst_num_queries=220 \
    --fst_query_dim=128 \
    --fst_num_scales=5

# ============================================================================
# PHASE 2: Fine-tune with SCR loss (from Phase 1 checkpoint)
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --phase_2 \
    --phase_1_ckpt_dir="outputs/fst_training/checkpoint_step_100000" \
    --scr_ckpt_path="ckpt/scr_210000.pth" \
    --experience_name="fontdiffuser_fst_phase2" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --output_dir="outputs/fst_training_phase2" \
    --freeze_original_encoders \
    --mixed_precision="fp16"

# ============================================================================
# BASELINE: Train original model (without FST) for comparison
# ============================================================================
accelerate launch train_fst.py \
    --experience_name="fontdiffuser_baseline" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --output_dir="outputs/baseline_training" \
    --mixed_precision="fp16"

# ============================================================================
# RESUME: Resume from checkpoint
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_resumed" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=150000 \
    --resume_from_checkpoint="outputs/fst_training/checkpoint_step_100000/training_state.pt" \
    --output_dir="outputs/fst_training"

# ============================================================================
# MULTI-GPU: Train on multiple GPUs
# ============================================================================
accelerate launch --multi_gpu --num_processes=4 train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_multigpu" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --output_dir="outputs/fst_training_multigpu" \
    --mixed_precision="fp16"

# ============================================================================
# CUSTOM FST CONFIGURATION: Experiment with different FST settings
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_custom" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/fst_custom" \
    --fst_feature_channels="128,256,512,1024" \
    --fst_num_queries=128 \
    --fst_query_dim=256 \
    --fst_num_scales=4 \
    --style_source_same_prob=0.7

# ============================================================================
# FINE-TUNE ONLY FST: Freeze original encoders, train only FST modules
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --freeze_original_encoders \
    --phase_1_ckpt_dir="pretrained/fontdiffuser_base" \
    --experience_name="fontdiffuser_fst_finetune" \
    --data_root="my_dataset" \
    --train_batch_size=8 \
    --max_train_steps=50000 \
    --learning_rate=1e-4 \
    --output_dir="outputs/fst_finetune"

# ============================================================================
# DEBUGGING: Quick test run
# ============================================================================
accelerate launch train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_debug" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --max_train_steps=100 \
    --ckpt_interval=50 \
    --log_interval=10 \
    --output_dir="outputs/debug"
"""
