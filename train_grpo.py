"""
Training runner for FontDiffuserGRPOTrainer.
Simple entry point that initializes and runs the GRPO trainer.

The GRPO trainer extends FontDiffuserDROTrainer by adding a Group Relative
Policy Optimization signal on top of the DRO reward losses. For each
(content, style) pair, G rollouts are generated, scored with the reward
module, normalized within the group, then used to compute a clipped
importance-weighted policy gradient loss.

Usage
-----
# GRPO fine-tuning from DRO checkpoint (single GPU):
accelerate launch train_grpo.py \
    --use_fst \
    --use_dro \
    --use_grpo \
    --phase_1_ckpt_dir outputs/dro_training/final \
    --experience_name fontdiffuser_grpo_phase1 \
    --data_root my_dataset \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 50000 \
    --learning_rate 1e-5 \
    --ckpt_interval 5000 \
    --log_interval 100 \
    --output_dir outputs/grpo_training \
    --dro_weight 0.05 \
    --dro_warmup_steps 500 \
    --grpo_group_size 4 \
    --grpo_sample_steps 5 \
    --grpo_pg_weight 0.01 \
    --grpo_kl_coeff 0.01 \
    --grpo_warmup_steps 1000 \
    --mixed_precision fp16

# GRPO + Phase 2 SCR contrastive loss:
accelerate launch train_grpo.py \
    --use_fst \
    --use_dro \
    --use_grpo \
    --phase_2 \
    --phase_1_ckpt_dir outputs/grpo_training/final \
    --scr_ckpt_path ckpt/scr_210000.pth \
    --experience_name fontdiffuser_grpo_phase2 \
    --data_root my_dataset \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 25000 \
    --learning_rate 5e-6 \
    --output_dir outputs/grpo_training_phase2 \
    --dro_weight 0.03 \
    --grpo_pg_weight 0.005 \
    --grpo_warmup_steps 500 \
    --mixed_precision fp16

# Multi-GPU:
accelerate launch --multi_gpu --num_processes 4 train_grpo.py \
    --use_fst \
    --use_dro \
    --use_grpo \
    --phase_1_ckpt_dir outputs/dro_training/final \
    --experience_name fontdiffuser_grpo_multigpu \
    --data_root my_dataset \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 50000 \
    --learning_rate 1e-5 \
    --output_dir outputs/grpo_multigpu \
    --dro_weight 0.05 \
    --grpo_group_size 4 \
    --grpo_pg_weight 0.01 \
    --mixed_precision fp16

# Resume from checkpoint:
accelerate launch train_grpo.py \
    --use_fst \
    --use_dro \
    --use_grpo \
    --experience_name fontdiffuser_grpo_resumed \
    --data_root my_dataset \
    --train_batch_size 2 \
    --max_train_steps 75000 \
    --resume_from_checkpoint outputs/grpo_training/checkpoint_step_50000 \
    --output_dir outputs/grpo_training

# Debugging: quick test run:
accelerate launch train_grpo.py \
    --use_fst \
    --use_dro \
    --use_grpo \
    --experience_name fontdiffuser_grpo_debug \
    --data_root my_dataset \
    --train_batch_size 1 \
    --max_train_steps 100 \
    --ckpt_interval 50 \
    --log_interval 10 \
    --grpo_group_size 2 \
    --grpo_sample_steps 3 \
    --grpo_warmup_steps 10 \
    --output_dir outputs/grpo_debug
"""

import argparse
import logging
import os
import sys

from src.configs.fontdiffuser import get_parser
from src.trainers.trainer_grpo import FontDiffuserGRPOTrainer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments for GRPO training.

    Returns:
        Parsed argument namespace with all FontDiffuser, DRO, and GRPO args.
    """
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()

    # Handle local rank for distributed training
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Convert image sizes to tuples to match FontDatasetFST expectations
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main() -> None:
    """Main GRPO training entry point."""
    try:
        args: argparse.Namespace = get_args()

        logger.info("=" * 80)
        logger.info("FontDiffuserGRPO Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Experiment     : {args.experience_name}")
        logger.info(f"Output dir     : {args.output_dir}")
        logger.info(f"FST enabled    : {args.use_fst}")
        if args.use_fst:
            logger.info(f"  Feature channels    : {args.fst_feature_channels}")
            logger.info(f"  Num queries         : {args.fst_num_queries}")
            logger.info(f"  Query dim           : {args.fst_query_dim}")
            logger.info(f"  Num scales          : {args.fst_num_scales}")
            logger.info(f"  Style source prob   : {args.style_source_same_prob}")
        logger.info(f"DRO enabled    : {args.use_dro}")
        if args.use_dro:
            logger.info(f"  dro_weight          : {args.dro_weight}")
            logger.info(f"  dro_ssim_weight     : {args.dro_ssim_weight}")
            logger.info(f"  dro_lpips_weight    : {args.dro_lpips_weight}")
            logger.info(f"  dro_reward_scale    : {args.dro_reward_scale}")
            logger.info(f"  dro_warmup_steps    : {args.dro_warmup_steps}")
            logger.info(f"  dro_max_ts_frac     : {args.dro_max_timestep_frac}")
            logger.info(f"  dro_sharp_weight    : {args.dro_sharp_weight}")
            logger.info(f"  dro_div_weight      : {args.dro_div_weight}")
            logger.info(f"  dro_normalise       : {args.dro_normalise_reward}")
        logger.info(f"GRPO enabled   : {args.use_grpo}")
        if args.use_grpo:
            logger.info(f"  grpo_group_size     : {args.grpo_group_size}")
            logger.info(f"  grpo_clip_eps       : {args.grpo_clip_eps}")
            logger.info(f"  grpo_pg_weight      : {args.grpo_pg_weight}")
            logger.info(f"  grpo_sample_steps   : {args.grpo_sample_steps}")
            logger.info(f"  grpo_warmup_steps   : {args.grpo_warmup_steps}")
            logger.info(f"  grpo_kl_coeff       : {args.grpo_kl_coeff}")
            logger.info(f"  grpo_reward_clip    : {args.grpo_reward_clip}")
        logger.info(f"Phase 2 (SCR)  : {args.phase_2}")
        logger.info(f"Batch size     : {args.train_batch_size}")
        logger.info(f"Max steps      : {args.max_train_steps}")
        logger.info(f"Learning rate  : {args.learning_rate}")
        logger.info("=" * 80)

        trainer = FontDiffuserGRPOTrainer(args)

        logger.info("Setting up training components...")
        trainer.setup()
        logger.info("[OK] Setup complete")

        logger.info("Starting GRPO training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("[OK] GRPO training completed successfully!")
        logger.info("=" * 80)

        if args.export_onnx:
            logger.info("Starting ONNX export...")
            trainer.export_to_onnx()
            logger.info("[OK] ONNX export finished!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()