"""
Enhanced training runner for FontDiffuserWithFST using proposed auxiliary losses.

This script integrates three novel loss functions:
  1. FreqBandContentStyleLoss: Enforces content (low-freq) vs style (high-freq) separation
  2. StrokeTopologyLoss: Penalizes stroke presence/absence mismatches
  3. FreqWeightedDiffusionLoss: Spatially weights diffusion loss emphasizing strokes

Usage:
    # Basic training with auxiliary losses enabled
    accelerate launch train_fst_enhanced.py \
        --use_fst \
        --use_aux_losses \
        --experience_name="fst_enhanced_v1" \
        --data_root="my_dataset" \
        --output_dir="outputs/enhanced_training" \
        --max_train_steps=100000

    # Fine-grained loss configuration
    accelerate launch train_fst_enhanced.py \
        --use_fst \
        --use_aux_losses \
        --aux_freq_band \
        --aux_stroke_topo \
        --aux_freq_diff \
        --aux_freq_weight=0.5 \
        --aux_topo_weight=0.3 \
        --aux_anneal_temperature \
        --aux_temperature_schedule=linear \
        --experience_name="fst_enhanced_custom" \
        --data_root="my_dataset" \
        --output_dir="outputs/enhanced_training"
"""

import argparse
import os
import sys

from src.configs.fontdiffuser import get_parser
from src.trainers.trainer_fst_enhanced import FontDiffuserFSTTrainerEnhanced

# Setup logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def add_enhanced_loss_arguments(parser: argparse.ArgumentParser) -> None:
    """Add auxiliary loss configuration arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to.
    """
    aux_group = parser.add_argument_group("Enhanced Auxiliary Loss Configuration")

    # Enable/disable auxiliary losses
    aux_group.add_argument(
        "--use_aux_losses",
        action="store_true",
        default=False,
        help="Enable auxiliary loss functions (FreqBand, StrokeTopology, FreqWeighted)",
    )

    # Loss component selection
    aux_group.add_argument(
        "--aux_freq_band",
        type=bool,
        default=True,
        help="Enable FreqBandContentStyleLoss (default: True)",
    )
    aux_group.add_argument(
        "--aux_stroke_topo",
        type=bool,
        default=True,
        help="Enable StrokeTopologyLoss (default: True)",
    )
    aux_group.add_argument(
        "--aux_freq_diff",
        type=bool,
        default=True,
        help="Enable FreqWeightedDiffusionLoss (default: True)",
    )

    # Loss weights
    aux_group.add_argument(
        "--aux_freq_weight",
        type=float,
        default=0.5,
        help="Weight on FreqBandContentStyleLoss (default: 0.5)",
    )
    aux_group.add_argument(
        "--aux_topo_weight",
        type=float,
        default=0.3,
        help="Weight on StrokeTopologyLoss (default: 0.3)",
    )

    # FreqBandContentStyleLoss configuration
    aux_group.add_argument(
        "--aux_lf_radius",
        type=float,
        default=0.1,
        help="Low-freq circle radius (0-1 fraction of image). Default: 0.1",
    )
    aux_group.add_argument(
        "--aux_hf_radius",
        type=float,
        default=0.4,
        help="High-freq annulus start radius (0-1). Default: 0.4",
    )
    aux_group.add_argument(
        "--aux_lf_weight",
        type=float,
        default=1.0,
        help="Weight on low-freq (content) component. Default: 1.0",
    )
    aux_group.add_argument(
        "--aux_hf_weight",
        type=float,
        default=0.5,
        help="Weight on high-freq (style) component. Default: 0.5",
    )

    # StrokeTopologyLoss configuration
    aux_group.add_argument(
        "--aux_topo_threshold",
        type=float,
        default=0.5,
        help="Stroke binarization threshold in [0,1]. Default: 0.5",
    )
    aux_group.add_argument(
        "--aux_topo_temperature",
        type=float,
        default=0.05,
        help="Sigmoid temperature for soft binarization. Lower=sharper. Default: 0.05",
    )
    aux_group.add_argument(
        "--aux_topo_topology_weight",
        type=float,
        default=1.0,
        help="Weight on per-pixel topology BCE. Default: 1.0",
    )
    aux_group.add_argument(
        "--aux_topo_density_weight",
        type=float,
        default=0.3,
        help="Weight on global ink density consistency. Default: 0.3",
    )
    aux_group.add_argument(
        "--aux_dark_ink",
        type=bool,
        default=True,
        help="Whether ink is dark (vs light). Default: True",
    )

    # FreqWeightedDiffusionLoss configuration
    aux_group.add_argument(
        "--aux_fw_lf_radius",
        type=float,
        default=0.15,
        help="Low-freq radius for diffusion weight map. Default: 0.15",
    )
    aux_group.add_argument(
        "--aux_fw_max_weight",
        type=float,
        default=3.0,
        help="Max weight for stroke pixels. Higher=more emphasis. Default: 3.0",
    )
    aux_group.add_argument(
        "--aux_fw_normalize_weights",
        type=bool,
        default=True,
        help="Normalize weight map so mean==1. Default: True",
    )

    # Temperature annealing
    aux_group.add_argument(
        "--aux_anneal_temperature",
        action="store_true",
        default=False,
        help="Anneal StrokeTopologyLoss temperature during training",
    )
    aux_group.add_argument(
        "--aux_temperature_schedule",
        type=str,
        default="linear",
        choices=["linear", "exponential", "cosine"],
        help="Temperature annealing schedule. Default: linear",
    )


def get_args():
    """Parse command line arguments with enhanced loss configuration."""
    parser: argparse.ArgumentParser = get_parser()

    # Add auxiliary loss arguments
    add_enhanced_loss_arguments(parser)

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
        logger.info("FontDiffusionWithFST Enhanced Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Experiment: {args.experience_name}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"FST enabled: {args.use_fst}")
        logger.info(f"Auxiliary losses enabled: {args.use_aux_losses}")

        if args.use_aux_losses:
            logger.info(f"  Frequency band loss: {args.aux_freq_band}")
            logger.info(f"  Stroke topology loss: {args.aux_stroke_topo}")
            logger.info(f"  Frequency-weighted diffusion: {args.aux_freq_diff}")
            logger.info(f"  Frequency weight: {args.aux_freq_weight}")
            logger.info(f"  Topology weight: {args.aux_topo_weight}")
            if args.aux_anneal_temperature:
                logger.info(
                    f"  Temperature annealing: {args.aux_temperature_schedule}"
                )

        logger.info(f"Phase 2 (SCR): {args.phase_2}")
        logger.info(f"Batch size: {args.train_batch_size}")
        logger.info(f"Max steps: {args.max_train_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)

        # Initialize enhanced trainer
        trainer = FontDiffuserFSTTrainerEnhanced(args)

        # Setup components
        logger.info("Setting up training components...")
        trainer.setup()
        logger.info("✓ Setup complete")

        # Run training
        logger.info("Starting enhanced training...")
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
==============================================================================
EXAMPLE COMMANDS FOR ENHANCED TRAINING
==============================================================================

# ============================================================================
# BASIC: Train with all auxiliary losses enabled (recommended defaults)
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --experience_name="fst_enhanced_basic" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --output_dir="outputs/enhanced_training" \
    --mixed_precision="fp16"

# ============================================================================
# WITH TEMPERATURE ANNEALING: Sharpen topology loss over time
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_anneal_temperature \
    --aux_temperature_schedule=linear \
    --experience_name="fst_enhanced_anneal" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced_training_anneal" \
    --mixed_precision="fp16"

# ============================================================================
# CUSTOM LOSS WEIGHTS: Adjust emphasis of each loss component
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_freq_band \
    --aux_stroke_topo \
    --aux_freq_diff \
    --aux_freq_weight=0.7 \
    --aux_topo_weight=0.5 \
    --aux_lf_radius=0.12 \
    --aux_hf_radius=0.35 \
    --aux_topo_threshold=0.4 \
    --experience_name="fst_enhanced_custom" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced_custom" \
    --mixed_precision="fp16"

# ============================================================================
# SELECTIVE LOSSES: Enable only specific auxiliary losses
# ============================================================================
# Only frequency band loss (content/style separation)
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_freq_band \
    --aux_stroke_topo=False \
    --aux_freq_diff=False \
    --experience_name="fst_enhanced_freqonly" \
    --data_root="my_dataset" \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced_freqonly" \
    --mixed_precision="fp16"

# Only stroke topology loss
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_freq_band=False \
    --aux_stroke_topo \
    --aux_freq_diff=False \
    --experience_name="fst_enhanced_topoonly" \
    --data_root="my_dataset" \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced_topoonly" \
    --mixed_precision="fp16"

# ============================================================================
# AGGRESSIVE STROKE EMPHASIS: High frequency weighting and max weight
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_freq_weight=1.0 \
    --aux_topo_weight=0.5 \
    --aux_fw_max_weight=5.0 \
    --aux_topo_topology_weight=1.5 \
    --experience_name="fst_enhanced_aggressive" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced_aggressive" \
    --mixed_precision="fp16"

# ============================================================================
# MULTI-GPU ENHANCED TRAINING: Distributed with auxiliary losses
# ============================================================================
accelerate launch --multi_gpu --num_processes=4 train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --aux_anneal_temperature \
    --experience_name="fst_enhanced_multigpu" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --output_dir="outputs/enhanced_multigpu" \
    --mixed_precision="fp16"

# ============================================================================
# PHASE 2 ENHANCED: Continue from Phase 1 checkpoint with auxiliary losses
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --phase_2 \
    --phase_1_ckpt_dir="outputs/phase1_checkpoint" \
    --scr_ckpt_path="ckpt/scr_210000.pth" \
    --experience_name="fst_enhanced_phase2" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --output_dir="outputs/enhanced_phase2" \
    --mixed_precision="fp16"

# ============================================================================
# DEBUGGING: Quick test with auxiliary losses
# ============================================================================
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --experience_name="fst_enhanced_debug" \
    --data_root="my_dataset" \
    --train_batch_size=2 \
    --max_train_steps=100 \
    --ckpt_interval=50 \
    --log_interval=10 \
    --output_dir="outputs/debug_enhanced"

# ============================================================================
# COMPARISON: Run both baseline and enhanced training
# ============================================================================
# Baseline (no auxiliary losses)
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --experience_name="fst_baseline" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/baseline" \
    --mixed_precision="fp16"

# Enhanced (with auxiliary losses)
accelerate launch train_fst_enhanced.py \
    --use_fst \
    --use_aux_losses \
    --experience_name="fst_enhanced" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/enhanced" \
    --mixed_precision="fp16"

"""
