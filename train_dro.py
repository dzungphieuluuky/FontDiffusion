"""
Training entrypoint for FontDiffuserDROTrainer.

Usage
-----
# Single-GPU DRO fine-tuning:
python train_dro.py \\
    --config_path configs/fontdiffuser.py \\
    --ckpt_dir ckpt \\
    --phase_1_ckpt_dir ckpt/fst_checkpoint \\
    --output_dir results/dro_run \\
    --use_fst \\
    --use_dro \\
    --dro_weight 0.1 \\
    --dro_ssim_weight 1.0 \\
    --dro_lpips_weight 1.0 \\
    --dro_warmup_steps 500

# Multi-GPU with accelerate:
accelerate launch --num_processes 4 train_dro.py \\
    --config_path configs/fontdiffuser.py \\
    --ckpt_dir ckpt \\
    --phase_1_ckpt_dir ckpt/fst_checkpoint \\
    --output_dir results/dro_run \\
    --use_fst \\
    --use_dro \\
    --dro_weight 0.05 \\
    --dro_warmup_steps 1000
"""

import logging
import sys

from src.configs.fontdiffuser import get_parser
from src.trainers.trainer_dro import FontDiffuserDROTrainer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    logger.info("Initialising FontDiffuserDROTrainer...")
    trainer = FontDiffuserDROTrainer(args)

    logger.info("Starting DRO training...")
    trainer.train()

    logger.info("✓ DRO training complete.")


if __name__ == "__main__":
    main()