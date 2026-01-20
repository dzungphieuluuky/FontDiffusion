"""
Training entry point with Hydra configuration.
"""

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.trainers.trainer_fst import FontDiffuserFSTTrainer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/training", config_name="fst", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra configuration.

    Example:
        python train_fst.py use_fst=true fst_num_queries=220
        python train_fst.py --config-name fst_distributed use_fst=true
    """
    try:
        logger.info("=" * 80)
        logger.info("FontDiffuserWithFST Training")
        logger.info("=" * 80)
        logger.info(OmegaConf.to_yaml(cfg))

        # Initialize trainer
        trainer = FontDiffuserFSTTrainer(cfg)

        logger.info("Setting up training components...")
        trainer.setup()
        logger.info("✓ Setup complete")

        logger.info("Starting training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()