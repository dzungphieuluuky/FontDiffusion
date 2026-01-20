"""
FontDiffuser training script with Hydra configuration support.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.trainers.trainer import FontDiffuserTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FontDiffuserTrainer")


@hydra.main(version_base="1.3", config_path="configs/training", config_name="default")
def main(cfg: DictConfig):
    """Main training entry point with Hydra."""
    
    # Log configuration
    logger.info("="*80)
    logger.info("FONTDIFFUSER TRAINING")
    logger.info("="*80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*80)
    
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing mandatory keys: {missing_keys}")
    
    # Create and run trainer
    try:
        trainer = FontDiffuserTrainer(cfg)
        logger.info("Setting up training...")
        trainer.setup()
        logger.info("Starting training...")
        trainer.train()
        logger.info("✅ Training completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()