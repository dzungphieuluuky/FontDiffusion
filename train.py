"""
FontDiffuser training script with Hydra configuration support.
Improved version with memory safety, error handling, and better organization.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.trainers.trainer import FontDiffuserTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FontDiffuserTrainer")


@hydra.main(version_base=None, config_path="configs/training", config_name="default")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    
    # Log configuration
    logger.info("="*80)
    logger.info("Training Configuration:")
    logger.info("="*80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*80)
    
    # Validate required fields
    if not cfg.data_root:
        raise ValueError("data_root must be specified in config")
    
    if not cfg.output_dir:
        raise ValueError("output_dir must be specified in config")
    
    # Create and run trainer
    try:
        trainer = FontDiffuserTrainer(cfg)
        logger.info("Setting up training...")
        trainer.setup()
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()