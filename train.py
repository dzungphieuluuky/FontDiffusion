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
    if not cfg.data.dataset_path:
        raise ValueError("dataset_path must be specified in config")
    
    # Process image sizes (convert to tuples if needed)
    if isinstance(cfg.model.style_encoder.image_size, int):
        cfg.model.style_encoder.image_size = [
            cfg.model.style_encoder.image_size,
            cfg.model.style_encoder.image_size
        ]
    
    if isinstance(cfg.model.content_encoder.image_size, int):
        cfg.model.content_encoder.image_size = [
            cfg.model.content_encoder.image_size,
            cfg.model.content_encoder.image_size
        ]
    
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