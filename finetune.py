"""
FontDiffuser training script with multi-phase support.
Improved version with memory safety, error handling, and better organization.
"""

import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm


from configs.fontdiffuser import get_parser
from dataset import FontDataset, CollateFN
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    StyleTransformationModule,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
)
from utils.utilities import (
    find_checkpoint,
    load_model_checkpoint,
    save_model_checkpoint,
    get_hf_bar,
)
from utils import (
    normalize_mean_std,
    reNormalize_img,
    save_args_to_yaml,
    x0_from_epsilon,
)

from training import FontDiffuserTrainer, TrainingConfig
logger = logging.getLogger("FontDiffuserTrainer")

def parse_args_training():
    """Parse command line arguments with additional options."""
    parser = get_parser()
    
    # Add new arguments for improved functionality
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help="Save full model checkpoint in addition to components"
    )
    parser.add_argument(
        "--phase-1",
        action="store_true",
        help="Enable Phase 1 training (content and style encoders only)"
    )

    args = parser.parse_args()
    
    # Handle environment variables
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank
    
    # Process image sizes
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    
    return args


def main():
    """Main entry point."""
    args = parse_args_training()
    
    # Create and run trainer
    trainer = FontDiffuserTrainer(args)
    trainer.setup()
    trainer.train()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()