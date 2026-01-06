import os
import logging
from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from logging_utils import setup_logging

logger = setup_logging(level=logging.INFO, name="QualityEvaluator")
try:
    import lpips

    LPIPS_AVAILABLE: bool = True
except ImportError:
    logger.info("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE: bool = False

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE: bool = True
except ImportError:
    logger.info(
        "Warning: pytorch-fid not available. Install with: pip install pytorch-fid"
    )
    FID_AVAILABLE: bool = False

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE: bool = True
except ImportError:
    logger.info(
        "Warning: scikit-image not available. Install with: pip install scikit-image"
    )
    SSIM_AVAILABLE: bool = False

try:
    import wandb

    WANDB_AVAILABLE: bool = True
except ImportError:
    logger.info("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE: bool = False


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID"""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device: str = device

        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn: Optional[Any] = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn: Optional[Any] = None

        self.transform_to_tensor: transforms.ToTensor = transforms.ToTensor()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0

        try:
            # Convert to tensors [-1, 1]
            img1_tensor: torch.Tensor = (
                self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
            )
            img2_tensor: torch.Tensor = (
                self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
            )

            with torch.inference_mode():
                lpips_value: float = self.lpips_fn(img1_tensor, img2_tensor).item()

            return lpips_value
        except Exception as e:
            logger.info(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0

        try:
            # Convert to grayscale numpy arrays
            img1_gray: np.ndarray = np.array(img1.convert("L"))
            img2_gray: np.ndarray = np.array(img2.convert("L"))

            ssim_value: float = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            logger.info(f"Error computing SSIM: {e}")
            return -1.0

    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images"""
        if not FID_AVAILABLE:
            return -1.0

        try:
            fid_value: float = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir], batch_size=50, device=self.device, dims=2048
            )
            return fid_value
        except Exception as e:
            logger.info(f"Error computing FID: {e}")
            return -1.0

    def save_image(self, image: Image.Image, path: str) -> None:
        """Save PIL image to path"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)
        except Exception as e:
            logger.info(f"Error saving image to {path}: {e}")
