"""
Direct Reward Optimization reward functions derived from evaluate.py metrics.

Rewards are computed differentiably where possible (SSIM, LPIPS-VGG);
FID is non-differentiable and used only for logging.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSIM reward (differentiable)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)


def compute_ssim_reward(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute differentiable SSIM reward between pred and target.

    Args:
        pred: (B, C, H, W) float tensor in [0, data_range].
        target: (B, C, H, W) float tensor in [0, data_range].
        kernel_size: Gaussian window size.
        sigma: Gaussian standard deviation.
        data_range: Value range of the inputs.
        k1: SSIM stability constant 1.
        k2: SSIM stability constant 2.

    Returns:
        Scalar tensor — mean SSIM in [0, 1] (higher = better content fidelity).
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(pred.device)
    padding = kernel_size // 2

    _, C, _, _ = pred.shape
    ssim_vals: list[torch.Tensor] = []

    for c in range(C):
        ch1 = pred[:, c: c + 1]
        ch2 = target[:, c: c + 1]

        mu1 = F.conv2d(ch1, kernel, padding=padding)
        mu2 = F.conv2d(ch2, kernel, padding=padding)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(ch1 * ch1, kernel, padding=padding) - mu1_sq
        sigma2_sq = F.conv2d(ch2 * ch2, kernel, padding=padding) - mu2_sq
        sigma12 = F.conv2d(ch1 * ch2, kernel, padding=padding) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_vals.append((numerator / denominator.clamp(min=1e-8)).mean())

    return torch.stack(ssim_vals).mean()


# ---------------------------------------------------------------------------
# LPIPS-VGG reward (differentiable)
# ---------------------------------------------------------------------------

class VGGRewardFeatures(nn.Module):
    """VGG16 feature extractor for differentiable LPIPS-style reward.

    Gradients are allowed to flow through the prediction only;
    the VGG backbone itself is kept frozen.
    """

    _LAYER_ENDS: list[int] = [4, 9, 16, 23]

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = backbone.features
        self.slices = nn.ModuleList()
        prev = 0
        for end in self._LAYER_ENDS:
            self.slices.append(nn.Sequential(*list(features.children())[prev: end + 1]))
            prev = end + 1

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale VGG features.

        Args:
            x: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            List of (B, C_i, H_i, W_i) feature tensors.
        """
        x = (x - self.mean) / self.std
        feats, h = [], x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


def compute_lpips_reward(
    pred: torch.Tensor,
    ref: torch.Tensor,
    vgg: VGGRewardFeatures,
) -> torch.Tensor:
    """Compute differentiable LPIPS-style perceptual distance (lower = more similar).

    Args:
        pred: (B, 3, H, W) float tensor in [0, 1].
        ref: (B, 3, H, W) float tensor in [0, 1] — reference (style or content).
        vgg: Pre-built :class:`VGGRewardFeatures`.

    Returns:
        Scalar tensor — mean LPIPS (lower = better style match).
    """
    feats_pred = vgg(pred)
    with torch.no_grad():
        feats_ref = vgg(ref)

    distances: list[torch.Tensor] = []
    for fp, fr in zip(feats_pred, feats_ref):
        fp_norm = F.normalize(fp, p=2, dim=1)
        fr_norm = F.normalize(fr.detach(), p=2, dim=1)
        distances.append(((fp_norm - fr_norm) ** 2).mean())

    return torch.stack(distances).mean()


# ---------------------------------------------------------------------------
# Combined DRO reward
# ---------------------------------------------------------------------------

class DRORewardModule(nn.Module):
    """Combines SSIM (content fidelity) and LPIPS (style similarity) into one reward.

    Reward = w_ssim * SSIM(pred, content) - w_lpips * LPIPS(pred, style)

    A higher reward means better content structure AND closer style distribution.

    Args:
        ssim_weight: Weight for the SSIM content-fidelity reward.
        lpips_weight: Weight for the LPIPS style-similarity penalty.
        reward_scale: Global scale applied to the final reward before returning.
    """

    def __init__(
        self,
        ssim_weight: float = 1.0,
        lpips_weight: float = 1.0,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.reward_scale = reward_scale
        self.vgg = VGGRewardFeatures()

    def forward(
        self,
        pred_images: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite DRO reward.

        Args:
            pred_images: (B, C, H, W) model predictions, normalised to [0, 1].
            content_images: (B, C, H, W) content references, normalised to [0, 1].
            style_images: (B, C, H, W) style references, normalised to [0, 1].

        Returns:
            Tuple of (scalar reward tensor, metrics dict).
        """
        # Expand grayscale → RGB for VGG
        if pred_images.shape[1] == 1:
            pred_rgb = pred_images.expand(-1, 3, -1, -1)
            content_rgb = content_images.expand(-1, 3, -1, -1)
            style_rgb = style_images.expand(-1, 3, -1, -1)
        else:
            pred_rgb, content_rgb, style_rgb = pred_images, content_images, style_images

        ssim_val = compute_ssim_reward(pred_rgb, content_rgb)
        lpips_val = compute_lpips_reward(pred_rgb, style_rgb, self.vgg)

        reward = self.reward_scale * (
            self.ssim_weight * ssim_val - self.lpips_weight * lpips_val
        )

        metrics = {
            "dro/ssim_reward": ssim_val.item(),
            "dro/lpips_penalty": lpips_val.item(),
            "dro/composite_reward": reward.item(),
        }
        return reward, metrics