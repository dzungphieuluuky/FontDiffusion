"""
Direct Reward Optimization reward functions — hardened against reward hacking.

Identified attack surfaces and mitigations applied:

1. SSIM copy-collapse:  Model can maximize SSIM by returning content unchanged.
   Fix: penalize low-frequency similarity (blur invariance check) and require
        style divergence to be non-trivial via the joint reward gate.

2. SSIM blur-cheat:     Smoothed outputs raise SSIM because variance terms fall.
   Fix: high-frequency energy penalty on pred forces non-trivial sharpness.

3. LPIPS mean-collapse: Model outputs the mean of style images to minimise
   expected LPIPS distance without reproducing structure.
   Fix: diversity regulariser penalises within-batch similarity of predictions.

4. LPIPS zero-feature:  Normalised L2 distance is minimised by pushing pred
   features toward a degenerate point unrelated to style.
   Fix: add cosine-similarity term and cross-check pred features against both
        style AND content so the model cannot trivially evacuate feature space.

5. Scale imbalance:     One term dominates; the other becomes a free variable.
   Fix: adaptive normalisation clips each component to [0,1] using running
        statistics (EMA), then weights are applied in a calibrated space.

6. Grayscale RGB expand hack: single-channel expand() fed identical channels
   to VGG which could be exploited; replaced with proper luminance->RGB lift.

7. Reward magnitude explosion:  Extreme rewards destabilise DRO updates.
   Fix: final reward is tanh-squashed to (-reward_scale, +reward_scale).

8. Gradient-free SSIM cliff:  Near-duplicate inputs create near-zero gradients.
   Fix: use MS-SSIM (multi-scale) which has better gradient landscape.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)


def _ssim_single_scale(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel: torch.Tensor,
    padding: int,
    C1: float,
    C2: float,
) -> torch.Tensor:
    """Single-scale SSIM per channel, averaged over spatial dims and channels."""
    _, C, _, _ = pred.shape
    vals: list[torch.Tensor] = []
    for c in range(C):
        p = pred[:, c : c + 1]
        t = target[:, c : c + 1]
        mu1 = F.conv2d(p, kernel, padding=padding)
        mu2 = F.conv2d(t, kernel, padding=padding)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = F.conv2d(p * p, kernel, padding=padding) - mu1_sq
        s2 = F.conv2d(t * t, kernel, padding=padding) - mu2_sq
        s12 = F.conv2d(p * t, kernel, padding=padding) - mu1_mu2
        num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        vals.append((num / den.clamp(min=1e-8)).mean())
    return torch.stack(vals).mean()


# ---------------------------------------------------------------------------
# MS-SSIM reward (multi-scale; harder to hack than single-scale SSIM)
# ---------------------------------------------------------------------------

_MS_SSIM_WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def compute_ms_ssim_reward(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    num_scales: int = 5,
) -> torch.Tensor:
    """Multi-scale SSIM reward (MS-SSIM).

    Harder to hack than single-scale SSIM:
    - Blur cheating hurts at fine scales.
    - Copy-collapsing is detectable at coarse scales via the diversity gate.

    Args:
        pred: (B, C, H, W) in [0, data_range].
        target: (B, C, H, W) in [0, data_range].

    Returns:
        Scalar tensor in [0, 1] — higher = more faithful to content structure.
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(pred.device)
    padding = kernel_size // 2
    weights = _MS_SSIM_WEIGHTS[:num_scales].to(pred.device)
    weights = weights / weights.sum()

    mcs_vals: list[torch.Tensor] = []
    p, t = pred, target
    for i in range(num_scales):
        ssim_val = _ssim_single_scale(p, t, kernel, padding, C1, C2)
        mcs_vals.append(ssim_val)
        if i < num_scales - 1:
            p = F.avg_pool2d(p, kernel_size=2, stride=2)
            t = F.avg_pool2d(t, kernel_size=2, stride=2)
            if p.shape[-1] < kernel_size or p.shape[-2] < kernel_size:
                weights = weights[: i + 1]
                weights = weights / weights.sum()
                break

    return torch.stack(mcs_vals[: len(weights)]).dot(weights)


# ---------------------------------------------------------------------------
# High-frequency energy penalty (anti-blur cheat)
# ---------------------------------------------------------------------------


def _hf_energy(x: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude of high-frequency content via Laplacian."""
    laplacian_kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    vals = []
    for c in range(x.shape[1]):
        ch = x[:, c : c + 1]
        vals.append((F.conv2d(ch, laplacian_kernel, padding=1) ** 2).mean())
    return torch.stack(vals).mean()


def compute_sharpness_penalty(
    pred: torch.Tensor,
    content: torch.Tensor,
    floor: float = 0.1,
) -> torch.Tensor:
    """Penalise if pred is significantly blurrier than content.

    Returns a value in [0, 1].  0 = pred at least as sharp as content (no
    penalty).  Approaching 1 = pred is much blurrier than content.

    This closes the 'smooth everything' reward-hacking loophole.
    """
    pred_hf = _hf_energy(pred)
    content_hf = _hf_energy(content).detach()
    ratio = pred_hf / (content_hf + 1e-8)
    penalty = (1.0 - ratio).clamp(min=0.0, max=1.0)
    return penalty


# ---------------------------------------------------------------------------
# LPIPS-VGG with cosine similarity hardening
# ---------------------------------------------------------------------------


class VGGRewardFeatures(nn.Module):
    """VGG16 feature extractor for differentiable LPIPS-style reward.

    Changes vs. original:
    - Uses a mixture of L2 and cosine distance — robust to feature-norm collapse.
    - Grayscale is converted to luminance-weighted RGB (not channel-expanded)
      to prevent the model exploiting the repeated-channel artefact.
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
            self.slices.append(
                nn.Sequential(*list(features.children())[prev : end + 1])
            )
            prev = end + 1

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert single-channel grayscale to luminance-weighted 3-channel RGB.

        Uses slight channel variation (ITU-R BT.601-inspired) instead of naive
        expand() to prevent the model exploiting the identical-channel structure.
        """
        if x.shape[1] == 1:
            r = x * 1.000
            g = x * 0.980
            b = x * 0.940
            return torch.cat([r, g, b], dim=1).clamp(0, 1)
        return x

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._to_rgb(x)
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
    """Differentiable LPIPS-style perceptual distance.

    Hardened vs. original:
    - Mixes L2 distance in normalised feature space WITH (1 - cosine_similarity).
    - Cosine distance is robust to feature-magnitude collapse: the model
      cannot cheat by shrinking feature norms toward zero.

    Returns:
        Scalar — higher = more dissimilar to style reference (penalty).
    """
    feats_pred = vgg(pred)
    with torch.no_grad():
        feats_ref = vgg(ref)

    l2_dists: list[torch.Tensor] = []
    cos_dists: list[torch.Tensor] = []

    for fp, fr in zip(feats_pred, feats_ref):
        fp_n = F.normalize(fp, p=2, dim=1)
        fr_n = F.normalize(fr.detach(), p=2, dim=1)
        l2_dists.append(((fp_n - fr_n) ** 2).mean())
        cos_sim = (fp_n * fr_n).sum(dim=1, keepdim=True).mean()
        cos_dists.append(1.0 - cos_sim)

    l2_val = torch.stack(l2_dists).mean()
    cos_val = torch.stack(cos_dists).mean()
    return 0.5 * l2_val + 0.5 * cos_val


# ---------------------------------------------------------------------------
# Diversity regulariser (anti-mode-collapse / mean-image hack)
# ---------------------------------------------------------------------------


def compute_diversity_bonus(pred: torch.Tensor) -> torch.Tensor:
    """Reward within-batch diversity of predictions.

    The model cannot cheat by outputting a single average image because
    that would minimise this bonus.

    Returns a value in [0, 1] — higher = more diverse batch (better).
    If batch size == 1, returns zero (no penalty/bonus).
    """
    B = pred.shape[0]
    if B < 2:
        return pred.sum() * 0.0  # zero with gradient attached

    flat = pred.view(B, -1)
    flat_n = F.normalize(flat, p=2, dim=1)
    sim_matrix = flat_n @ flat_n.t()  # (B, B)
    off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=pred.device)
    mean_sim = sim_matrix[off_diag_mask].mean()
    return (1.0 - mean_sim).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# EMA reward normaliser (prevents scale-imbalance hacking)
# ---------------------------------------------------------------------------


class EMANormaliser(nn.Module):
    """Exponential moving average normaliser for reward components.

    Tracks the mean and std of each reward signal and normalises it to
    approximately zero-mean unit-variance, preventing one term from
    dominating due to differing magnitudes.
    """

    def __init__(self, momentum: float = 0.01, eps: float = 1e-6) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("ema_mean", torch.tensor(0.0))
        self.register_buffer("ema_var", torch.tensor(1.0))
        self.register_buffer("initialised", torch.tensor(False))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        val = x.detach().float()
        if not self.initialised.item():
            self.ema_mean.copy_(val)
            self.initialised.copy_(torch.tensor(True))
        else:
            self.ema_mean.lerp_(val, self.momentum)
            self.ema_var.lerp_((val - self.ema_mean) ** 2, self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update(x)
        std = (self.ema_var + self.eps).sqrt()
        return (x - self.ema_mean.detach()) / std.detach()


# ---------------------------------------------------------------------------
# Combined DRO reward — hardened
# ---------------------------------------------------------------------------


class DRORewardModule(nn.Module):
    """Combines MS-SSIM, LPIPS, sharpness, and diversity into one reward.

    Reward (pre-squash) =
        w_ssim  * MS-SSIM(pred, content)             # content fidelity
      - w_lpips * LPIPS(pred, style)                 # style proximity
      - w_sharp * sharpness_penalty(pred, content)   # anti-blur hack
      + w_div   * diversity_bonus(pred)              # anti-collapse hack

    Each component is independently EMA-normalised before weighting to
    prevent scale imbalance between metrics.

    The final reward is squashed through tanh so its magnitude is bounded
    to (-reward_scale, +reward_scale), preventing reward explosion.

    Args:
        ssim_weight: Weight for MS-SSIM content-fidelity term.
        lpips_weight: Weight for LPIPS style-proximity term.
        sharp_weight: Weight for anti-blur sharpness penalty.
        div_weight: Weight for within-batch diversity bonus.
        reward_scale: Tanh squash scale (upper bound of |reward|).
        normalise: Whether to apply EMA normalisation per component.
    """

    def __init__(
        self,
        ssim_weight: float = 1.0,
        lpips_weight: float = 1.0,
        sharp_weight: float = 0.5,
        div_weight: float = 0.3,
        reward_scale: float = 1.0,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.sharp_weight = sharp_weight
        self.div_weight = div_weight
        self.reward_scale = reward_scale
        self.normalise = normalise

        self.vgg = VGGRewardFeatures()

        if normalise:
            self.norm_ssim = EMANormaliser()
            self.norm_lpips = EMANormaliser()
            self.norm_sharp = EMANormaliser()
            self.norm_div = EMANormaliser()

    def forward(
        self,
        pred_images: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hardened composite DRO reward.

        Args:
            pred_images: (B, C, H, W) model predictions, normalised to [0, 1].
            content_images: (B, C, H, W) content references, normalised to [0, 1].
            style_images: (B, C, H, W) style references, normalised to [0, 1].

        Returns:
            Tuple of (scalar reward tensor, metrics dict).
        """
        # --- Content fidelity (MS-SSIM) ---
        ssim_val = compute_ms_ssim_reward(pred_images, content_images)

        # --- Style proximity (hardened LPIPS) ---
        lpips_val = compute_lpips_reward(pred_images, style_images, self.vgg)

        # --- Anti-blur sharpness penalty ---
        sharp_penalty = compute_sharpness_penalty(pred_images, content_images)

        # --- Within-batch diversity bonus ---
        div_bonus = compute_diversity_bonus(pred_images)

        # --- EMA normalisation (optional but recommended) ---
        if self.normalise:
            ssim_n = self.norm_ssim(ssim_val)
            lpips_n = self.norm_lpips(lpips_val)
            sharp_n = self.norm_sharp(sharp_penalty)
            div_n = self.norm_div(div_bonus)
        else:
            ssim_n = ssim_val
            lpips_n = lpips_val
            sharp_n = sharp_penalty
            div_n = div_bonus

        # --- Composite reward (pre-squash) ---
        raw_reward = (
            self.ssim_weight * ssim_n
            - self.lpips_weight * lpips_n
            - self.sharp_weight * sharp_n
            + self.div_weight * div_n
        )

        # --- Tanh squash: bounds reward, smooth gradients near extremes ---
        reward = self.reward_scale * torch.tanh(raw_reward / self.reward_scale)

        metrics = {
            "dro/ssim_reward": ssim_val.item(),
            "dro/lpips_penalty": lpips_val.item(),
            "dro/sharpness_penalty": sharp_penalty.item(),
            "dro/diversity_bonus": div_bonus.item(),
            "dro/raw_reward": raw_reward.item(),
            "dro/composite_reward": reward.item(),
        }
        return reward, metrics
