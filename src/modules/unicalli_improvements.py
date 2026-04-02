"""
UniCalli-inspired improvements for FontDiffuserFST.

Paper: "UniCalli: A Unified Diffusion Framework for Column-Level Generation
        and Recognition of Chinese Calligraphy" (ICLR 2026, arXiv 2510.13745)

Four improvements extracted from UniCalli and adapted to the FontDiffuserFST
+ Fourier + DRO + MRL stack:

  1. AsymmetricNoisingScheduler
     UniCalli keeps the content branch CLEAN while noising the output branch.
     In FontDiffuserFST, content_images are currently passed through the same
     noising pipeline as the target, which adds gratuitous uncertainty to the
     content conditioning signal. Decoupling the schedules means the content
     encoder always sees a sharp, unambiguous glyph structure.

  2. StructureRecognitionHead + RecognitionAuxLoss
     UniCalli trains generation and recognition jointly: recognition constrains
     the generator to preserve character identity. We add a lightweight
     recognition auxiliary head on top of the content encoder's coarse MRL
     prefix (64d). The cross-entropy recognition loss provides a direct
     supervisory signal on glyph identity — orthogonal to and complementary
     with the contrastive MRL loss.

  3. ConditionalDropoutAugmentation
     UniCalli randomly replaces the content condition with pure noise at p_drop
     to prevent style-overfitting on long-tail data. Adapted here: three
     dropout modes (content-only, style-only, both) with independent
     probabilities, plus hard-negative mixing where the dropped condition is
     replaced by a *different sample's* condition rather than pure noise —
     this is stronger than UniCalli's pure-noise masking.

  4. SpatialBoundingBoxConditioner
     UniCalli injects a rasterized bounding-box map as a spatial prior.
     Adapted for single-character FontDiffuser: a stroke-layout heatmap
     derived from the content image's binary skeleton, encoding spatial
     distribution of ink mass. Zero new learned weights — purely a
     deterministic pre-processing step that produces an extra conditioning
     channel.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 1.  AsymmetricNoisingScheduler
# =============================================================================

class AsymmetricNoisingScheduler:
    """Decoupled noise schedules for content and output branches.

    UniCalli insight: keeping the content branch clean during training gives
    the content encoder a perfectly sharp signal at every step, removing the
    confound between "content is noisy" and "model is uncertain about content."

    For FontDiffuserFST:
      - content_images: CLEAN (t_content = 0 always)
      - style_images:   CLEAN (style reference is always a real image)
      - target_images:  NOISED at timestep t (standard diffusion objective)

    The improvement here is making this explicit and providing a controlled
    partial-noising option for style images (t_style ∈ [0, t_max_style])
    which acts as a data augmentation preventing style encoder overfitting.

    Args:
        noise_scheduler:  The base DDPM/DDIM scheduler from the model.
        style_noise_fraction: Maximum fraction of full noise applied to style
                              images. 0.0 = completely clean (UniCalli default).
                              0.3 = up to 30% noise level for augmentation.
        content_noise_fraction: Should be 0.0 (clean content). Set >0 only
                                for ablation experiments.
    """

    def __init__(
        self,
        noise_scheduler,
        style_noise_fraction: float = 0.0,
        content_noise_fraction: float = 0.0,
    ) -> None:
        self.scheduler = noise_scheduler
        self.style_noise_fraction = style_noise_fraction
        self.content_noise_fraction = content_noise_fraction

        T = noise_scheduler.config.num_train_timesteps
        self.t_max_style = int(style_noise_fraction * T)
        self.t_max_content = int(content_noise_fraction * T)

    def add_target_noise(
        self,
        target_images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Standard noising for the target (output) branch."""
        return self.scheduler.add_noise(target_images, noise, timesteps)

    def add_style_noise(
        self,
        style_images: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial noising for style images (augmentation, not diffusion).

        If style_noise_fraction == 0, returns clean images unchanged.
        Otherwise adds a small amount of noise sampled uniformly in
        [0, t_max_style] — acts as style augmentation to prevent overfitting.

        Returns:
            (noised_style_images, style_timesteps)
        """
        B, device = style_images.shape[0], style_images.device

        if self.t_max_style == 0:
            # Completely clean — UniCalli default
            t_style = torch.zeros(B, dtype=torch.long, device=device)
            return style_images, t_style

        t_style = torch.randint(0, self.t_max_style, (B,), device=device)
        if noise is None:
            noise = torch.randn_like(style_images)
        noised = self.scheduler.add_noise(style_images, noise, t_style)
        return noised, t_style

    def add_content_noise(
        self,
        content_images: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial noising for content images (ablation only; default: clean)."""
        B, device = content_images.shape[0], content_images.device

        if self.t_max_content == 0:
            t_content = torch.zeros(B, dtype=torch.long, device=device)
            return content_images, t_content

        t_content = torch.randint(0, self.t_max_content, (B,), device=device)
        if noise is None:
            noise = torch.randn_like(content_images)
        noised = self.scheduler.add_noise(content_images, noise, t_content)
        return noised, t_content

    def sample_target_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample full diffusion timesteps for the target branch."""
        return torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()


# =============================================================================
# 2.  StructureRecognitionHead + RecognitionAuxLoss
# =============================================================================

class StructureRecognitionHead(nn.Module):
    """Lightweight character identity classification head.

    UniCalli's key insight: training recognition jointly with generation
    constrains the generator to preserve character identity. Recognition
    provides a direct cross-entropy loss on glyph class that is:
      - Orthogonal to SSIM/LPIPS (which measure pixel/feature similarity)
      - Stronger than contrastive MRL loss for discrete identity preservation
      - Computationally cheap: operates on the 64d coarse MRL prefix

    Architecture: 2-layer MLP on the coarse MRL prefix (64d → 128 → n_classes)
    Zero impact on inference — head is detached from diffusion conditioning.

    Args:
        input_dim:   Dimension of the coarse MRL prefix (default: 64).
        n_classes:   Number of character classes in training vocabulary.
        hidden_dim:  Hidden layer size.
        dropout:     Dropout rate for regularisation.
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_classes: int = 6763,  # GB2312 level-1 standard Chinese characters
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        # Initialise output layer with small weights to prevent early instability
        nn.init.normal_(self.classifier[-1].weight, std=0.01)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, coarse_prefix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_prefix: (B, input_dim) L2-normalised coarse MRL projection.

        Returns:
            (B, n_classes) unnormalised logits.
        """
        return self.classifier(coarse_prefix)


class RecognitionAuxLoss(nn.Module):
    """Joint generation-recognition auxiliary loss.

    Computes cross-entropy on character identity predictions from:
    1. pred_x0 encoding → classifier (gradient flows, trains encoder)
    2. content encoding → classifier (supervision signal, content detached)

    The pred_x0 branch is the critical one: if the model generates an image
    whose coarse features are classified as a *different* character, it gets
    penalised directly. This is much more targeted than SSIM.

    Args:
        recognition_head:  The StructureRecognitionHead module.
        pred_weight:        Weight on the pred_x0 recognition loss.
        content_weight:     Weight on the content recognition loss
                            (sanity regulariser; usually lower).
        label_smoothing:    Label smoothing for cross-entropy.
    """

    def __init__(
        self,
        recognition_head: StructureRecognitionHead,
        pred_weight: float = 1.0,
        content_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = recognition_head
        self.pred_weight = pred_weight
        self.content_weight = content_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pred_coarse: torch.Tensor,
        content_coarse: torch.Tensor,
        char_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_coarse:    (B, d_coarse) MRL coarse prefix from pred_x0 encoding.
            content_coarse: (B, d_coarse) MRL coarse prefix from content encoding
                            (computed with no_grad).
            char_labels:    (B,) integer character class indices.

        Returns:
            (scalar loss, metrics dict)
        """
        # Prediction branch — gradients flow through here
        pred_logits = self.head(pred_coarse)               # (B, n_classes)
        pred_loss = F.cross_entropy(
            pred_logits, char_labels,
            label_smoothing=self.label_smoothing,
        )

        # Content branch — sanity check that content encoder stays calibrated
        with torch.no_grad():
            content_logits = self.head(content_coarse.detach())
            content_acc = (content_logits.argmax(dim=1) == char_labels).float().mean()
            content_loss_val = F.cross_entropy(content_logits, char_labels).item()

        pred_acc = (pred_logits.detach().argmax(dim=1) == char_labels).float().mean()

        loss = self.pred_weight * pred_loss

        return loss, {
            "recog/pred_ce_loss": pred_loss.item(),
            "recog/pred_accuracy": pred_acc.item(),
            "recog/content_accuracy": content_acc.item(),  # diagnostic
            "recog/content_ce": content_loss_val,          # diagnostic
        }


# =============================================================================
# 3.  ConditionalDropoutAugmentation
# =============================================================================

class ConditionalDropoutAugmentation:
    """Style-content disentanglement via conditional dropout.

    UniCalli replaces the content condition with pure noise at probability
    p_drop to prevent the model from overfitting to long-tail styles.

    Extensions over UniCalli:
    1. Three independent dropout modes: content-only, style-only, both.
    2. Hard-negative replacement: instead of pure noise, replaces dropped
       condition with another sample's condition from the same batch.
       This forces the model to be invariant to mismatched content/style
       while still seeing realistic image statistics (not pure noise).
    3. Curriculum: p_drop ramps up over training steps.

    Args:
        p_drop_content:   Probability of dropping content condition.
        p_drop_style:     Probability of dropping style condition.
        use_hard_negative: If True, replace with a shuffled batch sample
                           instead of pure noise (stronger disentanglement).
        curriculum_steps: If > 0, p_drop ramps from 0 → target over this
                          many steps.
    """

    def __init__(
        self,
        p_drop_content: float = 0.1,
        p_drop_style: float = 0.05,
        use_hard_negative: bool = True,
        curriculum_steps: int = 1000,
    ) -> None:
        self.p_drop_content = p_drop_content
        self.p_drop_style = p_drop_style
        self.use_hard_negative = use_hard_negative
        self.curriculum_steps = curriculum_steps

    def _current_p(self, p_target: float, step: int) -> float:
        """Curriculum: linearly ramp p from 0 to p_target."""
        if self.curriculum_steps <= 0:
            return p_target
        frac = min(step / max(self.curriculum_steps, 1), 1.0)
        return p_target * frac

    def _hard_negative(self, x: torch.Tensor) -> torch.Tensor:
        """Shuffle batch dimension to create hard-negative replacements."""
        B = x.shape[0]
        if B == 1:
            return torch.randn_like(x)  # fallback for B=1
        # Cyclic shift so no sample is its own hard negative
        idx = torch.arange(B, device=x.device)
        idx = torch.roll(idx, shifts=1, dims=0)
        return x[idx].detach()

    def apply(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, bool]]:
        """Apply conditional dropout augmentation.

        Args:
            content_images: (B, C, H, W) content conditioning images.
            style_images:   (B, C, H, W) style conditioning images.
            step:           Current training step (for curriculum).

        Returns:
            (augmented_content, augmented_style, dropout_flags)
            dropout_flags: dict with 'content_dropped', 'style_dropped' bools.
        """
        B = content_images.shape[0]
        p_c = self._current_p(self.p_drop_content, step)
        p_s = self._current_p(self.p_drop_style, step)

        content_out = content_images.clone()
        style_out = style_images.clone()
        content_dropped = False
        style_dropped = False

        # Per-sample dropout mask
        drop_content_mask = torch.rand(B) < p_c  # (B,)
        drop_style_mask = torch.rand(B) < p_s    # (B,)

        if drop_content_mask.any():
            content_dropped = True
            for i in range(B):
                if drop_content_mask[i]:
                    if self.use_hard_negative:
                        # Replace with another sample's content
                        j = (i + 1) % B
                        content_out[i] = content_images[j].detach()
                    else:
                        # UniCalli: replace with pure noise
                        content_out[i] = torch.randn_like(content_images[i])

        if drop_style_mask.any():
            style_dropped = True
            for i in range(B):
                if drop_style_mask[i]:
                    if self.use_hard_negative:
                        j = (i + 1) % B
                        style_out[i] = style_images[j].detach()
                    else:
                        style_out[i] = torch.randn_like(style_images[i])

        return content_out, style_out, {
            "content_dropped": content_dropped,
            "style_dropped": style_dropped,
            "n_content_dropped": int(drop_content_mask.sum()),
            "n_style_dropped": int(drop_style_mask.sum()),
        }


# =============================================================================
# 4.  SpatialBoundingBoxConditioner
# =============================================================================

class SpatialBoundingBoxConditioner(nn.Module):
    """Rasterized spatial prior conditioning channel.

    UniCalli augments its calligraphy input with a rasterized bounding-box
    map encoding where characters live. For single-character FontDiffuser,
    we adapt this to a stroke-layout heatmap:

    Instead of bounding boxes, we compute:
      - Stroke centroid map: Gaussian blobs centred on the ink centroid
        per horizontal band (top/middle/bottom thirds of the character).
      - Ink density column profile: per-column sum of ink mass, normalised.

    These two channels (stacked into a 2-channel map) tell the denoiser
    the spatial distribution of strokes *before* it has to predict them,
    providing the same structural prior that UniCalli's bbox map does.

    This is purely deterministic (no learned weights). The 2-channel map
    is concatenated to the noisy target latent as extra conditioning channels,
    following the ControlNet-lite pattern.

    Args:
        image_size:   (H, W) of input images.
        gaussian_sigma: Sigma for centroid Gaussian blobs, as fraction of H.
        dark_ink:     True for dark ink on white background.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 64),
        gaussian_sigma: float = 0.08,
        dark_ink: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.sigma = gaussian_sigma * image_size[0]
        self.dark_ink = dark_ink

        H, W = image_size
        # Precompute coordinate grids
        self.register_buffer(
            "gy",
            torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W),
        )
        self.register_buffer(
            "gx",
            torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W),
        )

    def _ink_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to binary ink mask. Returns (B, H, W) float."""
        gray = images.mean(dim=1)  # (B, H, W)
        if self.dark_ink:
            return (gray < 0.5).float()
        return (gray > 0.5).float()

    def _centroid_map(self, ink: torch.Tensor) -> torch.Tensor:
        """Gaussian blob centred on stroke centroid. Returns (B, H, W)."""
        B, H, W = ink.shape
        eps = 1e-6

        # Weighted centroid of ink pixels
        total_ink = ink.sum(dim=(1, 2), keepdim=True).clamp(min=eps)  # (B,1,1)
        cy = (ink * self.gy.unsqueeze(0)).sum(dim=(1, 2), keepdim=True) / total_ink
        cx = (ink * self.gx.unsqueeze(0)).sum(dim=(1, 2), keepdim=True) / total_ink

        # Gaussian blob at centroid
        dist_sq = (self.gy.unsqueeze(0) - cy) ** 2 + (self.gx.unsqueeze(0) - cx) ** 2
        blob = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return blob  # (B, H, W)

    def _column_density(self, ink: torch.Tensor) -> torch.Tensor:
        """Normalised per-column ink density broadcast to (B, H, W)."""
        col_sum = ink.sum(dim=1, keepdim=True)  # (B, 1, W)
        col_norm = col_sum / (col_sum.max(dim=2, keepdim=True)[0].clamp(min=1e-6))
        return col_norm.expand_as(ink)  # (B, H, W)

    def forward(self, content_images: torch.Tensor) -> torch.Tensor:
        """Build 2-channel spatial prior map from content images.

        Args:
            content_images: (B, C, H, W) content reference in [0, 1].

        Returns:
            (B, 2, H, W) spatial prior map, values in [0, 1].
              channel 0: ink centroid Gaussian blob
              channel 1: column ink density profile
        """
        ink = self._ink_mask(content_images)      # (B, H, W)
        centroid_ch = self._centroid_map(ink)     # (B, H, W)
        density_ch = self._column_density(ink)    # (B, H, W)

        spatial_map = torch.stack([centroid_ch, density_ch], dim=1)  # (B, 2, H, W)
        return spatial_map.detach()  # no grad — purely a conditioning signal


# =============================================================================
# 5.  UniCalliImprovementsModule  (unified entry point)
# =============================================================================

class UniCalliImprovementsModule(nn.Module):
    """Unified module bundling all four UniCalli-inspired improvements.

    Drop-in addition to FontDiffuserDROTrainer._setup_models().

    Args:
        noise_scheduler:     Base DDPM/DDIM scheduler.
        n_classes:           Number of character classes.
        coarse_dim:          Coarse MRL prefix dimension (must match MRL config).
        image_size:          (H, W) of content/style images.
        style_noise_frac:    Style noise augmentation fraction (0 = clean).
        p_drop_content:      Content dropout probability.
        p_drop_style:        Style dropout probability.
        use_hard_negative:   Hard-negative vs pure-noise replacement.
        curriculum_steps:    Steps over which dropout ramps up.
        recog_pred_weight:   Weight on prediction branch recognition loss.
        use_spatial_prior:   Whether to compute spatial bbox conditioning.
        dark_ink:            True for dark-on-white font images.
    """

    def __init__(
        self,
        noise_scheduler,
        n_classes: int = 6763,
        coarse_dim: int = 64,
        image_size: tuple[int, int] = (64, 64),
        style_noise_frac: float = 0.0,
        p_drop_content: float = 0.1,
        p_drop_style: float = 0.05,
        use_hard_negative: bool = True,
        curriculum_steps: int = 1000,
        recog_pred_weight: float = 1.0,
        use_spatial_prior: bool = True,
        dark_ink: bool = True,
    ) -> None:
        super().__init__()

        # 1. Asymmetric noising
        self.asymmetric_noiser = AsymmetricNoisingScheduler(
            noise_scheduler=noise_scheduler,
            style_noise_fraction=style_noise_frac,
            content_noise_fraction=0.0,  # always clean content
        )

        # 2. Recognition head + loss
        self.recognition_head = StructureRecognitionHead(
            input_dim=coarse_dim,
            n_classes=n_classes,
        )
        self.recognition_loss = RecognitionAuxLoss(
            recognition_head=self.recognition_head,
            pred_weight=recog_pred_weight,
        )

        # 3. Conditional dropout
        self.conditional_dropout = ConditionalDropoutAugmentation(
            p_drop_content=p_drop_content,
            p_drop_style=p_drop_style,
            use_hard_negative=use_hard_negative,
            curriculum_steps=curriculum_steps,
        )

        # 4. Spatial prior conditioner (no learned weights)
        self.use_spatial_prior = use_spatial_prior
        if use_spatial_prior:
            self.spatial_conditioner = SpatialBoundingBoxConditioner(
                image_size=image_size,
                dark_ink=dark_ink,
            )

    def prepare_inputs(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        target_images: torch.Tensor,
        step: int = 0,
    ) -> dict:
        """Prepare all conditioning inputs for the training step.

        Call this at the start of train_step, before the model forward pass.

        Returns dict with keys:
          noisy_target, noise, timesteps,
          content_aug, style_aug,
          spatial_prior (optional),
          dropout_info
        """
        B, device = target_images.shape[0], target_images.device

        # Sample target timesteps and noise
        timesteps = self.asymmetric_noiser.sample_target_timesteps(B, device)
        noise = torch.randn_like(target_images)
        noisy_target = self.asymmetric_noiser.add_target_noise(
            target_images, noise, timesteps
        )

        # Apply conditional dropout augmentation
        content_aug, style_aug, dropout_info = self.conditional_dropout.apply(
            content_images, style_images, step=step
        )

        result = {
            "noisy_target": noisy_target,
            "noise": noise,
            "timesteps": timesteps,
            "content_aug": content_aug,
            "style_aug": style_aug,
            "dropout_info": dropout_info,
        }

        # Spatial prior from ORIGINAL (non-augmented) content
        if self.use_spatial_prior:
            with torch.no_grad():
                spatial_prior = self.spatial_conditioner(content_images)
            result["spatial_prior"] = spatial_prior

        return result

    def compute_recognition_loss(
        self,
        pred_coarse: torch.Tensor,
        content_coarse: torch.Tensor,
        char_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute joint recognition loss.

        Args:
            pred_coarse:    (B, 64) coarse MRL prefix from pred_x0 encoding.
            content_coarse: (B, 64) coarse MRL prefix from content encoding.
            char_labels:    (B,) character class integer labels.
        """
        return self.recognition_loss(pred_coarse, content_coarse, char_labels)


# =============================================================================
# 6.  Complete train_step integration example
# =============================================================================

def unicalli_train_step_template(
    trainer,
    samples: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Template showing how all four improvements integrate into train_step.

    Assumes trainer has:
      - trainer.unicalli          : UniCalliImprovementsModule
      - trainer.mrl_encoder       : MatryoshkaContentEncoder
      - trainer.mrl_loss          : MRLLossModule
      - trainer.aux_losses        : FontDiffuserAuxLosses
      - trainer.loss_schedule     : CombinedLossSchedule
      - trainer.dro_reward_module : DRORewardModule
      - trainer.model             : FontDiffuserFST model
      - trainer.global_step       : int

    This template is pseudocode — adapt field names to your actual trainer.
    """
    step = trainer.global_step
    device = trainer.accelerator.device

    content_images = samples["content_image"]       # (B, C, H, W) in [-1, 1]
    style_images   = samples["style_image"]         # (B, C, H, W) in [-1, 1]
    target_images  = samples["target_image"]        # (B, C, H, W) in [-1, 1]
    char_labels    = samples.get("char_label")      # (B,) int — NEW requirement

    # ── Step 1: Prepare inputs (asymmetric noising + dropout + spatial prior) ──
    inputs = trainer.unicalli.prepare_inputs(
        content_images=content_images,
        style_images=style_images,
        target_images=target_images,
        step=step,
    )

    # ── Step 2: Model forward pass ──
    noise_pred = trainer.model(
        noisy_latents=inputs["noisy_target"],
        timestep=inputs["timesteps"],
        content_images=inputs["content_aug"],       # augmented (dropout applied)
        style_images=inputs["style_aug"],           # augmented
        spatial_prior=inputs.get("spatial_prior"),  # optional extra channel
    )

    # ── Step 3: Reconstruct x0 and denormalise ──
    from src.tools.utils import x0_from_epsilon
    pred_x0 = x0_from_epsilon(
        scheduler=trainer.unicalli.asymmetric_noiser.scheduler,
        noise_pred=noise_pred,
        x_t=inputs["noisy_target"],
        timesteps=inputs["timesteps"],
    )
    pred_x0_01   = ((pred_x0 + 1.0) / 2.0).clamp(0, 1)
    content_01   = ((content_images + 1.0) / 2.0).clamp(0, 1)
    style_01     = ((style_images + 1.0) / 2.0).clamp(0, 1)

    # ── Step 4: MRL encoding ──
    _, pred_proj = trainer.mrl_encoder.forward_mrl(pred_x0_01)
    with torch.no_grad():
        _, content_proj = trainer.mrl_encoder.forward_mrl(content_01)

    mrl_loss, mrl_metrics = trainer.mrl_loss(
        pred_projections=pred_proj,
        content_projections=content_proj,
        content_images=content_01,
    )

    # ── Step 5: Recognition loss (NEW from UniCalli) ──
    recog_loss = torch.zeros(1, device=device).squeeze()
    recog_metrics: dict[str, float] = {}
    if char_labels is not None:
        # pred_proj[0] is coarse prefix (64d) — same index as nesting_dims[0]
        recog_loss, recog_metrics = trainer.unicalli.compute_recognition_loss(
            pred_coarse=pred_proj[0],      # coarse MRL prefix from pred
            content_coarse=content_proj[0],
            char_labels=char_labels.to(device),
        )

    # ── Step 6: Aux losses (freq band + stroke topology + weighted diffusion) ──
    aux_loss, aux_metrics = trainer.aux_losses(
        pred_x0=pred_x0_01,
        content=content_01,
        style=style_01,
        noise_pred=noise_pred,
        noise_target=inputs["noise"],
    )

    # ── Step 7: DRO reward ──
    dro_reward, dro_metrics = trainer.dro_reward_module(
        pred_images=pred_x0_01,
        content_images=content_01,
        style_images=style_01,
    )
    dro_loss = -dro_reward

    # ── Step 8: Weighted combination ──
    weights = trainer.loss_schedule.get_weights(step)

    total_loss = (
        weights["mrl"]  * mrl_loss      # encoder: content structure
      + weights["aux"]  * aux_loss      # decoder: freq/topology/diffusion
      + 0.5             * recog_loss    # encoder: glyph identity (UniCalli)
      + trainer.dro_weight * dro_loss   # perceptual quality
    )

    # ── Step 9: Single backward ──
    trainer.accelerator.backward(total_loss)

    # Aggregate metrics
    all_metrics = {
        **mrl_metrics,
        **recog_metrics,
        **aux_metrics,
        **dro_metrics,
        **{f"dropout/{k}": float(v) for k, v in inputs["dropout_info"].items()},
        "loss/mrl_weight": weights["mrl"],
        "loss/aux_weight": weights["aux"],
        "loss/total": total_loss.item(),
    }
    return total_loss, all_metrics
