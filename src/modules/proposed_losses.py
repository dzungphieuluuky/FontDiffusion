"""
Proposed loss functions for FontDiffuserFST + Fourier model.

Design constraints:
  - Zero new trainable weights
  - Minimal RAM overhead (no large feature extractors)
  - Each loss adds <0.5s per step on GPU (Colab/Kaggle budget)

Three innovations:
  1. FreqBandContentStyleLoss  — exploits existing FFT pipeline to separate
                                 content (low-freq) and style (high-freq) loss
                                 in frequency space directly.

  2. StrokeTopologyLoss        — binarises pred and content via differentiable
                                 soft-thresholding, penalises stroke presence/
                                 absence mismatches (topology errors).

  3. FreqWeightedDiffusionLoss — reweights the standard epsilon-prediction MSE
                                 loss spatially using a frequency-derived stroke
                                 mask so stroke pixels dominate the diffusion
                                 objective.

Integration notes:
  - Pass pre-computed FFT tensors (fft_content, fft_style, fft_pred) to avoid
    recomputation when your base model already performs FFTs.
  - FreqWeightedDiffusionLoss is returned separately from the auxiliary total
    so the trainer can decide whether to add it or use it in place of its own
    MSE loss.
  - Monitor freq_diff/mean_weight and freq_diff/weight_std each step; if
    mean_weight drops below 0.5, consider disabling normalize_weights.
  - Anneal StrokeTopologyLoss temperature during training for sharper topology
    boundaries in later epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Shared FFT utility
# =============================================================================

def compute_fft2(x: torch.Tensor) -> torch.Tensor:
    """Compute normalised 2-D FFT of a real image tensor.

    Args:
        x: (B, C, H, W) real-valued tensor.

    Returns:
        Complex tensor of the same shape.
    """
    return torch.fft.fft2(x, norm="ortho")


# =============================================================================
# 1. FreqBandContentStyleLoss
# =============================================================================

class FreqBandContentStyleLoss(nn.Module):
    """Frequency-band decomposed content/style loss.

    Your model already computes FFT to disentangle content (low-freq) and
    style (high-freq). This loss enforces that separation in the prediction:

        L_freq = w_low  * || LF(pred) - LF(content) ||_1
               + w_high * || HF(pred) - HF(style)   ||_1

    LF = low-frequency band  (radius <= lf_radius in the Fourier domain)
    HF = high-frequency band (radius >  hf_radius in the Fourier domain)

    Using L1 rather than L2 — it is more robust to outlier frequency components
    and produces sharper reconstructions in practice.

    Args:
        lf_radius: Low-frequency circle radius as a fraction of image size.
                   Captures global structure / stroke layout. Default 0.1
                   means the inner 10% of the frequency domain.
        hf_radius: High-frequency annulus start radius (same scale).
                   Captures fine texture / style details. Default 0.4.
        lf_weight: Weight on the low-frequency content term.
        hf_weight: Weight on the high-frequency style term.
    """

    def __init__(
        self,
        lf_radius: float = 0.1,
        hf_radius: float = 0.4,
        lf_weight: float = 1.0,
        hf_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lf_radius = lf_radius
        self.hf_radius = hf_radius
        self.lf_weight = lf_weight
        self.hf_weight = hf_weight

        # Cache for masks — rebuilt if image size changes
        self._cached_size: Optional[tuple[int, int]] = None
        self._lf_mask: Optional[torch.Tensor] = None
        self._hf_mask: Optional[torch.Tensor] = None

    def _build_masks(self, H: int, W: int, device: torch.device) -> None:
        if self._cached_size == (H, W) and self._lf_mask is not None:
            if self._lf_mask.device == device:
                return

        # Normalised frequency grid in [-1, 1]
        fy = torch.fft.fftfreq(H, device=device)  # (H,)
        fx = torch.fft.fftfreq(W, device=device)  # (W,)
        fy2 = (fy.unsqueeze(1) ** 2)              # (H, 1)
        fx2 = (fx.unsqueeze(0) ** 2)              # (1, W)
        r = (fy2 + fx2).sqrt()                    # (H, W) — normalised radius

        self._lf_mask = (r <= self.lf_radius).float().unsqueeze(0).unsqueeze(0)
        self._hf_mask = (r > self.hf_radius).float().unsqueeze(0).unsqueeze(0)
        self._cached_size = (H, W)

    def _extract_band(
        self,
        X_fft: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply frequency mask to a pre-computed FFT and return real IFFT.

        Args:
            X_fft: Pre-computed complex FFT, shape (B, C, H, W).
            mask:  Real-valued frequency mask broadcastable to X_fft.

        Returns:
            Real-valued inverse FFT of the masked spectrum, same shape.
        """
        # Cast mask to match the real dtype of the complex tensor to satisfy
        # mixed-precision constraints.
        mask = mask.to(X_fft.real.dtype)
        return torch.fft.ifft2(X_fft * mask, norm="ortho").real

    def forward(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        fft_pred: Optional[torch.Tensor] = None,
        fft_content: Optional[torch.Tensor] = None,
        fft_style: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred:        (B, C, H, W) model prediction x0, in [0, 1].
            content:     (B, C, H, W) content reference, in [0, 1].
            style:       (B, C, H, W) style reference, in [0, 1].
            fft_pred:    Optional pre-computed FFT of pred. If supplied,
                         avoids recomputing fft2(pred).
            fft_content: Optional pre-computed FFT of content.
            fft_style:   Optional pre-computed FFT of style.

        Returns:
            (scalar loss, metrics dict)
        """
        B, C, H, W = pred.shape
        self._build_masks(H, W, pred.device)
        lf_mask = self._lf_mask.to(pred.device)
        hf_mask = self._hf_mask.to(pred.device)

        # Reuse caller-supplied FFTs to avoid redundant transforms
        X_pred    = fft_pred    if fft_pred    is not None else compute_fft2(pred)
        X_content = fft_content if fft_content is not None else compute_fft2(content.detach())
        X_style   = fft_style   if fft_style   is not None else compute_fft2(style.detach())

        # Low-freq: pred should match content structure
        pred_lf    = self._extract_band(X_pred,    lf_mask)
        content_lf = self._extract_band(X_content, lf_mask)
        lf_loss = F.l1_loss(pred_lf, content_lf)

        # High-freq: pred should match style texture
        pred_hf   = self._extract_band(X_pred,   hf_mask)
        style_hf  = self._extract_band(X_style,  hf_mask)
        hf_loss = F.l1_loss(pred_hf, style_hf)

        loss = self.lf_weight * lf_loss + self.hf_weight * hf_loss

        return loss, {
            "freq/lf_content_loss": lf_loss.item(),
            "freq/hf_style_loss": hf_loss.item(),
            "freq/total": loss.item(),
        }


# =============================================================================
# 2. StrokeTopologyLoss
# =============================================================================

class StrokeTopologyLoss(nn.Module):
    """Differentiable stroke topology consistency loss.

    Font characters are fundamentally binary: ink (stroke) vs. paper
    (background). Style transfer should modify *how* strokes look, not
    *where* they exist or whether they exist at all.

    This loss binarises pred and content using a differentiable soft-sigmoid
    threshold (no argmax, fully differentiable) and penalises:

        L_topo = BCE(sigma((pred - tau) / T), stroke_mask_content)

    where tau is the threshold and T is a temperature controlling sharpness.
    This directly penalises:
      - False positives: pred has ink where content has none (added strokes)
      - False negatives: pred has no ink where content has some (dropped strokes)

    Additionally computes a stroke-count consistency term:
        L_count = || mean(stroke_pred) - mean(stroke_content) ||

    which penalises global ink density mismatch without requiring per-pixel
    alignment (useful for style variations that legitimately change stroke width).

    Tip: anneal `temperature` from a large value (e.g. 0.2) toward the default
    (0.05) over training so the topology boundary sharpens gradually.

    Args:
        threshold: Binarisation threshold in [0, 1] for ink detection.
        temperature: Sigmoid sharpness. Lower = harder binary; higher = softer.
        topology_weight: Weight on the per-pixel topology BCE term.
        density_weight: Weight on the global ink density term.
        dark_ink: If True, ink pixels are dark (value < threshold). Set False
                  for light-on-dark datasets.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        temperature: float = 0.05,
        topology_weight: float = 1.0,
        density_weight: float = 0.3,
        dark_ink: bool = True,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.topology_weight = topology_weight
        self.density_weight = density_weight
        self.dark_ink = dark_ink

    def _soft_stroke_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return a soft [0,1] stroke probability map.

        For dark-ink images (ink < threshold): invert then threshold.
        For light-ink images (ink > threshold): threshold directly.
        """
        if self.dark_ink:
            # Ink is dark → low pixel value → invert before thresholding
            x_ink = 1.0 - x
        else:
            x_ink = x
        # Soft binarisation via sigmoid; cast threshold to input dtype for
        # mixed-precision compatibility.
        tau = torch.tensor(self.threshold, dtype=x_ink.dtype, device=x_ink.device)
        T   = torch.tensor(self.temperature, dtype=x_ink.dtype, device=x_ink.device)
        return torch.sigmoid((x_ink - tau) / T)

    def forward(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred:    (B, C, H, W) model prediction x0, in [0, 1].
            content: (B, C, H, W) content reference, in [0, 1].

        Returns:
            (scalar loss, metrics dict)
        """
        pred_strokes = self._soft_stroke_map(pred)
        with torch.no_grad():
            content_strokes = self._soft_stroke_map(content)

        # Per-pixel topology BCE — gradient flows through pred_strokes only
        topo_loss = F.binary_cross_entropy(
            pred_strokes,
            content_strokes,
            reduction="mean",
        )

        # Global ink density consistency
        pred_density = pred_strokes.mean()
        content_density = content_strokes.mean().detach()
        density_loss = (pred_density - content_density).abs()

        loss = self.topology_weight * topo_loss + self.density_weight * density_loss

        # Compute hard accuracy metric for logging (no grad)
        with torch.no_grad():
            pred_hard = (pred_strokes > 0.5).float()
            content_hard = (content_strokes > 0.5).float()
            iou_num = (pred_hard * content_hard).sum()
            iou_den = ((pred_hard + content_hard) > 0).float().sum().clamp(min=1)
            stroke_iou = (iou_num / iou_den).item()

        return loss, {
            "topo/topology_bce": topo_loss.item(),
            "topo/density_loss": density_loss.item(),
            "topo/total": loss.item(),
            "topo/stroke_iou": stroke_iou,  # diagnostic only
        }


# =============================================================================
# 3. FreqWeightedDiffusionLoss
# =============================================================================

class FreqWeightedDiffusionLoss(nn.Module):
    """Frequency-guided spatially-weighted diffusion noise prediction loss.

    Replaces the standard uniform MSE on epsilon prediction:

        L_standard = || eps_pred - eps ||^2

    with a spatially weighted version:

        L_weighted = || W(content) * (eps_pred - eps) ||^2

    where W(content) is a soft stroke-importance map derived from the
    low-frequency content of the content image. Stroke-dense regions get
    higher weight; background gets lower weight.

    This does NOT change the diffusion objective — it reweights it.  The model
    still learns to predict noise everywhere, but gradients from stroke regions
    dominate, so the model prioritises getting stroke structure right.

    The weight map is derived purely from the content image via FFT:
    1. Extract low-frequency reconstruction of content → coarse glyph shape.
    2. Compute per-pixel variance from the LF reconstruction → stroke salience.
    3. Normalise to [1, max_weight] so background gets weight 1 (not 0).

    Monitoring: log freq_diff/mean_weight and freq_diff/weight_std each step.
    If mean_weight falls below ~0.5, consider setting normalize_weights=False
    or clamping the weight map (weight_map.clamp(min=0.5)).

    Args:
        lf_radius: Low-frequency radius for stroke shape extraction.
        max_weight: Maximum spatial weight applied to stroke pixels.
                    Higher = more emphasis on strokes vs background.
        normalize_weights: If True, normalise weight map so mean == 1
                           (preserves overall loss magnitude).
    """

    def __init__(
        self,
        lf_radius: float = 0.15,
        max_weight: float = 3.0,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.lf_radius = lf_radius
        self.max_weight = max_weight
        self.normalize_weights = normalize_weights

        self._cached_size: Optional[tuple[int, int]] = None
        self._lf_mask: Optional[torch.Tensor] = None

    def _build_lf_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self._cached_size == (H, W) and self._lf_mask is not None:
            if self._lf_mask.device == device:
                return self._lf_mask
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.fftfreq(W, device=device)
        r = (fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2).sqrt()
        self._lf_mask = (r <= self.lf_radius).float().unsqueeze(0).unsqueeze(0)
        self._cached_size = (H, W)
        return self._lf_mask

    @torch.no_grad()
    def _build_weight_map(
        self,
        content: torch.Tensor,
        fft_content: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Derive spatial importance weights from content image LF band.

        Args:
            content:     (B, C, H, W) content reference image in [0, 1].
            fft_content: Optional pre-computed FFT of content to avoid
                         redundant transform.

        Returns:
            (B, 1, H, W) weight map with values in [1, max_weight].
            If normalize_weights=True the per-sample mean is clamped to >=0.5
            before division so extreme normalisation is prevented.
        """
        B, C, H, W = content.shape
        lf_mask = self._build_lf_mask(H, W, content.device)

        # Cast mask to content dtype for mixed-precision safety
        lf_mask = lf_mask.to(content.dtype)

        # Reuse caller-supplied FFT when available
        X = fft_content if fft_content is not None else compute_fft2(content)
        lf_recon = torch.fft.ifft2(X * lf_mask, norm="ortho").real  # (B,C,H,W)

        # Stroke salience: pixels where LF reconstruction departs from background
        lf_mean = lf_recon.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # Normalise to [0,1]
        b_min = lf_mean.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        b_max = lf_mean.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        salience = (lf_mean - b_min) / (b_max - b_min + 1e-8)  # (B,1,H,W)

        # Map to [1, max_weight]
        weight_map = 1.0 + (self.max_weight - 1.0) * salience

        if self.normalize_weights:
            # Clamp mean to >=0.5 to prevent extreme rescaling when stroke
            # coverage is very sparse (see monitoring guidance in docstring).
            mean_w = weight_map.flatten(2).mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            weight_map = weight_map / mean_w.clamp(min=0.5)

        return weight_map  # (B, 1, H, W)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        content: torch.Tensor,
        fft_content: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            noise_pred:   (B, C, H, W) model epsilon prediction.
            noise_target: (B, C, H, W) ground-truth noise added to target.
            content:      (B, C, H, W) content reference image in [0, 1].
                          Used ONLY for weight map derivation — no grad needed.
            fft_content:  Optional pre-computed FFT of content.

        Returns:
            (scalar loss, metrics dict)
        """
        weight_map = self._build_weight_map(content, fft_content=fft_content)

        # Spatially weighted MSE — weight broadcast over C
        sq_err = (noise_pred - noise_target) ** 2          # (B, C, H, W)
        weighted_err = weight_map * sq_err                 # broadcast over C
        loss = weighted_err.mean()

        # Standard MSE + weight diagnostics for logging
        with torch.no_grad():
            standard_mse = sq_err.mean().item()
            mean_weight   = weight_map.mean().item()
            std_weight    = weight_map.std().item()

        return loss, {
            "freq_diff/weighted_mse": loss.item(),
            "freq_diff/standard_mse": standard_mse,
            "freq_diff/mean_weight":  mean_weight,
            "freq_diff/weight_std":   std_weight,   # monitor for extreme values
        }


# =============================================================================
# Combined wrapper — drop-in for the trainer
# =============================================================================

class FontDiffuserAuxLosses(nn.Module):
    """Combines all three proposed losses into one module.

    Designed to be instantiated once in _setup_models and called from
    train_step alongside the existing diffusion + FST losses.

    FFT pre-computation
    -------------------
    Pass fft_pred, fft_content, and fft_style to avoid recomputing transforms
    that your base model may already have computed for content/style separation.
    If not supplied, each sub-loss will compute its own FFT.

    Diffusion loss decoupling
    -------------------------
    FreqWeightedDiffusionLoss is intended to *replace* the standard MSE, so
    the wrapper returns it as a dedicated ``diffusion_loss`` output rather than
    folding it into ``aux_total``.  The trainer can then do either:

        loss = aux_total + diffusion_loss          # add alongside existing MSE
        loss = aux_total + diffusion_loss          # replace existing MSE entirely

    Args:
        use_freq_band:    Enable FreqBandContentStyleLoss.
        use_stroke_topo:  Enable StrokeTopologyLoss.
        use_freq_diff:    Enable FreqWeightedDiffusionLoss.
        freq_weight:      Scalar weight on FreqBandContentStyleLoss.
        topo_weight:      Scalar weight on StrokeTopologyLoss.
        (FreqWeightedDiffusionLoss replaces rather than supplements the
         standard diffusion loss, so it has no external weight scalar.)
    """

    def __init__(
        self,
        use_freq_band: bool = True,
        use_stroke_topo: bool = True,
        use_freq_diff: bool = True,
        freq_weight: float = 0.5,
        topo_weight: float = 0.3,
        # FreqBandContentStyleLoss kwargs
        lf_radius: float = 0.1,
        hf_radius: float = 0.4,
        lf_weight: float = 1.0,
        hf_weight: float = 0.5,
        # StrokeTopologyLoss kwargs
        threshold: float = 0.5,
        temperature: float = 0.05,
        topology_weight: float = 1.0,
        density_weight: float = 0.3,
        dark_ink: bool = True,
        # FreqWeightedDiffusionLoss kwargs
        fw_lf_radius: float = 0.15,
        max_weight: float = 3.0,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()

        self.use_freq_band = use_freq_band
        self.use_stroke_topo = use_stroke_topo
        self.use_freq_diff = use_freq_diff
        self.freq_weight = freq_weight
        self.topo_weight = topo_weight

        if use_freq_band:
            self.freq_band_loss = FreqBandContentStyleLoss(
                lf_radius=lf_radius,
                hf_radius=hf_radius,
                lf_weight=lf_weight,
                hf_weight=hf_weight,
            )
        if use_stroke_topo:
            self.stroke_topo_loss = StrokeTopologyLoss(
                threshold=threshold,
                temperature=temperature,
                topology_weight=topology_weight,
                density_weight=density_weight,
                dark_ink=dark_ink,
            )
        if use_freq_diff:
            self.freq_diff_loss = FreqWeightedDiffusionLoss(
                lf_radius=fw_lf_radius,
                max_weight=max_weight,
                normalize_weights=normalize_weights,
            )

    def forward(
        self,
        pred_x0: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        noise_pred: Optional[torch.Tensor] = None,
        noise_target: Optional[torch.Tensor] = None,
        # Optional pre-computed FFTs — pass these if your base model already
        # has them to avoid redundant torch.fft.fft2 calls.
        fft_pred: Optional[torch.Tensor] = None,
        fft_content: Optional[torch.Tensor] = None,
        fft_style: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
        """Compute all enabled auxiliary losses.

        FFTs of pred_x0, content, and style are computed once here (or reused
        from the caller) and forwarded to each sub-loss to eliminate duplicate
        transforms.

        Args:
            pred_x0:      (B, C, H, W) reconstructed clean prediction in [0,1].
            content:      (B, C, H, W) content reference in [0, 1].
            style:        (B, C, H, W) style reference in [0, 1].
            noise_pred:   (B, C, H, W) epsilon prediction (required if
                          use_freq_diff=True).
            noise_target: (B, C, H, W) ground-truth noise (required if
                          use_freq_diff=True).
            fft_pred:     Optional pre-computed fft2(pred_x0).
            fft_content:  Optional pre-computed fft2(content).
            fft_style:    Optional pre-computed fft2(style).

        Returns:
            aux_total:      Scalar sum of freq_band and stroke_topo losses
                            (weighted). Does NOT include the diffusion loss.
            diffusion_loss: FreqWeightedDiffusionLoss scalar, or None if
                            use_freq_diff=False.  The trainer should decide
                            whether to add this to aux_total or use it instead
                            of its existing MSE.
            all_metrics:    Combined metrics dict for logging.

        Example trainer usage::

            aux_total, diff_loss, metrics = aux_losses(
                pred_x0, content, style, noise_pred, noise_target
            )
            # Option A — replace standard MSE:
            total_loss = aux_total + diff_loss
            # Option B — add alongside existing mse_loss:
            total_loss = mse_loss + aux_total + diff_loss
        """
        # Pre-compute FFTs once for all sub-losses that need them
        if self.use_freq_band or self.use_freq_diff:
            if fft_content is None:
                fft_content = compute_fft2(content.detach())
            if fft_pred is None and self.use_freq_band:
                fft_pred = compute_fft2(pred_x0)
            if fft_style is None and self.use_freq_band:
                fft_style = compute_fft2(style.detach())

        aux_total = torch.zeros(1, device=pred_x0.device, dtype=pred_x0.dtype).squeeze()
        all_metrics: dict[str, float] = {}

        if self.use_freq_band:
            fb_loss, fb_metrics = self.freq_band_loss(
                pred_x0, content, style,
                fft_pred=fft_pred,
                fft_content=fft_content,
                fft_style=fft_style,
            )
            aux_total = aux_total + self.freq_weight * fb_loss
            all_metrics.update(fb_metrics)

        if self.use_stroke_topo:
            st_loss, st_metrics = self.stroke_topo_loss(pred_x0, content)
            aux_total = aux_total + self.topo_weight * st_loss
            all_metrics.update(st_metrics)

        diffusion_loss: Optional[torch.Tensor] = None
        if self.use_freq_diff:
            if noise_pred is None or noise_target is None:
                raise ValueError(
                    "noise_pred and noise_target are required when use_freq_diff=True"
                )
            diffusion_loss, fd_metrics = self.freq_diff_loss(
                noise_pred, noise_target, content,
                fft_content=fft_content,
            )
            all_metrics.update(fd_metrics)

        all_metrics["aux/total_loss"] = aux_total.item()
        if diffusion_loss is not None:
            all_metrics["aux/diffusion_loss"] = diffusion_loss.item()

        return aux_total, diffusion_loss, all_metrics
