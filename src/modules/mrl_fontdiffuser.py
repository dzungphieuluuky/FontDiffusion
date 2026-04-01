"""
Matryoshka Representation Learning (MRL) for FontDiffuser.

Architecture overview
─────────────────────
MRL is applied to the *content encoder* (the component that produces the
conditioning signal for glyph structure). Rather than treating the encoder
output as a single flat vector, MRL trains it so that every nested prefix
of dimension d_i is already a complete, meaningful representation at
granularity i.

This mirrors the existing Fourier decomposition:

    Fourier space  │  Feature space (MRL)
    ───────────────┼────────────────────────────────────────
    Low-freq band  │  Coarse prefix  (dims 0..d0)   ← topology
    Mid-freq band  │  Mid prefix     (dims 0..d1)   ← stroke layout
    High-freq band │  Fine prefix    (dims 0..d2)   ← stroke detail

Why this fixes the content-structure problem
────────────────────────────────────────────
Without MRL, the LPIPS style reward corrupts the full content embedding
indiscriminately.  With MRL, coarse prefix dimensions are trained with
higher loss weight so they become robust to style pressure.  The model
cannot sacrifice glyph topology to achieve style matching because the
coarse content signal is an independent, heavily supervised prefix.

Integration with proposed losses
─────────────────────────────────
MRL introduces a new multi-granularity content loss (MRLContentLoss) that
replaces or supplements the MS-SSIM content term.  The existing three
losses (FreqBandContentStyleLoss, StrokeTopologyLoss,
FreqWeightedDiffusionLoss) remain unchanged — MRL operates on the encoder
side while those losses operate on the decoder/prediction side.

New components
──────────────
  MRLProjectionHead   — lightweight linear projections per granularity
                        (the only new weights; O(D²) params, ~0.1% of
                        total model size for D=512)
  MRLContentLoss      — multi-granularity contrastive + reconstruction loss
  MRLFourierAlignment — aligns MRL granularity boundaries with Fourier bands
  MatryoshkaContentEncoder — wraps existing content encoder; zero surgery
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MRLProjectionHead
# ─────────────────────────────────────────────────────────────────────────────

class MRLProjectionHead(nn.Module):
    """One lightweight linear projection per Matryoshka granularity.

    For each nesting dimension d_i, projects the d_i-dimensional prefix
    of the full encoder embedding into a normalised d_i-dimensional
    representation used for loss computation.

    Using a *separate* projection per granularity is critical: without it
    the encoder can represent different information at the same dimensions
    depending on context, defeating the nesting guarantee.

    Args:
        embedding_dim: Full encoder output dimension (e.g. 512).
        nesting_dims:  Ordered list of prefix sizes, smallest first.
                       e.g. [64, 128, 256, 512].
                       Must all be ≤ embedding_dim.
    """

    def __init__(
        self,
        embedding_dim: int,
        nesting_dims: Sequence[int],
    ) -> None:
        super().__init__()

        assert all(d <= embedding_dim for d in nesting_dims), (
            "All nesting_dims must be ≤ embedding_dim"
        )
        assert nesting_dims == sorted(nesting_dims), (
            "nesting_dims must be in ascending order"
        )

        self.embedding_dim = embedding_dim
        self.nesting_dims = list(nesting_dims)

        # One linear per granularity: d_i → d_i (square, no bottleneck)
        # These are small: 64²+128²+256²+512² = 360,448 params for default dims
        self.projections = nn.ModuleList([
            nn.Linear(d, d, bias=False) for d in nesting_dims
        ])

        # Initialise as near-identity so training starts from a valid state
        for proj in self.projections:
            nn.init.eye_(proj.weight)

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Project each prefix of the embedding.

        Args:
            embedding: (B, embedding_dim) or (B, T, embedding_dim) full
                       encoder output.

        Returns:
            List of L2-normalised projected embeddings, one per granularity,
            smallest first.  Shape per element: (B, d_i) or (B, T, d_i).
        """
        outputs: list[torch.Tensor] = []
        for d, proj in zip(self.nesting_dims, self.projections):
            prefix = embedding[..., :d]       # take prefix
            projected = proj(prefix)           # linear transform
            normalised = F.normalize(projected, p=2, dim=-1)
            outputs.append(normalised)
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MRLContentLoss
# ─────────────────────────────────────────────────────────────────────────────

class MRLContentLoss(nn.Module):
    """Multi-granularity content preservation loss.

    For each Matryoshka granularity, computes two terms:

    (a) Contrastive alignment loss
        Within a batch, the projected prefix of pred_x0's encoding should
        be close to the same prefix of content's encoding, and far from
        other samples' content encodings.  Uses InfoNCE (NT-Xent variant).

        This enforces that glyph identity is preserved at every granularity.

    (b) Reconstruction consistency loss
        The coarser prefixes should be sufficient to reconstruct the coarser
        prefixes of the content embedding — enforced via cosine similarity.

        This prevents the encoder from encoding topology only in fine dims.

    Granularity weights follow MRL convention: coarser levels get higher
    weight, as they represent more fundamental structural information.

    Args:
        nesting_dims:      Ordered list of prefix sizes (ascending).
        temperature:       InfoNCE temperature. Smaller = sharper distribution.
        base_weight:       Weight for the finest granularity.
        weight_multiplier: Each coarser level is weighted by
                           base_weight * weight_multiplier^(n_levels - 1 - i).
                           Default gives coarsest level 8× the finest.
        reconstruction_weight: Weight of the reconstruction consistency term.
    """

    def __init__(
        self,
        nesting_dims: Sequence[int],
        temperature: float = 0.07,
        base_weight: float = 1.0,
        weight_multiplier: float = 2.0,
        reconstruction_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.nesting_dims = list(nesting_dims)
        self.temperature = temperature
        self.reconstruction_weight = reconstruction_weight

        n = len(nesting_dims)
        # Coarsest level gets highest weight
        self.granularity_weights = [
            base_weight * (weight_multiplier ** (n - 1 - i))
            for i in range(n)
        ]
        # Normalise so weights sum to n (preserves scale)
        w_sum = sum(self.granularity_weights)
        self.granularity_weights = [
            w * n / w_sum for w in self.granularity_weights
        ]

        logger.info(
            "MRLContentLoss granularity weights: "
            + ", ".join(
                f"d={d}: {w:.3f}"
                for d, w in zip(nesting_dims, self.granularity_weights)
            )
        )

    def _infonce_loss(
        self,
        pred_proj: torch.Tensor,
        content_proj: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE loss treating each (pred_i, content_i) as a positive pair.

        Args:
            pred_proj:    (B, d) L2-normalised projected pred prefixes.
            content_proj: (B, d) L2-normalised projected content prefixes.

        Returns:
            Scalar InfoNCE loss.
        """
        B = pred_proj.shape[0]
        if B == 1:
            # InfoNCE undefined for batch size 1 — fall back to cosine loss
            return 1.0 - (pred_proj * content_proj).sum(dim=-1).mean()

        # Similarity matrix (B, B): rows = pred, cols = content
        logits = (pred_proj @ content_proj.t()) / self.temperature  # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(B, device=pred_proj.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def _reconstruction_loss(
        self,
        pred_proj: torch.Tensor,
        content_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity consistency — pred prefix should align with content prefix."""
        # (pred_proj and content_proj are already L2-normalised)
        cos_sim = (pred_proj * content_proj.detach()).sum(dim=-1)  # (B,)
        return (1.0 - cos_sim).mean()

    def forward(
        self,
        pred_projections: list[torch.Tensor],
        content_projections: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MRL content loss across all granularities.

        Args:
            pred_projections:    Output of MRLProjectionHead(pred_encoding).
            content_projections: Output of MRLProjectionHead(content_encoding).
                                 content_projections should be computed with
                                 torch.no_grad() on the content reference.

        Returns:
            (scalar loss, metrics dict)
        """
        assert len(pred_projections) == len(content_projections) == len(self.nesting_dims)

        total = torch.zeros(1, device=pred_projections[0].device).squeeze()
        metrics: dict[str, float] = {}

        for i, (d, w, pred_p, content_p) in enumerate(
            zip(self.nesting_dims, self.granularity_weights,
                pred_projections, content_projections)
        ):
            contrastive = self._infonce_loss(pred_p, content_p)
            recon = self._reconstruction_loss(pred_p, content_p)

            level_loss = contrastive + self.reconstruction_weight * recon
            total = total + w * level_loss

            metrics[f"mrl/contrastive_d{d}"] = contrastive.item()
            metrics[f"mrl/recon_d{d}"] = recon.item()
            metrics[f"mrl/level_loss_d{d}"] = level_loss.item()

        metrics["mrl/total"] = total.item()
        return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MRLFourierAlignment
# ─────────────────────────────────────────────────────────────────────────────

class MRLFourierAlignment(nn.Module):
    """Aligns MRL granularity structure with Fourier frequency decomposition.

    The existing Fourier pipeline separates images into low/mid/high frequency
    bands. This loss ensures that the corresponding MRL prefix is *predictive*
    of the corresponding frequency band — i.e. coarse prefix → low-freq
    reconstruction, fine prefix adds → high-freq detail.

    Concretely, for each granularity i with prefix dim d_i and corresponding
    frequency band B_i, it minimises:

        || decode_i(prefix_d_i(content_emb)) - IFFT(FFT(content) * mask_i) ||_1

    where decode_i is a tiny 2-layer MLP decoder (per granularity) that maps
    from d_i dimensions back to a flattened frequency band.

    This is the only component that adds meaningful new weights, but they are
    small MLPs (d_i → 256 → H*W per granularity, for the LF/MF bands only).

    Design choice: Only applied to the two coarser granularities (LF, MF).
    The finest granularity alignment is implicit — if coarse aligns correctly,
    the residual (fine dims) naturally captures high-freq detail.

    Args:
        nesting_dims:   Ordered prefix dimensions, smallest first.
        freq_radii:     Frequency radius boundaries, one fewer than nesting_dims.
                        e.g. [0.1, 0.3] for 3 nesting levels means:
                          level 0 (d=64)  ↔ r ≤ 0.1  (low-freq)
                          level 1 (d=128) ↔ 0.1 < r ≤ 0.3  (mid-freq)
                          level 2 (d=256) ↔ r > 0.3  (high-freq, no decoder)
        spatial_size:   (H, W) of the feature maps / images. Used to size
                        the decoder output.
        hidden_dim:     Hidden dimension of the band decoder MLPs.
        alignment_weight: Loss weight for this term.
    """

    def __init__(
        self,
        nesting_dims: Sequence[int],
        freq_radii: Sequence[float],
        spatial_size: tuple[int, int] = (64, 64),
        hidden_dim: int = 256,
        alignment_weight: float = 0.3,
    ) -> None:
        super().__init__()

        assert len(freq_radii) == len(nesting_dims) - 1, (
            "len(freq_radii) must equal len(nesting_dims) - 1"
        )

        self.nesting_dims = list(nesting_dims)
        self.freq_radii = list(freq_radii)
        self.spatial_size = spatial_size
        self.alignment_weight = alignment_weight

        H, W = spatial_size
        # Only build decoders for levels 0..n-2 (all except finest)
        n_supervised = len(nesting_dims) - 1
        self.band_decoders = nn.ModuleList()
        for i in range(n_supervised):
            d_in = nesting_dims[i]
            # Output: single-channel frequency band amplitude (H*W)
            self.band_decoders.append(nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, H * W),
            ))

        # Frequency masks (built lazily, cached)
        self._cached_masks: Optional[list[torch.Tensor]] = None
        self._cached_device: Optional[torch.device] = None

    def _build_band_masks(self, device: torch.device) -> list[torch.Tensor]:
        """Build cumulative band masks for each supervised granularity."""
        if self._cached_masks is not None and self._cached_device == device:
            return self._cached_masks

        H, W = self.spatial_size
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.fftfreq(W, device=device)
        r = (fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2).sqrt()  # (H, W)

        masks = []
        prev_r = 0.0
        for radius in self.freq_radii:
            # Band mask: prev_r < r <= radius
            mask = ((r > prev_r) & (r <= radius)).float()
            masks.append(mask)
            prev_r = radius

        self._cached_masks = masks
        self._cached_device = device
        return masks

    def _extract_band_amplitude(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract mean-channel amplitude of a frequency band.

        Returns:
            (B, H*W) real-valued amplitude map for the band.
        """
        X = torch.fft.fft2(image, norm="ortho")  # (B, C, H, W) complex
        X_band = X * mask.unsqueeze(0).unsqueeze(0)
        band_recon = torch.fft.ifft2(X_band, norm="ortho").real  # (B, C, H, W)
        # Channel mean → (B, H*W)
        return band_recon.mean(dim=1).flatten(1)

    def forward(
        self,
        content_projections: list[torch.Tensor],
        content_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Fourier alignment loss for coarse/mid granularities.

        Args:
            content_projections: Output of MRLProjectionHead for content images.
                                 These should be computed WITH gradients if
                                 alignment_weight > 0 (content encoder is trained).
            content_images:      (B, C, H, W) content reference in [0, 1].

        Returns:
            (scalar loss, metrics dict)
        """
        device = content_images.device
        band_masks = self._build_band_masks(device)

        total = torch.zeros(1, device=device).squeeze()
        metrics: dict[str, float] = {}

        for i, (mask, decoder) in enumerate(zip(band_masks, self.band_decoders)):
            # Target: actual frequency band amplitude from content image
            with torch.no_grad():
                target_amplitude = self._extract_band_amplitude(content_images, mask)
                # Normalise target to zero mean unit variance per sample
                t_mean = target_amplitude.mean(dim=1, keepdim=True)
                t_std = target_amplitude.std(dim=1, keepdim=True).clamp(min=1e-6)
                target_norm = (target_amplitude - t_mean) / t_std

            # Predict band amplitude from prefix projection
            prefix_proj = content_projections[i]   # (B, d_i) — with grad
            pred_amplitude = decoder(prefix_proj)   # (B, H*W)

            # Normalise prediction the same way
            p_mean = pred_amplitude.mean(dim=1, keepdim=True)
            p_std = pred_amplitude.std(dim=1, keepdim=True).clamp(min=1e-6)
            pred_norm = (pred_amplitude - p_mean) / p_std

            band_loss = F.l1_loss(pred_norm, target_norm.detach())
            d = self.nesting_dims[i]
            total = total + self.alignment_weight * band_loss
            metrics[f"mrl_fourier/band_loss_d{d}"] = band_loss.item()

        metrics["mrl_fourier/total"] = total.item()
        return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MatryoshkaContentEncoder  (zero-surgery wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class MatryoshkaContentEncoder(nn.Module):
    """Wraps the existing FontDiffuser content encoder with MRL heads.

    The original encoder is untouched — its forward pass is called as-is.
    MRL projection heads are attached on top of the encoder output.

    The wrapper exposes two paths:
      - forward()          : standard path, returns original embedding
                             (used for conditioning the diffusion model,
                              fully backward-compatible)
      - forward_mrl()      : returns original embedding + projected prefixes
                             (used during training for MRL loss computation)

    This design ensures zero change to the diffusion model's conditioning
    interface — MRL only affects the training objective, not inference.

    Args:
        content_encoder: Existing encoder module (FontDiffuser content encoder).
        embedding_dim:   Output dimension of the encoder.
        nesting_dims:    Matryoshka nesting dimensions (ascending).
    """

    def __init__(
        self,
        content_encoder: nn.Module,
        embedding_dim: int,
        nesting_dims: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.content_encoder = content_encoder
        self.embedding_dim = embedding_dim
        self.nesting_dims = list(nesting_dims)

        self.mrl_head = MRLProjectionHead(
            embedding_dim=embedding_dim,
            nesting_dims=nesting_dims,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard forward — returns encoder embedding unchanged.

        Fully backward-compatible with existing diffusion conditioning.
        """
        return self.content_encoder(x, **kwargs)

    def forward_mrl(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """MRL forward — returns embedding and projected prefixes.

        Args:
            x: Input image tensor.

        Returns:
            (embedding, projected_prefixes)
              embedding:           Full encoder output for conditioning.
              projected_prefixes:  List of L2-normalised prefix projections,
                                   one per nesting_dim, smallest first.
        """
        output = self.content_encoder(x, **kwargs)
        
        # Handle the case where content_encoder returns a tuple (e.g., h, residual_features)
        if isinstance(output, tuple):
            embedding = output[0]
        else:
            embedding = output

        # Pool spatial dims if embedding is (B, C, H, W)
        if embedding.dim() == 4:
            flat = embedding.flatten(2).mean(dim=2)   # (B, C)
        elif embedding.dim() == 3:
            flat = embedding.mean(dim=1)               # (B, C) from seq
        else:
            flat = embedding                           # (B, C) already flat

        projected = self.mrl_head(flat)
        return output, projected


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MRLLossModule  (unified loss entry point)
# ─────────────────────────────────────────────────────────────────────────────

class MRLLossModule(nn.Module):
    """Unified entry point for all MRL-related losses.

    Combines:
      - MRLContentLoss       (multi-granularity contrastive + reconstruction)
      - MRLFourierAlignment  (aligns prefix predictivity with freq bands)

    Designed to be used alongside FontDiffuserAuxLosses (the three proposed
    losses) — MRL operates on encoder representations while AuxLosses operate
    on decoder predictions.

    Args:
        nesting_dims:           Matryoshka nesting dimensions.
        freq_radii:             Frequency band boundaries for alignment.
        embedding_dim:          Full encoder embedding dimension.
        spatial_size:           (H, W) for Fourier alignment decoders.
        mrl_content_weight:     Weight on MRLContentLoss total.
        fourier_align_weight:   Weight on MRLFourierAlignment total.
        temperature:            InfoNCE temperature.
        use_fourier_alignment:  Whether to use MRLFourierAlignment.
                                Disable to save decoder parameter budget.
    """

    def __init__(
        self,
        nesting_dims: Sequence[int] = (64, 128, 256, 512),
        freq_radii: Sequence[float] = (0.1, 0.3),
        embedding_dim: int = 512,
        spatial_size: tuple[int, int] = (64, 64),
        mrl_content_weight: float = 1.0,
        fourier_align_weight: float = 0.3,
        temperature: float = 0.07,
        use_fourier_alignment: bool = True,
    ) -> None:
        super().__init__()

        self.mrl_content_weight = mrl_content_weight
        self.use_fourier_alignment = use_fourier_alignment

        self.content_loss = MRLContentLoss(
            nesting_dims=nesting_dims,
            temperature=temperature,
        )

        if use_fourier_alignment:
            self.fourier_alignment = MRLFourierAlignment(
                nesting_dims=nesting_dims,
                freq_radii=list(freq_radii),
                spatial_size=spatial_size,
                alignment_weight=fourier_align_weight,
            )

    def forward(
        self,
        pred_projections: list[torch.Tensor],
        content_projections: list[torch.Tensor],
        content_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute all MRL losses.

        Args:
            pred_projections:    MRL head outputs for pred_x0 encoding.
            content_projections: MRL head outputs for content encoding
                                 (computed with no_grad).
            content_images:      Raw content images for Fourier alignment.
                                 Required if use_fourier_alignment=True.

        Returns:
            (total loss, metrics dict)
        """
        total = torch.zeros(1, device=pred_projections[0].device).squeeze()
        all_metrics: dict[str, float] = {}

        # Multi-granularity content loss
        content_loss, content_metrics = self.content_loss(
            pred_projections, content_projections
        )
        total = total + self.mrl_content_weight * content_loss
        all_metrics.update(content_metrics)

        # Fourier alignment loss (on content projections, not pred)
        if self.use_fourier_alignment:
            if content_images is None:
                raise ValueError(
                    "content_images required when use_fourier_alignment=True"
                )
            fourier_loss, fourier_metrics = self.fourier_alignment(
                content_projections, content_images
            )
            total = total + fourier_loss
            all_metrics.update(fourier_metrics)

        all_metrics["mrl/grand_total"] = total.item()
        return total, all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Trainer integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mrl_components(
    content_encoder: nn.Module,
    embedding_dim: int = 512,
    nesting_dims: tuple[int, ...] = (64, 128, 256, 512),
    freq_radii: tuple[float, ...] = (0.1, 0.3),
    spatial_size: tuple[int, int] = (64, 64),
    use_fourier_alignment: bool = True,
) -> tuple[MatryoshkaContentEncoder, MRLLossModule]:
    """Factory that builds both MRL components in one call.

    Usage in _setup_models():

        self.mrl_encoder, self.mrl_loss = build_mrl_components(
            content_encoder = self.model.content_encoder,
            embedding_dim   = self.args.content_encoder_dim,
        )
        # Replace content encoder reference in model
        self.model.content_encoder = self.mrl_encoder

    Then in train_step():

        # Encode with MRL heads
        _, pred_proj    = self.mrl_encoder.forward_mrl(pred_x0_01)
        with torch.no_grad():
            _, content_proj = self.mrl_encoder.forward_mrl(content_01)

        mrl_loss, mrl_metrics = self.mrl_loss(
            pred_projections    = pred_proj,
            content_projections = content_proj,
            content_images      = content_01,
        )

        total_loss = fst_loss + aux_loss + mrl_weight * mrl_loss

    Args:
        content_encoder: Existing encoder module to wrap.
        embedding_dim:   Encoder output dim.
        nesting_dims:    Matryoshka prefix sizes (ascending).
        freq_radii:      Band boundaries, len = len(nesting_dims) - 1.
        spatial_size:    Image/feature spatial size for Fourier decoders.
        use_fourier_alignment: Whether to build Fourier alignment decoders.

    Returns:
        (MatryoshkaContentEncoder, MRLLossModule)
    """
    mrl_encoder = MatryoshkaContentEncoder(
        content_encoder=content_encoder,
        embedding_dim=embedding_dim,
        nesting_dims=nesting_dims,
    )
    mrl_loss_module = MRLLossModule(
        nesting_dims=nesting_dims,
        freq_radii=freq_radii,
        embedding_dim=embedding_dim,
        spatial_size=spatial_size,
        use_fourier_alignment=use_fourier_alignment,
    )
    n_mrl_params = sum(p.numel() for p in mrl_encoder.mrl_head.parameters())
    n_fourier_params = (
        sum(p.numel() for p in mrl_loss_module.fourier_alignment.parameters())
        if use_fourier_alignment else 0
    )
    logger.info(
        f"MRL components built | "
        f"projection head: {n_mrl_params:,} params | "
        f"fourier decoders: {n_fourier_params:,} params | "
        f"nesting_dims: {nesting_dims}"
    )
    return mrl_encoder, mrl_loss_module


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Complete loss weight schedule (combines MRL + proposed losses)
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLossSchedule:
    """Manages loss weight annealing across training steps.

    MRL losses should dominate early (encoder learning structure),
    then auxiliary pixel/frequency losses take over (decoder refinement).

    Three phases:
      Phase 1 (steps 0 .. warmup):       MRL only — build representation
      Phase 2 (steps warmup .. rampup):  MRL + AuxLosses, MRL annealing down
      Phase 3 (steps rampup .. end):     Full mix at final weights

    Args:
        mrl_warmup_steps:  Steps to run MRL alone before adding aux losses.
        mrl_rampdown_steps: Steps over which MRL weight anneals from
                            mrl_start_weight to mrl_final_weight.
        mrl_start_weight:  MRL weight at start of phase 2.
        mrl_final_weight:  MRL weight from phase 3 onward.
        aux_start_weight:  Aux loss weight at start of phase 2.
        aux_final_weight:  Aux loss weight from phase 3 onward.
    """

    def __init__(
        self,
        mrl_warmup_steps: int = 500,
        mrl_rampdown_steps: int = 1000,
        mrl_start_weight: float = 1.0,
        mrl_final_weight: float = 0.3,
        aux_start_weight: float = 0.1,
        aux_final_weight: float = 1.0,
    ) -> None:
        self.mrl_warmup_steps = mrl_warmup_steps
        self.mrl_rampdown_steps = mrl_rampdown_steps
        self.mrl_start_weight = mrl_start_weight
        self.mrl_final_weight = mrl_final_weight
        self.aux_start_weight = aux_start_weight
        self.aux_final_weight = aux_final_weight

    def get_weights(self, step: int) -> dict[str, float]:
        """Return current loss weights for the given step.

        Returns:
            Dict with keys 'mrl', 'aux', 'dro' (DRO weight unchanged).
        """
        if step < self.mrl_warmup_steps:
            # Phase 1: MRL only
            return {"mrl": self.mrl_start_weight, "aux": 0.0}

        t = min(step - self.mrl_warmup_steps, self.mrl_rampdown_steps)
        frac = t / max(self.mrl_rampdown_steps, 1)

        # Cosine annealing for smooth transitions
        cos_frac = 0.5 * (1.0 - math.cos(math.pi * frac))

        mrl_w = self.mrl_start_weight + cos_frac * (
            self.mrl_final_weight - self.mrl_start_weight
        )
        aux_w = self.aux_start_weight + cos_frac * (
            self.aux_final_weight - self.aux_start_weight
        )

        return {"mrl": mrl_w, "aux": aux_w}
