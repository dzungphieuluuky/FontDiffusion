"""
Evaluate generated font images against content and style references.

Metrics:
  - SSIM  (Structural Similarity Index)      — content fidelity
  - LPIPS (Learned Perceptual Image Patch Similarity) — perceptual quality vs style
  - FID   (Fréchet Inception Distance)       — distributional realism

Expected folder layout
----------------------
FontDiffusion/
├── ContentImage/
│   └── <char>.png / <char>.jpg
├── style_images/
│   └── cleaned_handwritten_enhanced/
│       └── *.png / *.jpg
├── results/                         <- --results_dir
│   └── *.png / *.jpg
└── results_checkpoint.json          <- preferred manifest (--checkpoint)

Manifest schema (results_checkpoint.json)
------------------------------------------
{
  "results": [
    {
      "content_char":  "A",
      "content_path":  "ContentImage/A.png",
      "style_path":    "style_images/cleaned_handwritten_enhanced/font_A.png",
      "result_path":   "results/abc123.png"
    },
    ...
  ]
}

If no manifest is provided, the script matches result files to content images
by the first character(s) of the result filename stem (e.g., "A_abc123.png" -> "A.png").

Usage
-----
# With manifest (preferred):
python evaluation/evaluate_generation.py \
    --checkpoint results_checkpoint.json \
    --content_dir ContentImage/ \
    --style_dir style_images/cleaned_handwritten_enhanced/ \
    --results_dir results/ \
    --output_csv evaluation/metrics.csv

# Without manifest (filename-stem matching):
python evaluation/evaluate_generation.py \
    --content_dir ContentImage/ \
    --style_dir style_images/cleaned_handwritten_enhanced/ \
    --results_dir results/ \
    --output_csv evaluation/metrics.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, Inception_V3_Weights

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate_generation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
INCEPTION_SIZE: int = 299
DEFAULT_EVAL_SIZE: int = 96  # matches FontDiffuser dm_size
LPIPS_FEATURE_SIZE: int = 64


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path, size: int) -> torch.Tensor:
    """Load an image as a normalised (0..1) RGB float tensor (C, H, W)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return TF.to_tensor(img)  # (3, H, W) in [0, 1]


def find_images(directory: Path) -> dict[str, Path]:
    """Return {stem: path} for all image files in *directory* (non-recursive)."""
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }


# ---------------------------------------------------------------------------
# SSIM (pure PyTorch, no external lib)
# ---------------------------------------------------------------------------


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 1-D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _create_2d_gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D separable Gaussian kernel (1, 1, size, size)."""
    k1d = _gaussian_kernel(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # (size, size)
    return k2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def compute_ssim_batch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute mean SSIM for a batch of image pairs.

    Args:
        img1: (B, C, H, W) float tensor in [0, data_range].
        img2: (B, C, H, W) float tensor in [0, data_range].
        kernel_size: Size of the Gaussian window.
        sigma: Gaussian standard deviation.
        data_range: Value range of the input images.
        k1: SSIM stability constant 1.
        k2: SSIM stability constant 2.

    Returns:
        Scalar tensor — mean SSIM across batch and channels.
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    device = img1.device
    kernel = _create_2d_gaussian_kernel(kernel_size, sigma).to(device)

    B, C, H, W = img1.shape
    padding = kernel_size // 2

    # Process each channel separately to keep memory low
    ssim_vals: list[torch.Tensor] = []
    for c in range(C):
        ch1 = img1[:, c : c + 1]  # (B, 1, H, W)
        ch2 = img2[:, c : c + 1]

        mu1 = F.conv2d(ch1, kernel, padding=padding)
        mu2 = F.conv2d(ch2, kernel, padding=padding)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(ch1 * ch1, kernel, padding=padding) - mu1_sq
        sigma2_sq = F.conv2d(ch2 * ch2, kernel, padding=padding) - mu2_sq
        sigma12 = F.conv2d(ch1 * ch2, kernel, padding=padding) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator.clamp(min=1e-8)
        ssim_vals.append(ssim_map.mean())

    return torch.stack(ssim_vals).mean()


# ---------------------------------------------------------------------------
# LPIPS — VGG-based perceptual loss (pure torchvision, no lpips package)
# ---------------------------------------------------------------------------


class VGGPerceptualFeatures(nn.Module):
    """Extract multi-scale VGG16 features for perceptual similarity.

    Uses relu1_2, relu2_2, relu3_3, relu4_3 — identical layer selection
    to the original LPIPS paper (Zhang et al., 2018).
    """

    _VGG_LAYERS: list[int] = [4, 9, 16, 23]  # relu indices in vgg16.features

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = backbone.features

        self.slices = nn.ModuleList()
        prev = 0
        for end in self._VGG_LAYERS:
            self.slices.append(nn.Sequential(*list(features.children())[prev : end + 1]))
            prev = end + 1

        # Freeze backbone — we only use features, no training
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of feature maps at each VGG scale.

        Args:
            x: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            List of (B, C_i, H_i, W_i) feature tensors.
        """
        x = (x - self.mean) / self.std
        feats: list[torch.Tensor] = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


def compute_lpips_batch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    vgg: VGGPerceptualFeatures,
) -> torch.Tensor:
    """Compute mean LPIPS distance for a batch of image pairs.

    Each scale's feature maps are L2-normalised per channel before
    computing squared L2 distance, then spatially averaged — matching
    the LPIPS formulation without requiring the external ``lpips`` package.

    Args:
        img1: (B, 3, H, W) float tensor in [0, 1].
        img2: (B, 3, H, W) float tensor in [0, 1].
        vgg:  Pre-built :class:`VGGPerceptualFeatures` module.

    Returns:
        Scalar tensor — mean LPIPS across batch and scales.
    """
    feats1 = vgg(img1)
    feats2 = vgg(img2)

    scale_distances: list[torch.Tensor] = []
    for f1, f2 in zip(feats1, feats2):
        # L2 normalise along channel dim
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)
        diff = (f1_norm - f2_norm) ** 2  # (B, C, H, W)
        scale_distances.append(diff.mean())

    return torch.stack(scale_distances).mean()


# ---------------------------------------------------------------------------
# FID — Fréchet Inception Distance (pure torchvision)
# ---------------------------------------------------------------------------


class InceptionFeatureExtractor(nn.Module):
    """Extract 2048-d pool features from Inception v3 for FID computation."""

    def __init__(self) -> None:
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()  # replace final classifier
        model.aux_logits = False
        self.model = model.eval()

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d features.

        Args:
            x: (B, 3, 299, 299) float tensor in [0, 1].

        Returns:
            (B, 2048) feature tensor.
        """
        # Inception expects [-1, 1]
        x = x * 2.0 - 1.0
        return self.model(x)


def _compute_fid_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute FID from pre-computed mean/covariance statistics.

    Uses the closed-form formula:
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))

    Args:
        mu1: Mean of real features (2048,).
        sigma1: Covariance of real features (2048, 2048).
        mu2: Mean of generated features (2048,).
        sigma2: Covariance of generated features (2048, 2048).
        eps: Regularisation term for numerical stability.

    Returns:
        FID scalar.
    """
    diff = mu1 - mu2
    diff_sq = diff @ diff

    # Matrix square root via eigendecomposition (more stable than scipy sqrtm
    # for large matrices and avoids an external dependency)
    covmean = _matrix_sqrt(sigma1 @ sigma2, eps=eps)
    
    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(diff_sq + trace_term)


def _matrix_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute matrix square root via eigendecomposition.

    Args:
        mat: Symmetric positive semi-definite (N, N) matrix.
        eps: Clamp threshold for negative eigenvalues (numerical noise).

    Returns:
        (N, N) matrix square root.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


@torch.no_grad()
def collect_inception_features(
    paths: list[Path],
    inception: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run images through Inception v3 and return all extracted features.

    Args:
        paths: List of image file paths.
        inception: Pre-built :class:`InceptionFeatureExtractor`.
        device: Target device.
        batch_size: Number of images per Inception forward pass.

    Returns:
        (N, 2048) float32 numpy array.
    """
    resize = T.Resize((INCEPTION_SIZE, INCEPTION_SIZE), antialias=True)
    all_feats: list[np.ndarray] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        imgs = torch.stack(
            [resize(load_image_rgb(p, INCEPTION_SIZE)) for p in batch_paths]
        ).to(device)
        feats = inception(imgs).cpu().numpy()  # (B, 2048)
        all_feats.append(feats)

    return np.concatenate(all_feats, axis=0)  # (N, 2048)


def compute_fid(
    real_paths: list[Path],
    generated_paths: list[Path],
    inception: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute FID between two sets of images.

    Args:
        real_paths: Paths to reference (style) images.
        generated_paths: Paths to generated images.
        inception: Pre-built :class:`InceptionFeatureExtractor`.
        device: Target device.
        batch_size: Inception batch size.

    Returns:
        FID score (lower is better).
    """
    logger.info("Collecting Inception features for real images …")
    real_feats = collect_inception_features(real_paths, inception, device, batch_size)

    logger.info("Collecting Inception features for generated images …")
    gen_feats = collect_inception_features(
        generated_paths, inception, device, batch_size
    )

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    return _compute_fid_from_stats(mu_r, sigma_r, mu_g, sigma_g)


# ---------------------------------------------------------------------------
# Manifest loading and pair matching
# ---------------------------------------------------------------------------


def load_pairs_from_manifest(
    checkpoint_path: Path,
) -> list[dict[str, Path]]:
    """Load evaluation triplets from ``results_checkpoint.json``.

    Args:
        checkpoint_path: Path to the manifest JSON file.

    Returns:
        List of dicts with keys ``content_path``, ``style_path``, ``result_path``.
    """
    with checkpoint_path.open() as fh:
        data = json.load(fh)

    records: list[dict[str, Path]] = []
    for entry in data.get("results", []):
        record = {
            "content_path": Path(entry["content_path"]),
            "style_path": Path(entry["style_path"]),
            "result_path": Path(entry["result_path"]),
        }
        # Validate all paths exist before adding
        missing = [k for k, v in record.items() if not v.exists()]
        if missing:
            logger.warning(
                "Skipping entry — missing files: %s",
                {k: str(record[k]) for k in missing},
            )
            continue
        records.append(record)

    logger.info("Loaded %d valid pairs from manifest.", len(records))
    return records


def match_pairs_by_stem(
    content_dir: Path,
    results_dir: Path,
) -> list[dict[str, Path]]:
    """Match result images to content images by filename stem convention.

    Convention: result file ``<style_name>+<char>.png`` maps to content
    ``<char>.png``, where ``+`` separates the style identifier from the
    content character.

    Args:
        content_dir: Directory of content reference images.
        results_dir: Directory of generated result images.

    Returns:
        List of dicts with keys ``content_path`` and ``result_path``.
        ``style_path`` is absent — caller must handle this case.
    """
    content_map = find_images(content_dir)
    results_map = find_images(results_dir)

    pairs: list[dict[str, Path]] = []
    for result_stem, result_path in results_map.items():
        if "+" not in result_stem:
            logger.debug(
                "Skipping result '%s' — does not match '<style>+<char>' convention.",
                result_stem,
            )
            continue

        # e.g. "handwritten+A" -> content key "A"
        content_key = result_stem.split("+", maxsplit=1)[1]

        if content_key in content_map:
            pairs.append(
                {
                    "content_path": content_map[content_key],
                    "result_path": result_path,
                }
            )
        else:
            logger.debug(
                "No content match for result '%s' (tried key '%s').",
                result_stem,
                content_key,
            )

    logger.info(
        "Stem-matched %d / %d result images to content images.",
        len(pairs),
        len(results_map),
    )
    return pairs


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pairs(
    pairs: list[dict[str, Path]],
    vgg: VGGPerceptualFeatures,
    device: torch.device,
    eval_size: int,
) -> list[dict]:
    """Compute per-pair SSIM and LPIPS scores.

    SSIM is computed between result and *content* image (structural fidelity).
    LPIPS is computed between result and *style* image (perceptual similarity),
    falling back to content if style is unavailable.

    Args:
        pairs: List of path dicts (keys: ``content_path``, ``result_path``,
               optional ``style_path``).
        vgg: Pre-built :class:`VGGPerceptualFeatures` module.
        device: Target device.
        eval_size: Spatial size to resize images to before comparison.

    Returns:
        List of per-pair result dicts with keys ``ssim``, ``lpips``,
        ``content_path``, ``result_path``.
    """
    records: list[dict] = []

    for pair in pairs:
        content_img = load_image_rgb(pair["content_path"], eval_size).to(device)
        result_img = load_image_rgb(pair["result_path"], eval_size).to(device)

        # Add batch dim: (1, 3, H, W)
        content_b = content_img.unsqueeze(0)
        result_b = result_img.unsqueeze(0)

        ssim_val = compute_ssim_batch(result_b, content_b).item()

        # LPIPS: against style if available, else against content
        if "style_path" in pair:
            style_img = load_image_rgb(pair["style_path"], eval_size).to(device)
            lpips_ref = style_img.unsqueeze(0)
        else:
            lpips_ref = content_b

        lpips_val = compute_lpips_batch(result_b, lpips_ref, vgg).item()

        records.append(
            {
                "content_path": str(pair["content_path"]),
                "result_path": str(pair["result_path"]),
                "style_path": str(pair.get("style_path", "")),
                "ssim": ssim_val,
                "lpips": lpips_val,
            }
        )

    return records


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write evaluation rows to a CSV file.

    Args:
        rows: List of per-pair metric dicts.
        output_path: Destination CSV path.
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["content_path", "style_path", "result_path", "ssim", "lpips"]

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Per-pair metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Argument parsing (extends codebase shared parser)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated font images with SSIM, LPIPS and FID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content_dir",
        type=Path,
        default=Path("ContentImage"),
        help="Directory containing content reference images.",
    )
    parser.add_argument(
        "--style_dir",
        type=Path,
        default=Path("style_images/cleaned_handwritten_enhanced"),
        help="Directory containing style reference images.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("TargetImage"),
        help="Directory containing generated images to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to results_checkpoint.json manifest. "
            "If omitted, pairs are matched by filename stem."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("evaluation/metrics.csv"),
        help="Destination CSV path for per-pair metrics.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help="Spatial size (H=W) to resize images to before SSIM/LPIPS.",
    )
    parser.add_argument(
        "--inception_batch_size",
        type=int,
        default=32,
        help="Batch size for Inception v3 feature extraction (FID).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation (faster; useful when result set is small).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Validate input directories
    # ------------------------------------------------------------------
    for label, path in [
        ("content_dir", args.content_dir),
        ("style_dir", args.style_dir),
        ("results_dir", args.results_dir),
    ]:
        if not path.exists():
            logger.error("Required directory not found: %s = %s", label, path)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build evaluation pairs
    # ------------------------------------------------------------------
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            logger.error("Manifest not found: %s", args.checkpoint)
            sys.exit(1)
        pairs = load_pairs_from_manifest(args.checkpoint)
    else:
        logger.info(
            "No manifest provided — matching pairs by filename stem in %s",
            args.results_dir,
        )
        pairs = match_pairs_by_stem(args.content_dir, args.results_dir)

    if not pairs:
        logger.error("No evaluation pairs found. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load perceptual models (lazy, on-demand)
    # ------------------------------------------------------------------
    logger.info("Loading VGG16 for LPIPS …")
    vgg = VGGPerceptualFeatures().to(device).eval()

    # ------------------------------------------------------------------
    # Per-pair SSIM + LPIPS
    # ------------------------------------------------------------------
    logger.info("Computing per-pair SSIM and LPIPS for %d pairs …", len(pairs))
    per_pair_records = evaluate_pairs(pairs, vgg, device, args.eval_size)

    ssim_scores = [r["ssim"] for r in per_pair_records]
    lpips_scores = [r["lpips"] for r in per_pair_records]

    logger.info(
        "SSIM  — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        np.mean(ssim_scores),
        np.std(ssim_scores),
        np.min(ssim_scores),
        np.max(ssim_scores),
    )
    logger.info(
        "LPIPS — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        np.mean(lpips_scores),
        np.std(lpips_scores),
        np.min(lpips_scores),
        np.max(lpips_scores),
    )

    # ------------------------------------------------------------------
    # FID (distributional metric — needs enough samples to be meaningful)
    # ------------------------------------------------------------------
    fid_score: float | None = None
    if not args.skip_fid:
        if len(pairs) < 50:
            logger.warning(
                "Only %d samples available for FID. "
                "FID is unreliable below ~2 000 samples; "
                "treat the result as indicative only.",
                len(pairs),
            )

        logger.info("Loading Inception v3 for FID …")
        inception = InceptionFeatureExtractor().to(device).eval()

        result_paths = [Path(r["result_path"]) for r in per_pair_records]
        style_paths = [
            Path(r["style_path"]) for r in per_pair_records if r["style_path"]
        ]

        if style_paths:
            fid_score = compute_fid(
                style_paths, result_paths, inception, device, args.inception_batch_size
            )
            logger.info("FID (generated vs style): %.4f", fid_score)
        else:
            content_paths = [Path(r["content_path"]) for r in per_pair_records]
            fid_score = compute_fid(
                content_paths, result_paths, inception, device, args.inception_batch_size
            )
            logger.info("FID (generated vs content): %.4f", fid_score)

        # Append FID to each row so it appears in the CSV
        for row in per_pair_records:
            row["fid"] = fid_score

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Pairs evaluated : %d", len(per_pair_records))
    logger.info("Mean SSIM       : %.4f  (↑ higher = better structural match)", np.mean(ssim_scores))
    logger.info("Mean LPIPS      : %.4f  (↓ lower  = better perceptual match)", np.mean(lpips_scores))
    if fid_score is not None:
        logger.info("FID             : %.4f  (↓ lower  = more realistic distribution)", fid_score)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    write_csv(per_pair_records, args.output_csv)


if __name__ == "__main__":
    main()