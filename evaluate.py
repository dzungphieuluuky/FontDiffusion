"""
Evaluate generated font images against content and style references.

Metrics:
  - SSIM  (Structural Similarity Index)      — content fidelity
  - LPIPS (Learned Perceptual Image Patch Similarity) — perceptual quality vs style
  - FID   (Fréchet Inception Distance)       — distributional realism

Expected folder layout
----------------------
FontDiffusion/
├── FST-FFT-test/
│   ├── ContentImage/
│   │   └── <char>.png / <char>.jpg
│   └── TargetImage/
│       └── <style_name>/
│           └── <style_name>+<char>.png
├── style_images/
│   └── cleaned_handwritten_enhanced/
│       └── <style_name>.png / <style_name>.jpg   <- matched by style_name
└── results_checkpoint.json                        <- preferred manifest (--checkpoint)

Manifest schema (results_checkpoint.json)
------------------------------------------
{
  "results": [
    {
      "content_char":  "A",
      "content_path":  "FST-FFT-test/ContentImage/A.png",
      "style_path":    "style_images/cleaned_handwritten_enhanced/backan.png",
      "result_path":   "FST-FFT-test/TargetImage/backan/backan+A.png"
    },
    ...
  ]
}

Usage
-----
# With manifest (preferred):
python evaluate.py \\
    --checkpoint results_checkpoint.json \\
    --content_dir FST-FFT-test/ContentImage \\
    --style_dir style_images/cleaned_handwritten_enhanced \\
    --results_dir FST-FFT-test/TargetImage \\
    --output_csv evaluation/metrics.csv

# Without manifest (filename-stem matching):
python evaluate.py \\
    --content_dir FST-FFT-test/ContentImage \\
    --style_dir style_images/cleaned_handwritten_enhanced \\
    --results_dir FST-FFT-test/TargetImage \\
    --output_csv evaluation/metrics.csv
"""

import argparse
import csv
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


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path, size: int) -> torch.Tensor:
    """Load an image as a normalised (0..1) RGB float tensor (C, H, W).

    Args:
        path: Path to the image file.
        size: Target spatial size (H = W).

    Returns:
        (3, size, size) float tensor in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return TF.to_tensor(img)


def find_images(directory: Path) -> dict[str, Path]:
    """Return {stem: path} for all image files directly inside *directory*.

    Args:
        directory: Directory to scan (non-recursive).

    Returns:
        Mapping of filename stem to absolute path.
    """
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


# ---------------------------------------------------------------------------
# SSIM (pure PyTorch)
# ---------------------------------------------------------------------------


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 1-D Gaussian kernel.

    Args:
        size: Kernel length.
        sigma: Standard deviation.

    Returns:
        (size,) normalised float tensor.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _create_2d_gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D separable Gaussian kernel shaped (1, 1, size, size).

    Args:
        size: Kernel spatial size.
        sigma: Standard deviation.

    Returns:
        (1, 1, size, size) float tensor.
    """
    k1d = _gaussian_kernel(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)


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

    SSIM is computed per channel with a Gaussian window, then averaged across
    batch and channels.

    Args:
        img1: (B, C, H, W) float tensor in [0, data_range].
        img2: (B, C, H, W) float tensor in [0, data_range].
        kernel_size: Size of the Gaussian window.
        sigma: Gaussian standard deviation.
        data_range: Value range of the inputs.
        k1: SSIM stability constant 1.
        k2: SSIM stability constant 2.

    Returns:
        Scalar tensor — mean SSIM (higher is better structural match).
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    device = img1.device
    kernel = _create_2d_gaussian_kernel(kernel_size, sigma).to(device)
    padding = kernel_size // 2

    _, C, _, _ = img1.shape
    ssim_vals: list[torch.Tensor] = []

    for c in range(C):
        ch1 = img1[:, c : c + 1]
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
# LPIPS — VGG16-based perceptual similarity (pure torchvision)
# ---------------------------------------------------------------------------


class VGGPerceptualFeatures(nn.Module):
    """Extract multi-scale VGG16 features for LPIPS computation.

    Uses relu1_2, relu2_2, relu3_3, relu4_3 — matching the layer selection
    in Zhang et al. (2018) without requiring the external ``lpips`` package.
    """

    _VGG_LAYERS: list[int] = [4, 9, 16, 23]

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = backbone.features

        self.slices = nn.ModuleList()
        prev = 0
        for end in self._VGG_LAYERS:
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return VGG feature maps at each selected layer.

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

    Features are L2-normalised per channel before computing squared L2
    distance, then spatially and scale averaged — matching the LPIPS
    formulation without the external ``lpips`` package.

    Args:
        img1: (B, 3, H, W) float tensor in [0, 1].
        img2: (B, 3, H, W) float tensor in [0, 1].
        vgg: Pre-built :class:`VGGPerceptualFeatures` module.

    Returns:
        Scalar tensor — mean LPIPS (lower is better perceptual match).
    """
    feats1 = vgg(img1)
    feats2 = vgg(img2)

    scale_distances: list[torch.Tensor] = []
    for f1, f2 in zip(feats1, feats2):
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)
        scale_distances.append(((f1_norm - f2_norm) ** 2).mean())

    return torch.stack(scale_distances).mean()


# ---------------------------------------------------------------------------
# FID — Fréchet Inception Distance (pure torchvision)
# ---------------------------------------------------------------------------


class InceptionFeatureExtractor(nn.Module):
    """Extract 2048-d pool features from Inception v3 for FID computation."""

    def __init__(self) -> None:
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.aux_logits = False
        self.model = model.eval()

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d features from (B, 3, 299, 299) input in [0, 1].

        Args:
            x: (B, 3, 299, 299) float tensor in [0, 1].

        Returns:
            (B, 2048) feature tensor.
        """
        return self.model(x * 2.0 - 1.0)


def _matrix_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute the matrix square root via eigendecomposition.

    More numerically stable than ``scipy.linalg.sqrtm`` for large matrices
    and avoids an external dependency.

    Args:
        mat: Symmetric positive semi-definite (N, N) matrix.
        eps: Clamp threshold for negative eigenvalues (numerical noise).

    Returns:
        (N, N) matrix square root.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _compute_fid_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute FID from pre-computed mean/covariance statistics.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))

    Args:
        mu1: Mean of real features (2048,).
        sigma1: Covariance of real features (2048, 2048).
        mu2: Mean of generated features (2048,).
        sigma2: Covariance of generated features (2048, 2048).

    Returns:
        FID scalar (lower is better).
    """
    diff_sq = float((mu1 - mu2) @ (mu1 - mu2))
    covmean = _matrix_sqrt(sigma1 @ sigma2)
    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return diff_sq + float(trace_term)


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
        batch_size: Number of images per forward pass.

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
        all_feats.append(inception(imgs).cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def compute_fid(
    real_paths: list[Path],
    generated_paths: list[Path],
    inception: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute FID between two sets of images.

    Args:
        real_paths: Paths to reference images (style or content).
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
# Manifest loading
# ---------------------------------------------------------------------------


def load_pairs_from_manifest(checkpoint_path: Path) -> list[dict[str, Path]]:
    """Load evaluation triplets from ``results_checkpoint.json``.

    Args:
        checkpoint_path: Path to the manifest JSON file.

    Returns:
        List of dicts with keys ``content_path``, ``style_path``,
        ``result_path``.  Entries with missing files are skipped with a
        warning.
    """
    with checkpoint_path.open() as fh:
        data = json.load(fh)

    records: list[dict[str, Path]] = []
    for entry in data.get("results", []):
        record: dict[str, Path] = {
            "content_path": Path(entry["content_path"]),
            "style_path": Path(entry["style_path"]),
            "result_path": Path(entry["result_path"]),
        }
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


# ---------------------------------------------------------------------------
# Stem-based pair matching (no manifest)
# ---------------------------------------------------------------------------


def match_pairs_by_stem(
    content_dir: Path,
    style_dir: Path,
    results_dir: Path,
) -> list[dict[str, Path]]:
    """Match result images to content and style images by filename convention.

    Convention
    ----------
    - Result:  ``<results_dir>/<style_name>/<style_name>+<char>.png``
    - Content: ``<content_dir>/<char>.png``
    - Style:   ``<style_dir>/<style_name>.png``

    For example, ``TargetImage/backan/backan+罐.png`` maps to:
    - content → ``ContentImage/罐.png``
    - style   → ``style_images/cleaned_handwritten_enhanced/backan.png``

    Args:
        content_dir: Directory containing flat content reference images.
        style_dir: Directory containing per-style reference images, one file
            per style named ``<style_name>.<ext>``.
        results_dir: Root directory whose immediate subdirectories are
            per-style result folders.

    Returns:
        List of dicts with keys ``content_path``, ``style_path``,
        ``result_path``, and ``style_name``.  Entries where either the
        content or style reference is missing are skipped with a warning.
    """
    content_map = find_images(content_dir)
    style_map = find_images(style_dir)

    style_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir()]
    if not style_dirs:
        logger.warning(
            "No style subdirectories found under %s. "
            "Expected layout: <results_dir>/<style_name>/<style_name>+<char>.png",
            results_dir,
        )

    pairs: list[dict[str, Path]] = []
    total_results = 0
    missing_styles: set[str] = set()

    for style_subdir in style_dirs:
        style_name = style_subdir.name

        # Resolve the style reference image once per style folder
        style_path = style_map.get(style_name)
        if style_path is None:
            if style_name not in missing_styles:
                logger.warning(
                    "No style reference image found for style '%s' in %s — "
                    "all results for this style will be skipped.",
                    style_name,
                    style_dir,
                )
                missing_styles.add(style_name)
            continue

        result_images = {
            p.stem: p
            for p in sorted(style_subdir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        total_results += len(result_images)

        for result_stem, result_path in result_images.items():
            if "+" not in result_stem:
                logger.debug(
                    "Skipping '%s' — does not match '<style>+<char>' convention.",
                    result_stem,
                )
                continue

            stem_style, content_key = result_stem.split("+", maxsplit=1)

            if stem_style != style_name:
                logger.debug(
                    "Style prefix mismatch: file '%s' is in folder '%s'.",
                    result_stem,
                    style_name,
                )

            content_path = content_map.get(content_key)
            if content_path is None:
                logger.debug(
                    "No content match for result '%s' (tried key '%s').",
                    result_stem,
                    content_key,
                )
                continue

            pairs.append(
                {
                    "content_path": content_path,
                    "style_path": style_path,
                    "result_path": result_path,
                    "style_name": style_name,
                }
            )

    logger.info(
        "Stem-matched %d / %d result images across %d style(s). "
        "%d style(s) skipped due to missing reference.",
        len(pairs),
        total_results,
        len(style_dirs),
        len(missing_styles),
    )
    return pairs


# ---------------------------------------------------------------------------
# Per-pair SSIM + LPIPS evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pairs(
    pairs: list[dict[str, Path]],
    vgg: VGGPerceptualFeatures,
    device: torch.device,
    eval_size: int,
) -> list[dict]:
    """Compute per-pair SSIM and LPIPS scores.

    - SSIM  is measured between ``result`` and ``content`` (structural fidelity).
    - LPIPS is measured between ``result`` and ``style``   (perceptual similarity).

    Args:
        pairs: List of path dicts with keys ``content_path``, ``style_path``,
               ``result_path``.
        vgg: Pre-built :class:`VGGPerceptualFeatures` module.
        device: Target device.
        eval_size: Spatial size to resize images to before comparison.

    Returns:
        List of per-pair result dicts with keys ``content_path``,
        ``style_path``, ``result_path``, ``style_name``, ``ssim``, ``lpips``.
    """
    records: list[dict] = []

    for pair in pairs:
        content_img = load_image_rgb(pair["content_path"], eval_size).to(device)
        style_img = load_image_rgb(pair["style_path"], eval_size).to(device)
        result_img = load_image_rgb(pair["result_path"], eval_size).to(device)

        content_b = content_img.unsqueeze(0)  # (1, 3, H, W)
        style_b = style_img.unsqueeze(0)
        result_b = result_img.unsqueeze(0)

        ssim_val = compute_ssim_batch(result_b, content_b).item()
        lpips_val = compute_lpips_batch(result_b, style_b, vgg).item()

        records.append(
            {
                "style_name": pair.get("style_name", ""),
                "content_path": str(pair["content_path"]),
                "style_path": str(pair["style_path"]),
                "result_path": str(pair["result_path"]),
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
        output_path: Destination CSV path (parent directories are created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "style_name",
        "content_path",
        "style_path",
        "result_path",
        "ssim",
        "lpips",
        "fid",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Per-pair metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Argument parsing
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
        default=Path("FST-FFT-test/ContentImage"),
        help="Directory containing flat content reference images.",
    )
    parser.add_argument(
        "--style_dir",
        type=Path,
        default=Path("style_images/cleaned_handwritten_enhanced"),
        help=(
            "Directory containing style reference images, one file per style "
            "named <style_name>.<ext> (e.g. backan.png)."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("FST-FFT-test/TargetImage"),
        help=(
            "Root directory of generated images. "
            "Expected layout: <results_dir>/<style_name>/<style_name>+<char>.png"
        ),
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
        pairs = match_pairs_by_stem(args.content_dir, args.style_dir, args.results_dir)

    if not pairs:
        logger.error("No evaluation pairs found. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Per-pair SSIM + LPIPS
    # ------------------------------------------------------------------
    logger.info("Loading VGG16 for LPIPS …")
    vgg = VGGPerceptualFeatures().to(device).eval()

    logger.info("Computing per-pair SSIM and LPIPS for %d pairs …", len(pairs))
    per_pair_records = evaluate_pairs(pairs, vgg, device, args.eval_size)

    ssim_scores = [r["ssim"] for r in per_pair_records]
    lpips_scores = [r["lpips"] for r in per_pair_records]

    logger.info(
        "SSIM  (result vs content) — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        np.mean(ssim_scores),
        np.std(ssim_scores),
        np.min(ssim_scores),
        np.max(ssim_scores),
    )
    logger.info(
        "LPIPS (result vs style)   — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        np.mean(lpips_scores),
        np.std(lpips_scores),
        np.min(lpips_scores),
        np.max(lpips_scores),
    )

    # ------------------------------------------------------------------
    # FID (result distribution vs style distribution)
    # ------------------------------------------------------------------
    fid_score: float | None = None
    if not args.skip_fid:
        if len(pairs) < 50:
            logger.warning(
                "Only %d samples available for FID. "
                "FID is unreliable below ~2 000 samples — treat result as indicative only.",
                len(pairs),
            )

        logger.info("Loading Inception v3 for FID …")
        inception = InceptionFeatureExtractor().to(device).eval()

        result_paths = [Path(r["result_path"]) for r in per_pair_records]
        style_paths = [Path(r["style_path"]) for r in per_pair_records]

        fid_score = compute_fid(
            style_paths, result_paths, inception, device, args.inception_batch_size
        )
        logger.info("FID (generated vs style): %.4f", fid_score)

        for row in per_pair_records:
            row["fid"] = fid_score

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Pairs evaluated : %d", len(per_pair_records))
    logger.info(
        "Mean SSIM       : %.4f  (↑ higher = better structural match vs content)",
        np.mean(ssim_scores),
    )
    logger.info(
        "Mean LPIPS      : %.4f  (↓ lower  = better perceptual match vs style)",
        np.mean(lpips_scores),
    )
    if fid_score is not None:
        logger.info(
            "FID             : %.4f  (↓ lower  = more realistic style distribution)",
            fid_score,
        )
    logger.info("=" * 60)

    write_csv(per_pair_records, args.output_csv)


if __name__ == "__main__":
    main()
