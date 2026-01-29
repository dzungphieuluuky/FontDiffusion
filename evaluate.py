import os
import sys
import json
import logging
import argparse
from pathlib import Path
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.tools.utilities import HFTqdm

# Optional dependencies
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Install with: pip install opencv-python")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FolderEvaluator:
    """Evaluates and compares two dataset folders."""

    def __init__(self, device: str = "cuda:0") -> None:
        """
        Initialize evaluator.

        Args:
            device: Device to use (cuda:0, cpu, etc.)
        """
        self.device: str = device

        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn: lpips.LPIPS = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

        self.transform_to_tensor: transforms.ToTensor = transforms.ToTensor()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images."""
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
            logger.warning(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images."""
        if not SSIM_AVAILABLE:
            return -1.0

        try:
            img1_gray: np.ndarray = np.array(img1.convert("L"))
            img2_gray: np.ndarray = np.array(img2.convert("L"))

            ssim_value: float = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            logger.warning(f"Error computing SSIM: {e}")
            return -1.0

    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images."""
        if not FID_AVAILABLE:
            return -1.0

        try:
            fid_value: float = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir], batch_size=50, device=self.device, dims=2048
            )
            return fid_value
        except Exception as e:
            logger.warning(f"Error computing FID: {e}")
            return -1.0

    def compute_histogram_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute histogram distance (Earth Mover's Distance)."""
        if not CV2_AVAILABLE:
            return -1.0

        try:
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            distance = 0.0
            for i in range(3):  # BGR channels
                hist1 = cv2.calcHist([img1_cv], [i], None, [256], [0, 256])
                hist2 = cv2.calcHist([img2_cv], [i], None, [256], [0, 256])

                # Normalize histograms
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()

                # Compute chi-square distance
                distance += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

            return distance / 3.0
        except Exception as e:
            logger.warning(f"Error computing histogram distance: {e}")
            return -1.0


class DatasetFolder:
    """Represents a dataset folder structure."""

    def __init__(self, root_dir: str) -> None:
        """
        Initialize dataset folder.

        Args:
            root_dir: Root directory containing ContentImage/ and TargetImage/
        """
        self.root_dir: Path = Path(root_dir)
        self.content_dir: Path = self.root_dir / "ContentImage"
        self.target_base_dir: Path = self.root_dir / "TargetImage"

        if not self.content_dir.exists():
            raise ValueError(f"ContentImage directory not found: {self.content_dir}")
        if not self.target_base_dir.exists():
            raise ValueError(f"TargetImage directory not found: {self.target_base_dir}")

        self.styles: list[str] = sorted(
            [d.name for d in self.target_base_dir.iterdir() if d.is_dir()]
        )
        self.content_images: dict[str, Path] = self._load_content_images()

    def _load_content_images(self) -> dict[str, Path]:
        """Load content image paths."""
        content_images: dict[str, Path] = {}
        for img_path in self.content_dir.glob("*.png"):
            content_images[img_path.stem] = img_path
        return content_images

    def get_target_image(self, style: str, filename: str) -> Path | None:
        """Get target image path."""
        target_path = self.target_base_dir / style / filename
        return target_path if target_path.exists() else None

    def get_all_target_images(self, style: str) -> list[tuple[str, Path]]:
        """Get all target images for a style."""
        style_dir = self.target_base_dir / style
        if not style_dir.exists():
            return []

        images: list[tuple[str, Path]] = []
        for img_path in style_dir.glob("*.png"):
            images.append((img_path.stem, img_path))
        return images


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Evaluate and compare two dataset folders"
    )

    parser.add_argument(
        "--folder1",
        type=str,
        required=True,
        help="First dataset folder path (e.g., handwritten_original/)",
    )
    parser.add_argument(
        "--folder2",
        type=str,
        required=True,
        help="Second dataset folder path (e.g., this_generations/)",
    )
    parser.add_argument(
        "--folder1_name",
        type=str,
        default="Folder1",
        help="Display name for folder1",
    )
    parser.add_argument(
        "--folder2_name",
        type=str,
        default="Folder2",
        help="Display name for folder2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        default=True,
        help="Compute LPIPS metric",
    )
    parser.add_argument(
        "--compute_ssim",
        action="store_true",
        default=True,
        help="Compute SSIM metric",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="Compute FID metric (slower)",
    )
    parser.add_argument(
        "--compute_histogram",
        action="store_true",
        default=True,
        help="Compute histogram distance",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default=None,
        help="Comma-separated list of styles to evaluate (None = all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save metric plots",
    )

    return parser.parse_args()


def evaluate_folders(
    folder1: DatasetFolder,
    folder2: DatasetFolder,
    evaluator: FolderEvaluator,
    folder1_name: str = "Folder1",
    folder2_name: str = "Folder2",
    styles: list[str] | None = None,
    compute_lpips: bool = True,
    compute_ssim: bool = True,
    compute_fid: bool = False,
    compute_histogram: bool = True,
) -> dict:
    """
    Evaluate two folders and compute comparison metrics.

    Returns:
        Dictionary with evaluation results
    """
    results: dict = {
        "folder1": str(folder1.root_dir),
        "folder2": str(folder2.root_dir),
        "folder1_name": folder1_name,
        "folder2_name": folder2_name,
        "metrics": {},
        "per_style_metrics": {},
        "per_image_metrics": [],
    }

    # Determine which styles to evaluate
    if styles is None:
        # Use intersection of styles
        styles_set1 = set(folder1.styles)
        styles_set2 = set(folder2.styles)
        common_styles = sorted(list(styles_set1 & styles_set2))
        logger.info(f"Found {len(common_styles)} common styles")
        logger.info(f"Folder1 unique: {styles_set1 - styles_set2}")
        logger.info(f"Folder2 unique: {styles_set2 - styles_set1}")
    else:
        common_styles = styles

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{'Evaluation Results':^60}")
    logger.info("=" * 60)
    logger.info(f"Folder1: {folder1_name} ({folder1.root_dir})")
    logger.info(f"Folder2: {folder2_name} ({folder2.root_dir})")
    logger.info(f"Styles to evaluate: {len(common_styles)}")
    logger.info(f"LPIPS: {compute_lpips}")
    logger.info(f"SSIM: {compute_ssim}")
    logger.info(f"FID: {compute_fid}")
    logger.info(f"Histogram: {compute_histogram}")
    logger.info("=" * 60)

    # Initialize metric accumulators
    all_lpips: list[float] = []
    all_ssim: list[float] = []
    all_histogram: list[float] = []

    # Evaluate per style
    for style in HFTqdm(common_styles, desc="🎨 Evaluating styles", colour="cyan"):
        logger.info(f"\nProcessing style: {style}")

        style_lpips: list[float] = []
        style_ssim: list[float] = []
        style_histogram: list[float] = []
        matched_count = 0
        missing_count = 0

        # Get all target images from folder1 for this style
        target_images_1 = folder1.get_all_target_images(style)

        for img_stem, img_path_1 in HFTqdm(
            target_images_1,
            desc=f"  📊 Comparing {style}",
            colour="green",
            leave=False,
        ):
            try:
                # Try to find matching image in folder2
                img_path_2 = folder2.get_target_image(style, f"{img_stem}.png")

                if img_path_2 is None:
                    missing_count += 1
                    logger.debug(f"    Missing in {folder2_name}: {img_stem}")
                    continue

                # Load images
                img1: Image.Image = Image.open(img_path_1).convert("RGB")
                img2: Image.Image = Image.open(img_path_2).convert("RGB")

                # Verify same size
                if img1.size != img2.size:
                    logger.warning(
                        f"    Size mismatch for {img_stem}: {img1.size} vs {img2.size}"
                    )
                    missing_count += 1
                    continue

                image_metrics: dict = {
                    "stem": img_stem,
                    "style": style,
                    "size": img1.size,
                }

                # Compute metrics
                if compute_lpips:
                    lpips_score: float = evaluator.compute_lpips(img1, img2)
                    if lpips_score >= 0:
                        style_lpips.append(lpips_score)
                        all_lpips.append(lpips_score)
                        image_metrics["lpips"] = lpips_score

                if compute_ssim:
                    ssim_score: float = evaluator.compute_ssim(img1, img2)
                    if ssim_score >= 0:
                        style_ssim.append(ssim_score)
                        all_ssim.append(ssim_score)
                        image_metrics["ssim"] = ssim_score

                if compute_histogram:
                    hist_score: float = evaluator.compute_histogram_distance(img1, img2)
                    if hist_score >= 0:
                        style_histogram.append(hist_score)
                        all_histogram.append(hist_score)
                        image_metrics["histogram_distance"] = hist_score

                results["per_image_metrics"].append(image_metrics)
                matched_count += 1

            except Exception as e:
                logger.error(f"    Error processing {img_stem}: {e}")
                missing_count += 1

        # Aggregate per-style metrics
        style_results: dict = {
            "matched": matched_count,
            "missing": missing_count,
        }

        if style_lpips:
            style_results["lpips"] = {
                "mean": float(np.mean(style_lpips)),
                "std": float(np.std(style_lpips)),
                "min": float(np.min(style_lpips)),
                "max": float(np.max(style_lpips)),
                "median": float(np.median(style_lpips)),
                "samples": len(style_lpips),
            }
            logger.info(
                f"  LPIPS: mean={style_results['lpips']['mean']:.4f}, "
                f"std={style_results['lpips']['std']:.4f}, "
                f"samples={len(style_lpips)}"
            )

        if style_ssim:
            style_results["ssim"] = {
                "mean": float(np.mean(style_ssim)),
                "std": float(np.std(style_ssim)),
                "min": float(np.min(style_ssim)),
                "max": float(np.max(style_ssim)),
                "median": float(np.median(style_ssim)),
                "samples": len(style_ssim),
            }
            logger.info(
                f"  SSIM: mean={style_results['ssim']['mean']:.4f}, "
                f"std={style_results['ssim']['std']:.4f}, "
                f"samples={len(style_ssim)}"
            )

        if style_histogram:
            style_results["histogram_distance"] = {
                "mean": float(np.mean(style_histogram)),
                "std": float(np.std(style_histogram)),
                "min": float(np.min(style_histogram)),
                "max": float(np.max(style_histogram)),
                "median": float(np.median(style_histogram)),
                "samples": len(style_histogram),
            }
            logger.info(
                f"  Histogram Distance: mean={style_results['histogram_distance']['mean']:.4f}, "
                f"std={style_results['histogram_distance']['std']:.4f}, "
                f"samples={len(style_histogram)}"
            )

        # Compute FID for this style if requested
        if compute_fid:
            style_dir_1 = folder1.target_base_dir / style
            style_dir_2 = folder2.target_base_dir / style

            if style_dir_1.exists() and style_dir_2.exists():
                logger.info(f"  Computing FID for {style}...")
                fid_value: float = evaluator.compute_fid(
                    str(style_dir_1), str(style_dir_2)
                )
                if fid_value >= 0:
                    style_results["fid"] = fid_value
                    logger.info(f"  FID: {fid_value:.4f}")

        results["per_style_metrics"][style] = style_results

    # Aggregate global metrics
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{'AGGREGATE METRICS':^60}")
    logger.info("=" * 60)

    if all_lpips:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(all_lpips)),
            "std": float(np.std(all_lpips)),
            "min": float(np.min(all_lpips)),
            "max": float(np.max(all_lpips)),
            "median": float(np.median(all_lpips)),
            "samples": len(all_lpips),
        }
        logger.info(f"LPIPS: mean={results['metrics']['lpips']['mean']:.4f}")

    if all_ssim:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(all_ssim)),
            "std": float(np.std(all_ssim)),
            "min": float(np.min(all_ssim)),
            "max": float(np.max(all_ssim)),
            "median": float(np.median(all_ssim)),
            "samples": len(all_ssim),
        }
        logger.info(f"SSIM: mean={results['metrics']['ssim']['mean']:.4f}")

    if all_histogram:
        results["metrics"]["histogram_distance"] = {
            "mean": float(np.mean(all_histogram)),
            "std": float(np.std(all_histogram)),
            "min": float(np.min(all_histogram)),
            "max": float(np.max(all_histogram)),
            "median": float(np.median(all_histogram)),
            "samples": len(all_histogram),
        }
        logger.info(
            f"Histogram Distance: mean={results['metrics']['histogram_distance']['mean']:.4f}"
        )

    logger.info("=" * 60)

    return results


def save_results(results: dict, output_dir: str) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to {output_path}")


def plot_results(results: dict, output_dir: str) -> None:
    """Plot evaluation metrics."""
    if not results["per_style_metrics"]:
        logger.warning("No per-style metrics to plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    styles = list(results["per_style_metrics"].keys())
    metrics_names = ["lpips", "ssim", "histogram_distance"]

    for metric_name in metrics_names:
        # Check if metric exists
        if metric_name not in results["per_style_metrics"][styles[0]]:
            continue

        metric_values = []
        for style in styles:
            if metric_name in results["per_style_metrics"][style]:
                metric_values.append(
                    results["per_style_metrics"][style][metric_name]["mean"]
                )
            else:
                metric_values.append(0)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(styles)), metric_values, color="blue", alpha=0.8)
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels(styles, rotation=45, ha="right")
        ax.set_ylabel(f"{metric_name.replace('_', ' ').title()} (mean)")
        ax.set_title(
            f"{metric_name.replace('_', ' ').title()} Comparison: "
            f"{results['folder1_name']} vs {results['folder2_name']}"
        )
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric_name}_per_style.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"✓ Saved plot: {plot_path}")

    # Overall metrics comparison
    if "lpips" in results["metrics"] and "ssim" in results["metrics"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # LPIPS
        ax1.bar(
            ["Mean", "Median"],
            [
                results["metrics"]["lpips"]["mean"],
                results["metrics"]["lpips"]["median"],
            ],
            color="blue",
            alpha=0.8,
        )
        ax1.set_ylabel("LPIPS")
        ax1.set_title("LPIPS Metrics")
        ax1.grid(axis="y", alpha=0.3)

        # SSIM
        ax2.bar(
            ["Mean", "Median"],
            [results["metrics"]["ssim"]["mean"], results["metrics"]["ssim"]["median"]],
            color="green",
            alpha=0.8,
        )
        ax2.set_ylabel("SSIM")
        ax2.set_title("SSIM Metrics")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "overall_metrics.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"✓ Saved plot: {plot_path}")


def main() -> None:
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Evaluation current model vs retrained model")
    logger.info("=" * 60)

    try:
        # Load folders
        logger.info("\n📂 Loading dataset folders...")
        folder1 = DatasetFolder(args.folder1)
        folder2 = DatasetFolder(args.folder2)

        logger.info(
            f"  Folder1: {len(folder1.content_images)} content images, {len(folder1.styles)} styles"
        )
        logger.info(
            f"  Folder2: {len(folder2.content_images)} content images, {len(folder2.styles)} styles"
        )

        # Initialize evaluator
        logger.info(f"\n🔧 Initializing evaluator on {args.device}...")
        evaluator = FolderEvaluator(device=args.device)

        # Parse styles if provided
        styles = None
        if args.styles:
            styles = [s.strip() for s in args.styles.split(",")]

        # Evaluate
        results = evaluate_folders(
            folder1,
            folder2,
            evaluator,
            folder1_name=args.folder1_name,
            folder2_name=args.folder2_name,
            styles=styles,
            compute_lpips=args.compute_lpips,
            compute_ssim=args.compute_ssim,
            compute_fid=args.compute_fid,
            compute_histogram=args.compute_histogram,
        )

        # Save results
        logger.info(f"\n💾 Saving results...")
        save_results(results, args.output_dir)

        # Plot results
        if args.save_plots:
            logger.info(f"📊 Generating plots...")
            plot_results(results, args.output_dir)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ Evaluation complete!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Example usage:
python evaluate_two_folders.py \\
    --folder1 handwritten_original/ \\
    --folder2 this_generations/ \\
    --folder1_name "Original" \\
    --folder2_name "Generated" \\
    --device cuda:0 \\
    --compute_fid \\
    --output_dir evaluation_results/

# Evaluate specific styles only
python evaluate_two_folders.py \\
    --folder1 handwritten_original/ \\
    --folder2 this_generations/ \\
    --styles "style0,style1,style2" \\
    --output_dir evaluation_results/
"""
