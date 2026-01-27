"""
Enhanced evaluation and comparison of generated images from two dataset folders.

Supports comprehensive metrics, statistical analysis, and rich visualizations.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

mp.set_start_method("spawn", force=True)  # <-- Add this line
_METRICS_COMPUTER = None
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

from src.tools.utilities import HFTqdm

# Optional dependencies with graceful fallback
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
    from skimage.metrics import peak_signal_noise_ratio as psnr

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Install with: pip install opencv-python")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available for dimensionality reduction")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available for interactive plots")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedEvaluator")

# Set style for matplotlib
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


class MetricsComputer:
    """Computes various image quality and similarity metrics."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device

        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

        # Initialize feature extractor for perceptual metrics
        if torch.cuda.is_available():
            self.feature_extractor = self._init_feature_extractor()
        else:
            self.feature_extractor = None

        self.transform_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _init_feature_extractor(self):
        """Initialize a pretrained model for feature extraction."""
        try:
            from torchvision.models import vgg16, VGG16_Weights

            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.features.children())[:23])
            model = model.to(self.device)
            model.eval()
            return model
        except:
            return None

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS perceptual distance."""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0
        try:
            img1_tensor = (
                self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
            )
            img2_tensor = (
                self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
            )
            with torch.inference_mode():
                lpips_value = self.lpips_fn(img1_tensor, img2_tensor).item()
            return lpips_value
        except Exception as e:
            logger.warning(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM."""
        if not SKIMAGE_AVAILABLE:
            return -1.0
        try:
            img1_gray = np.array(img1.convert("L"))
            img2_gray = np.array(img2.convert("L"))
            ssim_value = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            logger.warning(f"Error computing SSIM: {e}")
            return -1.0

    def compute_psnr(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute PSNR."""
        if not SKIMAGE_AVAILABLE:
            return -1.0
        try:
            img1_arr = np.array(img1)
            img2_arr = np.array(img2)
            psnr_value = psnr(img1_arr, img2_arr, data_range=255)
            return psnr_value
        except Exception as e:
            logger.warning(f"Error computing PSNR: {e}")
            return -1.0

    def compute_mse(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute Mean Squared Error."""
        try:
            img1_arr = np.array(img1).astype(np.float32)
            img2_arr = np.array(img2).astype(np.float32)
            mse = np.mean((img1_arr - img2_arr) ** 2)
            return float(mse)
        except Exception as e:
            logger.warning(f"Error computing MSE: {e}")
            return -1.0

    def compute_histogram_distance(
        self, img1: Image.Image, img2: Image.Image
    ) -> dict[str, float]:
        """Compute various histogram-based distances."""
        if not CV2_AVAILABLE:
            return {}
        try:
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            results = {}
            methods = {
                "chisqr": cv2.HISTCMP_CHISQR,
                "correlation": cv2.HISTCMP_CORREL,
                "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
            }

            for method_name, method in methods.items():
                distance = 0.0
                for i in range(3):
                    hist1 = cv2.calcHist([img1_cv], [i], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img2_cv], [i], None, [256], [0, 256])
                    hist1 = cv2.normalize(hist1, hist1).flatten()
                    hist2 = cv2.normalize(hist2, hist2).flatten()
                    distance += cv2.compareHist(hist1, hist2, method)
                results[method_name] = distance / 3.0

            return results
        except Exception as e:
            logger.warning(f"Error computing histogram distance: {e}")
            return {}

    def compute_color_statistics(
        self, img1: Image.Image, img2: Image.Image
    ) -> dict[str, float]:
        """Compute color distribution statistics."""
        try:
            img1_arr = np.array(img1).astype(np.float32)
            img2_arr = np.array(img2).astype(np.float32)

            results = {}
            for i, channel in enumerate(["R", "G", "B"]):
                mean_diff = abs(np.mean(img1_arr[:, :, i]) - np.mean(img2_arr[:, :, i]))
                std_diff = abs(np.std(img1_arr[:, :, i]) - np.std(img2_arr[:, :, i]))
                results[f"{channel}_mean_diff"] = float(mean_diff)
                results[f"{channel}_std_diff"] = float(std_diff)

            return results
        except Exception as e:
            logger.warning(f"Error computing color statistics: {e}")
            return {}

    def compute_edge_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute edge-based similarity using Canny edge detection."""
        if not CV2_AVAILABLE:
            return -1.0
        try:
            img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

            edges1 = cv2.Canny(img1_gray, 100, 200)
            edges2 = cv2.Canny(img2_gray, 100, 200)

            # Compute Jaccard similarity
            intersection = np.logical_and(edges1, edges2).sum()
            union = np.logical_or(edges1, edges2).sum()

            if union == 0:
                return 0.0

            jaccard = intersection / union
            return float(jaccard)
        except Exception as e:
            logger.warning(f"Error computing edge similarity: {e}")
            return -1.0

    def extract_features(self, img: Image.Image) -> Optional[np.ndarray]:
        """Extract deep features from image."""
        if self.feature_extractor is None:
            return None
        try:
            img_tensor = self.transform_to_tensor(img).unsqueeze(0).to(self.device)
            img_tensor = self.normalize(img_tensor)

            with torch.inference_mode():
                features = self.feature_extractor(img_tensor)

            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return None

    def compute_all_metrics(self, img1: Image.Image, img2: Image.Image) -> dict:
        """Compute all available metrics."""
        metrics = {}

        # Perceptual metrics
        metrics["lpips"] = self.compute_lpips(img1, img2)
        metrics["ssim"] = self.compute_ssim(img1, img2)
        metrics["psnr"] = self.compute_psnr(img1, img2)
        metrics["mse"] = self.compute_mse(img1, img2)

        # Histogram metrics
        hist_metrics = self.compute_histogram_distance(img1, img2)
        metrics.update(hist_metrics)

        # Color statistics
        color_stats = self.compute_color_statistics(img1, img2)
        metrics.update(color_stats)

        # Edge similarity
        metrics["edge_similarity"] = self.compute_edge_similarity(img1, img2)

        return metrics


class DatasetFolder:
    """Represents a dataset folder structure."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.content_dir = self.root_dir / "ContentImage"
        self.target_base_dir = self.root_dir / "TargetImage"

        if not self.content_dir.exists():
            raise ValueError(f"ContentImage directory not found: {self.content_dir}")
        if not self.target_base_dir.exists():
            raise ValueError(f"TargetImage directory not found: {self.target_base_dir}")

        self.styles = sorted(
            [d.name for d in self.target_base_dir.iterdir() if d.is_dir()]
        )
        self.content_images = self._load_content_images()

    def _load_content_images(self) -> dict[str, Path]:
        content_images = {}
        for img_path in self.content_dir.glob("*.png"):
            content_images[img_path.stem] = img_path
        return content_images

    def get_target_image(self, style: str, filename: str) -> Optional[Path]:
        target_path = self.target_base_dir / style / filename
        return target_path if target_path.exists() else None

    def get_all_target_images(self, style: str) -> list[tuple[str, Path]]:
        style_dir = self.target_base_dir / style
        if not style_dir.exists():
            return []
        images = [(img_path.stem, img_path) for img_path in style_dir.glob("*.png")]
        return images


def _init_metrics_computer(device):
    global _METRICS_COMPUTER
    _METRICS_COMPUTER = MetricsComputer(device=device)


def compute_image_pair_metrics(args_tuple):
    img_path_1, img_path_2, img_stem, style, device = args_tuple
    try:
        img1 = Image.open(img_path_1).convert("RGB")
        img2 = Image.open(img_path_2).convert("RGB")
        if img1.size != img2.size:
            return None
        # Use global instance
        global _METRICS_COMPUTER
        computer = _METRICS_COMPUTER
        metrics = computer.compute_all_metrics(img1, img2)
        features1 = computer.extract_features(img1)
        features2 = computer.extract_features(img2)
        result = {
            "stem": img_stem,
            "style": style,
            "size": img1.size,
            "metrics": metrics,
            "features1": features1,
            "features2": features2,
        }
        return result
    except Exception as e:
        logger.error(f"Error processing {img_stem}: {e}")
        return None


class StatisticalAnalyzer:
    """Performs statistical analysis on metrics."""

    @staticmethod
    def compute_summary_stats(values: list[float]) -> dict:
        """Compute comprehensive summary statistics."""
        if not values:
            return {}

        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
            "samples": len(values),
        }

    @staticmethod
    def compare_distributions(values1: list[float], values2: list[float]) -> dict:
        """Compare two distributions statistically."""
        if not values1 or not values2:
            return {}

        arr1 = np.array(values1)
        arr2 = np.array(values2)

        results = {}

        # t-test
        try:
            t_stat, p_value = stats.ttest_ind(arr1, arr2)
            results["t_test"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        except:
            pass

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_value = stats.mannwhitneyu(arr1, arr2)
            results["mann_whitney"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        except:
            pass

        # Cohen's d (effect size)
        try:
            pooled_std = np.sqrt((np.std(arr1) ** 2 + np.std(arr2) ** 2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std
                results["cohens_d"] = float(cohens_d)
        except:
            pass

        # Kolmogorov-Smirnov test
        try:
            ks_stat, p_value = stats.ks_2samp(arr1, arr2)
            results["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        except:
            pass

        return results


class Visualizer:
    """Creates comprehensive visualizations."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric_distributions(self, results: dict, metric_name: str):
        """Plot distribution comparison for a metric."""
        styles = list(results["per_style_metrics"].keys())

        if not styles:
            return

        # Check if metric exists
        if metric_name not in results["per_style_metrics"][styles[0]].get(
            "metrics", {}
        ):
            return

        # Extract values
        means = []
        stds = []
        for style in styles:
            style_data = results["per_style_metrics"][style]["metrics"].get(
                metric_name, {}
            )
            if style_data:
                means.append(style_data.get("mean", 0))
                stds.append(style_data.get("std", 0))
            else:
                means.append(0)
                stds.append(0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Bar plot with error bars
        x = np.arange(len(styles))
        ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(styles, rotation=45, ha="right")
        ax1.set_ylabel(f"{metric_name.replace('_', ' ').title()}")
        ax1.set_title(f"{metric_name.replace('_', ' ').title()} by Style (Mean ± Std)")
        ax1.grid(axis="y", alpha=0.3)

        # Box plot
        all_values = []
        labels = []
        for style in styles:
            if "per_image_metrics" in results:
                style_values = [
                    m["metrics"].get(metric_name, np.nan)
                    for m in results["per_image_metrics"]
                    if m["style"] == style and metric_name in m.get("metrics", {})
                ]
                if style_values:
                    all_values.append(style_values)
                    labels.append(style)

        if all_values:
            bp = ax2.boxplot(all_values, labels=labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
            ax2.set_xticklabels(labels, rotation=45, ha="right")
            ax2.set_ylabel(f"{metric_name.replace('_', ' ').title()}")
            ax2.set_title(
                f"{metric_name.replace('_', ' ').title()} Distribution by Style"
            )
            ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{metric_name}_distribution.png")
        plt.close()

    def plot_violin_comparison(self, results: dict, metrics: list[str]):
        """Create violin plots for multiple metrics."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            data = []
            for m in results.get("per_image_metrics", []):
                if metric in m.get("metrics", {}):
                    val = m["metrics"][metric]
                    if val >= 0:  # Filter out error values
                        data.append(val)

            if data:
                parts = ax.violinplot(
                    [data], positions=[0], showmeans=True, showmedians=True
                )
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
                ax.set_xticks([])
                ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "violin_plots.png")
        plt.close()

    def plot_correlation_heatmap(self, results: dict, metrics: list[str]):
        """Plot correlation heatmap between metrics."""
        # Collect data
        data_dict = {metric: [] for metric in metrics}

        for m in results.get("per_image_metrics", []):
            for metric in metrics:
                if metric in m.get("metrics", {}):
                    val = m["metrics"][metric]
                    if val >= 0:
                        data_dict[metric].append(val)

        # Find minimum length
        min_len = min(len(v) for v in data_dict.values() if v)
        if min_len == 0:
            return

        # Trim to same length
        for k in data_dict:
            data_dict[k] = data_dict[k][:min_len]

        # Create correlation matrix
        data_array = np.array([data_dict[m] for m in metrics])
        corr_matrix = np.corrcoef(data_array)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Labels
        metric_labels = [m.replace("_", " ").title() for m in metrics]
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(metric_labels, rotation=45, ha="right")
        ax.set_yticklabels(metric_labels)

        # Annotate
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

        ax.set_title("Metric Correlation Heatmap")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png")
        plt.close()

    def plot_radar_chart(self, results: dict, styles: list[str], metrics: list[str]):
        """Create radar chart comparing styles across metrics."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for radar charts")
            return

        # Normalize metrics to 0-1 scale for comparison
        metric_ranges = {}
        for metric in metrics:
            all_vals = []
            for style in styles:
                style_data = (
                    results["per_style_metrics"].get(style, {}).get("metrics", {})
                )
                if metric in style_data:
                    all_vals.append(style_data[metric].get("mean", 0))
            if all_vals:
                metric_ranges[metric] = (min(all_vals), max(all_vals))

        fig = go.Figure()

        for style in styles[:5]:  # Limit to 5 styles for readability
            style_data = results["per_style_metrics"].get(style, {}).get("metrics", {})
            r_values = []

            for metric in metrics:
                if metric in style_data and metric in metric_ranges:
                    val = style_data[metric].get("mean", 0)
                    min_val, max_val = metric_ranges[metric]
                    if max_val > min_val:
                        normalized = (val - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5
                    r_values.append(normalized)
                else:
                    r_values.append(0)

            fig.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=[m.replace("_", " ").title() for m in metrics],
                    fill="toself",
                    name=style,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Style Comparison Radar Chart (Normalized Metrics)",
        )

        fig.write_html(str(self.output_dir / "radar_chart.html"))

    def plot_tsne_embeddings(self, results: dict):
        """Plot t-SNE visualization of image embeddings."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for t-SNE")
            return

        # Collect features and labels
        features1 = []
        features2 = []
        styles = []

        for m in results.get("per_image_metrics", []):
            if m.get("features1") is not None and m.get("features2") is not None:
                features1.append(m["features1"])
                features2.append(m["features2"])
                styles.append(m["style"])

        if not features1:
            return

        # Combine features
        all_features = np.vstack([features1, features2])
        all_labels = styles + styles
        all_sources = ["Folder1"] * len(features1) + ["Folder2"] * len(features2)

        # Apply t-SNE
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1)
        )
        embeddings = tsne.fit_transform(all_features)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_styles = sorted(set(styles))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_styles)))
        style_to_color = {style: colors[i] for i, style in enumerate(unique_styles)}

        for i, (emb, style, source) in enumerate(
            zip(embeddings, all_labels, all_sources)
        ):
            marker = "o" if source == "Folder1" else "^"
            ax.scatter(
                emb[0],
                emb[1],
                c=[style_to_color[style]],
                marker=marker,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = []
        for style in unique_styles:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=style_to_color[style],
                    markersize=8,
                    label=style,
                )
            )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Folder1",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Folder2",
            )
        )

        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title("t-SNE Visualization of Image Embeddings")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")

        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_embeddings.png", bbox_inches="tight")
        plt.close()

    def create_summary_report(self, results: dict):
        """Create an HTML summary report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ font-weight: bold; color: #4CAF50; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .summary-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                .significant {{ color: #d32f2f; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Image Quality Evaluation Report</h1>
                
                <div class="summary-box">
                    <h3>Dataset Information</h3>
                    <p><strong>Folder 1:</strong> {results.get("folder1_name", "Unknown")} ({results.get("folder1", "Unknown")})</p>
                    <p><strong>Folder 2:</strong> {results.get("folder2_name", "Unknown")} ({results.get("folder2", "Unknown")})</p>
                    <p><strong>Total Styles:</strong> {len(results.get("per_style_metrics", {{}}))}</p>
                    <p><strong>Total Images Compared:</strong> {len(results.get("per_image_metrics", []))}</p>
                </div>
                
                <h2>Overall Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
        """

        # Add overall metrics
        for metric_name, metric_data in results.get("metrics", {}).items():
            if isinstance(metric_data, dict):
                html_content += f"""
                    <tr>
                        <td class="metric">{metric_name.replace("_", " ").title()}</td>
                        <td>{metric_data.get("mean", 0):.4f}</td>
                        <td>{metric_data.get("median", 0):.4f}</td>
                        <td>{metric_data.get("std", 0):.4f}</td>
                        <td>{metric_data.get("min", 0):.4f}</td>
                        <td>{metric_data.get("max", 0):.4f}</td>
                    </tr>
                """

        html_content += """
                </table>
                
                <h2>Per-Style Analysis</h2>
        """

        # Add per-style metrics
        for style, style_data in results.get("per_style_metrics", {}).items():
            html_content += f"""
                <h3>Style: {style}</h3>
                <p>Matched: {style_data.get("matched", 0)}, Missing: {style_data.get("missing", 0)}</p>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
            """

            for metric_name, metric_vals in style_data.get("metrics", {}).items():
                if isinstance(metric_vals, dict):
                    html_content += f"""
                        <tr>
                            <td>{metric_name.replace("_", " ").title()}</td>
                            <td>{metric_vals.get("mean", 0):.4f}</td>
                            <td>{metric_vals.get("std", 0):.4f}</td>
                            <td>{metric_vals.get("min", 0):.4f}</td>
                            <td>{metric_vals.get("max", 0):.4f}</td>
                        </tr>
                    """

            html_content += "</table>"

        # Add statistical tests
        if "statistical_tests" in results:
            html_content += """
                <h2>Statistical Significance Tests</h2>
                <p>Tests comparing the two folders across all metrics:</p>
            """

            for metric_name, tests in results["statistical_tests"].items():
                html_content += (
                    f"<h3>{metric_name.replace('_', ' ').title()}</h3><table>"
                )
                html_content += "<tr><th>Test</th><th>Statistic</th><th>P-Value</th><th>Significant</th></tr>"

                for test_name, test_data in tests.items():
                    if isinstance(test_data, dict):
                        sig_class = (
                            "significant" if test_data.get("significant", False) else ""
                        )
                        html_content += f"""
                            <tr>
                                <td>{test_name.replace("_", " ").title()}</td>
                                <td>{test_data.get("statistic", test_data.get("t_statistic", test_data.get("u_statistic", 0))):.4f}</td>
                                <td class="{sig_class}">{test_data.get("p_value", 0):.4f}</td>
                                <td class="{sig_class}">{"Yes" if test_data.get("significant", False) else "No"}</td>
                            </tr>
                        """

                html_content += "</table>"

        html_content += """
                <h2>Visualizations</h2>
                <p>Generated plots are saved in the output directory.</p>
            </div>
        </body>
        </html>
        """

        with open(self.output_dir / "report.html", "w") as f:
            f.write(html_content)

        logger.info(f"✓ HTML report saved: {self.output_dir / 'report.html'}")


def evaluate_folders_enhanced(
    folder1: DatasetFolder,
    folder2: DatasetFolder,
    folder1_name: str = "Folder1",
    folder2_name: str = "Folder2",
    styles: Optional[list[str]] = None,
    device: str = "cuda:0",
    n_workers: int = 4,
) -> dict:
    """Enhanced evaluation with parallel processing and comprehensive metrics."""

    results = {
        "folder1": str(folder1.root_dir),
        "folder2": str(folder2.root_dir),
        "folder1_name": folder1_name,
        "folder2_name": folder2_name,
        "metrics": {},
        "per_style_metrics": {},
        "per_image_metrics": [],
        "statistical_tests": {},
    }

    # Determine styles to evaluate
    if styles is None:
        styles_set1 = set(folder1.styles)
        styles_set2 = set(folder2.styles)
        common_styles = sorted(list(styles_set1 & styles_set2))
        logger.info(f"Found {len(common_styles)} common styles")
    else:
        common_styles = styles

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{'ENHANCED EVALUATION':^60}")
    logger.info("=" * 60)
    logger.info(f"Folder1: {folder1_name} ({folder1.root_dir})")
    logger.info(f"Folder2: {folder2_name} ({folder2.root_dir})")
    logger.info(f"Styles to evaluate: {len(common_styles)}")
    logger.info(f"Workers: {n_workers}")
    logger.info("=" * 60)

    # Collect all work items
    work_items = []
    for style in common_styles:
        target_images_1 = folder1.get_all_target_images(style)
        for img_stem, img_path_1 in target_images_1:
            img_path_2 = folder2.get_target_image(style, f"{img_stem}.png")
            if img_path_2 is not None:
                work_items.append((img_path_1, img_path_2, img_stem, style, device))

    logger.info(f"Total image pairs to compare: {len(work_items)}")

    # Process in parallel
    all_results = []
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_metrics_computer, initargs=(device,)
    ) as executor:
        futures = {
            executor.submit(compute_image_pair_metrics, item): item
            for item in work_items
        }

        for future in HFTqdm(
            as_completed(futures),
            total=len(work_items),
            desc="🔄 Computing metrics",
            colour="cyan",
        ):
            result = future.result()
            if result is not None:
                all_results.append(result)

    results["per_image_metrics"] = all_results
    logger.info(f"✓ Successfully processed {len(all_results)} image pairs")

    # Aggregate metrics by style
    style_metrics = defaultdict(lambda: defaultdict(list))

    for img_result in all_results:
        style = img_result["style"]
        for metric_name, metric_value in img_result["metrics"].items():
            if metric_value >= 0:  # Filter error values
                style_metrics[style][metric_name].append(metric_value)

    # Compute statistics per style
    analyzer = StatisticalAnalyzer()

    for style in common_styles:
        style_data = {
            "matched": len([r for r in all_results if r["style"] == style]),
            "missing": 0,  # Would need to track this separately
            "metrics": {},
        }

        for metric_name, values in style_metrics[style].items():
            if values:
                style_data["metrics"][metric_name] = analyzer.compute_summary_stats(
                    values
                )

        results["per_style_metrics"][style] = style_data

    # Aggregate global metrics
    global_metrics = defaultdict(list)
    for img_result in all_results:
        for metric_name, metric_value in img_result["metrics"].items():
            if metric_value >= 0:
                global_metrics[metric_name].append(metric_value)

    for metric_name, values in global_metrics.items():
        if values:
            results["metrics"][metric_name] = analyzer.compute_summary_stats(values)

    # Statistical comparison tests (if we had two separate datasets to compare)
    # For now, we're comparing images from the same pair, so we can do paired tests

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{'SUMMARY STATISTICS':^60}")
    logger.info("=" * 60)

    for metric_name, stats in results["metrics"].items():
        logger.info(f"{metric_name.replace('_', ' ').title()}:")
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    logger.info("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced evaluation of two dataset folders"
    )
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    parser.add_argument("--folder1_name", type=str, default="Folder1")
    parser.add_argument("--folder2_name", type=str, default="Folder2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument("--styles", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="evaluation_results_enhanced")
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # Load datasets
        logger.info("\n📂 Loading dataset folders...")
        folder1 = DatasetFolder(args.folder1)
        folder2 = DatasetFolder(args.folder2)

        # Determine workers
        n_workers = args.workers if args.workers else mp.cpu_count()

        # Parse styles
        styles = None
        if args.styles:
            styles = [s.strip() for s in args.styles.split(",")]

        # Evaluate
        results = evaluate_folders_enhanced(
            folder1,
            folder2,
            folder1_name=args.folder1_name,
            folder2_name=args.folder2_name,
            styles=styles,
            device=args.device,
            n_workers=n_workers,
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n💾 Saving results...")
        with open(output_dir / "evaluation_results.json", "w") as f:
            # Remove features from JSON (too large)
            results_copy = results.copy()
            results_copy["per_image_metrics"] = [
                {k: v for k, v in m.items() if k not in ["features1", "features2"]}
                for m in results["per_image_metrics"]
            ]
            json.dump(results_copy, f, indent=2)

        logger.info(f"✓ Results saved to {output_dir / 'evaluation_results.json'}")

        # Generate visualizations
        if not args.skip_visualizations:
            logger.info(f"\n📊 Generating visualizations...")
            viz = Visualizer(str(output_dir))

            # Get available metrics
            if results["metrics"]:
                available_metrics = list(results["metrics"].keys())

                # Distribution plots
                for metric in available_metrics[:8]:  # Limit to prevent too many plots
                    viz.plot_metric_distributions(results, metric)

                # Violin plots for key metrics
                key_metrics = [
                    m
                    for m in ["lpips", "ssim", "psnr", "mse"]
                    if m in available_metrics
                ]
                if key_metrics:
                    viz.plot_violin_comparison(results, key_metrics)

                # Correlation heatmap
                if len(available_metrics) >= 2:
                    viz.plot_correlation_heatmap(results, available_metrics[:10])

                # Radar chart
                styles_to_plot = list(results["per_style_metrics"].keys())[:8]
                if styles_to_plot and len(key_metrics) >= 3:
                    viz.plot_radar_chart(results, styles_to_plot, key_metrics[:6])

                # t-SNE embeddings
                viz.plot_tsne_embeddings(results)

                # HTML report
                viz.create_summary_report(results)

            logger.info(f"✓ Visualizations saved to {output_dir}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ EVALUATION COMPLETE!")
        logger.info(f"Results directory: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Example usage:

# Basic evaluation
python evaluate_two_folders_enhanced.py \\
    --folder1 handwritten_original/ \\
    --folder2 this_generations/ \\
    --folder1_name "Pretrained" \\
    --folder2_name "From Scratch" \\
    --output_dir enhanced_results/

# With specific styles and more workers
python evaluate_two_folders_enhanced.py \\
    --folder1 handwritten_original/ \\
    --folder2 this_generations/ \\
    --styles "style0,style1,style2" \\
    --workers 8 \\
    --device cuda:0 \\
    --output_dir enhanced_results/

# Skip visualizations for faster results
python evaluate_two_folders_enhanced.py \\
    --folder1 handwritten_original/ \\
    --folder2 this_generations/ \\
    --skip_visualizations
"""
