import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class AblationAnalyzer:
    """Analyze ablation study results across multiple configurations."""

    def __init__(self, ablation_dir: Path) -> None:
        """Initialize with ablation evaluation directory."""
        self.ablation_dir = Path(ablation_dir)
        self.results: Dict[str, Dict] = {}
        self.metrics_data: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(dict)

    def parse_config_from_dirname(self, dirname: str) -> Tuple[int | None, int | None]:
        """Extract scale values from folder name like 'scale3_generations_vs_scale10_generations'."""
        try:
            # Extract all numbers from directory name
            parts = dirname.split("_vs_")
            if len(parts) != 2:
                return None, None

            # Parse first config
            scale1 = None
            for part in parts[0].split("-"):
                if part.startswith("scale"):
                    scale1 = int(part.replace("scale", ""))

            # Parse second config
            scale2 = None
            for part in parts[1].split("-"):
                if part.startswith("scale"):
                    scale2 = int(part.replace("scale", ""))

            return scale1, scale2
        except (ValueError, IndexError):
            return None, None

    def load_results(self) -> None:
        """Load all evaluation_results.json files from ablation folders."""
        print(f"📂 Searching for evaluation results in: {self.ablation_dir}\n")

        # Find all folders starting with comparison prefix
        comparison_dirs = [
            d for d in self.ablation_dir.iterdir() if d.is_dir() and "_vs_" in d.name
        ]

        print(f"Found {len(comparison_dirs)} comparison folders\n")

        for comp_dir in sorted(comparison_dirs):
            results_file = comp_dir / "evaluation_results.json"

            if not results_file.exists():
                print(f"⚠️  Skipping {comp_dir.name}: no evaluation_results.json")
                continue

            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract scales from folder name
                scale1, scale2 = self.parse_config_from_dirname(comp_dir.name)

                if scale1 is None or scale2 is None:
                    print(f"⚠️  Could not parse scales from: {comp_dir.name}")
                    continue

                self.results[comp_dir.name] = {
                    "scale1": scale1,
                    "scale2": scale2,
                    "data": data,
                }

                print(f"✓ Loaded: {comp_dir.name}")
                print(f"  Scales: {scale1} vs {scale2}")

            except json.JSONDecodeError as e:
                print(f"❌ Error parsing {results_file}: {e}")
            except Exception as e:
                print(f"❌ Error loading {results_file}: {e}")

        print(f"\n✓ Successfully loaded {len(self.results)} comparisons\n")

    def extract_metrics(self) -> None:
        """Extract global metrics into structured format."""
        print("📊 Extracting metrics from all comparisons...\n")

        metric_names = ["lpips", "ssim", "histogram_distance", "fid"]

        for comp_name, comp_data in self.results.items():
            scale1 = comp_data["scale1"]
            scale2 = comp_data["scale2"]
            data = comp_data["data"]

            metrics = data.get("metrics", {})

            for metric_name in metric_names:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    mean_val = metric_dict.get("mean")

                    if mean_val is not None:
                        # Store with both key combinations for easier access
                        key = (min(scale1, scale2), max(scale1, scale2))
                        self.metrics_data[metric_name][key] = float(mean_val)

        print("✓ Metrics extracted\n")

    def _create_scale_matrix(self, metric_name: str, scales: List[int]) -> np.ndarray:
        """Create NxN matrix of metric values for given scales."""
        n = len(scales)
        matrix = np.full((n, n), np.nan)

        scale_to_idx = {s: i for i, s in enumerate(scales)}

        for (s1, s2), value in self.metrics_data[metric_name].items():
            if s1 in scale_to_idx and s2 in scale_to_idx:
                i1, i2 = scale_to_idx[s1], scale_to_idx[s2]
                matrix[i1, i2] = value
                matrix[i2, i1] = value  # Symmetric

        # Fill diagonal with NaN (no self-comparison)
        np.fill_diagonal(matrix, np.nan)

        return matrix

    def plot_metric_heatmaps(self, output_dir: Path) -> None:
        """Generate heatmaps for each metric across all scale combinations."""
        print("🎨 Generating metric heatmaps...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all unique scales
        all_scales = set()
        for metric_data in self.metrics_data.values():
            for scale1, scale2 in metric_data.keys():
                all_scales.add(scale1)
                all_scales.add(scale2)

        scales = sorted(list(all_scales))
        print(f"Scale range: {min(scales)} - {max(scales)}\n")

        metric_info = {
            "lpips": {"title": "LPIPS (Lower is Better)", "cmap": "RdYlGn_r"},
            "ssim": {"title": "SSIM (Higher is Better)", "cmap": "RdYlGn"},
            "histogram_distance": {
                "title": "Histogram Distance (Lower is Better)",
                "cmap": "RdYlGn_r",
            },
            "fid": {"title": "FID (Lower is Better)", "cmap": "RdYlGn_r"},
        }

        for metric_name, info in metric_info.items():
            if (
                metric_name not in self.metrics_data
                or not self.metrics_data[metric_name]
            ):
                print(f"⚠️  No data for {metric_name}, skipping")
                continue

            matrix = self._create_scale_matrix(metric_name, scales)

            fig, ax = plt.subplots(figsize=(14, 12))

            # Create heatmap
            sns.heatmap(
                matrix,
                xticklabels=scales,
                yticklabels=scales,
                annot=True,
                fmt=".4f",
                cmap=info["cmap"],
                cbar_kws={"label": metric_name.replace("_", " ").title()},
                ax=ax,
                square=True,
                linewidths=0.5,
                cbar=True,
                vmin=np.nanmin(matrix),
                vmax=np.nanmax(matrix),
            )

            ax.set_xlabel("Guidance Scale", fontsize=12, fontweight="bold")
            ax.set_ylabel("Guidance Scale", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{info['title']}\nAblation Study Heatmap",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            plt.tight_layout()

            output_file = output_dir / f"{metric_name}_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✓ Saved: {output_file.name}")

        print()

    def plot_metric_distributions(self, output_dir: Path) -> None:
        """Plot distributions of metric values across all comparisons."""
        print("📈 Generating metric distributions...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        metric_info = {
            "lpips": "LPIPS Score",
            "ssim": "SSIM Score",
            "histogram_distance": "Histogram Distance",
            "fid": "FID Score",
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (metric_name, display_name) in enumerate(metric_info.items()):
            if metric_name not in self.metrics_data:
                continue

            values = list(self.metrics_data[metric_name].values())

            if not values:
                print(f"⚠️  No data for {metric_name}")
                continue

            ax = axes[idx]

            # Histogram
            ax.hist(values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(
                np.mean(values),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(values):.4f}",
            )
            ax.axvline(
                np.median(values),
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(values):.4f}",
            )

            ax.set_xlabel(display_name, fontweight="bold")
            ax.set_ylabel("Frequency", fontweight="bold")
            ax.set_title(f"{display_name} Distribution", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Print stats
            print(f"{display_name}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Median: {np.median(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
            print()

        plt.tight_layout()
        output_file = output_dir / "metric_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved: {output_file.name}\n")

    def plot_scale_trends(self, output_dir: Path) -> None:
        """Plot how metrics change with guidance scale."""
        print("📊 Generating scale trend analysis...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect metrics per scale
        scale_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for metric_name in self.metrics_data:
            for (scale1, scale2), value in self.metrics_data[metric_name].items():
                scale_metrics[scale1][metric_name].append(value)
                scale_metrics[scale2][metric_name].append(value)

        metric_info = {
            "lpips": "LPIPS (Lower Better)",
            "ssim": "SSIM (Higher Better)",
            "histogram_distance": "Histogram Distance (Lower Better)",
            "fid": "FID (Lower Better)",
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (metric_name, display_name) in enumerate(metric_info.items()):
            ax = axes[idx]

            scales = sorted(scale_metrics.keys())
            means = []
            stds = []

            for scale in scales:
                if metric_name in scale_metrics[scale]:
                    values = scale_metrics[scale][metric_name]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # Plot with error bars
            ax.errorbar(
                scales,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
                label="Mean ± Std",
            )
            ax.fill_between(
                scales,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
            )

            ax.set_xlabel("Guidance Scale", fontweight="bold")
            ax.set_ylabel(display_name, fontweight="bold")
            ax.set_title(f"{display_name} vs Guidance Scale", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(scales)

        plt.tight_layout()
        output_file = output_dir / "scale_trends.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved: {output_file.name}\n")

    def export_to_csv(self, output_dir: Path) -> None:
        """Export all metrics to CSV."""
        print("💾 Exporting to CSV...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame
        records = []

        for comp_name, comp_data in self.results.items():
            scale1 = comp_data["scale1"]
            scale2 = comp_data["scale2"]
            metrics = comp_data["data"].get("metrics", {})

            record = {
                "comparison": comp_name,
                "scale1": scale1,
                "scale2": scale2,
                "lpips": metrics.get("lpips", {}).get("mean"),
                "ssim": metrics.get("ssim", {}).get("mean"),
                "histogram_distance": metrics.get("histogram_distance", {}).get("mean"),
                "fid": metrics.get("fid", {}).get("mean"),
            }
            records.append(record)

        df = pd.DataFrame(records)
        csv_file = output_dir / "ablation_metrics.csv"
        df.to_csv(csv_file, index=False)

        print(f"✓ Saved: {csv_file.name}\n")

    def generate_summary_report(self, output_dir: Path) -> None:
        """Generate text summary report."""
        print("📝 Generating summary report...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        all_scales = set()
        for metric_data in self.metrics_data.values():
            for scale1, scale2 in metric_data.keys():
                all_scales.add(scale1)
                all_scales.add(scale2)

        scales = sorted(list(all_scales))

        report = f"""
{"=" * 80}
ABLATION STUDY ANALYSIS REPORT
{"=" * 80}

SUMMARY
-------
Total Comparisons: {len(self.results)}
Guidance Scales Evaluated: {min(scales)} - {max(scales)}
Number of Scales: {len(scales)}

METRICS OVERVIEW
----------------
"""

        for metric_name in ["lpips", "ssim", "histogram_distance", "fid"]:
            if metric_name in self.metrics_data:
                values = list(self.metrics_data[metric_name].values())
                if values:
                    report += f"""
{metric_name.upper()}:
  Mean: {np.mean(values):.4f}
  Median: {np.median(values):.4f}
  Std: {np.std(values):.4f}
  Range: [{np.min(values):.4f}, {np.max(values):.4f}]
  Samples: {len(values)}
"""

        report += f"""
{"=" * 80}
SCALE DISTRIBUTION
{"=" * 80}
Scales: {", ".join(map(str, scales))}

{"=" * 80}
"""

        report_file = output_dir / "ablation_summary.txt"
        with open(report_file, "w") as f:
            f.write(report)

        print(report)
        print(f"✓ Saved: {report_file.name}\n")

    def run_full_analysis(self, output_dir: str = "ablation_analysis_results") -> None:
        """Run complete ablation analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("🔬 ABLATION STUDY COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")

        self.load_results()

        if not self.results:
            print("❌ No results found to analyze!")
            return

        self.extract_metrics()

        # Generate all visualizations and reports
        self.plot_metric_heatmaps(output_path)
        self.plot_metric_distributions(output_path)
        self.plot_scale_trends(output_path)
        self.export_to_csv(output_path)
        self.generate_summary_report(output_path)

        print("=" * 80)
        print(f"✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {output_path}")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - lpips_heatmap.png")
        print("  - ssim_heatmap.png")
        print("  - histogram_distance_heatmap.png")
        print("  - fid_heatmap.png")
        print("  - metric_distributions.png")
        print("  - scale_trends.png")
        print("  - ablation_metrics.csv")
        print("  - ablation_summary.txt")
        print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ablation study analyzer for guidance scale experiments"
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="ablation_evaluation_analysis",
        help="Path to ablation evaluation directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_analysis_results",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    analyzer = AblationAnalyzer(Path(args.ablation_dir))
    analyzer.run_full_analysis(args.output_dir)


if __name__ == "__main__":
    main()
