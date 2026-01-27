import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class InferenceStepsAnalyzer:
    """Analyze ablation study results across different inference step configurations."""

    def __init__(self, ablation_dir: Path) -> None:
        """Initialize with ablation evaluation directory."""
        self.ablation_dir = Path(ablation_dir)
        self.results: Dict[str, Dict] = {}
        self.metrics_data: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(dict)

    def parse_config_from_dirname(self, dirname: str) -> Tuple[int | None, int | None]:
        """Extract step values from folder name like 'step20_generations_vs_step50_generations'."""
        try:
            # Extract all numbers from directory name
            parts = dirname.split("_vs_")
            if len(parts) != 2:
                return None, None

            # Parse first config
            steps1 = None
            for part in parts[0].split("-"):
                if part.startswith("step"):
                    steps1 = int(part.replace("step", ""))

            # Parse second config
            steps2 = None
            for part in parts[1].split("-"):
                if part.startswith("step"):
                    steps2 = int(part.replace("step", ""))

            return steps1, steps2
        except (ValueError, IndexError):
            return None, None

    def load_results(self) -> None:
        """Load all evaluation_results.json files from ablation folders."""
        print(f"📂 Searching for evaluation results in: {self.ablation_dir}\n")

        # Find all folders containing step comparisons
        comparison_dirs = [
            d
            for d in self.ablation_dir.iterdir()
            if d.is_dir() and "_vs_" in d.name and "step" in d.name
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

                # Extract steps from folder name
                steps1, steps2 = self.parse_config_from_dirname(comp_dir.name)

                if steps1 is None or steps2 is None:
                    print(f"⚠️  Could not parse steps from: {comp_dir.name}")
                    continue

                self.results[comp_dir.name] = {
                    "steps1": steps1,
                    "steps2": steps2,
                    "data": data,
                }

                print(f"✓ Loaded: {comp_dir.name}")
                print(f"  Steps: {steps1} vs {steps2}")

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
            steps1 = comp_data["steps1"]
            steps2 = comp_data["steps2"]
            data = comp_data["data"]

            metrics = data.get("metrics", {})

            for metric_name in metric_names:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    mean_val = metric_dict.get("mean")

                    if mean_val is not None:
                        # Store with both key combinations for easier access
                        key = (min(steps1, steps2), max(steps1, steps2))
                        self.metrics_data[metric_name][key] = float(mean_val)

        print("✓ Metrics extracted\n")

    def _create_steps_matrix(
        self, metric_name: str, steps_list: List[int]
    ) -> np.ndarray:
        """Create NxN matrix of metric values for given inference steps."""
        n = len(steps_list)
        matrix = np.full((n, n), np.nan)

        steps_to_idx = {s: i for i, s in enumerate(steps_list)}

        for (s1, s2), value in self.metrics_data[metric_name].items():
            if s1 in steps_to_idx and s2 in steps_to_idx:
                i1, i2 = steps_to_idx[s1], steps_to_idx[s2]
                matrix[i1, i2] = value
                matrix[i2, i1] = value  # Symmetric

        # Fill diagonal with NaN (no self-comparison)
        np.fill_diagonal(matrix, np.nan)

        return matrix

    def plot_metric_heatmaps(self, output_dir: Path) -> None:
        """Generate heatmaps for each metric across all step combinations."""
        print("🎨 Generating metric heatmaps...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all unique steps
        all_steps = set()
        for metric_data in self.metrics_data.values():
            for steps1, steps2 in metric_data.keys():
                all_steps.add(steps1)
                all_steps.add(steps2)

        steps_list = sorted(list(all_steps))
        print(f"Step range: {min(steps_list)} - {max(steps_list)}\n")

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

            matrix = self._create_steps_matrix(metric_name, steps_list)

            fig, ax = plt.subplots(figsize=(14, 12))

            # Create heatmap
            sns.heatmap(
                matrix,
                xticklabels=steps_list,
                yticklabels=steps_list,
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

            ax.set_xlabel("Inference Steps", fontsize=12, fontweight="bold")
            ax.set_ylabel("Inference Steps", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{info['title']}\nInference Steps Ablation Heatmap",
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

    def plot_steps_trends(self, output_dir: Path) -> None:
        """Plot how metrics change with inference steps."""
        print("📊 Generating steps trend analysis...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect metrics per step
        steps_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for metric_name in self.metrics_data:
            for (steps1, steps2), value in self.metrics_data[metric_name].items():
                steps_metrics[steps1][metric_name].append(value)
                steps_metrics[steps2][metric_name].append(value)

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

            steps_list = sorted(steps_metrics.keys())
            means = []
            stds = []

            for step in steps_list:
                if metric_name in steps_metrics[step]:
                    values = steps_metrics[step][metric_name]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # Plot with error bars
            ax.errorbar(
                steps_list,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
                label="Mean ± Std",
            )
            ax.fill_between(
                steps_list,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
            )

            ax.set_xlabel("Inference Steps", fontweight="bold")
            ax.set_ylabel(display_name, fontweight="bold")
            ax.set_title(f"{display_name} vs Inference Steps", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(steps_list)

        plt.tight_layout()
        output_file = output_dir / "steps_trends.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved: {output_file.name}\n")

    def plot_quality_vs_efficiency(self, output_dir: Path) -> None:
        """Plot quality-efficiency tradeoff: metric quality vs inference speed."""
        print("📉 Generating quality vs efficiency analysis...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect metrics per step
        steps_metrics: Dict[int, Dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for metric_name in self.metrics_data:
            for (steps1, steps2), value in self.metrics_data[metric_name].items():
                steps_metrics[steps1][metric_name].append(value)
                steps_metrics[steps2][metric_name].append(value)

        steps_list = sorted(steps_metrics.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Normalize steps for colormap (0-1 range)
        steps_normalized = np.linspace(0, 1, len(steps_list))
        step_to_color = {step: steps_normalized[i] for i, step in enumerate(steps_list)}

        # For each metric, plot quality vs speed (relative inference time)
        metric_info = {
            "lpips": {"display": "LPIPS (Lower Better)", "direction": "lower"},
            "ssim": {"display": "SSIM (Higher Better)", "direction": "higher"},
            "histogram_distance": {
                "display": "Histogram Distance (Lower Better)",
                "direction": "lower",
            },
            "fid": {"display": "FID (Lower Better)", "direction": "lower"},
        }

        for idx, (metric_name, info) in enumerate(metric_info.items()):
            ax = axes[idx]

            quality_scores = []
            relative_times = []
            colors = []
            labels = []

            min_steps = min(steps_list)

            for step in steps_list:
                if metric_name in steps_metrics[step]:
                    values = steps_metrics[step][metric_name]
                    quality_scores.append(np.mean(values))
                    # Relative inference time (normalized to min_steps)
                    relative_times.append(step / min_steps)
                    colors.append(step_to_color[step])
                    labels.append(str(step))

            # Scatter plot with annotations
            scatter = ax.scatter(
                relative_times,
                quality_scores,
                s=200,
                alpha=0.6,
                c=colors,  # Normalized 0-1 values
                cmap="viridis",
                edgecolors="black",
                linewidth=1,
            )

            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (relative_times[i], quality_scores[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xlabel(
                "Relative Inference Time (normalized to min steps)", fontweight="bold"
            )
            ax.set_ylabel(info["display"], fontweight="bold")
            ax.set_title(f"Quality vs Efficiency: {info['display']}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Steps", fontweight="bold")

        plt.tight_layout()
        output_file = output_dir / "quality_vs_efficiency.png"
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
            steps1 = comp_data["steps1"]
            steps2 = comp_data["steps2"]
            metrics = comp_data["data"].get("metrics", {})

            record = {
                "comparison": comp_name,
                "steps1": steps1,
                "steps2": steps2,
                "lpips": metrics.get("lpips", {}).get("mean"),
                "ssim": metrics.get("ssim", {}).get("mean"),
                "histogram_distance": metrics.get("histogram_distance", {}).get("mean"),
                "fid": metrics.get("fid", {}).get("mean"),
            }
            records.append(record)

        df = pd.DataFrame(records)
        csv_file = output_dir / "steps_ablation_metrics.csv"
        df.to_csv(csv_file, index=False)

        print(f"✓ Saved: {csv_file.name}\n")

    def generate_summary_report(self, output_dir: Path) -> None:
        """Generate text summary report."""
        print("📝 Generating summary report...\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        all_steps = set()
        for metric_data in self.metrics_data.values():
            for steps1, steps2 in metric_data.keys():
                all_steps.add(steps1)
                all_steps.add(steps2)

        steps_list = sorted(list(all_steps))

        # Calculate per-step metrics
        steps_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for metric_name in self.metrics_data:
            for (steps1, steps2), value in self.metrics_data[metric_name].items():
                steps_metrics[steps1][metric_name].append(value)
                steps_metrics[steps2][metric_name].append(value)

        report = f"""
{"=" * 80}
INFERENCE STEPS ABLATION STUDY ANALYSIS REPORT
{"=" * 80}

SUMMARY
-------
Total Comparisons: {len(self.results)}
Inference Steps Evaluated: {min(steps_list)} - {max(steps_list)}
Number of Step Configurations: {len(steps_list)}

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
PER-STEP ANALYSIS
{"=" * 80}
"""

        for step in steps_list:
            report += f"\nSteps={step}:\n"
            for metric_name in ["lpips", "ssim", "histogram_distance", "fid"]:
                if metric_name in steps_metrics[step]:
                    values = steps_metrics[step][metric_name]
                    report += f"  {metric_name.upper():20s}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}\n"

        report += f"""
{"=" * 80}
EFFICIENCY ANALYSIS
{"=" * 80}
Relative Inference Times (normalized to {min(steps_list)} steps):
"""

        for step in steps_list:
            relative_time = step / min(steps_list)
            report += f"  {step:3d} steps: {relative_time:.2f}x\n"

        report += f"""
{"=" * 80}
"""

        report_file = output_dir / "steps_ablation_summary.txt"
        with open(report_file, "w") as f:
            f.write(report)

        print(report)
        print(f"✓ Saved: {report_file.name}\n")

    def run_full_analysis(self, output_dir: str = "steps_analysis_results") -> None:
        """Run complete inference steps ablation analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("🔬 INFERENCE STEPS ABLATION STUDY COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")

        self.load_results()

        if not self.results:
            print("❌ No results found to analyze!")
            return

        self.extract_metrics()

        # Generate all visualizations and reports
        self.plot_metric_heatmaps(output_path)
        self.plot_metric_distributions(output_path)
        self.plot_steps_trends(output_path)
        self.plot_quality_vs_efficiency(output_path)
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
        print("  - steps_trends.png")
        print("  - quality_vs_efficiency.png")
        print("  - steps_ablation_metrics.csv")
        print("  - steps_ablation_summary.txt")
        print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ablation study analyzer for inference steps experiments"
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="steps_evaluation_analysis",
        help="Path to ablation evaluation directory with step comparisons",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="steps_analysis_results",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    analyzer = InferenceStepsAnalyzer(Path(args.ablation_dir))
    analyzer.run_full_analysis(args.output_dir)


if __name__ == "__main__":
    main()
