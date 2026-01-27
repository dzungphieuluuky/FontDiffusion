import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


class MetricType(Enum):
    """Metric type enumeration."""

    LPIPS = "lpips"
    SSIM = "ssim"
    HISTOGRAM = "histogram_distance"
    FID = "fid"


class MetricDirection(Enum):
    """Whether higher or lower values are better."""

    HIGHER_BETTER = "higher"
    LOWER_BETTER = "lower"


METRIC_PROPERTIES = {
    MetricType.LPIPS: {"direction": MetricDirection.LOWER_BETTER, "display": "LPIPS"},
    MetricType.SSIM: {"direction": MetricDirection.HIGHER_BETTER, "display": "SSIM"},
    MetricType.HISTOGRAM: {
        "direction": MetricDirection.LOWER_BETTER,
        "display": "Histogram Distance",
    },
    MetricType.FID: {"direction": MetricDirection.LOWER_BETTER, "display": "FID"},
}


@dataclass
class MetricStats:
    """Statistical summary with validation."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    count: int
    iqr: float = field(init=False)
    cv: float = field(init=False)  # Coefficient of variation

    def __post_init__(self):
        """Calculate derived statistics and validate."""
        self.iqr = self.q75 - self.q25
        self.cv = (self.std / self.mean * 100) if self.mean != 0 else 0.0

        # Validation
        if self.count <= 0:
            raise ValueError(f"Invalid count: {self.count}")
        if self.min > self.max:
            raise ValueError(f"Min ({self.min}) > Max ({self.max})")
        if self.std < 0:
            raise ValueError(f"Negative std: {self.std}")

    def __str__(self) -> str:
        return (
            f"μ={self.mean:.4f}, σ={self.std:.4f}, "
            f"median={self.median:.4f}, range=[{self.min:.4f}, {self.max:.4f}], "
            f"IQR={self.iqr:.4f}, CV={self.cv:.2f}%, n={self.count}"
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OutlierInfo:
    """Information about outliers in data."""

    count: int
    percentage: float
    values: List[float]
    indices: List[int]

    def __str__(self) -> str:
        if self.count == 0:
            return "No outliers detected"
        return (
            f"{self.count} outliers ({self.percentage:.2f}%): "
            f"range=[{min(self.values):.4f}, {max(self.values):.4f}]"
        )


class DataValidator:
    """Validates and cleans evaluation data."""

    @staticmethod
    def validate_metric_value(value: Any, metric_name: str) -> Optional[float]:
        """Validate and convert metric value."""
        if value is None:
            return None

        try:
            val = float(value)

            # Range validation based on metric type
            if metric_name == "ssim":
                if not -1 <= val <= 1:
                    warnings.warn(f"SSIM value {val} outside expected range [-1, 1]")
            elif metric_name in ["lpips", "fid", "histogram_distance"]:
                if val < 0:
                    warnings.warn(f"{metric_name} value {val} is negative")

            # Check for invalid values
            if np.isnan(val) or np.isinf(val):
                return None

            return val
        except (ValueError, TypeError):
            warnings.warn(f"Invalid {metric_name} value: {value}")
            return None

    @staticmethod
    def validate_dict_structure(
        data: Dict[str, Any], required_keys: List[str], dict_name: str = "data"
    ) -> bool:
        """Validate dictionary has required keys."""
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            warnings.warn(f"{dict_name} missing keys: {missing_keys}")
            return False
        return True

    @staticmethod
    def clean_metric_dict(metric_data: Dict[str, Any]) -> Optional[MetricStats]:
        """Clean and validate metric dictionary."""
        required_keys = ["mean", "median", "std", "min", "max", "samples"]

        if not isinstance(metric_data, dict):
            return None

        # Extract values with defaults
        try:
            values = np.array(
                [
                    float(metric_data.get("mean", 0)),
                    float(metric_data.get("median", 0)),
                    float(metric_data.get("std", 0)),
                    float(metric_data.get("min", 0)),
                    float(metric_data.get("max", 0)),
                ]
            )

            # Check for invalid values
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                return None

            return MetricStats(
                mean=values[0],
                median=values[1],
                std=values[2],
                min=values[3],
                max=values[4],
                q25=float(metric_data.get("q25", values[0])),
                q75=float(metric_data.get("q75", values[0])),
                count=int(metric_data.get("samples", 0)),
            )
        except (ValueError, TypeError, KeyError) as e:
            warnings.warn(f"Error cleaning metric dict: {e}")
            return None


class StatisticalAnalyzer:
    """Performs statistical analysis on metrics."""

    @staticmethod
    def compute_stats(values: List[float]) -> Optional[MetricStats]:
        """Compute comprehensive statistics with validation."""
        if not values:
            return None

        # Remove invalid values
        clean_values = [v for v in values if v is not None and np.isfinite(v)]

        if not clean_values:
            return None

        arr = np.array(clean_values)

        try:
            return MetricStats(
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                std=float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                q25=float(np.percentile(arr, 25)),
                q75=float(np.percentile(arr, 75)),
                count=len(arr),
            )
        except Exception as e:
            warnings.warn(f"Error computing stats: {e}")
            return None

    @staticmethod
    def detect_outliers(
        values: List[float], method: str = "iqr", threshold: float = 1.5
    ) -> OutlierInfo:
        """Detect outliers using IQR or Z-score method."""
        if not values:
            return OutlierInfo(0, 0.0, [], [])

        clean_values = np.array([v for v in values if v is not None and np.isfinite(v)])

        if len(clean_values) == 0:
            return OutlierInfo(0, 0.0, [], [])

        outlier_mask = np.zeros(len(clean_values), dtype=bool)

        if method == "iqr":
            q1 = np.percentile(clean_values, 25)
            q3 = np.percentile(clean_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (clean_values < lower_bound) | (clean_values > upper_bound)

        elif method == "zscore":
            mean = np.mean(clean_values)
            std = np.std(clean_values)
            if std > 0:
                z_scores = np.abs((clean_values - mean) / std)
                outlier_mask = z_scores > threshold

        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = clean_values[outlier_mask].tolist()
        outlier_count = len(outlier_indices)
        outlier_pct = (outlier_count / len(clean_values)) * 100

        return OutlierInfo(
            count=outlier_count,
            percentage=outlier_pct,
            values=outlier_values,
            indices=outlier_indices,
        )

    @staticmethod
    def compare_distributions(
        values1: List[float], values2: List[float]
    ) -> Dict[str, float]:
        """Compare two distributions statistically."""
        clean1 = np.array([v for v in values1 if v is not None and np.isfinite(v)])
        clean2 = np.array([v for v in values2 if v is not None and np.isfinite(v)])

        results = {}

        # T-test
        if len(clean1) >= 2 and len(clean2) >= 2:
            t_stat, p_value = stats.ttest_ind(clean1, clean2)
            results["t_statistic"] = float(t_stat)
            results["t_test_p_value"] = float(p_value)

        # Mann-Whitney U test (non-parametric)
        if len(clean1) >= 1 and len(clean2) >= 1:
            u_stat, p_value = stats.mannwhitneyu(
                clean1, clean2, alternative="two-sided"
            )
            results["u_statistic"] = float(u_stat)
            results["mann_whitney_p_value"] = float(p_value)

        # Effect size (Cohen's d)
        if len(clean1) >= 2 and len(clean2) >= 2:
            pooled_std = np.sqrt((np.var(clean1, ddof=1) + np.var(clean2, ddof=1)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(clean1) - np.mean(clean2)) / pooled_std
                results["cohens_d"] = float(cohens_d)

        return results


class EvaluationAnalyzer:
    """Main analyzer class with enhanced safety."""

    def __init__(self, results_path: Union[str, Path]) -> None:
        """Initialize analyzer with validation."""
        self.results_path = Path(results_path)
        self.validator = DataValidator()
        self.stat_analyzer = StatisticalAnalyzer()
        self.data: Dict[str, Any] = {}
        self.is_valid = False

        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Load and validate JSON data."""
        if not self.results_path.exists():
            print(f"❌ File not found: {self.results_path}")
            return

        try:
            with open(self.results_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

            # Validate structure
            if not isinstance(self.data, dict):
                print("❌ Invalid JSON structure: expected dict")
                return

            self.is_valid = True
            print(f"✓ Loaded results from {self.results_path}")
            print(f"  Folder 1: {self.data.get('folder1_name', 'Unknown')}")
            print(f"  Folder 2: {self.data.get('folder2_name', 'Unknown')}")

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
        except Exception as e:
            print(f"❌ Error loading file: {e}")

    def _safe_get_metrics(self, key: str) -> Dict[str, Any]:
        """Safely retrieve metrics dictionary."""
        metrics = self.data.get(key, {})
        return metrics if isinstance(metrics, dict) else {}

    def _safe_get_list(self, key: str) -> List[Any]:
        """Safely retrieve list."""
        items = self.data.get(key, [])
        return items if isinstance(items, list) else []

    def print_section(self, title: str, width: int = 80) -> None:
        """Print formatted section header."""
        print(f"\n{'=' * width}")
        print(f"{title:^{width}}")
        print(f"{'=' * width}\n")

    def analyze_global_metrics(self) -> Dict[str, MetricStats]:
        """Analyze global aggregate metrics with validation."""
        self.print_section("GLOBAL METRICS SUMMARY")

        if not self.is_valid:
            print("❌ Cannot analyze: invalid data")
            return {}

        folder1 = self.data.get("folder1_name", "Folder1")
        folder2 = self.data.get("folder2_name", "Folder2")
        print(f"📊 Comparing: {folder1} vs {folder2}\n")

        metrics = self._safe_get_metrics("metrics")
        results = {}

        for metric_name, metric_data in metrics.items():
            stats = self.validator.clean_metric_dict(metric_data)

            if stats:
                results[metric_name] = stats
                display_name = METRIC_PROPERTIES.get(MetricType(metric_name), {}).get(
                    "display", metric_name.upper().replace("_", " ")
                )

                print(f"📈 {display_name}:")
                print(f"   {stats}")
                print()
            else:
                print(f"⚠️  {metric_name}: Invalid or missing data")
                print()

        return results

    def analyze_per_style_metrics(self) -> Dict[str, Dict[str, MetricStats]]:
        """Analyze per-style metrics with validation."""
        self.print_section("PER-STYLE METRICS BREAKDOWN")

        if not self.is_valid:
            print("❌ Cannot analyze: invalid data")
            return {}

        per_style = self._safe_get_metrics("per_style_metrics")

        if not per_style:
            print("⚠️  No per-style metrics available")
            return {}

        print(f"🎨 Total styles evaluated: {len(per_style)}\n")

        results = {}
        metric_rankings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        for style_name, style_data in sorted(per_style.items()):
            if not isinstance(style_data, dict):
                continue

            print(f"Style: {style_name}")
            matched = style_data.get("matched", 0)
            missing = style_data.get("missing", 0)
            total = matched + missing

            print(f"  Samples: {matched}/{total} matched ({missing} missing)")

            style_results = {}

            for metric_type in MetricType:
                metric_name = metric_type.value

                if metric_name in style_data:
                    metric_stats = self.validator.clean_metric_dict(
                        style_data[metric_name]
                    )

                    if metric_stats:
                        style_results[metric_name] = metric_stats
                        display_name = METRIC_PROPERTIES[metric_type]["display"]
                        print(
                            f"  {display_name}: {metric_stats.mean:.4f} ± {metric_stats.std:.4f} (n={metric_stats.count})"
                        )

                        # Track for ranking
                        metric_rankings[metric_name].append(
                            (style_name, metric_stats.mean)
                        )

            results[style_name] = style_results
            print()

        # Print rankings
        self._print_style_rankings(metric_rankings)

        return results

    def _print_style_rankings(
        self, rankings: Dict[str, List[Tuple[str, float]]]
    ) -> None:
        """Print style rankings by metric."""
        print("\n" + "=" * 80)
        print("STYLE RANKINGS BY METRIC")
        print("=" * 80 + "\n")

        for metric_name, style_values in rankings.items():
            if not style_values:
                continue

            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
                direction = METRIC_PROPERTIES[metric_type]["direction"]

                # Sort based on metric direction
                reverse = direction == MetricDirection.HIGHER_BETTER
                sorted_styles = sorted(
                    style_values, key=lambda x: x[1], reverse=reverse
                )

                print(
                    f"📊 {display_name} ({'higher is better' if reverse else 'lower is better'}):"
                )

                for rank, (style, value) in enumerate(sorted_styles[:10], 1):  # Top 10
                    indicator = "⭐" if rank <= 3 else " "
                    print(f"  {indicator} {rank:2d}. {style:20s} → {value:.4f}")

                if len(sorted_styles) > 10:
                    print(f"  ... and {len(sorted_styles) - 10} more styles")

                print()
            except (ValueError, KeyError):
                continue

    def analyze_per_image_metrics(self) -> Dict[str, Tuple[MetricStats, OutlierInfo]]:
        """Analyze per-image distributions with outlier detection."""
        self.print_section("PER-IMAGE METRICS DISTRIBUTION")

        if not self.is_valid:
            print("❌ Cannot analyze: invalid data")
            return {}

        per_image = self._safe_get_list("per_image_metrics")

        if not per_image:
            print("⚠️  No per-image metrics available")
            return {}

        print(f"📸 Total images evaluated: {len(per_image)}\n")

        # Extract values by metric
        metrics_data: Dict[str, List[float]] = defaultdict(list)

        for img_data in per_image:
            if not isinstance(img_data, dict):
                continue

            for metric_type in MetricType:
                metric_name = metric_type.value
                if metric_name in img_data:
                    val = self.validator.validate_metric_value(
                        img_data[metric_name], metric_name
                    )
                    if val is not None:
                        metrics_data[metric_name].append(val)

        results = {}

        for metric_name, values in metrics_data.items():
            if not values:
                continue

            stats = self.stat_analyzer.compute_stats(values)
            outliers = self.stat_analyzer.detect_outliers(
                values, method="iqr", threshold=1.5
            )

            if stats:
                results[metric_name] = (stats, outliers)

                try:
                    metric_type = MetricType(metric_name)
                    display_name = METRIC_PROPERTIES[metric_type]["display"]
                except (ValueError, KeyError):
                    display_name = metric_name.upper().replace("_", " ")

                print(f"📊 {display_name}:")
                print(f"   {stats}")
                print(f"   {outliers}")

                # Distribution shape analysis
                if len(values) >= 30:
                    skewness = float(stats_module.skew(values))
                    kurtosis = float(stats_module.kurtosis(values))
                    print(f"   Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")

                print()

        return results

    def analyze_cross_style_consistency(self) -> Dict[str, MetricStats]:
        """Analyze metric consistency across styles."""
        self.print_section("CROSS-STYLE CONSISTENCY ANALYSIS")

        if not self.is_valid:
            print("❌ Cannot analyze: invalid data")
            return {}

        per_style = self._safe_get_metrics("per_style_metrics")

        if not per_style:
            print("⚠️  No per-style metrics for consistency analysis")
            return {}

        # Collect mean values across styles for each metric
        cross_style_means: Dict[str, List[float]] = defaultdict(list)

        for style_data in per_style.values():
            if not isinstance(style_data, dict):
                continue

            for metric_type in MetricType:
                metric_name = metric_type.value
                if metric_name in style_data:
                    metric_dict = style_data[metric_name]
                    if isinstance(metric_dict, dict) and "mean" in metric_dict:
                        val = self.validator.validate_metric_value(
                            metric_dict["mean"], metric_name
                        )
                        if val is not None:
                            cross_style_means[metric_name].append(val)

        results = {}

        for metric_name, means in cross_style_means.items():
            if len(means) < 2:
                continue

            stats = self.stat_analyzer.compute_stats(means)

            if stats:
                results[metric_name] = stats

                try:
                    metric_type = MetricType(metric_name)
                    display_name = METRIC_PROPERTIES[metric_type]["display"]
                except (ValueError, KeyError):
                    display_name = metric_name.upper().replace("_", " ")

                consistency = (
                    "High" if stats.cv < 10 else "Medium" if stats.cv < 25 else "Low"
                )

                print(f"📈 {display_name}:")
                print(f"   Cross-style mean: {stats.mean:.4f}")
                print(f"   Cross-style std: {stats.std:.4f}")
                print(f"   Coefficient of variation: {stats.cv:.2f}%")
                print(f"   Consistency: {consistency}")
                print(f"   Range: [{stats.min:.4f}, {stats.max:.4f}]")
                print()

        return results

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export per-image metrics to pandas DataFrame."""
        if not self.is_valid:
            return pd.DataFrame()

        per_image = self._safe_get_list("per_image_metrics")

        if not per_image:
            return pd.DataFrame()

        # Clean and validate data
        clean_records = []

        for img_data in per_image:
            if not isinstance(img_data, dict):
                continue

            record = {
                "stem": img_data.get("stem", "unknown"),
                "style": img_data.get("style", "unknown"),
            }

            for metric_type in MetricType:
                metric_name = metric_type.value
                if metric_name in img_data:
                    val = self.validator.validate_metric_value(
                        img_data[metric_name], metric_name
                    )
                    record[metric_name] = val

            clean_records.append(record)

        return pd.DataFrame(clean_records)

    def generate_visualizations(self, output_dir: Union[str, Path]) -> None:
        """Generate comprehensive visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.is_valid:
            print("❌ Cannot generate visualizations: invalid data")
            return

        print("\n🎨 Generating visualizations...")

        # Get DataFrame
        df = self.export_to_dataframe()

        if df.empty:
            print("⚠️  No data available for visualization")
            return

        # 1. Distribution plots
        self._plot_distributions(df, output_path)

        # 2. Box plots by style
        self._plot_boxplots_by_style(df, output_path)

        # 3. Correlation heatmap
        self._plot_correlation_heatmap(df, output_path)

        # 4. Violin plots
        self._plot_violin_plots(df, output_path)

        print(f"✓ Visualizations saved to {output_path}")

    def _plot_distributions(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot metric distributions."""
        metric_cols = [m.value for m in MetricType if m.value in df.columns]

        if not metric_cols:
            return

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, metric_cols):
            values = df[metric_name].dropna()

            if len(values) == 0:
                continue

            ax.hist(values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")

            # Add statistics
            mean_val = values.mean()
            median_val = values.median()
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.4f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.4f}",
            )

            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
            except (ValueError, KeyError):
                display_name = metric_name.upper().replace("_", " ")

            ax.set_xlabel(display_name)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{display_name} Distribution")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "metric_distributions.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ metric_distributions.png")

    def _plot_boxplots_by_style(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot box plots grouped by style."""
        metric_cols = [m.value for m in MetricType if m.value in df.columns]

        if not metric_cols or "style" not in df.columns:
            return

        for metric_name in metric_cols:
            if metric_name not in df.columns:
                continue

            # Filter out NaN and get top 15 styles by sample count
            df_metric = df[["style", metric_name]].dropna()

            if len(df_metric) == 0:
                continue

            style_counts = df_metric["style"].value_counts()
            top_styles = style_counts.head(15).index.tolist()
            df_filtered = df_metric[df_metric["style"].isin(top_styles)]

            fig, ax = plt.subplots(figsize=(14, 6))

            df_filtered.boxplot(column=metric_name, by="style", ax=ax)

            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
            except (ValueError, KeyError):
                display_name = metric_name.upper().replace("_", " ")

            ax.set_title(f"{display_name} by Style")
            ax.set_xlabel("Style")
            ax.set_ylabel(display_name)
            plt.suptitle("")  # Remove default title
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            filename = f"{metric_name}_by_style_boxplot.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {filename}")

    def _plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot correlation heatmap between metrics."""
        metric_cols = [m.value for m in MetricType if m.value in df.columns]

        if len(metric_cols) < 2:
            return

        corr_df = df[metric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Metric Correlation Heatmap")

        plt.tight_layout()
        plt.savefig(
            output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ correlation_heatmap.png")

    def _plot_violin_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot violin plots for metrics."""
        metric_cols = [m.value for m in MetricType if m.value in df.columns]

        if not metric_cols:
            return

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, metric_cols):
            values = df[metric_name].dropna()

            if len(values) == 0:
                continue

            parts = ax.violinplot(
                [values], positions=[1], showmeans=True, showmedians=True
            )

            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.7)

            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
            except (ValueError, KeyError):
                display_name = metric_name.upper().replace("_", " ")

            ax.set_ylabel(display_name)
            ax.set_title(f"{display_name} Violin Plot")
            ax.set_xticks([1])
            ax.set_xticklabels(["All Data"])
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "metric_violin_plots.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ metric_violin_plots.png")

    def generate_html_report(self, output_path: Union[str, Path]) -> None:
        """Generate comprehensive HTML report."""
        if not self.is_valid:
            print("❌ Cannot generate report: invalid data")
            return

        html_path = Path(output_path)

        # Analyze all metrics
        global_stats = self.analyze_global_metrics()
        per_style_stats = self.analyze_per_style_metrics()
        per_image_stats = self.analyze_per_image_metrics()
        consistency_stats = self.analyze_cross_style_consistency()

        folder1 = self.data.get("folder1_name", "Folder 1")
        folder2 = self.data.get("folder2_name", "Folder 2")

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            color: #333; 
            background: #f5f7fa;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 4px solid #3498db; 
            padding-bottom: 15px; 
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 40px; 
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{ 
            color: #555; 
            margin-top: 25px; 
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .meta-info {{ 
            background: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .meta-info strong {{ color: #2c3e50; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            padding: 14px; 
            text-align: left; 
            border: 1px solid #ddd; 
        }}
        th {{ 
            background: #3498db; 
            color: white; 
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #e8f4f8; transition: background 0.3s; }}
        .metric-value {{ 
            font-weight: bold; 
            font-family: 'Courier New', monospace;
            color: #2c3e50;
        }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ 
            color: white; 
            margin-top: 0;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{ 
            font-size: 2.5em; 
            font-weight: bold; 
            margin: 10px 0;
        }}
        .stat-label {{ 
            opacity: 0.9; 
            font-size: 0.9em;
        }}
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Evaluation Analysis Report</h1>
        
        <div class="meta-info">
            <p><strong>Folder 1:</strong> {folder1}</p>
            <p><strong>Folder 2:</strong> {folder2}</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <h2>🎯 Global Metrics Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
                <th>Range</th>
                <th>Samples</th>
            </tr>
"""

        for metric_name, stats in global_stats.items():
            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
            except (ValueError, KeyError):
                display_name = metric_name.upper().replace("_", " ")

            html += f"""
            <tr>
                <td><strong>{display_name}</strong></td>
                <td class="metric-value">{stats.mean:.4f}</td>
                <td class="metric-value">{stats.median:.4f}</td>
                <td class="metric-value">{stats.std:.4f}</td>
                <td class="metric-value">[{stats.min:.4f}, {stats.max:.4f}]</td>
                <td>{stats.count}</td>
            </tr>
"""

        html += """
        </table>
        
        <h2>🎨 Per-Style Performance</h2>
"""

        if per_style_stats:
            html += """
        <table>
            <tr>
                <th>Style</th>
                <th>LPIPS</th>
                <th>SSIM</th>
                <th>Histogram Distance</th>
                <th>FID</th>
            </tr>
"""

            for style_name, metrics in sorted(per_style_stats.items()):
                html += f"<tr><td><strong>{style_name}</strong></td>"

                for metric_type in [
                    MetricType.LPIPS,
                    MetricType.SSIM,
                    MetricType.HISTOGRAM,
                    MetricType.FID,
                ]:
                    metric_name = metric_type.value
                    if metric_name in metrics:
                        val = metrics[metric_name].mean
                        html += f'<td class="metric-value">{val:.4f}</td>'
                    else:
                        html += "<td>-</td>"

                html += "</tr>\n"

            html += "</table>"

        html += """
        <h2>📈 Distribution Analysis</h2>
"""

        if per_image_stats:
            for metric_name, (stats, outliers) in per_image_stats.items():
                try:
                    metric_type = MetricType(metric_name)
                    display_name = METRIC_PROPERTIES[metric_type]["display"]
                except (ValueError, KeyError):
                    display_name = metric_name.upper().replace("_", " ")

                outlier_badge = ""
                if outliers.count > 0:
                    badge_class = (
                        "badge-warning" if outliers.percentage < 5 else "badge-danger"
                    )
                    outlier_badge = f'<span class="badge {badge_class}">{outliers.count} outliers ({outliers.percentage:.1f}%)</span>'

                html += f"""
        <h3>{display_name} {outlier_badge}</h3>
        <p><strong>Statistics:</strong> Mean = {stats.mean:.4f}, Median = {stats.median:.4f}, Std = {stats.std:.4f}</p>
        <p><strong>Range:</strong> [{stats.min:.4f}, {stats.max:.4f}]</p>
        <p><strong>IQR:</strong> {stats.iqr:.4f}, <strong>CV:</strong> {stats.cv:.2f}%</p>
"""

        html += """
        <h2>🔄 Cross-Style Consistency</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Cross-Style Mean</th>
                <th>Cross-Style Std</th>
                <th>CV (%)</th>
                <th>Consistency</th>
            </tr>
"""

        for metric_name, stats in consistency_stats.items():
            try:
                metric_type = MetricType(metric_name)
                display_name = METRIC_PROPERTIES[metric_type]["display"]
            except (ValueError, KeyError):
                display_name = metric_name.upper().replace("_", " ")

            consistency = (
                "High" if stats.cv < 10 else "Medium" if stats.cv < 25 else "Low"
            )
            badge_class = (
                "badge-success"
                if stats.cv < 10
                else "badge-warning" if stats.cv < 25 else "badge-danger"
            )

            html += f"""
            <tr>
                <td><strong>{display_name}</strong></td>
                <td class="metric-value">{stats.mean:.4f}</td>
                <td class="metric-value">{stats.std:.4f}</td>
                <td class="metric-value">{stats.cv:.2f}</td>
                <td><span class="badge {badge_class}">{consistency}</span></td>
            </tr>
"""

        html += """
        </table>
        
        <div class="footer">
            <p>Generated by Enhanced Evaluation Analyzer</p>
        </div>
    </div>
</body>
</html>
"""

        html_path.write_text(html, encoding="utf-8")
        print(f"✓ HTML report saved: {html_path}")

    def export_to_csv(self, output_dir: Union[str, Path]) -> None:
        """Export data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = self.export_to_dataframe()

        if not df.empty:
            csv_path = output_path / "per_image_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV exported: {csv_path}")

    def run_full_analysis(
        self, output_dir: Union[str, Path] = "analysis_results"
    ) -> None:
        """Run comprehensive analysis pipeline."""
        if not self.is_valid:
            print("❌ Cannot run analysis: invalid data")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("🔬 RUNNING FULL ANALYSIS PIPELINE")
        print("=" * 80)

        # Statistical analyses
        self.analyze_global_metrics()
        self.analyze_per_style_metrics()
        self.analyze_per_image_metrics()
        self.analyze_cross_style_consistency()

        # Visualizations
        self.generate_visualizations(output_path)

        # Reports
        self.generate_html_report(output_path / "analysis_report.html")
        self.export_to_csv(output_path)

        # Summary JSON
        summary = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "folder1_name": self.data.get("folder1_name", "Unknown"),
            "folder2_name": self.data.get("folder2_name", "Unknown"),
            "total_images": len(self._safe_get_list("per_image_metrics")),
            "total_styles": len(self._safe_get_metrics("per_style_metrics")),
        }

        with open(output_path / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print(f"✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {output_path}")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - analysis_report.html (Interactive HTML report)")
        print("  - per_image_metrics.csv (Raw data export)")
        print("  - analysis_summary.json (Summary statistics)")
        print("  - metric_distributions.png")
        print("  - correlation_heatmap.png")
        print("  - metric_violin_plots.png")
        print("  - [metric]_by_style_boxplot.png")
        print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced evaluation results analyzer with type safety",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_results.py --results evaluation_results/evaluation_results.json
  
  # Custom output directory
  python analyze_results.py --results results.json --output custom_analysis/
  
  # Generate only visualizations
  python analyze_results.py --results results.json --viz-only
        """,
    )

    parser.add_argument(
        "--results", type=str, required=True, help="Path to evaluation_results.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--viz-only", action="store_true", help="Generate only visualizations"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("🔬 ENHANCED EVALUATION RESULTS ANALYZER")
    print("=" * 80)
    print()

    analyzer = EvaluationAnalyzer(args.results)

    if not analyzer.is_valid:
        print("\n❌ Analysis failed due to invalid data")
        return

    if args.viz_only:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        analyzer.generate_visualizations(output_path)
        print(f"\n✅ Visualizations saved to: {output_path}")
    else:
        analyzer.run_full_analysis(args.output)


if __name__ == "__main__":
    # For scipy.stats compatibility
    from scipy import stats as stats_module

    main()


"""
OUTPUT STRUCTURE:
================
analysis_results/
├── analysis_report.html          # Comprehensive HTML report
├── analysis_summary.json          # Summary statistics
├── per_image_metrics.csv          # Raw data export
├── metric_distributions.png       # Distribution histograms
├── correlation_heatmap.png        # Metric correlations
├── metric_violin_plots.png        # Violin plots
├── lpips_by_style_boxplot.png    # Per-style LPIPS
├── ssim_by_style_boxplot.png     # Per-style SSIM
├── histogram_distance_by_style_boxplot.png
└── fid_by_style_boxplot.png      # Per-style FID

KEY IMPROVEMENTS:
================
1. Type Safety: Full type hints, validation, and error handling
2. Data Validation: Robust checking of all inputs
3. Statistics: Comprehensive stats including IQR, CV, skewness, kurtosis
4. Outlier Detection: IQR and Z-score methods
5. Visualizations: Multiple plot types with professional styling
6. HTML Report: Beautiful, interactive report
7. Export Options: CSV, JSON, HTML formats
8. Error Resilience: Graceful handling of missing/invalid data
"""
