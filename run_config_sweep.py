"""
Configuration sweep for FontDiffuser inference with multi-GPU support.

Runs sample_distributed.py with different combinations of:
  - num_inference_steps
  - guidance_scale (if applicable)
  - Other configurable parameters

Automatically organizes results and runs evaluation analysis on each configuration.

Usage:
  python scripts/run_config_sweep.py \\
    --config_combinations configs/sweep_config.json \\
    --output_base sweep_results/ \\
    --num_processes 4
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from itertools import product

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.utilities import HFTqdm
from analyze_evaluation_results import EvaluationAnalyzer

logger = logging.getLogger("ConfigSweep")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class InferenceConfig:
    """Single inference configuration."""
    num_inference_steps: int
    guidance_scale: float
    t_start: float = 0.0
    t_end: float = 1.0
    order: int = 2
    algorithm_type: str = "dpmsolver++"
    skip_type: str = "time-uniform"
    method: str = "multistep"
    correcting_x0_fn: Optional[str] = None
    
    def get_name(self) -> str:
        """Get configuration name for directory."""
        return f"steps{self.num_inference_steps}_gs{self.guidance_scale:.1f}"
    
    def get_display_name(self) -> str:
        """Get human-readable config name."""
        return f"Steps={self.num_inference_steps}, GS={self.guidance_scale:.1f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        return asdict(self)


@dataclass
class SweepResult:
    """Result of a single configuration run."""
    config: InferenceConfig
    output_dir: Path
    inference_time: float
    generation_success: bool
    evaluation_success: bool
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'config_name': self.config.get_name(),
            'output_dir': str(self.output_dir),
            'inference_time': self.inference_time,
            'generation_success': self.generation_success,
            'evaluation_success': self.evaluation_success,
            'metrics': self.metrics,
            'error_message': self.error_message,
        }


class ConfigSweepRunner:
    """Manages configuration sweep execution."""
    
    def __init__(
        self,
        base_output_dir: Path,
        base_args: Dict[str, Any],
        num_processes: int = 1,
        dry_run: bool = False,
    ):
        """Initialize sweep runner.
        
        Args:
            base_output_dir: Base directory for all sweep results
            base_args: Base arguments for sample_distributed.py
            num_processes: Number of GPU processes
            dry_run: If True, don't actually run commands
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_args = base_args
        self.num_processes = num_processes
        self.dry_run = dry_run
        self.results: List[SweepResult] = []
        
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Sweep output directory: {self.base_output_dir}")
    
    def _build_command(
        self,
        config: InferenceConfig,
        output_dir: Path,
    ) -> List[str]:
        """Build accelerate launch command for a configuration.
        
        Args:
            config: Inference configuration
            output_dir: Output directory for this config
            
        Returns:
            List of command arguments
        """
        # Base accelerate command for multi-GPU
        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "FontDiffusion/run_inference.py",
            "--mode", "sample_distributed",
        ]
        
        # Add base arguments
        for key, value in self.base_args.items():
            if value is not None and value is not False:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        # Override with sweep configuration
        cmd.extend([
            "--output_dir", str(output_dir),
            "--num_inference_steps", str(config.num_inference_steps),
            "--t_start", str(config.t_start),
            "--t_end", str(config.t_end),
            "--order", str(config.order),
            "--algorithm_type", config.algorithm_type,
            "--skip_type", config.skip_type,
            "--method", config.method,
        ])
        
        if config.correcting_x0_fn:
            cmd.extend(["--correcting_x0_fn", config.correcting_x0_fn])
        
        return cmd
    
    def run_config(
        self,
        config: InferenceConfig,
        config_idx: int,
        total_configs: int,
    ) -> SweepResult:
        """Run a single configuration.
        
        Args:
            config: Configuration to run
            config_idx: Current configuration index
            total_configs: Total number of configurations
            
        Returns:
            SweepResult with execution results
        """
        config_name = config.get_name()
        output_dir = self.base_output_dir / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info(f"[{config_idx + 1}/{total_configs}] Running: {config.get_display_name()}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)
        
        result = SweepResult(
            config=config,
            output_dir=output_dir,
            inference_time=0.0,
            generation_success=False,
            evaluation_success=False,
        )
        
        try:
            # Build command
            cmd = self._build_command(config, output_dir)
            
            if self.dry_run:
                logger.info(f"DRY RUN: {' '.join(cmd)}")
                result.generation_success = True
                return result
            
            # Run generation
            logger.info(f"Running: {' '.join(cmd)}")
            start_time = datetime.now()
            
            proc = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,  # Run from repo root
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            result.inference_time = elapsed
            
            if proc.returncode != 0:
                error_msg = f"Generation failed with code {proc.returncode}"
                logger.error(error_msg)
                logger.error(f"STDOUT:\n{proc.stdout[-500:]}")  # Last 500 chars
                logger.error(f"STDERR:\n{proc.stderr[-500:]}")
                result.error_message = error_msg
                return result
            
            logger.info(f"✓ Generation completed in {elapsed:.2f}s")
            result.generation_success = True
            
            # Run evaluation
            logger.info("Running evaluation analysis...")
            results_json = output_dir / "results_checkpoint.json"
            
            if not results_json.exists():
                logger.warning(f"Results file not found: {results_json}")
                return result
            
            # Run analysis
            analyzer = EvaluationAnalyzer(str(results_json))
            
            if not analyzer.is_valid:
                logger.warning("Evaluation data is invalid")
                return result
            
            # Extract metrics
            global_metrics = analyzer.analyze_global_metrics()
            result.metrics = {
                metric_name: {
                    'mean': stats.mean,
                    'median': stats.median,
                    'std': stats.std,
                    'count': stats.count,
                }
                for metric_name, stats in global_metrics.items()
            }
            
            result.evaluation_success = True
            logger.info(f"✓ Evaluation completed")
            
        except subprocess.TimeoutExpired:
            result.error_message = "Execution timeout (1 hour)"
            logger.error(result.error_message)
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Error: {e}", exc_info=True)
        
        return result
    
    def run_sweep(self, configs: List[InferenceConfig]) -> List[SweepResult]:
        """Run sweep over all configurations.
        
        Args:
            configs: List of configurations to run
            
        Returns:
            List of results
        """
        logger.info("=" * 80)
        logger.info(f"Starting configuration sweep with {len(configs)} configurations")
        logger.info("=" * 80)
        
        self.results = []
        
        for idx, config in enumerate(HFTqdm(configs, desc="🔄 Running sweep", colour="cyan")):
            result = self.run_config(config, idx, len(configs))
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
        
        logger.info("=" * 80)
        logger.info("✅ Sweep completed!")
        logger.info("=" * 80)
        
        return self.results
    
    def _save_results(self) -> None:
        """Save sweep results to JSON."""
        results_json = self.base_output_dir / "sweep_results.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'num_processes': self.num_processes,
            'total_configs': len(self.results),
            'successful': sum(1 for r in self.results if r.generation_success),
            'evaluated': sum(1 for r in self.results if r.evaluation_success),
            'results': [r.to_dict() for r in self.results],
        }
        
        with open(results_json, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_json}")


class SweepComparator:
    """Compares results across configurations."""
    
    def __init__(self, results: List[SweepResult]):
        """Initialize comparator.
        
        Args:
            results: List of sweep results
        """
        self.results = results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison DataFrame.
        
        Returns:
            DataFrame with all results
        """
        data = []
        
        for result in self.results:
            row = {
                'Config': result.config.get_display_name(),
                'Steps': result.config.num_inference_steps,
                'GuidanceScale': result.config.guidance_scale,
                'InferenceTime(s)': result.inference_time,
                'Success': '✓' if result.generation_success else '✗',
                'Evaluated': '✓' if result.evaluation_success else '✗',
            }
            
            if result.metrics:
                for metric_name, metric_data in result.metrics.items():
                    row[f'{metric_name}_mean'] = metric_data.get('mean', None)
                    row[f'{metric_name}_std'] = metric_data.get('std', None)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_comparison(self) -> None:
        """Print formatted comparison table."""
        df = self.create_comparison_table()
        
        print("\n" + "=" * 120)
        print("SWEEP COMPARISON TABLE")
        print("=" * 120 + "\n")
        
        print(df.to_string(index=False))
        print()
    
    def find_best_config(self, metric: str = 'lpips_mean', lower_is_better: bool = True) -> Optional[SweepResult]:
        """Find best configuration by metric.
        
        Args:
            metric: Metric column name
            lower_is_better: If True, lower values are better
            
        Returns:
            Best result or None
        """
        df = self.create_comparison_table()
        
        # Filter only evaluated results
        df_eval = df[df['Evaluated'] == '✓'].copy()
        
        if df_eval.empty or metric not in df_eval.columns:
            return None
        
        # Remove NaN values
        df_eval = df_eval.dropna(subset=[metric])
        
        if df_eval.empty:
            return None
        
        idx = df_eval[metric].idxmin() if lower_is_better else df_eval[metric].idxmax()
        best_row = df_eval.iloc[idx]
        
        # Find corresponding result
        config_name = best_row['Config']
        for result in self.results:
            if result.config.get_display_name() == config_name:
                return result
        
        return None
    
    def save_comparison_csv(self, output_path: Path) -> None:
        """Save comparison to CSV.
        
        Args:
            output_path: Path to save CSV
        """
        df = self.create_comparison_table()
        df.to_csv(output_path, index=False)
        logger.info(f"Comparison saved to {output_path}")
    
    def generate_summary_report(self, output_dir: Path) -> None:
        """Generate comprehensive summary report.
        
        Args:
            output_dir: Directory to save report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        self.save_comparison_csv(output_dir / 'sweep_comparison.csv')
        
        # Save text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SWEEP COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        df = self.create_comparison_table()
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
        
        # Best configurations
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BEST CONFIGURATIONS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for metric in ['lpips_mean', 'ssim_mean', 'histogram_distance_mean']:
            lower_better = metric != 'ssim_mean'
            best = self.find_best_config(metric, lower_better)
            
            if best:
                direction = "lower" if lower_better else "higher"
                report_lines.append(
                    f"Best {metric.replace('_', ' ').title()} "
                    f"({direction} is better): {best.config.get_display_name()}"
                )
                if best.metrics:
                    if metric in [f'{k}_mean' for k in best.metrics.keys()]:
                        val = best.metrics[metric.replace('_mean', '')].get('mean', 'N/A')
                        report_lines.append(f"  Value: {val}")
        
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("\n" + "=" * 80)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        successful = sum(1 for r in self.results if r.generation_success)
        evaluated = sum(1 for r in self.results if r.evaluation_success)
        
        report_lines.append(f"Total configurations: {len(self.results)}")
        report_lines.append(f"Successful generations: {successful}")
        report_lines.append(f"Evaluated: {evaluated}")
        
        if self.results:
            inference_times = [r.inference_time for r in self.results if r.inference_time > 0]
            if inference_times:
                report_lines.append(f"Avg inference time: {np.mean(inference_times):.2f}s")
                report_lines.append(f"Min inference time: {np.min(inference_times):.2f}s")
                report_lines.append(f"Max inference time: {np.max(inference_times):.2f}s")
        
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        report_path = output_dir / 'sweep_summary.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        print("\n" + report_text)


def generate_config_combinations(
    steps_list: List[int],
    guidance_scales: List[float],
) -> List[InferenceConfig]:
    """Generate all configuration combinations.
    
    Args:
        steps_list: List of inference steps
        guidance_scales: List of guidance scales
        
    Returns:
        List of configurations
    """
    configs = []
    
    for steps, gs in product(steps_list, guidance_scales):
        config = InferenceConfig(
            num_inference_steps=steps,
            guidance_scale=gs,
        )
        configs.append(config)
    
    return configs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run configuration sweep for FontDiffuser inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sweep with predefined configs
  python FontDiffusion/run_config_sweep.py \\
    --steps 10 20 30 \\
    --guidance_scales 1.0 1.5 2.0 \\
    --output sweep_results/ \\
    --num_processes 4

  # Dry run to test commands
  python FontDiffusion/run_config_sweep.py \\
    --steps 10 20 \\
    --guidance_scales 1.0 \\
    --dry_run

  # Run with custom base arguments
  python FontDiffusion/run_config_sweep.py \\
    --steps 15 20 25 \\
    --guidance_scales 0.5 1.0 1.5 \\
    --output sweep_results/ \\
    --characters data/characters.txt \\
    --style_images data/styles/ \\
    --ttf_path data/fonts/
        """
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[10, 15, 20],
        help="Inference steps to test"
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
        help="Guidance scales to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sweep_results",
        help="Base output directory"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of GPU processes"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (show commands without executing)"
    )
    
    # Base arguments for sample_distributed.py
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpt",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--characters",
        type=str,
        default="data/characters.txt",
        help="Characters file"
    )
    parser.add_argument(
        "--style_images",
        type=str,
        default="data/styles/",
        help="Style images directory"
    )
    parser.add_argument(
        "--ttf_path",
        type=str,
        default="data/fonts/",
        help="TTF fonts directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation"
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Ground truth directory for evaluation"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate configurations
    configs = generate_config_combinations(args.steps, args.guidance_scales)
    
    logger.info(f"Generated {len(configs)} configurations")
    for config in configs:
        logger.info(f"  - {config.get_display_name()}")
    
    # Prepare base arguments
    base_args = {
        'ckpt_dir': args.ckpt_dir,
        'characters': args.characters,
        'style_images': args.style_images,
        'ttf_path': args.ttf_path,
        'batch_size': args.batch_size,
        'device': args.device,
        'fp16': args.fp16,
        'evaluate': args.evaluate,
        'ground_truth_dir': args.ground_truth_dir,
    }
    
    # Create sweep runner
    runner = ConfigSweepRunner(
        base_output_dir=Path(args.output),
        base_args=base_args,
        num_processes=args.num_processes,
        dry_run=args.dry_run,
    )
    
    # Run sweep
    results = runner.run_sweep(configs)
    
    # Compare results
    comparator = SweepComparator(results)
    comparator.print_comparison()
    comparator.generate_summary_report(Path(args.output) / "comparison")
    
    logger.info(f"✅ Sweep complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()