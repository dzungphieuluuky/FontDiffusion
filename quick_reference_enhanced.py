#!/usr/bin/env python3
"""
Quick reference guide and example runner for enhanced FontDiffuser training.

This script provides:
1. Pre-configured training commands
2. Quick benchmark setups
3. Configuration templates
4. Validation helper functions
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import dict, list, Optional


# Pre-configured training setups
TRAINING_CONFIGS = {
    "baseline": {
        "description": "Standard FST training without auxiliary losses",
        "use_aux_losses": False,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_basic": {
        "description": "Enhanced training with recommended defaults",
        "use_aux_losses": True,
        "aux_freq_band": True,
        "aux_stroke_topo": True,
        "aux_freq_diff": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_aggressive": {
        "description": "Aggressive stroke emphasis (high loss weights)",
        "use_aux_losses": True,
        "aux_freq_weight": 1.0,
        "aux_topo_weight": 0.5,
        "aux_fw_max_weight": 5.0,
        "aux_topo_topology_weight": 1.5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_conservative": {
        "description": "Conservative auxiliary loss weights (safe for tuning)",
        "use_aux_losses": True,
        "aux_freq_weight": 0.3,
        "aux_topo_weight": 0.2,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_with_annealing": {
        "description": "Enhanced training with temperature annealing",
        "use_aux_losses": True,
        "aux_anneal_temperature": True,
        "aux_temperature_schedule": "linear",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_freq_only": {
        "description": "Only frequency band loss (content/style separation)",
        "use_aux_losses": True,
        "aux_freq_band": True,
        "aux_stroke_topo": False,
        "aux_freq_diff": False,
        "aux_freq_weight": 0.8,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "enhanced_topo_only": {
        "description": "Only stroke topology loss",
        "use_aux_losses": True,
        "aux_freq_band": False,
        "aux_stroke_topo": True,
        "aux_freq_diff": False,
        "aux_topo_weight": 0.5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 100000,
        "learning_rate": "5e-5",
    },
    "quick_test": {
        "description": "Quick test configuration (10 steps for debugging)",
        "use_aux_losses": True,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_train_steps": 10,
        "log_interval": 1,
        "ckpt_interval": 5,
        "learning_rate": "5e-5",
    },
}


def config_to_args(config: dict) -> list[str]:
    """Convert configuration dictionary to command-line arguments.

    Args:
        config: Configuration dictionary

    Returns:
        List of command-line arguments
    """
    args = []

    # Map config keys to CLI argument names
    key_map = {
        "batch_size": "--train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "max_train_steps": "--max_train_steps",
        "learning_rate": "--learning_rate",
        "log_interval": "--log_interval",
        "ckpt_interval": "--ckpt_interval",
    }

    for key, value in config.items():
        if key in key_map:
            args.append(key_map[key])
            args.append(str(value))
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif key not in ["description"]:
            args.append(f"--{key}")
            args.append(str(value))

    return args


def build_command(
    config_name: str,
    config: dict,
    data_root: str,
    output_dir: str,
    use_fst: bool = True,
    multi_gpu: Optional[int] = None,
    experience_name: Optional[str] = None,
    mixed_precision: str = "fp16",
) -> str:
    """Build complete training command.

    Args:
        config_name: Name of configuration
        config: Configuration dictionary
        data_root: Path to training data
        output_dir: Path for outputs
        use_fst: Enable FST (default: True)
        multi_gpu: Number of GPUs for multi-GPU training (None = single GPU)
        experience_name: Custom experiment name (uses config_name if None)
        mixed_precision: Mixed precision type (default: fp16)

    Returns:
        Complete accelerate launch command string
    """
    if experience_name is None:
        experience_name = f"enhanced_{config_name}"

    # Base accelerate command
    if multi_gpu:
        cmd = f"accelerate launch --multi_gpu --num_processes={multi_gpu}"
    else:
        cmd = "accelerate launch"

    # Add script
    cmd += " train_fst_enhanced.py"

    # Add core arguments
    cmd += f" --use_fst" if use_fst else ""
    cmd += f" --experience_name={experience_name}"
    cmd += f" --data_root={data_root}"
    cmd += f" --output_dir={output_dir}"
    cmd += f" --mixed_precision={mixed_precision}"

    # Add config-specific arguments
    for arg in config_to_args(config):
        cmd += f" {arg}"

    return cmd


def print_configs():
    """Print all available configurations."""
    print("\n" + "=" * 80)
    print("Available Training Configurations")
    print("=" * 80)
    for name, config in TRAINING_CONFIGS.items():
        description = config.get("description", "No description")
        print(f"\n{name:30s} - {description}")
        print(f"  Usage: python quick_reference.py --config {name}")

    print("\n" + "=" * 80)


def print_command(
    config_name: str,
    data_root: str = "my_dataset",
    output_dir: str = "outputs",
    **kwargs
):
    """Print example command for a configuration.

    Args:
        config_name: Name of configuration
        data_root: Path to data
        output_dir: Path for outputs
        **kwargs: Additional arguments
    """
    if config_name not in TRAINING_CONFIGS:
        print(f"Error: Configuration '{config_name}' not found")
        print_configs()
        return

    config = TRAINING_CONFIGS[config_name]
    cmd = build_command(
        config_name,
        config,
        data_root,
        f"{output_dir}/{config_name}",
        **kwargs,
    )

    print("\n" + "=" * 80)
    print(f"Command for configuration: {config_name}")
    print("=" * 80)
    print(f"Description: {config.get('description', 'N/A')}")
    print("\n" + cmd + "\n")
    print("=" * 80)


def create_benchmark_script(
    output_file: str = "benchmark_enhanced_training.sh",
    data_root: str = "my_dataset",
    results_dir: str = "benchmark_results",
):
    """Create a shell script to run benchmark comparisons.

    Args:
        output_file: Output script filename
        data_root: Path to training data
        results_dir: Directory for benchmark results
    """
    configs = ["baseline", "enhanced_basic", "enhanced_aggressive", "enhanced_conservative"]

    script_content = """#!/bin/bash
# Auto-generated benchmark script for enhanced FontDiffuser training
# Compares baseline and various enhanced configurations

set -e

echo "=========================================="
echo "FontDiffuser Enhanced Training Benchmark"
echo "=========================================="

"""

    for config_name in configs:
        if config_name not in TRAINING_CONFIGS:
            continue

        config = TRAINING_CONFIGS[config_name]
        output_path = f"{results_dir}/{config_name}"

        cmd = build_command(
            config_name,
            config,
            data_root,
            output_path,
            use_fst=True,
            multi_gpu=None,
            experience_name=f"bench_{config_name}",
            mixed_precision="fp16",
        )

        script_content += f"""
echo ""
echo "=========================================="
echo "Running: {config_name}"
echo "=========================================="
echo "{config.get('description', '')}"
echo ""

{cmd}

echo ""
echo "Completed: {config_name}"
echo "Results saved to: {output_path}"
echo ""
"""

    script_content += """
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Compare results in benchmark_results/ directory"
echo ""
echo "To analyze results:"
echo "  - Check loss curves in TensorBoard"
echo "  - Compare final checkpoint metrics"
echo "  - Evaluate visual quality on validation set"
"""

    with open(output_file, "w") as f:
        f.write(script_content)

    print(f"✓ Benchmark script created: {output_file}")
    print(f"  Run with: bash {output_file}")


def suggest_config(
    dataset_size: Optional[int] = None,
    hardware: str = "single_gpu",
    priority: str = "quality",
) -> str:
    """Suggest a configuration based on hardware and priorities.

    Args:
        dataset_size: Number of training samples (optional)
        hardware: "single_gpu", "multi_gpu", or "colab"
        priority: "speed", "quality", or "balanced"

    Returns:
        Recommended configuration name
    """
    if hardware == "colab":
        if priority in ["speed", "balanced"]:
            return "enhanced_conservative"
        else:
            return "enhanced_basic"

    elif hardware == "multi_gpu":
        if priority == "speed":
            return "enhanced_conservative"
        elif priority == "quality":
            return "enhanced_aggressive"
        else:
            return "enhanced_with_annealing"

    else:  # single_gpu
        if priority == "speed":
            return "enhanced_freq_only"
        elif priority == "quality":
            return "enhanced_basic"
        else:
            return "enhanced_conservative"


def validate_config(config_name: str) -> bool:
    """Validate a configuration.

    Args:
        config_name: Configuration name to validate

    Returns:
        True if valid, False otherwise
    """
    if config_name not in TRAINING_CONFIGS:
        print(f"Error: Configuration '{config_name}' not found")
        return False

    config = TRAINING_CONFIGS[config_name]

    # Check for required fields
    if config.get("description") is None:
        print(f"Warning: {config_name} missing description")
        return False

    print(f"✓ Configuration '{config_name}' is valid")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Quick reference and example runner for enhanced FontDiffuser training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all configurations
  python quick_reference.py --list

  # Print command for a specific configuration
  python quick_reference.py --config enhanced_basic

  # Create benchmark script
  python quick_reference.py --benchmark --output benchmark.sh

  # Get recommendation for your hardware
  python quick_reference.py --suggest --hardware colab --priority quality

  # Validate a configuration
  python quick_reference.py --validate enhanced_aggressive
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Print command for specific configuration",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="my_dataset",
        help="Path to training data (default: my_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Create benchmark comparison script",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_enhanced_training.sh",
        help="Output file for benchmark script",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Get configuration recommendation",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="single_gpu",
        choices=["single_gpu", "multi_gpu", "colab"],
        help="Hardware setup (default: single_gpu)",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="balanced",
        choices=["speed", "quality", "balanced"],
        help="Training priority (default: balanced)",
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate a configuration",
    )
    parser.add_argument(
        "--multi-gpu",
        type=int,
        help="Number of GPUs for multi-GPU training",
    )

    args = parser.parse_args()

    if args.list:
        print_configs()

    elif args.validate:
        validate_config(args.validate)

    elif args.suggest:
        recommended = suggest_config(
            hardware=args.hardware,
            priority=args.priority,
        )
        print(f"\n✓ Recommended configuration for {args.hardware} ({args.priority}):")
        print(f"  {recommended}")
        if recommended in TRAINING_CONFIGS:
            config = TRAINING_CONFIGS[recommended]
            print(f"  {config.get('description', '')}")

    elif args.benchmark:
        create_benchmark_script(
            args.output,
            args.data_root,
            "benchmark_results",
        )

    elif args.config:
        print_command(
            args.config,
            args.data_root,
            args.output_dir,
            multi_gpu=args.multi_gpu,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
