"""
Export Hugging Face dataset back to FontDiffusion directory structure.

Ultra-parallel version with automatic performance optimization.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import argparse

from datasets import Dataset, load_dataset
from PIL import Image
from utilities import HFTqdm
from filename_utils import (
    compute_file_hash,
    get_content_filename,
    get_target_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    output_dir: Path
    repo_id: Optional[str] = None
    local_dataset_path: Optional[Path] = None
    split: str = "train"
    config_name: Optional[str] = None
    token: Optional[str] = None

    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.local_dataset_path, str):
            self.local_dataset_path = Path(self.local_dataset_path)

        if not self.repo_id and not self.local_dataset_path:
            raise ValueError(
                "Must provide either repo_id (Hub) or local_dataset_path (disk)"
            )


class UltraParallelExporter:
    """Ultra-parallel exporter with automatic performance optimization."""

    def __init__(self, config: ExportConfig):
        """Initialize with auto-tuned performance settings."""
        self.config = config
        self.output_dir = config.output_dir
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"
        
        # Auto-tune performance parameters
        self.cpu_count = os.cpu_count() or 4
        self.num_workers = max(1, self.cpu_count * 2)  # Double CPU count for I/O
        self.batch_size = 2000  # Optimal for memory locality
        self.save_workers = max(1, self.cpu_count * 4)  # Many I/O workers
        
        logger.info(f"Auto-tuned: {self.num_workers} metadata workers, "
                   f"{self.save_workers} save workers, batch_size={self.batch_size}")

    def _load_dataset(self) -> Dataset:
        """Load dataset with streaming support."""
        if self.config.local_dataset_path:
            logger.info(f"Loading local dataset from {self.config.local_dataset_path}")
            load_start = time.time()
            dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
            load_time = time.time() - load_start
            logger.info(f"Loaded {len(dataset)} samples from disk in {load_time:.2f}s")
            return dataset

        config_msg = f" (config: {self.config.config_name})" if self.config.config_name else ""
        logger.info(f"Loading dataset from Hub: {self.config.repo_id} "
                   f"(split: {self.config.split}){config_msg}")
        load_start = time.time()
        dataset = load_dataset(
            self.config.repo_id,
            name=self.config.config_name,
            split=self.config.split,
            token=self.config.token,
        )
        load_time = time.time() - load_start
        logger.info(f"Loaded {len(dataset)} samples from Hub in {load_time:.2f}s")
        return dataset

    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")

    def _prepare_sample_batch(self, batch_samples: list) -> tuple:
        """Prepare metadata for a batch of samples (parallelizable)."""
        content_tasks = []
        target_tasks = []
        generations = []
        exported_content = set()
        
        for sample in batch_samples:
            char = sample["character"]
            style = sample["style"]
            font = sample.get("font", "unknown")
            
            # Content image
            content_filename = get_content_filename(char)
            content_img = sample.get("content_image")
            if isinstance(content_img, Image.Image):
                content_path = self.content_dir / content_filename
                if content_filename not in exported_content:
                    content_tasks.append((content_img, content_path))
                    exported_content.add(content_filename)
            
            # Target image
            target_img = sample.get("target_image")
            if isinstance(target_img, Image.Image):
                style_dir = self.target_dir / style
                style_dir.mkdir(parents=True, exist_ok=True)
                target_filename = get_target_filename(char, style)
                target_path = style_dir / target_filename
                target_tasks.append((target_img, target_path))
            
            # Generation record
            generations.append({
                "character": char,
                "style": style,
                "font": font,
                "content_image_path": f"ContentImage/{content_filename}",
                "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            })
        
        return content_tasks, target_tasks, generations

    def _save_image_batch(self, image_tasks: list) -> int:
        """Save a batch of images in parallel."""
        def save_single(args):
            img, path = args
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
                return True
            except Exception as e:
                logger.warning(f"Failed to save {path}: {e}")
                return False
        
        with ThreadPoolExecutor(max_workers=self.save_workers) as executor:
            futures = [executor.submit(save_single, task) for task in image_tasks]
            return sum(future.result() for future in as_completed(futures))

    def export(self) -> dict[str, Any]:
        """Execute ultra-parallel export process."""
        logger.info("Starting ultra-parallel dataset export...")
        total_start = time.time()

        # Phase 1: Load dataset
        dataset = self._load_dataset()
        self._create_directories()
        
        dataset_size = len(dataset)
        all_generations = []
        exported_content = set()
        total_saved = 0
        
        # Phase 2: Process in parallel batches
        logger.info(f"Processing {dataset_size} samples in ultra-parallel mode...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as process_pool:
            # Submit all batches for parallel processing
            batch_futures = []
            for batch_start in range(0, dataset_size, self.batch_size):
                batch_end = min(batch_start + self.batch_size, dataset_size)
                batch = [dataset[i] for i in range(batch_start, batch_end)]
                future = process_pool.submit(self._prepare_sample_batch, batch)
                batch_futures.append((future, batch_start, batch_end))
            
            # Process completed batches
            for future, batch_start, batch_end in batch_futures:
                batch_start_time = time.time()
                
                # Get prepared batch data
                content_tasks, target_tasks, batch_generations = future.result()
                all_generations.extend(batch_generations)
                
                # Save images in parallel
                all_tasks = content_tasks + target_tasks
                saved_count = self._save_image_batch(all_tasks)
                total_saved += saved_count
                
                # Track exported content for deduplication
                for img, path in content_tasks:
                    exported_content.add(path.name)
                
                # Progress logging
                batch_time = time.time() - batch_start_time
                elapsed = time.time() - total_start
                progress_pct = batch_end / dataset_size * 100
                rate = batch_end / elapsed if elapsed > 0 else 0
                eta = (dataset_size - batch_end) / rate if rate > 0 else 0
                
                logger.info(
                    f"Batch {batch_start}-{batch_end}: "
                    f"{saved_count}/{len(all_tasks)} images in {batch_time:.2f}s, "
                    f"Overall: {progress_pct:.1f}%, {rate:.1f} samples/s, ETA: {eta:.0f}s"
                )
        
        # Phase 3: Save metadata
        checkpoint_path = self.output_dir / "results_checkpoint.json"
        characters = sorted({g["character"] for g in all_generations})
        styles = sorted({g["style"] for g in all_generations})
        fonts = sorted({g["font"] for g in all_generations if g["font"] != "unknown"})
        
        metadata = {
            "generations": all_generations,
            "characters": characters,
            "styles": styles,
            "fonts": fonts if fonts else ["unknown"],
            "total_chars": len(characters),
            "total_styles": len(styles),
        }
        
        logger.info("Writing checkpoint file...")
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Final statistics
        total_time = time.time() - total_start
        logger.info(
            f"Export completed in {total_time:.2f}s\n"
            f"• {len(exported_content)} unique content images\n"
            f"• {len(all_generations)} target images\n"
            f"• {len(characters)} characters across {len(styles)} styles\n"
            f"• Overall rate: {dataset_size/total_time:.1f} samples/s"
        )
        
        return metadata


def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    split: str = "train",
    config_name: Optional[str] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Ultra-parallel dataset export with automatic optimization.
    
    Just provide the essential arguments - performance is auto-tuned!
    """
    config = ExportConfig(
        output_dir=Path(output_dir),
        repo_id=repo_id,
        local_dataset_path=Path(local_dataset_path) if local_dataset_path else None,
        split=split,
        config_name=config_name,
        token=token,
    )
    
    exporter = UltraParallelExporter(config)
    return exporter.export()


def main():
    """Simple CLI entry point - just provide essential arguments."""
    parser = argparse.ArgumentParser(
        description="Ultra-parallel HuggingFace dataset exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from HuggingFace Hub (auto-tuned for max speed)
  python export_ultra_parallel.py --output-dir ./output --repo-id username/dataset-name
  
  # Export from local cache
  python export_ultra_parallel.py --output-dir ./output --local-path /path/to/dataset
  
  # Export specific split and config
  python export_ultra_parallel.py --output-dir ./output --repo-id username/dataset-name \\
    --split test --config-name v2
    
  # Private dataset with token
  python export_ultra_parallel.py --output-dir ./output --repo-id org/private-dataset \\
    --token hf_xxxxxx
        """,
    )
    
    # Essential arguments
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to export to (will be created if it doesn't exist)"
    )
    
    # Dataset source (exactly one required)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--repo-id",
        help="HuggingFace repository ID (e.g., 'username/dataset-name')"
    )
    source_group.add_argument(
        "--local-path",
        help="Local dataset path (alternative to --repo-id)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--config-name",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token for private datasets"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run export with auto-tuned parallelization
        metadata = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            config_name=args.config_name,
            token=args.token,
        )
        
        # Success summary
        print(f"\n✅ Export completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🔤 Characters: {metadata['total_chars']}")
        print(f"🎨 Styles: {metadata['total_styles']}")
        print(f"🖼️  Images: {len(metadata['generations'])}")
        
    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()