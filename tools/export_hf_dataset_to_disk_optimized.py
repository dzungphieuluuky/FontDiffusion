"""
High-performance export of Hugging Face dataset back to FontDiffusion directory structure.

This module reconstructs the original directory layout from a HuggingFace dataset,
with parallel processing and efficient I/O operations.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias
import multiprocessing as mp
import threading
from queue import Queue

import psutil
from datasets import Dataset, DatasetDict, load_dataset, IterableDataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from fsspec.implementations.local import LocalFileSystem
import pickle
import hashlib

# Local imports
from utilities import get_hf_bar
from filename_utils import compute_file_hash, get_content_filename, get_target_filename

logger = logging.getLogger("DatasetExporter")

# Type aliases for modern Python
ImageType: TypeAlias = Image.Image
DatasetSplit: TypeAlias = str | Dataset | IterableDataset
ExportStats: TypeAlias = dict[str, int | list[str]]


@dataclass
class ExportConfig:
    """High-performance configuration for dataset export."""
    
    output_dir: Path
    repo_id: str | None = None
    local_dataset_path: Path | None = None
    split: str = "train"
    token: str | None = None
    num_workers: int = field(default_factory=lambda: min(16, mp.cpu_count()))
    batch_size: int = 100
    use_compression: bool = True
    image_quality: int = 95
    image_format: str = "PNG"
    verify_integrity: bool = True
    max_memory_mb: int = 4096
    
    def __post_init__(self) -> None:
        """Validate and convert paths with performance optimizations."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.local_dataset_path, str):
            self.local_dataset_path = Path(self.local_dataset_path)
        
        if not self.repo_id and not self.local_dataset_path:
            raise ValueError("Must provide either repo_id (Hub) or local_dataset_path (disk)")
        
        # Optimize workers based on system resources
        available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
        if available_memory < self.max_memory_mb:
            self.num_workers = max(1, self.num_workers // 2)
            logger.info(f"Reduced workers to {self.num_workers} due to memory constraints")
        
        # Create output directory early to avoid race conditions
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ParallelImageSaver:
    """High-performance parallel image saver with smart batching."""
    
    def __init__(self, max_workers: int, batch_size: int, quality: int = 95, format: str = "PNG"):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.quality = quality
        self.format = format
        self._saved_files = set()
        self._lock = threading.Lock()
        self._fs = LocalFileSystem()
    
    def save_images_batch(self, images_batch: list[tuple[Path, Image.Image]]) -> tuple[int, list[str]]:
        """Save a batch of images in parallel."""
        saved = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for path, img in images_batch:
                # Check if already saved (avoid duplicates)
                with self._lock:
                    if str(path) in self._saved_files:
                        continue
                
                future = executor.submit(self._save_single_image, path, img)
                futures[future] = path
            
            # Collect results
            for future in as_completed(futures):
                path = futures[future]
                try:
                    success = future.result()
                    if success:
                        saved += 1
                        with self._lock:
                            self._saved_files.add(str(path))
                except Exception as e:
                    errors.append(f"{path}: {e}")
        
        return saved, errors
    
    def _save_single_image(self, path: Path, image: Image.Image) -> bool:
        """Save a single image with error handling."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with optimized parameters
            if self.format == "JPEG":
                image.save(path, format=self.format, quality=self.quality, optimize=True, progressive=True)
            else:
                image.save(path, format=self.format, optimize=True)
            
            # Verify the saved file
            if path.stat().st_size > 0:
                return True
            else:
                path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            logger.debug(f"Failed to save {path}: {e}")
            return False


class MetadataCollector:
    """Efficient metadata collection with deduplication."""
    
    def __init__(self):
        self.generations: list[dict[str, Any]] = []
        self.characters: set[str] = set()
        self.styles: set[str] = set()
        self.fonts: set[str] = set()
        self.content_files: set[str] = set()
        self._lock = threading.Lock()
    
    def add_generation(self, 
                      char: str, 
                      style: str, 
                      font: str, 
                      content_path: str, 
                      target_path: str) -> None:
        """Add a generation record with thread safety."""
        with self._lock:
            # Compute hashes
            content_hash = compute_file_hash(char, "", font)
            target_hash = compute_file_hash(char, style, font)
            
            self.generations.append({
                "character": char,
                "style": style,
                "font": font,
                "content_image_path": content_path,
                "target_image_path": target_path,
                "content_hash": content_hash,
                "target_hash": target_hash,
            })
            
            # Update sets
            self.characters.add(char)
            self.styles.add(style)
            if font != "unknown":
                self.fonts.add(font)
            
            # Track content files for deduplication
            self.content_files.add(content_path)
    
    def get_metadata(self) -> ExportStats:
        """Get final metadata with sorted collections."""
        return {
            "generations": self.generations,
            "characters": sorted(self.characters),
            "styles": sorted(self.styles),
            "fonts": sorted(self.fonts) if self.fonts else ["unknown"],
            "total_chars": len(self.characters),
            "total_styles": len(self.styles),
            "total_generations": len(self.generations),
            "unique_content_files": len(self.content_files),
        }


class DatasetExporter:
    """High-performance dataset exporter with parallel processing."""
    
    def __init__(self, config: ExportConfig):
        """Initialize the high-performance exporter."""
        self.config = config
        self.output_dir = config.output_dir
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"
        
        # Performance components
        self.image_saver = ParallelImageSaver(
            max_workers=config.num_workers,
            batch_size=config.batch_size,
            quality=config.image_quality,
            format=config.image_format
        )
        self.metadata = MetadataCollector()
        
        # Statistics
        self.stats = {
            "content_images_saved": 0,
            "target_images_saved": 0,
            "errors": [],
            "skipped_duplicates": 0,
        }
    
    def _load_dataset(self) -> Dataset:
        """Load dataset with streaming support for large datasets."""
        if self.config.local_dataset_path:
            logger.info(f"Loading local dataset from {self.config.local_dataset_path}")
            
            try:
                # Try streaming first for large datasets
                if (self.config.local_dataset_path / "dataset_info.json").exists():
                    dataset = load_dataset(
                        str(self.config.local_dataset_path),
                        split=self.config.split,
                        streaming=True  # Use streaming for memory efficiency
                    )
                    # Convert to regular dataset for processing
                    # Note: For very large datasets, keep as streaming and process in batches
                    dataset = Dataset.from_generator(
                        lambda: iter(dataset),
                        features=dataset.features
                    )
                else:
                    dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
                
                logger.info(f"Loaded {len(dataset):,} samples from disk")
                return dataset
                
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e
        
        # Load from Hugging Face Hub
        logger.info(f"Loading dataset from Hub: {self.config.repo_id} (split: {self.config.split})")
        
        try:
            # Use streaming for large Hub datasets
            dataset = load_dataset(
                self.config.repo_id,
                split=self.config.split,
                token=self.config.token,
                streaming=True  # Stream to avoid downloading entire dataset
            )
            
            # Convert to regular dataset for batch processing
            dataset = Dataset.from_generator(
                lambda: iter(dataset),
                features=dataset.features
            )
            
            logger.info(f"Loaded {len(dataset):,} samples from Hub")
            return dataset
            
        except Exception as e:
            raise ValueError(f"Failed to load from Hub {self.config.repo_id}: {e}") from e
    
    def _process_batch(self, batch: list[dict[str, Any]]) -> tuple[list, list, list]:
        """Process a batch of samples into save operations."""
        content_batch = []
        target_batch = []
        metadata_batch = []
        
        for sample in batch:
            char = sample.get("character", "").strip()
            style = sample.get("style", "").strip()
            font = sample.get("font", "unknown").strip()
            
            if not char or not style:
                continue
            
            # Prepare content image path
            content_filename = get_content_filename(char)
            content_path = self.content_dir / content_filename
            
            # Add to content batch if not already in metadata
            content_img = sample.get("content_image")
            if isinstance(content_img, Image.Image):
                content_batch.append((content_path, content_img))
            
            # Prepare target image path
            style_dir = self.target_dir / style
            target_filename = get_target_filename(char, style)
            target_path = style_dir / target_filename
            
            # Add to target batch
            target_img = sample.get("target_image")
            if isinstance(target_img, Image.Image):
                target_batch.append((target_path, target_img))
            
            # Prepare metadata
            metadata_batch.append({
                "char": char,
                "style": style,
                "font": font,
                "content_path": f"ContentImage/{content_filename}",
                "target_path": f"TargetImage/{style}/{target_filename}",
            })
        
        return content_batch, target_batch, metadata_batch
    
    def _export_batch_parallel(self, dataset: Dataset) -> ExportStats:
        """Export dataset in parallel batches for maximum performance."""
        logger.info(f"Starting parallel export of {len(dataset):,} samples...")
        
        # Process in batches
        total_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                
                # Process batch
                content_batch, target_batch, metadata_batch = self._process_batch(batch)
                
                # Save images in parallel
                if content_batch:
                    saved, errors = self.image_saver.save_images_batch(content_batch)
                    self.stats["content_images_saved"] += saved
                    self.stats["errors"].extend(errors)
                
                if target_batch:
                    saved, errors = self.image_saver.save_images_batch(target_batch)
                    self.stats["target_images_saved"] += saved
                    self.stats["errors"].extend(errors)
                
                # Update metadata
                for meta in metadata_batch:
                    self.metadata.add_generation(
                        meta["char"], meta["style"], meta["font"],
                        meta["content_path"], meta["target_path"]
                    )
                
                pbar.update(1)
                
                # Log progress periodically
                if i % (self.config.batch_size * 10) == 0:
                    logger.info(
                        f"Progress: {i:,}/{len(dataset):,} samples, "
                        f"{self.stats['content_images_saved']} content, "
                        f"{self.stats['target_images_saved']} target images"
                    )
        
        return self.metadata.get_metadata()
    
    def _verify_export_integrity(self, metadata: ExportStats) -> bool:
        """Verify that all exported files are valid."""
        if not self.config.verify_integrity:
            return True
        
        logger.info("Verifying export integrity...")
        
        # Check that all referenced files exist
        missing_files = []
        
        for gen in metadata["generations"]:
            content_path = self.output_dir / gen["content_image_path"]
            target_path = self.output_dir / gen["target_image_path"]
            
            if not content_path.exists():
                missing_files.append(str(content_path))
            
            if not target_path.exists():
                missing_files.append(str(target_path))
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files")
            for f in missing_files[:10]:  # Show first 10
                logger.debug(f"Missing: {f}")
            if len(missing_files) > 10:
                logger.debug(f"... and {len(missing_files) - 10} more")
            
            # Try to recover by checking if files exist with different case (Windows)
            # or if they need to be regenerated
            return False
        
        logger.info("Export integrity verified successfully")
        return True
    
    def _save_checkpoint_optimized(self, metadata: ExportStats) -> Path:
        """Save results_checkpoint.json with compression and integrity check."""
        checkpoint_path = self.output_dir / "results_checkpoint.json"
        
        import datetime
        # Add export statistics
        export_info = {
            "export_timestamp": datetime.now().isoformat(),
            "export_config": {
                "num_workers": self.config.num_workers,
                "batch_size": self.config.batch_size,
                "image_format": self.config.image_format,
                "image_quality": self.config.image_quality,
            },
            "export_stats": {
                "content_images_saved": self.stats["content_images_saved"],
                "target_images_saved": self.stats["target_images_saved"],
                "skipped_duplicates": self.stats["skipped_duplicates"],
                "errors_count": len(self.stats["errors"]),
            },
            **metadata,
        }
        
        # Save with compression if requested
        if self.config.use_compression:
            # Use JSON with minimal whitespace
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(export_info, f, ensure_ascii=False, separators=(",", ":"))
        else:
            # Pretty print for readability
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        # Add checksum for integrity
        checksum = hashlib.md5(checkpoint_path.read_bytes()).hexdigest()
        checksum_path = checkpoint_path.with_suffix(".json.md5")
        checksum_path.write_text(checksum)
        
        logger.info(
            f"Saved checkpoint with {len(metadata['generations']):,} generations: "
            f"{metadata['total_chars']} chars, {metadata['total_styles']} styles"
        )
        
        return checkpoint_path
    
    def export(self) -> ExportStats:
        """Execute the full high-performance export process.
        
        Returns:
            Complete metadata dictionary with export statistics
            
        Raises:
            ValueError: If dataset loading or export fails
        """
        logger.info("Starting high-performance dataset export...")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Create directory structure
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")
        
        # Export in parallel batches
        metadata = self._export_batch_parallel(dataset)
        
        # Verify integrity
        if self.config.verify_integrity:
            if not self._verify_export_integrity(metadata):
                logger.warning("Export integrity check failed - some files may be missing")
        
        # Save checkpoint
        checkpoint_path = self._save_checkpoint_optimized(metadata)
        
        # Log summary
        logger.info(
            f"Export completed successfully!\n"
            f"  Content images: {self.stats['content_images_saved']:,}\n"
            f"  Target images:  {self.stats['target_images_saved']:,}\n"
            f"  Total samples:  {len(metadata['generations']):,}\n"
            f"  Errors:         {len(self.stats['errors'])}\n"
            f"  Checkpoint:     {checkpoint_path}"
        )
        
        if self.stats["errors"]:
            logger.warning(f"Encountered {len(self.stats['errors'])} errors during export")
            for error in self.stats["errors"][:5]:  # Show first 5 errors
                logger.debug(f"Error: {error}")
        
        return metadata


def export_dataset(
    output_dir: str | Path,
    repo_id: str | None = None,
    local_dataset_path: str | Path | None = None,
    split: str = "train",
    token: str | None = None,
    num_workers: int | None = None,
    batch_size: int = 100,
    verify_integrity: bool = True,
) -> ExportStats:
    """High-performance export of HuggingFace dataset to disk.
    
    Args:
        output_dir: Directory to export to
        repo_id: HuggingFace repository ID
        local_dataset_path: Local dataset path (alternative to repo_id)
        split: Dataset split name
        token: HuggingFace API token for private datasets
        num_workers: Number of parallel workers (auto-detected if None)
        batch_size: Batch size for parallel processing
        verify_integrity: Verify exported files exist
        
    Returns:
        Metadata dictionary from results_checkpoint.json with export stats
        
    Raises:
        ValueError: If neither repo_id nor local_dataset_path is provided,
                   or if dataset cannot be loaded
    """
    config = ExportConfig(
        output_dir=Path(output_dir),
        repo_id=repo_id,
        local_dataset_path=Path(local_dataset_path) if local_dataset_path else None,
        split=split,
        token=token,
        num_workers=num_workers or min(16, mp.cpu_count()),
        batch_size=batch_size,
        verify_integrity=verify_integrity,
    )
    
    exporter = DatasetExporter(config)
    return exporter.export()


def main() -> None:
    """CLI entry point with performance options."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description="High-performance export of HuggingFace dataset to FontDiffusion directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Examples:
  # Export from Hub with 8 parallel workers
  python export_hf_dataset.py --output-dir ./output --repo-id user/dataset --workers 8
  
  # Export from local cache with large batches
  python export_hf_dataset.py --output-dir ./output --local-path ./dataset --batch-size 500
  
  # Export with custom image quality
  python export_hf_dataset.py --output-dir ./output --repo-id user/dataset --image-format JPEG --quality 90
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to export to",
    )
    
    # Dataset source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID",
    )
    source_group.add_argument(
        "--local-path",
        type=str,
        help="Local dataset path (alternative to --repo-id)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token for private datasets",
    )
    
    # Performance tuning
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing (default: 100)",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        choices=["PNG", "JPEG", "WEBP"],
        default="PNG",
        help="Image format for export (default: PNG)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Image quality for JPEG/WEBP (1-100, default: 95)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip integrity verification (faster but less safe)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        start_time = datetime.now()
        
        metadata = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            token=args.token,
            num_workers=args.workers,
            batch_size=args.batch_size,
            verify_integrity=not args.no_verify,
        )
        
        elapsed = datetime.now() - start_time
        
        # Calculate performance metrics
        total_samples = metadata["total_generations"]
        seconds = elapsed.total_seconds()
        samples_per_second = total_samples / seconds if seconds > 0 else 0
        
        print(f"\n{'='*60}")
        print("EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Output directory:     {args.output_dir}")
        print(f"Total samples:        {total_samples:,}")
        print(f"Unique characters:    {metadata['total_chars']:,}")
        print(f"Unique styles:        {metadata['total_styles']:,}")
        print(f"Elapsed time:         {elapsed}")
        print(f"Processing rate:      {samples_per_second:.1f} samples/sec")
        print(f"Checkpoint saved:     {args.output_dir}/results_checkpoint.json")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()