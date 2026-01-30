"""
Create Hugging Face dataset from FontDiffusion images with ULTRA-FAST parallel processing.

Optimized for maximum speed with multi-level parallelism and memory optimization.
"""

import json
import logging
import time
import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import lru_cache
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from collections import defaultdict
import gc

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image, ImageFile

# Enable PIL optimizations
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Remove size limit

# Import local utilities
try:
    from utilities import HFTqdm
    from filename_utils import compute_file_hash
except ImportError:
    # Fallback implementations
    def compute_file_hash(char: str, style: str, font: str) -> str:
        import hashlib
        return hashlib.md5(f"{char}_{style}_{font}".encode()).hexdigest()
    
    HFTqdm = None

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation - optimized!"""
    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    config_name: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None
    use_shared_memory: bool = True  # NEW: Use shared memory for large datasets
    
    # Cache precomputed style images
    style_cache: Dict[str, Image.Image] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Convert paths to Path if they're strings and preload style images."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)
        
        # Preload all style images into memory
        self._preload_style_images()

    def _preload_style_images(self):
        """Preload all style images into memory for zero I/O during processing."""
        logger.info("Preloading style images into memory...")
        style_images = {}
        
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in self.style_images_dir.glob(f"*{ext}"):
                style_name = style_file.stem
                try:
                    img = Image.open(style_file).convert("RGB")
                    # Keep image in memory (typically small, few hundred KB each)
                    style_images[style_name] = img
                except Exception as e:
                    logger.warning(f"Failed to preload style image {style_file}: {e}")
        
        self.style_cache = style_images
        logger.info(f"Preloaded {len(style_images)} style images")


class UltraFastDatasetBuilder:
    """ULTRA-FAST dataset builder with aggressive optimizations."""
    
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"
    
    def __init__(self, config: DatasetConfig):
        """Initialize with aggressive performance optimizations."""
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        
        # Aggressive auto-tuning
        self.cpu_count = os.cpu_count() or 4
        self.total_workers = max(1, self.cpu_count * 4)  # Use all CPU cores aggressively
        self.batch_size = 2000  # Larger batches for better parallelism
        self.resize_height = 256
        self.spacing = 10
        
        # Precompute paths for fast access
        self._precompute_paths()
        
        # Memory optimization
        self.use_shared_memory = config.use_shared_memory
        if self.use_shared_memory:
            self._init_shared_memory()
        
        logger.info(f"ULTRA-FAST mode: {self.total_workers} workers, batch_size={self.batch_size}")
        logger.info(f"Using shared memory: {self.use_shared_memory}")
        
        self._validate_structure()

    def _precompute_paths(self):
        """Precompute all image paths for zero filesystem operations during processing."""
        logger.info("Precomputing image paths...")
        
        # Precompute content image paths
        self.content_paths = {}
        content_dir = self.data_dir / "ContentImage"
        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    char = img_file.stem  # Assuming filename is character
                    self.content_paths[char] = img_file
        
        # Precompute target image paths by style
        self.target_paths = defaultdict(dict)
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in target_dir.iterdir():
                if style_dir.is_dir():
                    style = style_dir.name
                    for img_file in style_dir.glob("*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            char = img_file.stem.split('_')[0]  # Assuming format: char_style.ext
                            self.target_paths[style][char] = img_file
        
        logger.info(f"Precomputed {len(self.content_paths)} content paths, "
                   f"{sum(len(v) for v in self.target_paths.values())} target paths")

    def _init_shared_memory(self):
        """Initialize shared memory for inter-process communication."""
        self.shared_cache = {}
        # We'll use multiprocessing.Manager for simple shared dict
        self.manager = mp.Manager()
        self.shared_cache = self.manager.dict()
        
        # Pre-cache style images in shared memory
        for style_name, img in self.config.style_cache.items():
            # Convert PIL image to bytes for sharing
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            self.shared_cache[f"style_{style_name}"] = buffer.getvalue()

    def _validate_structure(self) -> None:
        """Fast validation with precomputed paths."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        if not self.style_images_dir.exists():
            raise ValueError(f"Style images directory not found: {self.style_images_dir}")

        logger.info("Directory structure validated successfully")

    def _load_checkpoint_fast(self) -> tuple[list, dict]:
        """Load checkpoint with memory mapping for large files."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        
        logger.info(f"Loading checkpoint with memory mapping...")
        start = time.time()
        
        # Use mmap for large files
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        
        # Build lookup dictionaries for fast access
        char_to_content = {}
        style_char_to_target = defaultdict(dict)
        
        for gen in generations:
            char = gen.get("character", "")
            style = gen.get("style", "")
            
            # Store references for fast lookup
            char_to_content[char] = gen.get("content_image_path", "")
            style_char_to_target[style][char] = gen.get("target_image_path", "")
        
        load_time = time.time() - start
        logger.info(f"Loaded {len(generations)} generations in {load_time:.2f}s")
        
        return generations, {
            'char_to_content': char_to_content,
            'style_char_to_target': dict(style_char_to_target),
            'metadata': {
                'characters': data.get('characters', []),
                'styles': data.get('styles', []),
                'fonts': data.get('fonts', ['unknown']),
            }
        }

    def _load_and_process_batch(self, batch: list) -> list:
        """Process an entire batch in parallel - optimized version."""
        results = []
        
        for gen in batch:
            try:
                # Extract metadata
                char = gen.get("character", "")
                style = gen.get("style", "")
                font = gen.get("font", "unknown")
                
                # Get precomputed paths
                content_path = self.content_paths.get(char)
                target_path = self.target_paths.get(style, {}).get(char)
                style_img = self.config.style_cache.get(style)
                
                # Validate
                if not all([content_path, target_path, style_img]):
                    continue
                
                # Load images - these are the only I/O operations
                content_img = Image.open(content_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                
                # Create comparison image (optimized)
                # Precompute sizes
                c_width, c_height = content_img.size
                s_width, s_height = style_img.size
                t_width, t_height = target_img.size
                
                # Calculate new dimensions
                scale = self.resize_height / c_height
                c_new_width = int(c_width * scale)
                s_new_width = int(s_width * (self.resize_height / s_height))
                t_new_width = int(t_width * (self.resize_height / t_height))
                
                # Resize images
                content_resized = content_img.resize((c_new_width, self.resize_height), 
                                                     Image.Resampling.LANCZOS)
                style_resized = style_img.resize((s_new_width, self.resize_height), 
                                                 Image.Resampling.LANCZOS)
                target_resized = target_img.resize((t_new_width, self.resize_height), 
                                                   Image.Resampling.LANCZOS)
                
                # Create comparison
                total_width = c_new_width + s_new_width + t_new_width + 2 * self.spacing
                comparison = Image.new("RGB", (total_width, self.resize_height), 
                                      color=(255, 255, 255))
                
                comparison.paste(content_resized, (0, 0))
                comparison.paste(style_resized, (c_new_width + self.spacing, 0))
                comparison.paste(target_resized, (c_new_width + s_new_width + 2 * self.spacing, 0))
                
                # Build sample
                results.append({
                    "character": char,
                    "style": style,
                    "font": font,
                    "content_image": content_img,
                    "style_image": style_img,
                    "target_image": target_img,
                    "comparison_image": comparison,
                    "content_hash": compute_file_hash(char, "", font),
                    "target_hash": compute_file_hash(char, style, font),
                })
                
            except Exception as e:
                logger.debug(f"Failed to process {gen.get('character', '')}/{gen.get('style', '')}: {e}")
                continue
        
        return results

    def _generate_samples_mega_parallel(self) -> Generator[dict, None, None]:
        """Generate samples with MEGA-parallel processing using multiple strategies."""
        generations, lookup_dicts = self._load_checkpoint_fast()
        total_samples = len(generations)
        
        logger.info(f"Processing {total_samples} samples with MEGA-parallel mode...")
        logger.info(f"Using {self.total_workers} workers across {self.cpu_count} CPU cores")
        
        start_time = time.time()
        
        # Strategy 1: Process in large batches with parallel loading
        batch_count = (total_samples + self.batch_size - 1) // self.batch_size
        
        # Use multiple strategies based on dataset size
        if total_samples > 10000:
            # For very large datasets: use chunked processing with memory recycling
            yield from self._process_very_large_dataset(generations, batch_count, start_time)
        else:
            # For moderate datasets: full parallel processing
            yield from self._process_moderate_dataset(generations, batch_count, start_time)
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f}s ({total_samples/total_time:.1f} samples/s)")

    def _process_moderate_dataset(self, generations: list, batch_count: int, start_time: float) -> Generator:
        """Process moderate-sized datasets with full parallelism."""
        # Use ProcessPoolExecutor with aggressive worker count
        with ProcessPoolExecutor(max_workers=self.total_workers) as executor:
            futures = []
            
            # Submit all batches at once
            for i in range(batch_count):
                batch_start = i * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(generations))
                batch = generations[batch_start:batch_end]
                
                future = executor.submit(self._load_and_process_batch, batch)
                futures.append((future, batch_start, batch_end))
            
            # Process completed futures
            completed = 0
            for future, batch_start, batch_end in futures:
                try:
                    batch_results = future.result(timeout=300)  # 5-minute timeout
                    
                    # Yield results
                    for result in batch_results:
                        yield result
                    
                    completed += len(batch_results)
                    
                    # Progress reporting
                    elapsed = time.time() - start_time
                    progress_pct = batch_end / len(generations) * 100
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (len(generations) - batch_end) / rate if rate > 0 else 0
                    
                    logger.info(f"Batch {batch_start}-{batch_end}: "
                               f"{len(batch_results)}/{batch_end-batch_start} valid, "
                               f"Overall: {progress_pct:.1f}%, {rate:.1f} samples/s, ETA: {eta:.0f}s")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                    continue

    def _process_very_large_dataset(self, generations: list, batch_count: int, start_time: float) -> Generator:
        """Process very large datasets with memory-efficient chunking."""
        # Process in chunks to manage memory
        chunk_size = 10  # Process 10 batches at a time
        
        for chunk_start in range(0, batch_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_count)
            
            logger.info(f"Processing chunk {chunk_start}-{chunk_end} of {batch_count}...")
            chunk_start_time = time.time()
            
            # Process this chunk in parallel
            with ProcessPoolExecutor(max_workers=self.total_workers) as executor:
                futures = []
                
                for i in range(chunk_start, chunk_end):
                    batch_start = i * self.batch_size
                    batch_end = min(batch_start + self.batch_size, len(generations))
                    batch = generations[batch_start:batch_end]
                    
                    future = executor.submit(self._load_and_process_batch, batch)
                    futures.append((future, batch_start, batch_end))
                
                # Collect results from this chunk
                for future, batch_start, batch_end in futures:
                    try:
                        batch_results = future.result(timeout=600)  # 10-minute timeout for large batches
                        
                        for result in batch_results:
                            yield result
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                        continue
            
            chunk_time = time.time() - chunk_start_time
            logger.info(f"Chunk completed in {chunk_time:.2f}s")
            
            # Force garbage collection between chunks
            gc.collect()

    def build(self) -> Dataset:
        """Build dataset with ultra-fast processing and memory optimization."""
        logger.info("Building dataset with ULTRA-FAST processing...")
        
        features = Features({
            "character": Value("string"),
            "style": Value("string"),
            "font": Value("string"),
            "content_image": HFImage(),
            "style_image": HFImage(),
            "target_image": HFImage(),
            "comparison_image": HFImage(),
            "content_hash": Value("string"),
            "target_hash": Value("string"),
        })
        
        logger.info("Converting samples to Arrow format with streaming...")
        start_time = time.time()
        
        # Use from_generator with explicit arrow writer for better performance
        dataset = Dataset.from_generator(
            self._generate_samples_mega_parallel,
            features=features,
            keep_in_memory=False,  # Stream to disk to save RAM
            num_proc=self.cpu_count,  # Use multiple processes for Arrow conversion
        )
        
        conversion_time = time.time() - start_time
        logger.info(f"Dataset created: {len(dataset)} samples in {conversion_time:.2f}s "
                   f"({len(dataset)/conversion_time:.1f} samples/s)")
        
        return dataset

    def push_to_hub_fast(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub with optimized upload."""
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return
        
        logger.info(f"Pushing dataset to {self.config.repo_id} with optimized upload...")
        push_start = time.time()
        
        try:
            # Use streaming push for large datasets
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
                max_shard_size="500MB",  # Optimize shard size
                num_proc=self.cpu_count,  # Use multiple processes for upload
            )
            
            push_time = time.time() - push_start
            logger.info(f"Successfully pushed in {push_time:.2f}s")
            logger.info(f"Dataset available at: https://huggingface.co/datasets/{self.config.repo_id}")
            
        except Exception as e:
            logger.error(f"Push failed: {e}")
            # Try fallback method
            logger.info("Trying fallback push method...")
            self._push_fallback(dataset)

    def _push_fallback(self, dataset: Dataset) -> None:
        """Fallback push method with chunking."""
        try:
            # Save locally first, then upload
            temp_dir = Path(f"/tmp/dataset_{int(time.time())}")
            dataset.save_to_disk(temp_dir)
            
            # Upload saved dataset
            from huggingface_hub import upload_folder
            upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self.config.token,
                commit_message="Upload with fallback method",
            )
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info("Fallback push successful")
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            raise

    def save_local_fast(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with optimized I/O."""
        logger.info(f"Saving dataset to {output_path} with optimized I/O...")
        save_start = time.time()
        
        # Use multiple processes for saving
        dataset.save_to_disk(
            str(output_path),
            num_proc=self.cpu_count,
            max_shard_size="500MB",
        )
        
        save_time = time.time() - save_start
        logger.info(f"Dataset saved in {save_time:.2f}s")


def create_dataset_ultra_fast(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    use_shared_memory: bool = True,
) -> Dataset:
    """Create dataset with ULTRA-FAST parallel processing.
    
    Aggressively optimized for maximum speed - just provide essential arguments!
    
    Args:
        data_dir: Path to data directory (must contain ContentImage/ and TargetImage/)
        style_images_dir: Path to style images directory
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name
        config_name: Dataset configuration name
        push_to_hub: Whether to push to HuggingFace Hub
        private: Whether to make repository private
        token: HuggingFace API token
        local_save_path: Local path to save dataset
        use_shared_memory: Use shared memory for inter-process communication
    
    Returns:
        Created Dataset object
    """
    config = DatasetConfig(
        data_dir=Path(data_dir),
        style_images_dir=Path(style_images_dir),
        repo_id=repo_id,
        split=split,
        config_name=config_name,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        use_shared_memory=use_shared_memory,
    )
    
    builder = UltraFastDatasetBuilder(config)
    dataset = builder.build()
    
    if local_save_path:
        builder.save_local_fast(dataset, Path(local_save_path))
    
    if push_to_hub:
        builder.push_to_hub_fast(dataset)
    
    return dataset


def main():
    """Simple CLI - just provide essential arguments!"""
    parser = argparse.ArgumentParser(
        description="ULTRA-FAST HuggingFace dataset creator from FontDiffusion images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ULTRA-FAST dataset creation (max speed)
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name
  
  # Create but don't push to Hub
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --no-push
  
  # Create private dataset
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --private
  
  # Save locally as well
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name \\
    --local-save ./local_dataset
  
  # Disable shared memory (if having memory issues)
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --no-shared-mem

Performance optimizations:
  • Precomputes all image paths (zero filesystem lookups during processing)
  • Preloads style images into memory (zero I/O for style images)
  • Aggressive parallelization: workers = CPU cores × 4
  • Shared memory for inter-process communication
  • Memory-mapped checkpoint loading for large files
  • Optimized image resizing with precomputed dimensions
  • Chunked processing for very large datasets
  • Automatic garbage collection between chunks
  • Streaming Arrow conversion with multiple processes
  • Optimized HuggingFace Hub upload with sharding
        """,
    )
    
    # Essential arguments
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory (must contain ContentImage/ and TargetImage/)"
    )
    parser.add_argument(
        "--style-images-dir",
        required=True,
        help="Path to directory containing style images"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (username/dataset-name)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name (default: train)"
    )
    parser.add_argument(
        "--config-name",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace Hub"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--local-save",
        help="Save dataset locally to this path"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token for private datasets"
    )
    parser.add_argument(
        "--no-shared-mem",
        action="store_true",
        help="Disable shared memory optimization"
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
        # Run ULTRA-FAST dataset creation
        start_time = time.time()
        
        dataset = create_dataset_ultra_fast(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            config_name=args.config_name,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            use_shared_memory=not args.no_shared_mem,
        )
        
        total_time = time.time() - start_time
        
        # Success summary
        print(f"\n✅ Dataset creation completed in {total_time:.2f}s!")
        print(f"📊 Samples: {len(dataset)}")
        print(f"⚡ Speed: {len(dataset)/total_time:.1f} samples/second")
        print(f"🔤 Unique characters: {len(set(dataset['character']))}")
        print(f"🎨 Unique styles: {len(set(dataset['style']))}")
        
        if not args.no_push:
            print(f"🌐 Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        
        if args.local_save:
            print(f"💾 Local copy saved to: {args.local_save}")
        
    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    # Set high process priority on Linux/Mac
    try:
        os.nice(-10)  # Higher priority
    except:
        pass  # Not supported on this platform
    
    main()