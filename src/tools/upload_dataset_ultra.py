import json
import logging
import time
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, Generator, Tuple
import gc
import hashlib

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from PIL import Image, ImageFile
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    from filename_utils import compute_file_hash
except ImportError:
    def compute_file_hash(char: str, style: str, font: str) -> str:
        return hashlib.md5(f"{char}_{style}_{font}".encode()).hexdigest()

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset building and upload."""

    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    config_name: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None
    resize_height: int = 256
    spacing: int = 10
    jpeg_quality: int = 90
    num_shards: int = 8

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


def _load_cv2_resize(path: str, new_width: int, new_height: int) -> np.ndarray:
    """Load and resize an image using OpenCV. Returns RGB uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _encode_jpeg(arr: np.ndarray, quality: int) -> bytes:
    """Encode a numpy RGB array to JPEG bytes via OpenCV."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


class GeneratorDatasetBuilder:
    """Builds and uploads font style transfer datasets using a generator for true streaming."""
    
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        self.resize_height = config.resize_height
        self.spacing = config.spacing
        self.jpeg_quality = config.jpeg_quality
        
        self.style_paths: dict[str, str] = {}
        self.content_paths: dict[str, dict] = {}
        self.target_paths: dict[str, dict[str, dict]] = {}
        self.generations = self._load_checkpoint()
        
        self._validate_structure()
        self._build_style_path_index()
        self._build_path_caches()
        
        self.features = Features({
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
        
        logger.info("Generator pipeline initialized:")
        logger.info(f"  Total generations: {len(self.generations)}")
        logger.info(f"  Style images indexed: {len(self.style_paths)}")
        logger.info(f"  Content images: {len(self.content_paths)}")
        logger.info(f"  Target images: {sum(len(v) for v in self.target_paths.values())}")

    def _validate_structure(self) -> None:
        """Validate that required directories and checkpoint file exist."""
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
        if not self.style_images_dir.exists():
            raise ValueError(
                f"Style images directory not found: {self.style_images_dir}"
            )
        logger.info("Directory structure validated")

    def _build_style_path_index(self) -> None:
        """Index style image file paths."""
        logger.info("Indexing style image paths...")
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in sorted(self.style_images_dir.glob(f"*{ext}")):
                self.style_paths[style_file.stem] = str(style_file)
        logger.info(f"Indexed {len(self.style_paths)} style image paths")

    def _build_path_caches(self) -> None:
        """Build file path caches with pre-read image dimensions."""
        logger.info("Building path caches with dimensions...")
        
        # Content images
        content_dir = self.data_dir / "ContentImage"
        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                char = img_file.stem
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                    self.content_paths[char] = {
                        "path": str(img_file),
                        "width": width,
                        "height": height,
                    }
                except Exception as exc:
                    logger.debug(f"Failed to read dimensions for {img_file}: {exc}")

        # Target images
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in sorted(target_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                style = style_dir.name
                style_char_paths: dict[str, dict] = {}
                for img_file in style_dir.glob("*"):
                    if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    parts = img_file.stem.split("+")
                    if len(parts) < 2:
                        continue
                    char = parts[1]
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                        style_char_paths[char] = {
                            "path": str(img_file),
                            "width": width,
                            "height": height,
                        }
                    except Exception as exc:
                        logger.debug(f"Failed to read dimensions for {img_file}: {exc}")
                self.target_paths[style] = style_char_paths

        total_targets = sum(len(v) for v in self.target_paths.values())
        logger.info(
            f"Path caches built: {len(self.content_paths)} content, "
            f"{total_targets} target paths"
        )

    def _load_checkpoint(self) -> list[dict]:
        """Load generation records from checkpoint file."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        logger.info(f"Loaded {len(generations)} generations from checkpoint")
        return generations

    def _validate_item(self, item: dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate if an item can be processed. Returns (is_valid, char, style)."""
        char = item.get("character", "")
        style = item.get("style", "")
        
        if not char or not style:
            return False, char, style
            
        if char not in self.content_paths:
            return False, char, style
            
        if style not in self.target_paths or char not in self.target_paths[style]:
            return False, char, style
            
        if style not in self.style_paths:
            return False, char, style
            
        return True, char, style

    def _process_item(self, char: str, style: str, font: str) -> Optional[Dict[str, Any]]:
        """Process a single generation item. Returns None if processing fails."""
        try:
            # Get paths and info
            content_info = self.content_paths[char]
            target_info = self.target_paths[style][char]
            style_path = self.style_paths[style]
            
            # Load style image
            style_raw = Path(style_path).read_bytes()
            style_arr = cv2.imdecode(np.frombuffer(style_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if style_arr is None:
                return None
            style_arr = cv2.cvtColor(style_arr, cv2.COLOR_BGR2RGB)
            sh, sw = style_arr.shape[:2]

            # Calculate resize widths
            rh = self.resize_height
            c_width = max(1, int(content_info["width"] * (rh / content_info["height"])))
            s_width = max(1, int(sw * (rh / sh)))
            t_width = max(1, int(target_info["width"] * (rh / target_info["height"])))

            # Load and resize images
            content_arr = _load_cv2_resize(content_info["path"], c_width, rh)
            target_arr = _load_cv2_resize(target_info["path"], t_width, rh)
            style_arr = cv2.resize(style_arr, (s_width, rh), interpolation=cv2.INTER_LINEAR)

            # Build comparison
            spacer = np.full((rh, self.spacing, 3), 255, dtype=np.uint8)
            comparison = np.concatenate(
                [content_arr, spacer, style_arr, spacer, target_arr], axis=1
            )

            # Encode images
            content_bytes = _encode_jpeg(content_arr, self.jpeg_quality)
            style_bytes = style_raw  # Use original bytes to avoid re-encoding
            target_bytes = _encode_jpeg(target_arr, self.jpeg_quality)
            comparison_bytes = _encode_jpeg(comparison, self.jpeg_quality)

            return {
                "character": char,
                "style": style,
                "font": font,
                "content_image": {"bytes": content_bytes},
                "style_image": {"bytes": style_bytes},
                "target_image": {"bytes": target_bytes},
                "comparison_image": {"bytes": comparison_bytes},
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            }
        except Exception as e:
            logger.debug(f"Failed to process {char}/{style}: {e}")
            return None

    def _generator_fn(self) -> Generator[Dict[str, Any], None, None]:
        """Generator function that yields processed items one by one."""
        total = len(self.generations)
        processed = 0
        skipped = 0
        last_log_time = time.time()
        failure_samples = []
        
        for idx, item in enumerate(self.generations):
            is_valid, char, style = self._validate_item(item)
            
            if not is_valid:
                skipped += 1
                if len(failure_samples) < 10:
                    failure_samples.append(f"{item.get('character', '')}/{item.get('style', '')}")
                continue
            
            font = item.get("font", "unknown")
            result = self._process_item(char, style, font)
            
            if result:
                yield result
                processed += 1
            else:
                skipped += 1
                if len(failure_samples) < 10:
                    failure_samples.append(f"{char}/{style}")
            
            # Log progress periodically
            if time.time() - last_log_time > 5:  # Every 5 seconds
                logger.info(f"Progress: {idx+1}/{total} items processed ({processed} valid, {skipped} skipped)")
                last_log_time = time.time()
            
            # Force garbage collection periodically
            if idx % 1000 == 0:
                gc.collect()
        
        logger.info(f"Generator completed: {processed} valid, {skipped} skipped")
        if skipped > 0 and failure_samples:
            logger.warning(f"Sample failures: {failure_samples[:5]}")

    def build_and_upload(self) -> Dataset:
        """Build dataset using generator and upload directly without storing full dataset."""
        logger.info("Starting generator-based dataset creation...")
        start_time = time.time()

        try:
            # Create dataset from generator
            logger.info("Creating dataset from generator...")
            dataset = Dataset.from_generator(
                self._generator_fn,
                features=self.features,
                cache_dir=None  # Don't cache to disk
            )
            
            build_time = time.time() - start_time
            logger.info(f"Dataset generated: {len(dataset)} samples in {build_time:.2f}s")
            
            # Push to hub if requested
            if self.config.push_to_hub:
                self._push_to_hub(dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}", exc_info=True)
            raise

    def _push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to HuggingFace Hub with memory-efficient streaming."""
        if len(dataset) == 0:
            logger.error("Dataset is empty — skipping upload.")
            return

        logger.info(f"Uploading dataset to {self.config.repo_id}...")
        upload_start = time.time()
        
        try:
            # Use streaming upload with minimal memory footprint
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
                embed_external_files=False,
                num_shards=self.config.num_shards,
                num_proc=1,  # Single process for memory efficiency
                commit_message="Dataset upload via generator pipeline",
            )
            
            upload_time = time.time() - upload_start
            logger.info(f"Upload completed in {upload_time:.2f}s")
            logger.info(f"Dataset: https://huggingface.co/datasets/{self.config.repo_id}")
            
        except Exception as exc:
            logger.error(f"Upload failed: {exc}")
            raise

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with memory-efficient streaming."""
        logger.info(f"Saving dataset to {output_path}...")
        start_time = time.time()
        
        dataset.save_to_disk(
            str(output_path),
            num_proc=1,
        )
        
        logger.info(f"Dataset saved in {time.time() - start_time:.2f}s")


def create_dataset_generator(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    resize_height: int = 256,
    spacing: int = 10,
    jpeg_quality: int = 90,
    num_shards: int = 8,
) -> Dataset:
    """Create and optionally upload a font style transfer dataset using generator for true streaming.
    
    This implementation uses a generator function to yield samples one by one,
    minimizing memory usage and allowing true streaming to HuggingFace Hub.
    
    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ContentImage/ and TargetImage/.
    style_images_dir : str or Path
        Directory containing style reference images.
    repo_id : str
        HuggingFace repository ID (username/dataset-name).
    split : str
        Dataset split name.
    config_name : str, optional
        Dataset configuration name for multi-config repos.
    push_to_hub : bool
        Whether to upload to HuggingFace Hub.
    private : bool
        Whether the Hub repo should be private.
    token : str, optional
        HuggingFace API token.
    local_save_path : str or Path, optional
        Path to save dataset locally.
    resize_height : int
        Target height for resized output images.
    spacing : int
        Pixel spacing between panels in comparison images.
    jpeg_quality : int
        JPEG encoding quality (1-100).
    num_shards : int
        Number of Parquet shards for Hub upload.
    
    Returns
    -------
    Dataset
        The built HuggingFace Dataset.
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
        resize_height=resize_height,
        spacing=spacing,
        jpeg_quality=jpeg_quality,
        num_shards=num_shards,
    )
    
    builder = GeneratorDatasetBuilder(config)
    dataset = builder.build_and_upload()
    
    if local_save_path and len(dataset) > 0:
        builder.save_local(dataset, Path(local_save_path))
    
    return dataset


def main():
    """CLI entry point for generator-based dataset creation."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient HuggingFace dataset creator with generator streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/upload_dataset_generator.py \\
    --data-dir my_dataset \\
    --style-images-dir style_images \\
    --repo-id username/fontdiffusion-dataset
        """,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory (must contain ContentImage/ and TargetImage/)",
    )
    parser.add_argument(
        "--style-images-dir",
        required=True,
        help="Path to style images directory",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (username/dataset-name)",
    )
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--config-name", help="Dataset configuration name")
    parser.add_argument(
        "--no-push", action="store_true", help="Skip pushing to HuggingFace Hub"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    parser.add_argument("--local-save", help="Save dataset locally to this path")
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument(
        "--resize-height",
        type=int,
        default=256,
        help="Height for output images (default: 256)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in comparison (default: 10)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality (1-100, default: 90)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of shards for dataset upload (default: 8)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        start_time = time.time()
        dataset = create_dataset_generator(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            config_name=args.config_name,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            resize_height=args.resize_height,
            spacing=args.spacing,
            jpeg_quality=args.jpeg_quality,
            num_shards=args.num_shards,
        )
        total_time = time.time() - start_time
        print(f"\nDataset creation completed in {total_time:.2f}s")
        print(f"Samples: {len(dataset)}")
        print(f"Speed: {len(dataset) / total_time:.1f} samples/second")
        print(f"Unique characters: {len(set(dataset['character']))}")
        print(f"Unique styles: {len(set(dataset['style']))}")
        if not args.no_push:
            print(f"Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        if args.local_save:
            print(f"Local copy saved to: {args.local_save}")

    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        logger.exception(f"Dataset creation failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()