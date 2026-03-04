"""Ultra-fast HuggingFace dataset creator using datasets library parallelism.

Builds a font style transfer dataset from ContentImage/, TargetImage/, and
style reference images, then uploads to HuggingFace Hub.

Performance strategy:
  - Pre-cache style image bytes (each style loaded once, not per-character)
  - Use datasets.map() with num_proc for CPU parallelism
  - OpenCV for fast resizing, numpy for fast comparison image assembly
  - Encode resized images (smaller) rather than originals
"""
import argparse
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    from filename_utils import compute_file_hash
except ImportError:

    def compute_file_hash(char: str, style: str, font: str) -> str:
        import hashlib

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

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


def _load_and_resize_cv2(path: str, new_width: int, new_height: int) -> np.ndarray:
    """Load image from disk and resize using OpenCV. Returns RGB uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _encode_array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a numpy RGB array to PNG bytes via OpenCV (faster than PIL)."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _assemble_comparison_numpy(
    content: np.ndarray,
    style: np.ndarray,
    target: np.ndarray,
    spacing: int,
) -> np.ndarray:
    """Build side-by-side comparison image using numpy (no PIL paste)."""
    height = content.shape[0]
    spacer = np.full((height, spacing, 3), 255, dtype=np.uint8)
    return np.concatenate([content, spacer, style, spacer, target], axis=1)


def _process_batch(
    batch: dict,
    path_cache: dict,
    style_bytes_cache: dict,
    resize_height: int,
    spacing: int,
) -> dict:
    """Process a batch of generations into dataset rows.

    Parameters
    ----------
    batch : dict
        Columnar batch from datasets.map() with character/style/font lists.
    path_cache : dict
        Pre-built cache with content paths, target paths, style dims, style paths.
    style_bytes_cache : dict
        Pre-encoded style image bytes keyed by style name.
    resize_height : int
        Target height for all resized images.
    spacing : int
        Pixel spacing between panels in comparison image.

    Returns
    -------
    dict
        Columnar results matching the dataset features schema.
    """
    batch_size = len(batch["character"])

    content_cache = path_cache["content"]
    target_cache = path_cache["target"]
    style_dims_cache = path_cache["style_dims"]

    # Collect valid items in a single pass
    chars = []
    styles = []
    fonts = []
    content_infos = []
    target_infos = []
    style_dim_list = []

    skipped = 0
    failure_samples = []

    for i in range(batch_size):
        char = batch["character"][i]
        style = batch["style"][i]
        font = batch["font"][i]

        content_info = content_cache.get(char)
        if not content_info:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no content_info"))
            skipped += 1
            continue

        target_info = target_cache.get(style, {}).get(char)
        if not target_info:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no target_info"))
            skipped += 1
            continue

        sdims = style_dims_cache.get(style)
        if not sdims:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_dims"))
            skipped += 1
            continue

        if style not in style_bytes_cache:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_bytes"))
            skipped += 1
            continue

        chars.append(char)
        styles.append(style)
        fonts.append(font)
        content_infos.append(content_info)
        target_infos.append(target_info)
        style_dim_list.append(sdims)

    # Early exit for entirely invalid batch
    if not chars:
        if skipped > 0 and failure_samples:
            logger.warning(
                f"Batch skipped {batch_size}/{batch_size} items. "
                f"Examples: {failure_samples}"
            )
        return {
            "character": [],
            "style": [],
            "font": [],
            "content_image": [],
            "style_image": [],
            "target_image": [],
            "comparison_image": [],
            "content_hash": [],
            "target_hash": [],
        }

    num_valid = len(chars)
    rh = resize_height

    # Pre-compute resize dimensions
    c_widths = []
    s_widths = []
    t_widths = []
    for idx in range(num_valid):
        ci = content_infos[idx]
        ti = target_infos[idx]
        sw, sh = style_dim_list[idx]

        c_widths.append(int(ci["width"] * (rh / ci["height"])))
        s_widths.append(int(sw * (rh / sh)))
        t_widths.append(int(ti["width"] * (rh / ti["height"])))

    # Load, resize, and encode content images
    content_bytes_list = []
    for idx in range(num_valid):
        arr = _load_and_resize_cv2(content_infos[idx]["path"], c_widths[idx], rh)
        content_bytes_list.append((_encode_array_to_png_bytes(arr), arr))

    # Load, resize, and encode target images
    target_bytes_list = []
    for idx in range(num_valid):
        arr = _load_and_resize_cv2(target_infos[idx]["path"], t_widths[idx], rh)
        target_bytes_list.append((_encode_array_to_png_bytes(arr), arr))

    # Resize style images (use cached bytes for style_image column,
    # but still need resized array for comparison)
    style_resized_arrays = []
    for idx in range(num_valid):
        style_name = styles[idx]
        style_path = path_cache["style_paths"][style_name]
        arr = _load_and_resize_cv2(style_path, s_widths[idx], rh)
        style_resized_arrays.append(arr)

    # Build comparison images and encode
    comparison_bytes_list = []
    for idx in range(num_valid):
        comp = _assemble_comparison_numpy(
            content_bytes_list[idx][1],
            style_resized_arrays[idx],
            target_bytes_list[idx][1],
            spacing,
        )
        comparison_bytes_list.append(_encode_array_to_png_bytes(comp))

    # Build result columns
    content_hashes = [
        compute_file_hash(chars[i], "", fonts[i]) for i in range(num_valid)
    ]
    target_hashes = [
        compute_file_hash(chars[i], styles[i], fonts[i]) for i in range(num_valid)
    ]

    results = {
        "character": chars,
        "style": styles,
        "font": fonts,
        "content_image": [{"bytes": content_bytes_list[i][0]} for i in range(num_valid)],
        "style_image": [{"bytes": style_bytes_cache[styles[i]]} for i in range(num_valid)],
        "target_image": [{"bytes": target_bytes_list[i][0]} for i in range(num_valid)],
        "comparison_image": [{"bytes": comparison_bytes_list[i]} for i in range(num_valid)],
        "content_hash": content_hashes,
        "target_hash": target_hashes,
    }

    if skipped > 0 and failure_samples:
        logger.warning(
            f"Batch skipped {skipped}/{batch_size} items. "
            f"Examples: {failure_samples}"
        )

    return results


class UltraFastDatasetBuilder:
    """Builds and uploads font style transfer datasets using datasets parallelism."""

    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        self.resize_height = config.resize_height
        self.spacing = config.spacing

        self.cpu_count = os.cpu_count() or 4
        self.num_proc = min(self.cpu_count, 8)
        self.process_batch_size = 1000

        self.style_paths: dict[str, Path] = {}
        self.path_cache: dict[str, dict] = {}
        self.style_bytes_cache: dict[str, bytes] = {}

        self._validate_structure()
        self._build_style_path_index()
        self._build_path_cache_with_dims()
        self._preload_style_bytes()
        self.generations = self._load_checkpoint()

        logger.info("Ultra-fast pipeline initialized:")
        logger.info(f"  Total generations: {len(self.generations)}")
        logger.info(f"  CPU workers: {self.num_proc} processes")
        logger.info(f"  Process batch size: {self.process_batch_size}")
        logger.info(f"  Style images: {len(self.style_paths)}")

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
        """Build index of style image paths for direct loading."""
        logger.info("Building style image path index...")
        self.style_paths = {}
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in self.style_images_dir.glob(f"*{ext}"):
                style_name = style_file.stem
                self.style_paths[style_name] = style_file
        logger.info(f"Indexed {len(self.style_paths)} style images")

    def _build_path_cache_with_dims(self) -> None:
        """Build path cache with image dimensions for content and target images."""
        logger.info("Building path cache with dimensions...")
        content_dir = self.data_dir / "ContentImage"
        content_paths = {}
        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    char = img_file.stem
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                        content_paths[char] = {
                            "path": str(img_file),
                            "width": width,
                            "height": height,
                        }
                    except Exception as e:
                        logger.debug(f"Failed to read dimensions for {img_file}: {e}")

        target_paths = {}
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in target_dir.iterdir():
                if style_dir.is_dir():
                    style = style_dir.name
                    style_char_paths = {}
                    for img_file in style_dir.glob("*"):
                        if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                            filename_parts = img_file.stem.split("+")
                            if len(filename_parts) >= 2:
                                char = filename_parts[1]
                                try:
                                    with Image.open(img_file) as img:
                                        width, height = img.size
                                    style_char_paths[char] = {
                                        "path": str(img_file),
                                        "width": width,
                                        "height": height,
                                    }
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to read dimensions for {img_file}: {e}"
                                    )
                    target_paths[style] = style_char_paths

        # Build style dimensions from direct file access
        style_dims = {}
        for style_name, style_path in self.style_paths.items():
            try:
                with Image.open(style_path) as img:
                    style_dims[style_name] = img.size
            except Exception as e:
                logger.debug(f"Failed to read style dimensions for {style_path}: {e}")

        self.path_cache = {
            "content": content_paths,
            "target": target_paths,
            "style_dims": style_dims,
            "style_paths": {k: str(v) for k, v in self.style_paths.items()},
        }
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(
            f"Path cache built: {len(content_paths)} content, "
            f"{total_targets} target paths"
        )

    def _preload_style_bytes(self) -> None:
        """Pre-load and encode all style images once.

        Since there are only ~40 style images but ~48k generations,
        caching these bytes avoids re-encoding the same image thousands
        of times during batch processing.
        """
        logger.info("Pre-loading style image bytes...")
        for style_name, style_path in self.style_paths.items():
            try:
                img_bytes = style_path.read_bytes()
                self.style_bytes_cache[style_name] = img_bytes
            except Exception as e:
                logger.warning(f"Failed to pre-load style image {style_name}: {e}")
        logger.info(f"Pre-loaded {len(self.style_bytes_cache)} style images")

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

    def build(self) -> Dataset:
        """Build the dataset using datasets.map() for parallelism.

        Returns
        -------
        Dataset
            HuggingFace Dataset with all image columns populated.
        """
        logger.info("Building dataset with batched map() pipeline...")
        start_time = time.time()

        metadata = {
            "character": [g.get("character", "") for g in self.generations],
            "style": [g.get("style", "") for g in self.generations],
            "font": [g.get("font", "unknown") for g in self.generations],
        }
        thin_dataset = Dataset.from_dict(metadata)

        features = Features(
            {
                "character": Value("string"),
                "style": Value("string"),
                "font": Value("string"),
                "content_image": HFImage(),
                "style_image": HFImage(),
                "target_image": HFImage(),
                "comparison_image": HFImage(),
                "content_hash": Value("string"),
                "target_hash": Value("string"),
            }
        )

        logger.info(
            f"Processing with {self.num_proc} workers "
            f"(batch_size={self.process_batch_size})..."
        )

        dataset = thin_dataset.map(
            _process_batch,
            batched=True,
            batch_size=self.process_batch_size,
            num_proc=self.num_proc,
            features=features,
            remove_columns=thin_dataset.column_names,
            desc="Processing images",
            fn_kwargs={
                "path_cache": self.path_cache,
                "style_bytes_cache": self.style_bytes_cache,
                "resize_height": self.resize_height,
                "spacing": self.spacing,
            },
        )

        build_time = time.time() - start_time
        num_samples = len(dataset)
        speed = num_samples / build_time if build_time > 0 else 0.0
        logger.info(f"Dataset built: {num_samples} samples in {build_time:.2f}s")
        logger.info(f"Processing speed: {speed:.1f} samples/s")
        return dataset

    def push_to_hub_streaming(self, dataset: Dataset) -> None:
        """Push dataset to HuggingFace Hub.

        Parameters
        ----------
        dataset : Dataset
            The built dataset to upload.
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        if len(dataset) == 0:
            logger.error(
                "Dataset is empty, skipping upload. "
                "Check path_cache alignment with checkpoint."
            )
            return

        logger.info(f"Streaming dataset to {self.config.repo_id}...")
        start_time = time.time()

        try:
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
                embed_external_files=False,
                commit_message="Dataset upload",
            )
            upload_time = time.time() - start_time
            speed = len(dataset) / upload_time if upload_time > 0 else 0.0
            logger.info(
                f"Upload completed in {upload_time:.2f}s ({speed:.1f} samples/s)"
            )
            logger.info(
                f"Dataset: https://huggingface.co/datasets/{self.config.repo_id}"
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk.

        Parameters
        ----------
        dataset : Dataset
            The built dataset.
        output_path : Path
            Destination directory.
        """
        logger.info(f"Saving dataset to {output_path}...")
        start_time = time.time()
        dataset.save_to_disk(str(output_path))
        save_time = time.time() - start_time
        logger.info(f"Dataset saved in {save_time:.2f}s")


def create_dataset_ultra(
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
) -> Dataset:
    """Create and optionally upload a font style transfer dataset.

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
        Target height for resized/comparison images.
    spacing : int
        Pixel spacing between panels in comparison images.

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
    )

    builder = UltraFastDatasetBuilder(config)
    dataset = builder.build()

    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push_to_hub_streaming(dataset)

    return dataset


def main():
    """CLI entry point for ultra-fast dataset creation."""
    parser = argparse.ArgumentParser(
        description="Ultra-fast HuggingFace dataset creator with map() pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/upload_dataset_ultra.py \\
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
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument("--config-name", help="Dataset configuration name")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )
    parser.add_argument("--local-save", help="Save dataset locally to this path")
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument(
        "--resize-height",
        type=int,
        default=256,
        help="Height for comparison images (default: 256)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in comparison (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        start_time = time.time()

        dataset = create_dataset_ultra(
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
        )

        total_time = time.time() - start_time
        num_samples = len(dataset)
        speed = num_samples / total_time if total_time > 0 else 0.0
        print(f"\nDataset creation completed in {total_time:.2f}s")
        print(f"Samples: {num_samples}")
        print(f"Speed: {speed:.1f} samples/second")
        print(f"Unique characters: {len(set(dataset['character']))}")
        print(f"Unique styles: {len(set(dataset['style']))}")

        if not args.no_push:
            print(f"Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        if args.local_save:
            print(f"Local copy saved to: {args.local_save}")

    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()