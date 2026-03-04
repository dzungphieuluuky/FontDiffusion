import json
import logging
import time
import os
import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image, ImageFile
import cv2
import numpy as np

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


def _process_batch(
    batch: dict,
    path_cache: dict,
    style_bytes_cache: dict,
    style_dims_cache: dict,
    resize_height: int,
    spacing: int,
    jpeg_quality: int,
) -> dict:
    """Process a batch of generation records into dataset rows.

    Receives style images as raw bytes to avoid pickling PIL objects
    across subprocess boundaries, which causes worker OOM crashes.

    Parameters
    ----------
    batch : dict
        Columnar batch from datasets.map() with character/style/font lists.
    path_cache : dict
        Pre-built cache: content paths+dims, target paths+dims.
    style_bytes_cache : dict
        Raw PNG/JPEG bytes for each style image, keyed by style name.
        Bytes are safe to pickle across processes; PIL objects are not.
    style_dims_cache : dict
        (width, height) tuples for each style image, keyed by style name.
    resize_height : int
        Target height for all output images.
    spacing : int
        Pixel gap between panels in the comparison image.
    jpeg_quality : int
        JPEG quality for encoding (1-100).

    Returns
    -------
    dict
        Columnar results matching the dataset features schema.
    """
    batch_size = len(batch["character"])
    content_cache = path_cache["content"]
    target_cache = path_cache["target"]

    chars, styles, fonts = [], [], []
    content_infos, target_infos = [], []
    skipped = 0
    failure_samples: list[tuple[str, str]] = []

    # Single validation pass — collect valid items only
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

        if style not in style_bytes_cache:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_bytes"))
            skipped += 1
            continue

        if style not in style_dims_cache:
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_dims"))
            skipped += 1
            continue

        chars.append(char)
        styles.append(style)
        fonts.append(font)
        content_infos.append(content_info)
        target_infos.append(target_info)

    # Empty batch — return early with useful warning
    if not chars:
        if skipped > 0 and failure_samples:
            logger.warning(
                f"Batch skipped {batch_size}/{batch_size} items. "
                f"Examples: {failure_samples}"
            )
        return {
            "character": [], "style": [], "font": [],
            "content_image": [], "style_image": [], "target_image": [],
            "comparison_image": [], "content_hash": [], "target_hash": [],
        }

    num_valid = len(chars)
    rh = resize_height

    # Pre-compute resize widths for all valid items
    c_widths, s_widths, t_widths = [], [], []
    for idx in range(num_valid):
        ci = content_infos[idx]
        ti = target_infos[idx]
        sw, sh = style_dims_cache[styles[idx]]
        c_widths.append(max(1, int(ci["width"] * (rh / ci["height"]))))
        s_widths.append(max(1, int(sw * (rh / sh))))
        t_widths.append(max(1, int(ti["width"] * (rh / ti["height"]))))

    # Decode style bytes into arrays once per unique style in this batch
    # (avoids re-decoding the same style image for every character)
    unique_styles = list(dict.fromkeys(styles))
    style_arr_cache: dict[str, np.ndarray] = {}
    for uname in unique_styles:
        raw = style_bytes_cache[uname]
        arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            style_arr_cache[uname] = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    content_image_bytes, style_image_bytes = [], []
    target_image_bytes, comparison_bytes = [], []
    content_hashes, target_hashes = [], []
    valid_chars, valid_styles, valid_fonts = [], [], []

    for idx in range(num_valid):
        char = chars[idx]
        style = styles[idx]
        font = fonts[idx]

        try:
            content_arr = _load_cv2_resize(
                content_infos[idx]["path"], c_widths[idx], rh
            )
            target_arr = _load_cv2_resize(
                target_infos[idx]["path"], t_widths[idx], rh
            )

            style_full = style_arr_cache.get(style)
            if style_full is None:
                skipped += 1
                continue
            style_arr = cv2.resize(
                style_full, (s_widths[idx], rh), interpolation=cv2.INTER_LINEAR
            )

            # Build comparison with numpy concatenation (faster than PIL paste)
            spacer = np.full((rh, spacing, 3), 255, dtype=np.uint8)
            comparison = np.concatenate(
                [content_arr, spacer, style_arr, spacer, target_arr], axis=1
            )

            content_image_bytes.append({"bytes": _encode_jpeg(content_arr, jpeg_quality)})
            # Use original style bytes for style_image column (no quality loss)
            style_image_bytes.append({"bytes": style_bytes_cache[style]})
            target_image_bytes.append({"bytes": _encode_jpeg(target_arr, jpeg_quality)})
            comparison_bytes.append({"bytes": _encode_jpeg(comparison, jpeg_quality)})
            content_hashes.append(compute_file_hash(char, "", font))
            target_hashes.append(compute_file_hash(char, style, font))
            valid_chars.append(char)
            valid_styles.append(style)
            valid_fonts.append(font)

        except Exception as exc:
            logger.debug(f"Failed to process {char}/{style}: {exc}")
            skipped += 1
            continue

    if skipped > 0 and failure_samples:
        logger.warning(
            f"Batch skipped {skipped}/{batch_size} items. "
            f"Examples: {failure_samples}"
        )

    return {
        "character": valid_chars,
        "style": valid_styles,
        "font": valid_fonts,
        "content_image": content_image_bytes,
        "style_image": style_image_bytes,
        "target_image": target_image_bytes,
        "comparison_image": comparison_bytes,
        "content_hash": content_hashes,
        "target_hash": target_hashes,
    }


class UltraFastDatasetBuilder:
    """Builds and uploads font style transfer datasets using datasets.map() parallelism."""

    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        self.resize_height = config.resize_height
        self.spacing = config.spacing
        self.jpeg_quality = config.jpeg_quality

        self.cpu_count = os.cpu_count() or 4
        self.num_proc = min(self.cpu_count, 8)
        self.process_batch_size = 1000

        # Stores raw bytes — safe to pickle across processes
        self.style_bytes_cache: dict[str, bytes] = {}
        self.style_dims_cache: dict[str, tuple[int, int]] = {}
        self.path_cache: dict[str, dict] = {}

        self._validate_structure()
        self._build_style_index()
        self._build_path_cache_with_dims()
        self.generations = self._load_checkpoint()

        logger.info("Ultra-fast pipeline initialized:")
        logger.info(f"  Total generations: {len(self.generations)}")
        logger.info(f"  CPU workers: {self.num_proc} processes")
        logger.info(f"  Process batch size: {self.process_batch_size}")
        logger.info(f"  JPEG quality: {self.jpeg_quality}")
        logger.info(f"  Style images: {len(self.style_bytes_cache)}")

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

    def _build_style_index(self) -> None:
        """Load style images as raw bytes and record their dimensions.

        Storing bytes instead of PIL Image objects ensures the style cache
        can be safely pickled and sent to worker subprocesses without OOM.
        """
        logger.info("Loading style images as bytes...")
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in sorted(self.style_images_dir.glob(f"*{ext}")):
                style_name = style_file.stem
                try:
                    # Read raw bytes for pickling safety
                    raw_bytes = style_file.read_bytes()
                    # Decode once just to get dimensions
                    arr = cv2.imdecode(
                        np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if arr is None:
                        logger.warning(f"Could not decode style image: {style_file}")
                        continue
                    h, w = arr.shape[:2]
                    self.style_bytes_cache[style_name] = raw_bytes
                    self.style_dims_cache[style_name] = (w, h)
                except Exception as exc:
                    logger.warning(f"Failed to load style image {style_file}: {exc}")
        logger.info(f"Loaded {len(self.style_bytes_cache)} style images")

    def _build_path_cache_with_dims(self) -> None:
        """Build file path cache with pre-read image dimensions."""
        logger.info("Building path cache with dimensions...")
        content_dir = self.data_dir / "ContentImage"
        content_paths: dict[str, dict] = {}

        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                char = img_file.stem
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                    content_paths[char] = {
                        "path": str(img_file),
                        "width": width,
                        "height": height,
                    }
                except Exception as exc:
                    logger.debug(f"Failed to read dimensions for {img_file}: {exc}")

        target_paths: dict[str, dict] = {}
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
                        logger.debug(
                            f"Failed to read dimensions for {img_file}: {exc}"
                        )
                target_paths[style] = style_char_paths

        self.path_cache = {
            "content": content_paths,
            "target": target_paths,
        }
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(
            f"Path cache built: {len(content_paths)} content, "
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
                "style_dims_cache": self.style_dims_cache,
                "resize_height": self.resize_height,
                "spacing": self.spacing,
                "jpeg_quality": self.jpeg_quality,
            },
        )

        build_time = time.time() - start_time
        num_samples = len(dataset)
        speed = num_samples / build_time if build_time > 0 else 0.0
        logger.info(f"Dataset built: {num_samples} samples in {build_time:.2f}s")
        logger.info(f"Processing speed: {speed:.1f} samples/s")

        if num_samples == 0:
            logger.error(
                "Dataset is empty. Check that style names in checkpoint match "
                "style image filenames in style_images_dir."
            )

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
            logger.error("Dataset is empty — skipping upload.")
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
                num_shards=self.config.num_shards,
                num_proc=1,
                commit_message="Ultra-fast dataset upload with pre-encoded bytes",
            )
            upload_time = time.time() - start_time
            speed = len(dataset) / upload_time if upload_time > 0 else 0.0
            logger.info(
                f"Upload completed in {upload_time:.2f}s ({speed:.1f} samples/s)"
            )
            logger.info(
                f"Dataset: https://huggingface.co/datasets/{self.config.repo_id}"
            )
        except Exception as exc:
            logger.error(f"Upload failed: {exc}")
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
        logger.info(f"Dataset saved in {time.time() - start_time:.2f}s")


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
    jpeg_quality: int = 90,
    num_shards: int = 8,
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
            jpeg_quality=args.jpeg_quality,
            num_shards=args.num_shards,
        )
        total_time = time.time() - start_time
        print(f"\nDataset creation completed in {total_time:.2f}s")
        print(f"Samples: {len(dataset)}")
        print(f"Speed: {len(dataset)/total_time:.1f} samples/second")
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