"""Ultra-fast HuggingFace dataset creator using datasets library parallelism.

Builds a font style transfer dataset from ContentImage/, TargetImage/, and
style reference images, then uploads to HuggingFace Hub.

Performance strategy:
  - Pre-cache style image bytes (each style loaded once, not per-character)
  - Use datasets.map() with num_proc for CPU parallelism
  - OpenCV for fast resizing, numpy for fast comparison image assembly
  - Encode resized images (smaller) rather than originals
  - Batch PNG encoding via cv2.imencode with pre-allocated output buffers
  - Avoid repeated Python attribute lookups in hot loops
  - ThreadPoolExecutor for parallel I/O within each batch
  - Pre-compute all resize dimensions vectorized before any I/O
  - Reuse numpy array allocations for comparison images
"""
import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
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

# Module-level encode params (avoid repeated dict creation)
_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # level 1 = fast, small enough


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


# ---------------------------------------------------------------------------
# Low-level image helpers (module-level for multiprocessing picklability)
# ---------------------------------------------------------------------------

def _load_and_resize_cv2(path: str, new_width: int, new_height: int) -> np.ndarray:
    """Load + resize with OpenCV. Returns contiguous RGB uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    # In-place color conversion where possible
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _encode_array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode RGB array → PNG bytes (compression level 1 for speed)."""
    # cv2.imencode expects BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr, _PNG_PARAMS)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _assemble_comparison_numpy(
    content: np.ndarray,
    style: np.ndarray,
    target: np.ndarray,
    spacing: int,
) -> np.ndarray:
    """Build side-by-side comparison with a single np.concatenate call."""
    height = content.shape[0]
    spacer = np.full((height, spacing, 3), 255, dtype=np.uint8)
    return np.concatenate([content, spacer, style, spacer, target], axis=1)


def _load_single(args):
    """Worker target: load + resize one image. Returns (index, tag, array)."""
    idx, tag, path, w, h = args
    return idx, tag, _load_and_resize_cv2(path, w, h)


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
        Columnar batch from datasets.map().
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
    # Local aliases — avoid repeated global dict lookups in hot loops
    content_cache = path_cache["content"]
    target_cache = path_cache["target"]
    style_dims_cache = path_cache["style_dims"]
    style_paths_cache = path_cache["style_paths"]
    rh = resize_height

    batch_chars: list = batch["character"]
    batch_styles: list = batch["style"]
    batch_fonts: list = batch["font"]
    batch_size = len(batch_chars)

    # --- Validation pass (single loop, no repeated dict lookups) -----------
    chars: list = []
    styles: list = []
    fonts: list = []
    content_infos: list = []
    target_infos: list = []
    style_dim_list: list = []
    skipped = 0
    failure_samples: list = []

    for i in range(batch_size):
        char = batch_chars[i]
        style = batch_styles[i]

        content_info = content_cache.get(char)
        if not content_info:
            skipped += 1
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no content_info"))
            continue

        target_info = target_cache.get(style, {}).get(char)
        if not target_info:
            skipped += 1
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no target_info"))
            continue

        sdims = style_dims_cache.get(style)
        if not sdims:
            skipped += 1
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_dims"))
            continue

        if style not in style_bytes_cache:
            skipped += 1
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no style_bytes"))
            continue

        chars.append(char)
        styles.append(style)
        fonts.append(batch_fonts[i])
        content_infos.append(content_info)
        target_infos.append(target_info)
        style_dim_list.append(sdims)

    if not chars:
        if skipped and failure_samples:
            logger.warning(
                "Batch skipped %d/%d items. Examples: %s",
                batch_size, batch_size, failure_samples,
            )
        return {
            "character": [], "style": [], "font": [],
            "content_image": [], "style_image": [],
            "target_image": [], "comparison_image": [],
            "content_hash": [], "target_hash": [],
        }

    num_valid = len(chars)

    # --- Vectorised dimension computation ----------------------------------
    c_heights = np.array([ci["height"] for ci in content_infos], dtype=np.float32)
    c_widths_orig = np.array([ci["width"] for ci in content_infos], dtype=np.float32)
    c_widths = (c_widths_orig * (rh / c_heights)).astype(np.int32).tolist()

    t_heights = np.array([ti["height"] for ti in target_infos], dtype=np.float32)
    t_widths_orig = np.array([ti["width"] for ti in target_infos], dtype=np.float32)
    t_widths = (t_widths_orig * (rh / t_heights)).astype(np.int32).tolist()

    s_dims = style_dim_list  # list of (sw, sh) tuples
    s_widths = [int(sw * (rh / sh)) for sw, sh in s_dims]

    # --- Parallel I/O: load content + target + style in one thread pool ----
    # Build a flat task list; tag: 'c'=content, 't'=target, 's'=style
    tasks = []
    for idx in range(num_valid):
        tasks.append((idx, "c", content_infos[idx]["path"], c_widths[idx], rh))
        tasks.append((idx, "t", target_infos[idx]["path"], t_widths[idx], rh))
        tasks.append((idx, "s", style_paths_cache[styles[idx]], s_widths[idx], rh))

    content_arrays: list = [None] * num_valid
    target_arrays: list = [None] * num_valid
    style_arrays: list = [None] * num_valid

    # Use min(num_valid*3, 16) threads — beyond 16 gives diminishing returns
    max_workers = min(num_valid * 3, 16)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, tag, arr in pool.map(_load_single, tasks):
            if tag == "c":
                content_arrays[idx] = arr
            elif tag == "t":
                target_arrays[idx] = arr
            else:
                style_arrays[idx] = arr

    # --- Encode all images in parallel -------------------------------------
    def _encode_content(i):
        return _encode_array_to_png_bytes(content_arrays[i])

    def _encode_target(i):
        return _encode_array_to_png_bytes(target_arrays[i])

    def _encode_comparison(i):
        comp = _assemble_comparison_numpy(
            content_arrays[i], style_arrays[i], target_arrays[i], spacing
        )
        return _encode_array_to_png_bytes(comp)

    indices = range(num_valid)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        content_bytes_list = list(pool.map(_encode_content, indices))
        target_bytes_list = list(pool.map(_encode_target, indices))
        comparison_bytes_list = list(pool.map(_encode_comparison, indices))

    # --- Hashes (pure Python, cheap) ---------------------------------------
    content_hashes = [compute_file_hash(chars[i], "", fonts[i]) for i in indices]
    target_hashes = [compute_file_hash(chars[i], styles[i], fonts[i]) for i in indices]

    if skipped and failure_samples:
        logger.warning(
            "Batch skipped %d/%d items. Examples: %s",
            skipped, batch_size, failure_samples,
        )

    return {
        "character": chars,
        "style": styles,
        "font": fonts,
        "content_image": [{"bytes": b} for b in content_bytes_list],
        "style_image": [{"bytes": style_bytes_cache[styles[i]]} for i in indices],
        "target_image": [{"bytes": b} for b in target_bytes_list],
        "comparison_image": [{"bytes": b} for b in comparison_bytes_list],
        "content_hash": content_hashes,
        "target_hash": target_hashes,
    }


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
        # Use all available CPUs for the map() workers; thread-level parallelism
        # handles the I/O inside each worker, so we want maximum CPU processes.
        self.num_proc = self.cpu_count
        # Larger batch → amortises Python overhead; 2000 is a good sweet-spot.
        self.process_batch_size = 2000

        self.style_paths: dict[str, Path] = {}
        self.path_cache: dict[str, dict] = {}
        self.style_bytes_cache: dict[str, bytes] = {}

        self._validate_structure()
        self._build_style_path_index()
        self._build_path_cache_with_dims()
        self._preload_style_bytes()
        self.generations = self._load_checkpoint()

        logger.info("Ultra-fast pipeline initialized:")
        logger.info("  Total generations: %d", len(self.generations))
        logger.info("  CPU workers: %d processes", self.num_proc)
        logger.info("  Process batch size: %d", self.process_batch_size)
        logger.info("  Style images: %d", len(self.style_paths))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
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
        logger.info("Building style image path index...")
        self.style_paths = {
            style_file.stem: style_file
            for ext in (".png", ".jpg", ".jpeg")
            for style_file in self.style_images_dir.glob(f"*{ext}")
        }
        logger.info("Indexed %d style images", len(self.style_paths))

    def _build_path_cache_with_dims(self) -> None:
        """Build path cache with image dimensions using parallel PIL opens."""
        logger.info("Building path cache with dimensions...")

        content_dir = self.data_dir / "ContentImage"
        target_dir = self.data_dir / "TargetImage"

        # --- Content images ------------------------------------------------
        content_files = [
            f for f in content_dir.glob("*")
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ] if content_dir.exists() else []

        def _read_content(img_file: Path):
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
                return img_file.stem, {"path": str(img_file), "width": w, "height": h}
            except Exception as e:
                logger.debug("Failed to read %s: %s", img_file, e)
                return None, None

        content_paths: dict = {}
        with ThreadPoolExecutor(max_workers=min(len(content_files), 32)) as pool:
            for stem, info in pool.map(_read_content, content_files):
                if stem is not None:
                    content_paths[stem] = info

        # --- Target images -------------------------------------------------
        target_paths: dict = {}

        def _read_target(img_file: Path):
            parts = img_file.stem.split("+")
            if len(parts) < 2:
                return None
            char = parts[1]
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
                return img_file.parent.name, char, {
                    "path": str(img_file), "width": w, "height": h
                }
            except Exception as e:
                logger.debug("Failed to read %s: %s", img_file, e)
                return None

        target_files = [
            f
            for style_dir in (target_dir.iterdir() if target_dir.exists() else [])
            if style_dir.is_dir()
            for f in style_dir.glob("*")
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

        with ThreadPoolExecutor(max_workers=min(len(target_files) or 1, 32)) as pool:
            for result in pool.map(_read_target, target_files):
                if result is not None:
                    style_name, char, info = result
                    target_paths.setdefault(style_name, {})[char] = info

        # --- Style dimensions ----------------------------------------------
        def _read_style_dims(item):
            name, path = item
            try:
                with Image.open(path) as img:
                    return name, img.size
            except Exception as e:
                logger.debug("Failed style dims for %s: %s", path, e)
                return None, None

        style_dims: dict = {}
        with ThreadPoolExecutor(max_workers=min(len(self.style_paths), 32)) as pool:
            for name, dims in pool.map(_read_style_dims, self.style_paths.items()):
                if name is not None:
                    style_dims[name] = dims

        self.path_cache = {
            "content": content_paths,
            "target": target_paths,
            "style_dims": style_dims,
            "style_paths": {k: str(v) for k, v in self.style_paths.items()},
        }
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(
            "Path cache built: %d content, %d target paths",
            len(content_paths), total_targets,
        )

    def _preload_style_bytes(self) -> None:
        """Pre-load all style images in parallel (read_bytes is pure I/O)."""
        logger.info("Pre-loading style image bytes...")

        def _load(item):
            name, path = item
            try:
                return name, path.read_bytes()
            except Exception as e:
                logger.warning("Failed to pre-load style image %s: %s", name, e)
                return None, None

        with ThreadPoolExecutor(max_workers=min(len(self.style_paths), 32)) as pool:
            for name, data in pool.map(_load, self.style_paths.items()):
                if name is not None:
                    self.style_bytes_cache[name] = data

        logger.info("Pre-loaded %d style images", len(self.style_bytes_cache))

    def _load_checkpoint(self) -> list[dict]:
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        logger.info("Loaded %d generations from checkpoint", len(generations))
        return generations

    # ------------------------------------------------------------------
    # Build & upload
    # ------------------------------------------------------------------

    def build(self) -> Dataset:
        """Build the dataset using datasets.map() for parallelism.

        Returns
        -------
        Dataset
            HuggingFace Dataset with all image columns populated.
        """
        logger.info("Building dataset with batched map() pipeline...")
        start_time = time.time()

        # Build thin dataset with only scalar columns (fast pickling)
        gens = self.generations
        metadata = {
            "character": [g.get("character", "") for g in gens],
            "style": [g.get("style", "") for g in gens],
            "font": [g.get("font", "unknown") for g in gens],
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
            "Processing with %d workers (batch_size=%d)...",
            self.num_proc, self.process_batch_size,
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
        logger.info("Dataset built: %d samples in %.2fs", num_samples, build_time)
        logger.info("Processing speed: %.1f samples/s", speed)
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

        logger.info("Streaming dataset to %s...", self.config.repo_id)
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
                "Upload completed in %.2fs (%.1f samples/s)", upload_time, speed
            )
            logger.info(
                "Dataset: https://huggingface.co/datasets/%s", self.config.repo_id
            )
        except Exception as e:
            logger.error("Upload failed: %s", e)
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
        logger.info("Saving dataset to %s...", output_path)
        start_time = time.time()
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved in %.2fs", time.time() - start_time)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_dataset_ultra(
    data_dir: "str | Path",
    style_images_dir: "str | Path",
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: "Optional[str | Path]" = None,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        logger.exception("Dataset creation failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()