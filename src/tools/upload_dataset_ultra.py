"""Ultra-fast HuggingFace dataset creator using datasets library parallelism.

Builds a font style transfer dataset from ContentImage/, TargetImage/, and
style reference images, then uploads to HuggingFace Hub.

Performance strategy:
  - Pre-cache style image bytes (each style loaded once, not per-character)
  - Use datasets.map() with num_proc for CPU parallelism — no manual mp/threads
  - OpenCV for fast resizing + PNG encoding (compression level 1)
  - Vectorised resize-dimension computation with numpy
  - Encode resized images (smaller) rather than originals
"""
import argparse
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

# Fast PNG params: compression level 1 is ~3-5x faster than the default (6)
_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


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
    num_proc: int = 1
    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


# ---------------------------------------------------------------------------
# Image helpers — module-level for datasets.map() picklability
# ---------------------------------------------------------------------------

def _load_resize_bgr(path: str, width: int, height: int) -> np.ndarray:
    """Load + resize with OpenCV, keeping BGR layout."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def _encode_bgr(arr: np.ndarray) -> bytes:
    """Encode a BGR array to PNG bytes at compression level 1."""
    ok, buf = cv2.imencode(".png", arr, _PNG_PARAMS)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _assemble_and_encode(
    content: np.ndarray,
    style: np.ndarray,
    target: np.ndarray,
    spacing: int,
) -> bytes:
    """Concatenate three BGR arrays side-by-side and return PNG bytes."""
    h = content.shape[0]
    spacer = np.full((h, spacing, 3), 255, dtype=np.uint8)
    comp = np.concatenate([content, spacer, style, spacer, target], axis=1)
    ok, buf = cv2.imencode(".png", comp, _PNG_PARAMS)
    if not ok:
        raise RuntimeError("cv2.imencode failed for comparison image")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Batch processor — called by datasets.map(), one process per shard
# ---------------------------------------------------------------------------

def _process_batch(
    batch: dict,
    path_cache: dict,
    style_bytes_cache: dict,
    resize_height: int,
    spacing: int,
) -> dict:
    """Process a columnar batch into dataset rows.

    Parameters
    ----------
    batch : dict
        Columnar batch from datasets.map() with character/style/font lists.
    path_cache : dict
        Pre-built mapping: content paths+dims, target paths+dims,
        style dims, style paths.
    style_bytes_cache : dict
        Pre-loaded style image bytes keyed by style name.
    resize_height : int
        Target height for all resized images.
    spacing : int
        Pixel spacing between panels in the comparison image.

    Returns
    -------
    dict
        Columnar results matching the dataset Features schema.
    """
    content_cache = path_cache["content"]
    target_cache = path_cache["target"]
    style_dims_cache = path_cache["style_dims"]
    style_paths_cache = path_cache["style_paths"]
    rh = resize_height

    batch_chars: list = batch["character"]
    batch_styles: list = batch["style"]
    batch_fonts: list = batch["font"]
    batch_size = len(batch_chars)

    # ------------------------------------------------------------------
    # 1. Single-pass validation — collect only processable items
    # ------------------------------------------------------------------
    chars, styles, fonts = [], [], []
    content_infos, target_infos, style_dim_list = [], [], []
    skipped = 0
    failure_samples: list = []

    for i in range(batch_size):
        char = batch_chars[i]
        style = batch_styles[i]

        ci = content_cache.get(char)
        if not ci:
            skipped += 1
            if len(failure_samples) < 3:
                failure_samples.append((f"{char}/{style}", "no content_info"))
            continue

        ti = target_cache.get(style, {}).get(char)
        if not ti:
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
        content_infos.append(ci)
        target_infos.append(ti)
        style_dim_list.append(sdims)

    if not chars:
        if skipped and failure_samples:
            logger.warning(
                "Batch skipped %d/%d items. Examples: %s",
                batch_size, batch_size, failure_samples,
            )
        return {k: [] for k in (
            "character", "style", "font",
            "content_image", "style_image", "target_image",
            "comparison_image", "content_hash", "target_hash",
        )}

    n = len(chars)

    # ------------------------------------------------------------------
    # 2. Vectorised resize-dimension computation (numpy, no Python loop)
    # ------------------------------------------------------------------
    c_h = np.array([ci["height"] for ci in content_infos], dtype=np.float32)
    c_w = (np.array([ci["width"] for ci in content_infos], dtype=np.float32)
           * (rh / c_h)).astype(np.int32)

    t_h = np.array([ti["height"] for ti in target_infos], dtype=np.float32)
    t_w = (np.array([ti["width"] for ti in target_infos], dtype=np.float32)
           * (rh / t_h)).astype(np.int32)

    s_w = np.array(
        [int(sw * (rh / sh)) for sw, sh in style_dim_list], dtype=np.int32
    )

    # ------------------------------------------------------------------
    # 3. Load → resize → encode sequentially within this worker.
    #    datasets.map(num_proc=N) provides the outer parallelism across
    #    shards; no manual threads or processes needed here.
    # ------------------------------------------------------------------
    content_bytes, target_bytes, comparison_bytes = [], [], []

    for i in range(n):
        c_arr = _load_resize_bgr(content_infos[i]["path"], int(c_w[i]), rh)
        t_arr = _load_resize_bgr(target_infos[i]["path"], int(t_w[i]), rh)
        s_arr = _load_resize_bgr(style_paths_cache[styles[i]], int(s_w[i]), rh)

        content_bytes.append(_encode_bgr(c_arr))
        target_bytes.append(_encode_bgr(t_arr))
        comparison_bytes.append(_assemble_and_encode(c_arr, s_arr, t_arr, spacing))

    # ------------------------------------------------------------------
    # 4. Assemble result columns
    # ------------------------------------------------------------------
    if skipped and failure_samples:
        logger.warning(
            "Batch skipped %d/%d items. Examples: %s",
            skipped, batch_size, failure_samples,
        )

    return {
        "character": chars,
        "style": styles,
        "font": fonts,
        "content_image":    [{"bytes": b} for b in content_bytes],
        "style_image":      [{"bytes": style_bytes_cache[styles[i]]} for i in range(n)],
        "target_image":     [{"bytes": b} for b in target_bytes],
        "comparison_image": [{"bytes": b} for b in comparison_bytes],
        "content_hash": [compute_file_hash(chars[i], "", fonts[i]) for i in range(n)],
        "target_hash":  [compute_file_hash(chars[i], styles[i], fonts[i]) for i in range(n)],
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

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

        # Use all CPUs; datasets.map() manages the worker pool entirely
        self.num_proc = config.num_proc
        # Large batches amortise Python/IPC overhead per map() worker
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
        logger.info("  Total generations : %d", len(self.generations))
        logger.info("  CPU workers       : %d processes", self.num_proc)
        logger.info("  Batch size        : %d", self.process_batch_size)
        logger.info("  Style images      : %d", len(self.style_paths))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
        for d in self.REQUIRED_DIRS:
            p = self.data_dir / d
            if not p.exists():
                raise ValueError(f"Required directory not found: {p}")
        cp = self.data_dir / self.CHECKPOINT_FILE
        if not cp.exists():
            raise ValueError(f"Checkpoint file not found: {cp}")
        if not self.style_images_dir.exists():
            raise ValueError(
                f"Style images directory not found: {self.style_images_dir}"
            )
        logger.info("Directory structure validated")

    def _build_style_path_index(self) -> None:
        logger.info("Building style image path index...")
        self.style_paths = {
            f.stem: f
            for ext in (".png", ".jpg", ".jpeg")
            for f in self.style_images_dir.glob(f"*{ext}")
        }
        logger.info("Indexed %d style images", len(self.style_paths))

    def _build_path_cache_with_dims(self) -> None:
        """Build path + dimension cache for content, target, and style images."""
        logger.info("Building path cache with dimensions...")

        content_paths: dict = {}
        content_dir = self.data_dir / "ContentImage"
        if content_dir.exists():
            for f in content_dir.glob("*"):
                if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                try:
                    with Image.open(f) as img:
                        w, h = img.size
                    content_paths[f.stem] = {"path": str(f), "width": w, "height": h}
                except Exception as e:
                    logger.debug("Failed to read %s: %s", f, e)

        target_paths: dict = {}
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in target_dir.iterdir():
                if not style_dir.is_dir():
                    continue
                style_char_paths: dict = {}
                for f in style_dir.glob("*"):
                    if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                    parts = f.stem.split("+")
                    if len(parts) < 2:
                        continue
                    try:
                        with Image.open(f) as img:
                            w, h = img.size
                        style_char_paths[parts[1]] = {
                            "path": str(f), "width": w, "height": h,
                        }
                    except Exception as e:
                        logger.debug("Failed to read %s: %s", f, e)
                target_paths[style_dir.name] = style_char_paths

        style_dims: dict = {}
        for name, path in self.style_paths.items():
            try:
                with Image.open(path) as img:
                    style_dims[name] = img.size
            except Exception as e:
                logger.debug("Failed style dims for %s: %s", path, e)

        self.path_cache = {
            "content":     content_paths,
            "target":      target_paths,
            "style_dims":  style_dims,
            "style_paths": {k: str(v) for k, v in self.style_paths.items()},
        }
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(
            "Path cache built: %d content, %d target paths",
            len(content_paths), total_targets,
        )

    def _preload_style_bytes(self) -> None:
        """Load all style images into memory once — reused across all batches."""
        logger.info("Pre-loading style image bytes...")
        for name, path in self.style_paths.items():
            try:
                self.style_bytes_cache[name] = path.read_bytes()
            except Exception as e:
                logger.warning("Failed to pre-load style image %s: %s", name, e)
        logger.info("Pre-loaded %d style images", len(self.style_bytes_cache))

    def _load_checkpoint(self) -> list[dict]:
        cp = self.data_dir / self.CHECKPOINT_FILE
        with open(cp, "r", encoding="utf-8") as f:
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
        """Build the dataset via datasets.map() parallelism.

        Returns
        -------
        Dataset
            HuggingFace Dataset with all image columns populated.
        """
        logger.info("Building dataset with batched map() pipeline...")
        start = time.time()

        gens = self.generations
        thin = Dataset.from_dict({
            "character": [g.get("character", "")      for g in gens],
            "style":     [g.get("style", "")          for g in gens],
            "font":      [g.get("font", "unknown")    for g in gens],
        })

        features = Features({
            "character":        Value("string"),
            "style":            Value("string"),
            "font":             Value("string"),
            "content_image":    HFImage(),
            "style_image":      HFImage(),
            "target_image":     HFImage(),
            "comparison_image": HFImage(),
            "content_hash":     Value("string"),
            "target_hash":      Value("string"),
        })

        logger.info(
            "Processing with %d workers (batch_size=%d)...",
            self.num_proc, self.process_batch_size,
        )

        dataset = thin.map(
            _process_batch,
            batched=True,
            batch_size=self.process_batch_size,
            num_proc=self.num_proc,
            features=features,
            remove_columns=thin.column_names,
            desc="Processing images",
            fn_kwargs={
                "path_cache":        self.path_cache,
                "style_bytes_cache": self.style_bytes_cache,
                "resize_height":     self.resize_height,
                "spacing":           self.spacing,
            },
        )

        elapsed = time.time() - start
        n = len(dataset)
        logger.info(
            "Dataset built: %d samples in %.2fs (%.1f samples/s)",
            n, elapsed, n / elapsed if elapsed else 0,
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
            logger.error(
                "Dataset is empty — skipping upload. "
                "Check path_cache alignment with checkpoint."
            )
            return

        logger.info("Uploading to %s...", self.config.repo_id)
        start = time.time()
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
            elapsed = time.time() - start
            n = len(dataset)
            logger.info(
                "Upload completed in %.2fs (%.1f samples/s)",
                elapsed, n / elapsed if elapsed else 0,
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
        start = time.time()
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved in %.2fs", time.time() - start)


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
    num_proc: int = 1,
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
        num_proc=num_proc,
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
    parser.add_argument("--data-dir", required=True,
        help="Path to data directory (must contain ContentImage/ and TargetImage/)")
    parser.add_argument("--style-images-dir", required=True,
        help="Path to style images directory")
    parser.add_argument("--repo-id", required=True,
        help="HuggingFace repository ID (username/dataset-name)")
    parser.add_argument("--split", default="train",
        help="Dataset split name (default: train)")
    parser.add_argument("--config-name", help="Dataset configuration name")
    parser.add_argument("--no-push", action="store_true",
        help="Skip pushing to HuggingFace Hub")
    parser.add_argument("--private", action="store_true",
        help="Make repository private")
    parser.add_argument("--local-save", help="Save dataset locally to this path")
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument("--resize-height", type=int, default=256,
        help="Height for comparison images (default: 256)")
    parser.add_argument("--spacing", type=int, default=10,
        help="Spacing between images in comparison (default: 10)")
    parser.add_argument("--verbose", action="store_true",
        help="Enable verbose logging")
    parser.add_argument("--num-proc", type=int, default=1,
        help="Number of processes to use for parallel processing (default: 1)")
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        start = time.time()
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
            num_proc=args.num_proc,
        )
        total = time.time() - start
        n = len(dataset)
        print(f"\nDataset creation completed in {total:.2f}s")
        print(f"Samples : {n}")
        print(f"Speed   : {n / total if total else 0:.1f} samples/second")
        print(f"Unique characters : {len(set(dataset['character']))}")
        print(f"Unique styles     : {len(set(dataset['style']))}")
        if not args.no_push:
            print(f"Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        if args.local_save:
            print(f"Local copy saved to: {args.local_save}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception("Dataset creation failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()