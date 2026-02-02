import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from datasets import Dataset, load_dataset
from PIL import Image
from filename_utils import (
    compute_file_hash,
    get_content_filename,
    get_target_filename,
)
from utilities import HFTqdm
logger = logging.getLogger(__name__)

def _validate_dataset_structure(dataset: Dataset) -> None:
    required_columns = {"character", "style"}
    image_columns = {"content_image", "target_image"}
    dataset_cols = set(dataset.column_names)
    if not required_columns.issubset(dataset_cols):
        missing = required_columns - dataset_cols
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Has columns: {dataset_cols}"
        )
    if not image_columns.intersection(dataset_cols):
        raise ValueError(
            f"Dataset must have at least one of {image_columns}. "
            f"Has columns: {dataset_cols}"
        )
    logger.info(f"✓ Dataset structure validated. Columns: {dataset_cols}")

def _atomic_save_image(img: Image.Image, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=final_path.suffix, dir=final_path.parent
    )
    try:
        with os.fdopen(temp_fd, "wb") as f:
            img.save(f, format=img.format or "PNG")
        os.replace(temp_path, final_path)
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e

@staticmethod
def _process_batch(batch: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    batch_size = len(batch["character"])
    content_saved = []
    target_saved = []
    content_filenames = []
    target_filenames = []
    content_hashes = []
    target_hashes = []
    for i in HFTqdm(range(batch_size), desc="Processing batch"):
        char = batch["character"][i]
        style = batch["style"][i]
        font = batch.get("font", ["unknown"] * batch_size)[i]
        content_img = batch.get("content_image", [None] * batch_size)[i]
        content_filename = get_content_filename(char)
        if isinstance(content_img, Image.Image):
            try:
                final_path = output_path / "ContentImage" / content_filename
                _atomic_save_image(content_img, final_path)
                content_saved.append(True)
            except Exception as e:
                logger.debug(f"Failed to save content {content_filename}: {e}")
                content_saved.append(False)
        else:
            content_saved.append(False)
        target_img = batch.get("target_image", [None] * batch_size)[i]
        target_filename = get_target_filename(char, style)
        if isinstance(target_img, Image.Image):
            try:
                final_path = output_path / "TargetImage" / style / target_filename
                _atomic_save_image(target_img, final_path)
                target_saved.append(True)
            except Exception as e:
                logger.debug(f"Failed to save target {target_filename}: {e}")
                target_saved.append(False)
        else:
            target_saved.append(False)
        content_filenames.append(content_filename)
        target_filenames.append(target_filename)
        content_hashes.append(compute_file_hash(char, "", font))
        target_hashes.append(compute_file_hash(char, style, font))
    return {
        "content_saved": content_saved,
        "target_saved": target_saved,
        "content_filename": content_filenames,
        "target_filename": target_filenames,
        "content_hash": content_hashes,
        "target_hash": target_hashes,
    }

@dataclass
class ExportConfig:
    output_dir: Path
    repo_id: Optional[str] = None
    local_dataset_path: Optional[Path] = None
    split: str = "train"
    config_name: Optional[str] = None
    token: Optional[str] = None
    num_workers: int = 4
    batch_size: int = 1000
    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.local_dataset_path, str):
            self.local_dataset_path = Path(self.local_dataset_path)
        if not self.repo_id and not self.local_dataset_path:
            raise ValueError(
                "Must provide either repo_id (Hub) or local_dataset_path (disk)"
            )

def _process_single_export(sample: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    char = sample["character"]
    style = sample["style"]
    font = sample.get("font", "unknown")
    result = {
        "content_saved": False,
        "target_saved": False,
        "content_filename": get_content_filename(char),
        "target_filename": get_target_filename(char, style),
        "content_hash": compute_file_hash(char, "", font),
        "target_hash": compute_file_hash(char, style, font),
    }
    content_img = sample.get("content_image")
    if isinstance(content_img, Image.Image):
        try:
            final_path = output_path / "ContentImage" / result["content_filename"]
            _atomic_save_image(content_img, final_path)
            result["content_saved"] = True
        except Exception as e:
            logger.debug(f"Failed to save content {result['content_filename']}: {e}")
    target_img = sample.get("target_image")
    if isinstance(target_img, Image.Image):
        try:
            final_path = output_path / "TargetImage" / style / result["target_filename"]
            _atomic_save_image(target_img, final_path)
            result["target_saved"] = True
        except Exception as e:
            logger.debug(f"Failed to save target {result['target_filename']}: {e}")
    return result

def _process_batch_export(batch: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    batch_size = len(batch["character"])
    results = {
        "content_saved": [],
        "target_saved": [],
        "content_filename": [],
        "target_filename": [],
        "content_hash": [],
        "target_hash": [],
    }
    for i in range(batch_size):
        char = batch["character"][i]
        style = batch["style"][i]
        font = batch.get("font", ["unknown"] * batch_size)[i]
        content_filename = get_content_filename(char)
        target_filename = get_target_filename(char, style)
        content_saved = False
        content_img = batch.get("content_image", [None] * batch_size)[i]
        if isinstance(content_img, Image.Image):
            try:
                final_path = output_path / "ContentImage" / content_filename
                _atomic_save_image(content_img, final_path)
                content_saved = True
            except Exception as e:
                logger.debug(f"Failed to save content {content_filename}: {e}")
        target_saved = False
        target_img = batch.get("target_image", [None] * batch_size)[i]
        if isinstance(target_img, Image.Image):
            try:
                final_path = output_path / "TargetImage" / style / target_filename
                _atomic_save_image(target_img, final_path)
                target_saved = True
            except Exception as e:
                logger.debug(f"Failed to save target {target_filename}: {e}")
        results["content_saved"].append(content_saved)
        results["target_saved"].append(target_saved)
        results["content_filename"].append(content_filename)
        results["target_filename"].append(target_filename)
        results["content_hash"].append(compute_file_hash(char, "", font))
        results["target_hash"].append(compute_file_hash(char, style, font))
    return results

class UltraFastDatasetExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"
        self.cpu_count = os.cpu_count() or 4
        self.num_workers = min(config.num_workers, self.cpu_count)
        self.batch_size = config.batch_size
        logger.info(f"Ultra-fast exporter initialized:")
        logger.info(f"  Workers: {self.num_workers} parallel processes")
        logger.info(f"  Output: {self.output_dir}")

    def _load_dataset(self) -> Dataset:
        if self.config.local_dataset_path:
            logger.info(f"Loading local dataset from {self.config.local_dataset_path}")
            if not self.config.local_dataset_path.exists():
                raise ValueError(
                    f"Local dataset path does not exist: {self.config.local_dataset_path}"
                )
            try:
                dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
                _validate_dataset_structure(dataset)
                logger.info(f"Loaded {len(dataset)} samples from disk")
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e
        config_msg = (
            f" (config: {self.config.config_name})" if self.config.config_name else ""
        )
        logger.info(
            f"Loading dataset from Hub: {self.config.repo_id} "
            f"(split: {self.config.split}){config_msg}"
        )
        try:
            dataset = load_dataset(
                self.config.repo_id,
                name=self.config.config_name,
                split=self.config.split,
                token=self.config.token,
                num_proc=self.num_workers,
            )
            _validate_dataset_structure(dataset)
            logger.info(f"Loaded {len(dataset)} samples from Hub")
            return dataset
        except Exception as e:
            logger.error(
                f"Failed to load from Hub '{self.config.repo_id}': {e}\n"
                f"Check that:\n"
                f"  - Dataset exists and is public (or token is valid)\n"
                f"  - Config name '{self.config.config_name}' is correct\n"
                f"  - Split '{self.config.split}' exists"
            )
            raise ValueError(f"Failed to load dataset: {e}") from e

    def _precreate_directories(self, dataset: Dataset) -> None:
        logger.info("Pre-creating directory structure...")
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        unique_styles = set(dataset["style"])
        for style in unique_styles:
            style_dir = self.target_dir / style
            style_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Pre-created {len(unique_styles)} style directories")

    def _export_with_map(self, dataset: Dataset) -> dict[str, Any]:
        logger.info(f"Exporting with {self.num_workers} workers (batch_size={self.batch_size})...")
        checkpoint_path = self.output_dir / "results_checkpoint.json"
        processed_dataset = dataset.map(
            _process_batch_export,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers,
            desc="Exporting images",
            fn_kwargs={"output_dir": str(self.output_dir)},
        )
        logger.info("Writing results_checkpoint.json...")
        with checkpoint_path.open("w", encoding="utf-8") as json_file:
            json_file.write('{\n  "generations": [\n')
            generations_count = 0
            characters = set()
            styles = set()
            fonts = set()
            for i, sample in enumerate(processed_dataset):
                char = sample["character"]
                style = sample["style"]
                font = sample.get("font", "unknown")
                characters.add(char)
                styles.add(style)
                fonts.add(font)
                if sample.get("target_saved", False):
                    generation = {
                        "character": char,
                        "style": style,
                        "font": font,
                        "content_image_path": f"ContentImage/{sample['content_filename']}",
                        "target_image_path": f"TargetImage/{style}/{sample['target_filename']}",
                        "content_hash": sample["content_hash"],
                        "target_hash": sample["target_hash"],
                    }
                    if generations_count > 0:
                        json_file.write(",\n")
                    json_file.write("    ")
                    json.dump(generation, json_file, ensure_ascii=False)
                    generations_count += 1
                if (i + 1) % 1000 == 0:
                    progress_pct = (i + 1) * 100 // len(processed_dataset)
                    logger.info(
                        f"Progress: {i + 1}/{len(processed_dataset)} samples "
                        f"({progress_pct}%) - {generations_count} targets written"
                    )
            json_file.write("\n  ],\n")
            json_file.write(
                f'  "characters": {json.dumps(sorted(characters), ensure_ascii=False)},\n'
            )
            json_file.write(
                f'  "styles": {json.dumps(sorted(styles), ensure_ascii=False)},\n'
            )
            json_file.write(
                f'  "fonts": {json.dumps(sorted(fonts) if fonts else ["unknown"], ensure_ascii=False)},\n'
            )
            json_file.write(f'  "total_chars": {len(characters)},\n')
            json_file.write(f'  "total_styles": {len(styles)}\n')
            json_file.write("}\n")
        content_count = len(list(self.content_dir.glob("*.png")))
        logger.info(
            f"Exported {content_count} content images, {generations_count} target images"
        )
        return {
            "generations_count": generations_count,
            "content_count": content_count,
            "characters": sorted(characters),
            "styles": sorted(styles),
            "fonts": sorted(fonts) if fonts else ["unknown"],
            "total_chars": len(characters),
            "total_styles": len(styles),
        }
    
    def export(self) -> dict[str, Any]:
        logger.info("Starting ultra-fast dataset export...")
        dataset = self._load_dataset()
        self._precreate_directories(dataset)
        metadata = self._export_with_map(dataset)
        logger.info("Export completed successfully")
        return metadata

def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    split: str = "train",
    config_name: Optional[str] = None,
    token: Optional[str] = None,
    num_workers: int = 4,
    batch_size: int = 1000,
) -> dict[str, Any]:
    config = ExportConfig(
        output_dir=Path(output_dir),
        repo_id=repo_id,
        local_dataset_path=Path(local_dataset_path) if local_dataset_path else None,
        split=split,
        config_name=config_name,
        token=token,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    exporter = UltraFastDatasetExporter(config)
    return exporter.export()

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ultra-fast HuggingFace dataset exporter using dataset.map(num_proc=...)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from local dataset
  python src/tools/export_dataset_ultra.py --output-dir ./output --local-path my_dataset/
  
  # Export from Hub
  python src/tools/export_dataset_ultra.py --output-dir ./output --repo-id user/dataset --workers 8
        """,
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repository ID")
    parser.add_argument("--local-path", type=str, help="Local dataset path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--config-name", type=str, help="Dataset config name")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        metadata = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            config_name=args.config_name,
            token=args.token,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )
        print(f"\n✅ Export completed!")
        print(f"📊 Content images: {metadata['content_count']}")
        print(f"📊 Target images: {metadata['generations_count']}")
        print(f"🔤 Characters: {metadata['total_chars']}")
        print(f"🎨 Styles: {metadata['total_styles']}")
    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()