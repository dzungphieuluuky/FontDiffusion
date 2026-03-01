import json, logging, os, tempfile, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from datasets import Dataset, load_dataset
from PIL import Image
from filename_utils import compute_file_hash, get_content_filename, get_target_filename
from utilities import HFTqdm

logger = logging.getLogger(__name__)


def _validate_dataset_structure(dataset: Dataset) -> None:
    cols = set(dataset.column_names)
    if not {"character", "style"}.issubset(cols):
        raise ValueError(f"Missing required columns. Has: {cols}")
    if not {"content_image", "target_image"}.intersection(cols):
        raise ValueError(f"Missing image columns. Has: {cols}")


def _atomic_save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            img.save(f, format=img.format or "PNG")
        os.replace(tmp, path)
    except:
        os.unlink(tmp)
        raise


def _process_batch_export(batch: dict, output_dir: str) -> dict:
    out_path = Path(output_dir)
    size = len(batch["character"])
    fonts = batch.get("font", ["unknown"] * size)
    results = {
        k: []
        for k in [
            "content_saved",
            "target_saved",
            "content_filename",
            "target_filename",
            "content_hash",
            "target_hash",
        ]
    }

    for i in range(size):
        char, style, font = batch["character"][i], batch["style"][i], fonts[i]
        c_fn, t_fn = get_content_filename(char), get_target_filename(char, style)

        results["content_filename"].append(c_fn)
        results["target_filename"].append(t_fn)
        results["content_hash"].append(compute_file_hash(char, "", font))
        results["target_hash"].append(compute_file_hash(char, style, font))

        # Content Image
        c_img = batch.get("content_image", [None])[i]
        c_saved = False
        if isinstance(c_img, Image.Image):
            try:
                _atomic_save_image(c_img, out_path / "ContentImage" / c_fn)
                c_saved = True
            except Exception as e:
                logger.debug(f"Err content {c_fn}: {e}")
        results["content_saved"].append(c_saved)

        # Target Image
        t_img = batch.get("target_image", [None])[i]
        t_saved = False
        if isinstance(t_img, Image.Image):
            try:
                _atomic_save_image(t_img, out_path / "TargetImage" / style / t_fn)
                t_saved = True
            except Exception as e:
                logger.debug(f"Err target {t_fn}: {e}")
        results["target_saved"].append(t_saved)

    return results


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
        self.output_dir = Path(self.output_dir)
        if self.local_dataset_path:
            self.local_dataset_path = Path(self.local_dataset_path)
        if not self.repo_id and not self.local_dataset_path:
            raise ValueError("Require repo_id or local_dataset_path")


class DatasetExporter:
    def __init__(self, config: ExportConfig):
        self.cfg = config
        self.num_workers = min(config.num_workers, os.cpu_count() or 4)

    def _load_dataset(self) -> Dataset:
        if self.cfg.local_dataset_path:
            ds = Dataset.load_from_disk(str(self.cfg.local_dataset_path))
        else:
            ds = load_dataset(
                self.cfg.repo_id,
                name=self.cfg.config_name,
                split=self.cfg.split,
                token=self.cfg.token,
                num_proc=self.num_workers,
            )
        _validate_dataset_structure(ds)
        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _export_with_map(self, dataset: Dataset) -> dict[str, Any]:
        logger.info(f"Exporting with {self.num_workers} workers...")
        processed = dataset.map(
            _process_batch_export,
            batched=True,
            batch_size=self.cfg.batch_size,
            num_proc=self.num_workers,
            desc="Exporting",
            fn_kwargs={"output_dir": str(self.cfg.output_dir)},
        )

        ckpt_path = self.cfg.output_dir / "results_checkpoint.json"
        chars, styles, fonts, gen_count = set(), set(), set(), 0

        with ckpt_path.open("w", encoding="utf-8") as f:
            f.write('{\n  "generations": [\n')
            for i, s in enumerate(processed):
                c, st, fn = s["character"], s["style"], s.get("font", "unknown")
                chars.add(c)
                styles.add(st)
                fonts.add(fn)

                if s.get("target_saved"):
                    rec = {
                        "character": c,
                        "style": st,
                        "font": fn,
                        "content_image_path": f"ContentImage/{s['content_filename']}",
                        "target_image_path": f"TargetImage/{st}/{s['target_filename']}",
                        "content_hash": s["content_hash"],
                        "target_hash": s["target_hash"],
                    }
                    if gen_count > 0:
                        f.write(",\n")
                    f.write("    " + json.dumps(rec, ensure_ascii=False))
                    gen_count += 1

                if (i + 1) % 1000 == 0:
                    logger.info(f"Progress: {i + 1}/{len(processed)}")

            f.write(f'\n  ],\n  "characters": {json.dumps(sorted(chars))},\n')
            f.write(f'  "styles": {json.dumps(sorted(styles))},\n')
            f.write(f'  "fonts": {json.dumps(sorted(fonts))},\n')
            f.write(
                f'  "total_chars": {len(chars)},\n  "total_styles": {len(styles)}\n}}\n'
            )

        content_count = len(list((self.cfg.output_dir / "ContentImage").glob("*.png")))
        return {
            "generations_count": gen_count,
            "content_count": content_count,
            "total_chars": len(chars),
            "total_styles": len(styles),
            "characters": sorted(chars),
            "styles": sorted(styles),
            "fonts": sorted(fonts),
        }

    def export(self) -> dict[str, Any]:
        ds = self._load_dataset()
        # Pre-create dirs
        (self.cfg.output_dir / "ContentImage").mkdir(parents=True, exist_ok=True)
        for s in set(ds["style"]):
            (self.cfg.output_dir / "TargetImage" / s).mkdir(parents=True, exist_ok=True)
        return self._export_with_map(ds)


def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    **kwargs,
) -> dict[str, Any]:
    cfg = ExportConfig(
        output_dir=output_dir,
        repo_id=repo_id,
        local_dataset_path=local_dataset_path,
        **kwargs,
    )
    return DatasetExporter(cfg).export()


def main():
    parser = argparse.ArgumentParser(description="Fast Dataset Exporter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--local-path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name")
    parser.add_argument("--token")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1)
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        meta = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            config_name=args.config_name,
            token=args.token,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )
        print(
            f"\n✅ Done! Content: {meta['content_count']}, Targets: {meta['generations_count']}"
        )
    except Exception as e:
        logger.exception(f"Failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
