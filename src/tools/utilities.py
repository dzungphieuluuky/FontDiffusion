#!/usr/bin/env python3
"""Utility functions for model inspection, file‑system helpers, checkpoint conversion and CLI."""

from __future__ import annotations
import argparse, json, logging, shutil, sys, time, os
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar, Generic, Iterator
import torch
from safetensors.torch import save_file, load_file as safe_load
from datasets.utils import tqdm as hf_tqdm, enable_progress_bar
from tqdm.auto import tqdm as auto_tqdm
enable_progress_bar()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

logger = setup_logger(__name__)

# --------------------------------------------------------------------------- #
# Model utilities
# --------------------------------------------------------------------------- #

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def log_model_info(model: torch.nn.Module, name: str = "Model") -> None:
    if hasattr(model, "log_model_info"):
        model.log_model_info()
    else:
        total, trainable = count_parameters(model)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"{name} Parameter Summary")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Non‑trainable parameters: {total - trainable:,}")
        logger.info(f"{'=' * 80}\n")

# --------------------------------------------------------------------------- #
# Hugging‑Face style progress bar
# --------------------------------------------------------------------------- #

HF_BLUE, HF_GREEN, HF_ORANGE, HF_RED, HF_CYAN, HF_INDIGO = (
    "#1055C9", "#41A67E", "#FF8C00", "#E03E3E", "#00B8D9", "#6554C0"
)
HF_BAR_FORMAT = ("{desc}: {percentage:3.0f}%|{bar}| "
                 "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

T = TypeVar("T")

class HFTqdm(auto_tqdm, Generic[T]):
    def __init__(self, iterable: Optional[Iterable[T]] = None,
                 desc: str = "Processing", total: Optional[int] = None,
                 unit: str = "iteration", disable: bool = False, **kwargs: Any):
        kwargs.setdefault("unit", unit); kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("bar_format", HF_BAR_FORMAT); kwargs.setdefault("colour", HF_INDIGO)
        kwargs.setdefault("ascii", False); kwargs.setdefault("ncols", 100)
        kwargs.setdefault("leave", True); kwargs["disable"] = disable
        super().__init__(iterable=iterable, desc=desc, total=total, **kwargs)
        self._base_desc = desc; self._start_time = time.time(); self._warning_shown = False

    def __iter__(self) -> Iterator[T]: return super().__iter__()          # type: ignore[return-value]
    def update(self, n: int = 1) -> None:
        super().update(n)
        if self.total:
            self.colour = HF_INDIGO if self.n / self.total < 1.0 else HF_GREEN
    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        if desc: self._base_desc = desc
        super().set_description(desc, refresh=refresh)
    def set_postfix(self, ordered_dict: Optional[dict[str, Any]] = None,
                    refresh: bool = True, **kwargs: Any) -> None:
        super().set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)
    def close(self) -> None:
        if self.total and self.n >= self.total:
            self.colour = HF_GREEN
            elapsed = time.time() - self._start_time
            time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}min"
            self.set_description(f"✓ {self._base_desc}", refresh=False)
        else:
            self.colour = HF_ORANGE
        self.refresh(); super().close()
    def __enter__(self) -> HFTqdm: return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type:
            self.colour = HF_RED
            self.set_description(f"✗ {self._base_desc} (failed)", refresh=False)
        self.close(); return False

# --------------------------------------------------------------------------- #
# Path utilities
# --------------------------------------------------------------------------- #

def _ensure_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)

# --------------------------------------------------------------------------- #
# Checkpoint utilities
# --------------------------------------------------------------------------- #

def load_model_checkpoint(path: Path | str) -> dict[str, Any]:
    path = _ensure_path(path)
    if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
    return safe_load(path) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")

def save_model_checkpoint(state: dict[str, Any], path: Path | str) -> None:
    path = _ensure_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file as safe_save
    if path.suffix == ".safetensors": safe_save(state, path)
    else: torch.save(state, path)

def find_checkpoint(dir_: Path | str, name: str) -> Path:
    dir_ = _ensure_path(dir_)
    for ext in (".safetensors", ".pth"):
        cand = dir_ / f"{name}{ext}"
        if cand.exists(): return cand
    raise FileNotFoundError(f"Checkpoint not found for '{name}' in {dir_}")

# --------------------------------------------------------------------------- #
# File‑system helpers
# --------------------------------------------------------------------------- #

def flatten_folder(root: Path | str) -> None:
    root = _ensure_path(root)
    for sub, _, files in os.walk(root):
        if Path(sub) == root: continue
        for f in files:
            src = Path(sub) / f
            dst = root / f
            if dst.exists():
                base, ext = dst.stem, dst.suffix
                i = 1
                while (root / f"{base}_{i}{ext}").exists(): i += 1
                dst = root / f"{base}_{i}{ext}"
            shutil.move(str(src), str(dst))
    for sub, _, _ in os.walk(root, topdown=False):
        sp = Path(sub)
        if sp != root and not any(sp.iterdir()): sp.rmdir()

def rename_images(json_file: Path | str) -> None:
    json_file = _ensure_path(json_file)
    with json_file.open("r", encoding="utf-8") as f: data = json.load(f)
    for entry in data.get("generations", []):
        char, style = entry.get("character"), entry.get("style")
        for key in ("content_image_path", "target_image_path"):
            old = entry.get(key)
            if not old: continue
            old_path = Path(old)
            if not old_path.exists(): print(f"Skipping: File not found {old_path}"); continue
            new_name = f"{char}.png" if "content" in key else f"{style}+{char}.png"
            new_path = old_path.parent / new_name
            try:
                old_path.rename(new_path)
                entry[key] = str(new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except OSError as e: print(f"Error renaming {old_path}: {e}")
    out = json_file.with_name("updated_generations.json")
    with out.open("w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nProcessing complete. Updated JSON saved as: {out}")

def rename_content_images(dir_: Path | str) -> None:
    dir_ = _ensure_path(dir_) / "ContentImage"
    for f in dir_.iterdir():
        if "+" in f.name:
            new = dir_ / f.name.split("+")[1]
            f.rename(new)
            print(f"Renamed: {f} -> {new}")

def update_paths(inp: Path | str, out: Optional[Path | str] = None) -> None:
    inp = _ensure_path(inp); out = _ensure_path(out) if out else inp
    with inp.open("r", encoding="utf-8") as f: data = json.load(f)
    for item in data.get("generations", []):
        char, style = item["character"], item["style"]
        item["content_image_path"] = f"ContentImage/{char}.png"
        item["target_image_path"] = f"TargetImage/{style}/{style}+{char}.png"
    with out.open("w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def print_font_glyph_counts(dir_: Path | str) -> None:
    from fontTools.ttLib import TTFont
    dir_ = _ensure_path(dir_)
    exts = {".ttf", ".otf"}
    fonts = [p for p in dir_.iterdir() if p.suffix.lower() in exts]
    print(f"\nFont glyph count summary for directory: {dir_}")
    for fp in fonts:
        try:
            cmap = TTFont(fp).getBestCmap()
            print(f"  {fp.stem}: {len(cmap)} glyphs")
        except Exception as e: print(f"  {fp.name}: Failed to read ({e})")

# --------------------------------------------------------------------------- #
# Conversion utilities
# --------------------------------------------------------------------------- #

def pth_to_safetensors(pth: Path | str, out: Path | str) -> None:
    pth, out = _ensure_path(pth), _ensure_path(out)
    print(f"\nConverting {pth.name} to safetensors...")
    state = torch.load(pth, map_location="cpu")
    save_file(state, out)
    size_pth = pth.stat().st_size / (1024**3)
    size_safe = out.stat().st_size / (1024**3)
    print(f"✓ Converted: {pth}")
    print(f"  .pth size:  {size_pth:.2f} GB")
    print(f"  .safetensors size: {size_safe:.2f} GB")

def convert_checkpoint_folder(src_dir: Path | str, dst_dir: Optional[Path | str] = None) -> Path:
    src_dir = _ensure_path(src_dir); dst_dir = _ensure_path(dst_dir) if dst_dir else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nCONVERTING CHECKPOINTS TO SAFETENSORS\n{'=' * 60}")
    pths = list(src_dir.glob("*.pth"))
    if not pths:
        print(f"⚠ No .pth files found in {src_dir}")
        return dst_dir
    print(f"Found {len(pths)} .pth files")
    for pth in pths:
        pth_to_safetensors(pth, dst_dir / f"{pth.stem}.safetensors")
    print(f"\n✓ Conversion complete!")
    return dst_dir

# --------------------------------------------------------------------------- #
# Command‑line interface
# --------------------------------------------------------------------------- #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Utility functions for font diffusion project")
    parser.add_argument("--flatten_dir", type=str, help="Flatten directory")
    parser.add_argument("--font_glyphs_dir", type=str, help="Print glyph counts")
    parser.add_argument("--convert_ckpt_dir", type=str, help="Convert .pth checkpoints")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for safetensors")
    parser.add_argument("--rename_images_json", type=str, help="Rename images from JSON")
    parser.add_argument("--rename_content_images_dir", type=str, help="Rename content images")
    parser.add_argument("--update_paths", type=str, nargs=2, help="Update JSON paths")
    args = parser.parse_args()

    if args.flatten_dir: flatten_folder(args.flatten_dir); print(f"✓ Flattened directory: {args.flatten_dir}")
    if args.font_glyphs_dir: print_font_glyph_counts(args.font_glyphs_dir)
    if args.convert_ckpt_dir: convert_checkpoint_folder(args.convert_ckpt_dir, args.output_dir)
    if args.rename_images_json: rename_images(args.rename_images_json)
    if args.rename_content_images_dir: rename_content_images(args.rename_content_images_dir)
    if args.update_paths: update_paths(*args.update_paths); print(f"✓ Updated paths in JSON file: {args.update_paths}")
    if not any(vars(args).values()): print("No arguments provided. Use --help for usage information.")

if __name__ == "__main__":
    _cli()