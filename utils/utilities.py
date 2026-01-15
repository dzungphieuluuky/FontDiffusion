from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional
import os

import torch
from safetensors.torch import save_file
from tqdm.rich import tqdm_rich as rich_tqdm
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm import tqdm as auto_tqdm


# --------------------------------------------------------------------------- #
# Hugging‑Face style progress bar
# --------------------------------------------------------------------------- #

HF_BLUE = "#1055C9"
HF_GREEN = "#41A67E"
HF_ORANGE = "#FF8C00"
HF_RED = "#E03E3E"

HF_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


class HFTqdm(auto_tqdm):
    """
    Enhanced tqdm progress bar that mimics the Hugging‑Face download UI.
    Optimized for Kaggle notebooks (eliminates duplicate bars).

    Features
    -------
    * Smooth animation (100 ms updates)
    * Dynamic colour changes (blue → green on completion)
    * Emoji‑friendly description updates
    * Single progress bar display on Kaggle
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Default values matching the Hugging‑Face style
        kwargs.setdefault("unit", "it")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("bar_format", HF_BAR_FORMAT)
        kwargs.setdefault("colour", HF_BLUE)
        kwargs.setdefault("ascii", False)
        kwargs.setdefault("ncols", 100)
        kwargs.setdefault("smoothing", 0.7)
        kwargs.setdefault("leave", True)

        self._base_desc = kwargs.get("desc", "Processing")
        self._start_time = time.time()
        self._warning_shown = False

    def update(self, n: int = 1) -> None:  # type: ignore[override]
        super().update(n)

        if self.total:
            progress = self.n / self.total
            self.colour = HF_BLUE if progress < 1.0 else HF_GREEN

    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        if desc:
            self._base_desc = desc
        super().set_description(desc, refresh=refresh)

    def set_postfix(
        self,
        ordered_dict: Optional[dict[str, Any]] = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        super().set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)

    def close(self) -> None:
        if self.total and self.n >= self.total:
            self.colour = HF_GREEN
            elapsed = time.time() - self._start_time
            time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}min"
            self.set_description(f"✓ {self._base_desc}", refresh=False)
        else:
            self.colour = HF_ORANGE
        self.refresh()
        super().close()

    def __enter__(self) -> "HFTqdm":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.colour = HF_RED
            self.set_description(f"✗ {self._base_desc} (failed)", refresh=False)
        self.close()
        return False


def get_hf_bar(
    iterable: Optional[Iterable[Any]] = None,
    desc: str = "Processing",
    total: Optional[int] = None,
    unit: str = "it",
    disable: bool = False,
    **kwargs: Any,
) -> HFTqdm:
    """
    Factory for a Hugging‑Face style progress bar.

    Parameters
    ----------
    iterable : Iterable | None
        The iterable to wrap (if any).
    desc : str
        Initial description.
    total : int | None
        Total number of steps.
    unit : str
        Unit name for the bar.
    """
    kwargs["unit"] = unit
    kwargs["disable"] = disable
    return HFTqdm(iterable=iterable, desc=desc, total=total, **kwargs)


# --------------------------------------------------------------------------- #
# Checkpoint utilities
# --------------------------------------------------------------------------- #


def _ensure_path(path: Path | str) -> Path:
    """Return a Path object regardless of input type."""
    return path if isinstance(path, Path) else Path(path)


def load_model_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    """
    Load a model checkpoint from disk.

    Supports both .pth (torch) and .safetensors formats.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to the checkpoint file.

    Returns
    -------
    dict[str, Any]
        The state dictionary.
    """
    checkpoint_path = _ensure_path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as safe_load

        return safe_load(checkpoint_path, device="cpu")
    return torch.load(checkpoint_path, map_location="cpu")


def save_model_checkpoint(
    model_state_dict: dict[str, Any], checkpoint_path: Path | str
) -> None:
    """
    Save a model state dictionary to disk.

    Parameters
    ----------
    model_state_dict : dict[str, Any]
        The state dictionary to save.
    checkpoint_path : Path | str
        Destination path.
    """
    checkpoint_path = _ensure_path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import save_file as safe_save

        safe_save(model_state_dict, checkpoint_path)
    else:
        torch.save(model_state_dict, checkpoint_path)


def find_checkpoint(checkpoint_dir: Path | str, checkpoint_name: str) -> Path:
    """
    Find a checkpoint file in a directory.

    Prefers .safetensors over .pth.

    Parameters
    ----------
    checkpoint_dir : Path | str
        Directory containing checkpoints.
    checkpoint_name : str
        Base name of the checkpoint (without extension).

    Returns
    -------
    Path
        Full path to the found checkpoint file.

    Raises
    ------
    FileNotFoundError
        If neither format is present.
    """
    checkpoint_dir = _ensure_path(checkpoint_dir)
    safetensors_path = checkpoint_dir / f"{checkpoint_name}.safetensors"
    pth_path = checkpoint_dir / f"{checkpoint_name}.pth"

    if safetensors_path.exists():
        return safetensors_path
    if pth_path.exists():
        return pth_path

    raise FileNotFoundError(
        f"Checkpoint not found for '{checkpoint_name}' in {checkpoint_dir}\n"
        f"  Expected: {safetensors_path} or {pth_path}"
    )


# --------------------------------------------------------------------------- #
# File‑system helpers
# --------------------------------------------------------------------------- #


def flatten_folder(root_dir: Path | str) -> None:
    """
    Move all files from nested subdirectories into the root directory.

    Parameters
    ----------
    root_dir : Path | str
        The directory to flatten.
    """
    root_dir = _ensure_path(root_dir)

    for subdir, _, files in os.walk(root_dir):
        if Path(subdir) == root_dir:
            continue
        for file in files:
            src = Path(subdir) / file
            dst = root_dir / file
            if dst.exists():
                base, ext = dst.stem, dst.suffix
                i = 1
                while (root_dir / f"{base}_{i}{ext}").exists():
                    i += 1
                dst = root_dir / f"{base}_{i}{ext}"
            shutil.move(str(src), str(dst))

    # Remove empty sub‑directories
    for subdir, _, files in os.walk(root_dir, topdown=False):
        sub_path = Path(subdir)
        if sub_path != root_dir and not any(sub_path.iterdir()):
            sub_path.rmdir()


def rename_images(json_file: Path | str) -> None:
    """
    Rename image files referenced in a JSON metadata file.

    The new filenames follow the pattern ``style+char.png`` for target images
    and ``char.png`` for content images.

    Parameters
    ----------
    json_file : Path | str
        Path to the JSON file containing ``generations`` entries.
    """
    json_file = _ensure_path(json_file)

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    generations = data.get("generations", [])
    for entry in generations:
        char = entry.get("character")
        style = entry.get("style")
        new_filename = f"{style}+{char}.png"

        for key in ("content_image_path", "target_image_path"):
            old_path = entry.get(key)
            if not old_path:
                continue
            old_path = Path(old_path)
            if not old_path.exists():
                print(f"Skipping: File not found {old_path}")
                continue

            directory = old_path.parent
            if "content" in key:
                new_filename = f"{char}.png"
            new_path = directory / new_filename

            try:
                old_path.rename(new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                entry[key] = str(new_path)
            except OSError as e:
                print(f"Error renaming {old_path}: {e}")

    output_file = json_file.with_name("updated_generations.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nProcessing complete. Updated JSON saved as: {output_file}")


def rename_content_images(path: Path | str) -> None:
    """
    Rename all files in ``ContentImage/`` from ``1+char.png`` to ``char.png``.

    Parameters
    ----------
    path : Path | str
        Root directory containing the ``ContentImage`` sub‑folder.
    """
    path = _ensure_path(path)
    content_dir = path / "ContentImage"

    for filename in content_dir.iterdir():
        if "+" in filename.name:
            char = filename.name.split("+")[1]
            new_path = content_dir / char
            filename.rename(new_path)
            print(f"Renamed: {filename} -> {new_path}")


def update_paths(
    input_file: Path | str, output_file: Optional[Path | str] = None
) -> None:
    """
    Update ``content_image_path`` and ``target_image_path`` fields in a JSON file.

    Parameters
    ----------
    input_file : Path | str
        Path to the source JSON file.
    output_file : Path | str | None
        Destination path. If ``None``, overwrites ``input_file``.
    """
    input_file = _ensure_path(input_file)
    output_file = _ensure_path(output_file) if output_file else input_file

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("generations", [])
    for item in items:
        char, style = item["character"], item["style"]
        item["content_image_path"] = f"ContentImage/{char}.png"
        item["target_image_path"] = f"TargetImage/{style}/{style}+{char}.png"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_font_glyph_counts(fonts_dir: Path | str) -> None:
    """
    Print the number of glyphs (supported Unicode characters) for each font in a directory.

    Parameters
    ----------
    fonts_dir : Path | str
        Directory containing font files (.ttf, .otf).
    """
    from fontTools.ttLib import TTFont

    fonts_dir = _ensure_path(fonts_dir)
    font_extensions = {".ttf", ".otf", ".ttf", ".otf"}

    font_files = [p for p in fonts_dir.iterdir() if p.suffix.lower() in font_extensions]

    print(f"\nFont glyph count summary for directory: {fonts_dir}")
    for font_path in font_files:
        try:
            font = TTFont(font_path)
            cmap = font.getBestCmap()
            num_glyphs = len(cmap)
            font_name = font_path.stem
            print(f"  {font_name}: {num_glyphs} glyphs")
        except Exception as e:
            print(f"  {font_path.name}: Failed to read ({e})")


# --------------------------------------------------------------------------- #
# Conversion utilities
# --------------------------------------------------------------------------- #


def pth_to_safetensors(pth_path: Path | str, output_path: Path | str) -> None:
    """
    Convert a .pth checkpoint to .safetensors format.

    Parameters
    ----------
    pth_path : Path | str
        Source .pth file.
    output_path : Path | str
        Destination .safetensors file.
    """
    pth_path = _ensure_path(pth_path)
    output_path = _ensure_path(output_path)

    print(f"\nConverting {pth_path.name} to safetensors...")
    try:
        state_dict = torch.load(pth_path, map_location="cpu")
        save_file(state_dict, output_path)

        size_pth = pth_path.stat().st_size / (1024**3)
        size_safe = output_path.stat().st_size / (1024**3)

        print(f"✓ Converted: {pth_path}")
        print(f"  .pth size:  {size_pth:.2f} GB")
        print(f"  .safetensors size: {size_safe:.2f} GB")
    except Exception as e:
        print(f"✗ Error converting {pth_path}: {e}")
        raise


def convert_checkpoint_folder(
    ckpt_dir: Path | str,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """
    Convert all .pth files in a directory to .safetensors.

    Parameters
    ----------
    ckpt_dir : Path | str
        Source directory containing .pth checkpoints.
    output_dir : Path | str | None
        Destination directory for .safetensors files.
        If ``None`` the source directory is used.

    Returns
    -------
    Path
        The directory that contains the converted files.
    """
    ckpt_dir = _ensure_path(ckpt_dir)
    output_dir = _ensure_path(output_dir) if output_dir else ckpt_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("CONVERTING CHECKPOINTS TO SAFETENSORS")
    print("=" * 60)

    pth_files = list(ckpt_dir.glob("*.pth"))
    if not pth_files:
        print(f"⚠ No .pth files found in {ckpt_dir}")
        return output_dir

    print(f"Found {len(pth_files)} .pth files")
    for pth_file in pth_files:
        safe_file = output_dir / f"{pth_file.stem}.safetensors"
        pth_to_safetensors(pth_file, safe_file)

    print(f"\n✓ Conversion complete!")
    return output_dir


# --------------------------------------------------------------------------- #
# Command‑line interface
# --------------------------------------------------------------------------- #


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Utility functions for font diffusion project"
    )
    parser.add_argument(
        "--flatten_dir",
        type=str,
        help="Path to directory to flatten (move all files from subdirs to root)",
    )
    parser.add_argument(
        "--font_glyphs_dir",
        type=str,
        help="Path to directory containing fonts to print glyph counts",
    )
    parser.add_argument(
        "--convert_ckpt_dir",
        type=str,
        help="Path to directory containing .pth checkpoints to convert to .safetensors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for converted safetensors (if not specified, uses input dir)",
    )
    parser.add_argument(
        "--rename_images_json",
        type=str,
        help="Path to JSON file for renaming images based on character and style",
    )
    parser.add_argument(
        "--rename_content_images_dir",
        type=str,
        help="Path to directory to rename content images from '1+char.png' to 'char.png'",
    )
    parser.add_argument(
        "--update_paths",
        type=str,
        nargs=2,
        help="Path to JSON file to update content and target image paths",
    )
    args = parser.parse_args()

    if args.flatten_dir:
        flatten_folder(args.flatten_dir)
        print(f"✓ Flattened directory: {args.flatten_dir}")

    if args.font_glyphs_dir:
        print_font_glyph_counts(args.font_glyphs_dir)

    if args.convert_ckpt_dir:
        convert_checkpoint_folder(args.convert_ckpt_dir, args.output_dir)

    if args.rename_images_json:
        rename_images(args.rename_images_json)

    if args.rename_content_images_dir:
        rename_content_images(args.rename_content_images_dir)

    if args.update_paths:
        update_paths(args.update_paths[0], args.update_paths[1])
        print(f"✓ Updated paths in JSON file: {args.update_paths}")

    if not any(vars(args).values()):
        print("No arguments provided. Use --help for usage information.")


if __name__ == "__main__":
    _cli()
