"""Strip the underscore suffix from style image filenames.

Example: backan1_cleaned.png → backan1.png

Usage
-----
    python tools/rename_style_images.py            # dry-run
    python tools/rename_style_images.py --apply    # apply renames
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("rename_style_images")

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def build_rename_plan(style_dir: Path) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for files whose stem contains an underscore.

    Skips files where the destination already exists or would collide.
    """
    plan: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    for src in sorted(p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        if "_" not in src.stem:
            continue
        dst = src.with_stem(src.stem.split("_", maxsplit=1)[0])
        if dst == src:
            continue
        if dst in seen or dst.exists():
            logger.warning("Skipping '%s' — destination '%s' already taken.", src.name, dst.name)
            continue
        plan.append((src, dst))
        seen.add(dst)

    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename style images by stripping underscore suffix.")
    parser.add_argument("--style_dir", type=Path, default=Path("style_images/cleaned_handwritten_enhanced"))
    parser.add_argument("--apply", action="store_true", help="Apply renames (default: dry-run).")
    args = parser.parse_args()

    if not args.style_dir.exists():
        logger.error("Directory not found: %s", args.style_dir)
        sys.exit(1)

    plan = build_rename_plan(args.style_dir)
    if not plan:
        logger.info("Nothing to rename in %s.", args.style_dir)
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("[%s] %d file(s) planned:", mode, len(plan))
    for src, dst in plan:
        logger.info("  %-50s → %s", src.name, dst.name)

    if not args.apply:
        logger.info("Re-run with --apply to apply renames.")
        return

    failed = 0
    for src, dst in plan:
        try:
            src.rename(dst)
        except OSError as exc:
            logger.error("Failed '%s': %s", src.name, exc)
            failed += 1

    logger.info("Done — %d renamed, %d failed.", len(plan) - failed, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()