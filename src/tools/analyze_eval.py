"""
Analyze evaluation metrics across checkpoint CSVs and recommend the best checkpoint.

Reads one or more CSV files produced by evaluate.py, aggregates per-checkpoint
statistics, and ranks checkpoints by a weighted composite score.

Composite score (lower is better)
----------------------------------
  score = w_ssim * (1 - ssim_mean) + w_lpips * lpips_mean + w_fid * fid_norm

  - ssim  : inverted so that higher SSIM → lower penalty
  - lpips : lower is better (perceptual match vs style)
  - fid   : normalised to [0, 1] across all checkpoints; lower is better

Usage
-----
# Single CSV:
python tools/analyze_eval.py --eval_dir evaluation/

# Multiple CSVs (each file = one checkpoint):
python tools/analyze_eval.py \\
    --eval_dir evaluation/ \\
    --output_csv evaluation/checkpoint_rankings.csv \\
    --w_ssim 0.4 --w_lpips 0.4 --w_fid 0.2

# Skip FID in ranking (files have no fid column or --skip_fid was used):
python tools/analyze_eval.py \\
    --eval_dir evaluation/ --w_fid 0.0
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_W_SSIM: float = 0.7
DEFAULT_W_LPIPS: float = 0.3
DEFAULT_W_FID: float = 0.001

FLOAT_SENTINEL: float = float("nan")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict]:
    """Load a metrics CSV produced by evaluate.py.

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dicts with numeric fields cast to float where possible.
    """
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key in ("ssim", "lpips", "fid"):
                raw = row.get(key, "").strip()
                try:
                    row[key] = float(raw)
                except (ValueError, TypeError):
                    row[key] = FLOAT_SENTINEL
            rows.append(row)
    return rows


def discover_csvs(eval_dir: Path, pattern: str) -> list[Path]:
    """Return sorted CSV paths matching *pattern* inside *eval_dir*.

    Args:
        eval_dir: Directory to scan (non-recursive).
        pattern: Glob pattern, e.g. ``"*.csv"``.

    Returns:
        Sorted list of matching Path objects.
    """
    paths = sorted(eval_dir.glob(pattern))
    if not paths:
        logger.warning("No files matching '%s' found in %s.", pattern, eval_dir)
    return paths


# ---------------------------------------------------------------------------
# Per-checkpoint aggregation
# ---------------------------------------------------------------------------


def aggregate_checkpoint(
    name: str,
    rows: list[dict],
) -> dict:
    """Aggregate per-pair rows into checkpoint-level statistics.

    Args:
        name: Checkpoint identifier (typically the CSV stem).
        rows: All per-pair metric rows for this checkpoint.

    Returns:
        Dict with keys:
        ``checkpoint``, ``n_pairs``,
        ``ssim_mean``, ``ssim_std``, ``ssim_min``, ``ssim_max``,
        ``lpips_mean``, ``lpips_std``, ``lpips_min``, ``lpips_max``,
        ``fid`` (scalar or NaN).
    """
    ssim_vals = np.array([r["ssim"] for r in rows if not np.isnan(r["ssim"])])
    lpips_vals = np.array([r["lpips"] for r in rows if not np.isnan(r["lpips"])])

    # FID is stored the same for every row (dataset-level scalar); take mean
    fid_vals = np.array([r["fid"] for r in rows if not np.isnan(r["fid"])])
    fid = float(np.mean(fid_vals)) if len(fid_vals) > 0 else FLOAT_SENTINEL

    def _stats(arr: np.ndarray) -> tuple[float, float, float, float]:
        if len(arr) == 0:
            return (FLOAT_SENTINEL,) * 4
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    ssim_mean, ssim_std, ssim_min, ssim_max = _stats(ssim_vals)
    lpips_mean, lpips_std, lpips_min, lpips_max = _stats(lpips_vals)

    return {
        "checkpoint": name,
        "n_pairs": len(rows),
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "ssim_min": ssim_min,
        "ssim_max": ssim_max,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "lpips_min": lpips_min,
        "lpips_max": lpips_max,
        "fid": fid,
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_composite_scores(
    stats: list[dict],
    w_ssim: float,
    w_lpips: float,
    w_fid: float,
) -> list[dict]:
    """Attach a composite ranking score to each checkpoint stats dict.

    Normalisation is min-max across all checkpoints so every metric
    contributes on the same scale.

    Score formula (higher = better)
    --------------------------------
    ::
        score = w_ssim  * ssim_norm
              + w_lpips * (1 - lpips_norm)
              + w_fid   * (1 - fid_norm)

    Where ``*_norm`` is the min-max normalised value in [0, 1].

    Args:
        stats: List of per-checkpoint aggregated stat dicts.
        w_ssim: Weight for SSIM (higher SSIM → higher score).
        w_lpips: Weight for LPIPS penalty (lower LPIPS → higher score).
        w_fid: Weight for FID penalty (lower FID → higher score).

    Returns:
        Same list with ``composite_score`` and ``rank`` keys added,
        sorted descending by composite score.
    """
    def _minmax_norm(values: np.ndarray) -> np.ndarray:
        lo, hi = values.min(), values.max()
        if np.isclose(lo, hi):
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    ssim_arr = np.array([s["ssim_mean"] for s in stats])
    lpips_arr = np.array([s["lpips_mean"] for s in stats])
    fid_arr = np.array([s["fid"] for s in stats])

    # Handle NaN FID gracefully — zero-weight it per entry
    has_fid = ~np.isnan(fid_arr)

    ssim_norm = _minmax_norm(np.where(np.isnan(ssim_arr), 0.5, ssim_arr))
    lpips_norm = _minmax_norm(np.where(np.isnan(lpips_arr), 0.5, lpips_arr))
    fid_norm = np.where(
        has_fid,
        _minmax_norm(np.where(np.isnan(fid_arr), 0.0, fid_arr)),
        0.5,
    )

    effective_w_fid = np.where(has_fid, w_fid, 0.0)
    effective_w_fid_sum = w_ssim + w_lpips + effective_w_fid
    # Renormalise weights per checkpoint when FID is absent
    w_ssim_eff = w_ssim / effective_w_fid_sum
    w_lpips_eff = w_lpips / effective_w_fid_sum
    w_fid_eff = effective_w_fid / effective_w_fid_sum

    composite = (
        w_ssim_eff * ssim_norm
        + w_lpips_eff * (1.0 - lpips_norm)
        + w_fid_eff * (1.0 - fid_norm)
    )

    for i, s in enumerate(stats):
        s["composite_score"] = float(composite[i])

    ranked = sorted(stats, key=lambda s: s["composite_score"], reverse=True)
    for rank, s in enumerate(ranked, start=1):
        s["rank"] = rank

    return ranked


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(ranked: list[dict], w_ssim: float, w_lpips: float, w_fid: float) -> None:
    """Pretty-print the ranking table and recommendation.

    Args:
        ranked: Checkpoint stats sorted by composite score (best first).
        w_ssim: Effective SSIM weight used.
        w_lpips: Effective LPIPS weight used.
        w_fid: Effective FID weight used.
    """
    sep = "=" * 100
    logger.info("\n%s", sep)
    logger.info("CHECKPOINT RANKING REPORT")
    logger.info(
        "Weights — SSIM: %.2f (↑ content fidelity)  "
        "LPIPS: %.2f (↓ style match)  "
        "FID: %.2f (↓ realism)",
        w_ssim,
        w_lpips,
        w_fid,
    )
    logger.info("%s", sep)

    header = (
        f"{'Rank':<5} {'Checkpoint':<35} {'Pairs':>6} "
        f"{'SSIM↑':>8} {'±':>6} "
        f"{'LPIPS↓':>8} {'±':>6} "
        f"{'FID↓':>9} "
        f"{'Score↑':>8}"
    )
    logger.info(header)
    logger.info("-" * 100)

    for s in ranked:
        fid_str = f"{s['fid']:9.4f}" if not np.isnan(s["fid"]) else f"{'N/A':>9}"
        row = (
            f"{s['rank']:<5} {s['checkpoint']:<35} {s['n_pairs']:>6} "
            f"{s['ssim_mean']:>8.4f} {s['ssim_std']:>6.4f} "
            f"{s['lpips_mean']:>8.4f} {s['lpips_std']:>6.4f} "
            f"{fid_str} "
            f"{s['composite_score']:>8.4f}"
        )
        logger.info(row)

    logger.info("%s", sep)

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    logger.info("\n🏆  RECOMMENDED CHECKPOINT: %s", best["checkpoint"])
    logger.info(
        "    SSIM  = %.4f  |  LPIPS = %.4f  |  FID = %s  |  Score = %.4f",
        best["ssim_mean"],
        best["lpips_mean"],
        f"{best['fid']:.4f}" if not np.isnan(best["fid"]) else "N/A",
        best["composite_score"],
    )

    if runner_up is not None:
        score_gap = best["composite_score"] - runner_up["composite_score"]
        logger.info(
            "    Runner-up: %s  (score gap: %.4f)",
            runner_up["checkpoint"],
            score_gap,
        )

    # Per-metric winners
    valid_ssim = [s for s in ranked if not np.isnan(s["ssim_mean"])]
    valid_lpips = [s for s in ranked if not np.isnan(s["lpips_mean"])]
    valid_fid = [s for s in ranked if not np.isnan(s["fid"])]

    if valid_ssim:
        best_ssim = max(valid_ssim, key=lambda s: s["ssim_mean"])
        logger.info("\n📐  Best SSIM  (content fidelity) : %s  (%.4f)", best_ssim["checkpoint"], best_ssim["ssim_mean"])
    if valid_lpips:
        best_lpips = min(valid_lpips, key=lambda s: s["lpips_mean"])
        logger.info("🎨  Best LPIPS (style match)       : %s  (%.4f)", best_lpips["checkpoint"], best_lpips["lpips_mean"])
    if valid_fid:
        best_fid = min(valid_fid, key=lambda s: s["fid"])
        logger.info("📊  Best FID   (realism)           : %s  (%.4f)", best_fid["checkpoint"], best_fid["fid"])

    logger.info("%s\n", sep)


def print_per_style_breakdown(all_rows_by_ckpt: dict[str, list[dict]]) -> None:
    """Print per-style SSIM/LPIPS averages for the best checkpoint.

    Args:
        all_rows_by_ckpt: Mapping of checkpoint name → raw row list.
    """
    logger.info("PER-STYLE BREAKDOWN (all checkpoints)")
    logger.info("-" * 80)

    for ckpt_name, rows in all_rows_by_ckpt.items():
        styles: dict[str, list] = {}
        for row in rows:
            sn = row.get("style_name", "unknown") or "unknown"
            styles.setdefault(sn, []).append(row)

        logger.info("\n  Checkpoint: %s", ckpt_name)
        logger.info(
            "  %-30s %8s %8s %6s",
            "Style",
            "SSIM↑",
            "LPIPS↓",
            "Pairs",
        )
        logger.info("  " + "-" * 56)

        for style_name, style_rows in sorted(styles.items()):
            ssim_vals = [r["ssim"] for r in style_rows if not np.isnan(r["ssim"])]
            lpips_vals = [r["lpips"] for r in style_rows if not np.isnan(r["lpips"])]
            ssim_m = np.mean(ssim_vals) if ssim_vals else float("nan")
            lpips_m = np.mean(lpips_vals) if lpips_vals else float("nan")
            logger.info(
                "  %-30s %8.4f %8.4f %6d",
                style_name[:30],
                ssim_m,
                lpips_m,
                len(style_rows),
            )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_rankings_csv(ranked: list[dict], output_path: Path) -> None:
    """Write checkpoint rankings to CSV.

    Args:
        ranked: Sorted list of checkpoint stat dicts.
        output_path: Destination path (parent dirs are created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "checkpoint",
        "n_pairs",
        "composite_score",
        "ssim_mean",
        "ssim_std",
        "ssim_min",
        "ssim_max",
        "lpips_mean",
        "lpips_std",
        "lpips_min",
        "lpips_max",
        "fid",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)

    logger.info("Rankings written to %s", output_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for checkpoint analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze evaluate.py CSV outputs and recommend the best checkpoint "
            "using a weighted composite score (SSIM + LPIPS + FID)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("evaluation"),
        help=(
            "Directory containing evaluate.py CSV outputs. "
            "Each CSV is treated as one checkpoint. "
            "Files are matched by the glob pattern --csv_pattern."
        ),
    )
    parser.add_argument(
        "--csv_pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match CSV files inside --eval_dir.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("evaluation/checkpoint_rankings.csv"),
        help="Destination CSV path for the ranked checkpoint summary.",
    )
    parser.add_argument(
        "--w_ssim",
        type=float,
        default=DEFAULT_W_SSIM,
        help="Weight for SSIM in the composite score (higher SSIM = better content fidelity).",
    )
    parser.add_argument(
        "--w_lpips",
        type=float,
        default=DEFAULT_W_LPIPS,
        help="Weight for LPIPS in the composite score (lower LPIPS = better style match).",
    )
    parser.add_argument(
        "--w_fid",
        type=float,
        default=DEFAULT_W_FID,
        help="Weight for FID in the composite score (lower FID = more realistic output). "
             "Set to 0 to ignore FID.",
    )
    parser.add_argument(
        "--style_breakdown",
        action="store_true",
        help="Print per-style SSIM/LPIPS averages for every checkpoint.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="CHECKPOINT",
        help="Checkpoint names (CSV stems) to exclude from analysis.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not args.eval_dir.exists():
        logger.error("eval_dir not found: %s", args.eval_dir)
        sys.exit(1)

    weight_sum = args.w_ssim + args.w_lpips + args.w_fid
    if weight_sum <= 0:
        logger.error("At least one weight must be positive.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover and load CSVs
    # ------------------------------------------------------------------
    csv_paths = discover_csvs(args.eval_dir, args.csv_pattern)

    # Filter out the rankings file itself and any user-excluded stems
    excluded = set(args.exclude) | {"checkpoint_rankings"}
    csv_paths = [p for p in csv_paths if p.stem not in excluded]

    if not csv_paths:
        logger.error(
            "No evaluation CSVs found in %s matching '%s' (after exclusions). "
            "Run evaluate.py first.",
            args.eval_dir,
            args.csv_pattern,
        )
        sys.exit(1)

    logger.info("Found %d checkpoint CSV(s) to analyse.", len(csv_paths))

    all_rows_by_ckpt: dict[str, list[dict]] = {}
    for csv_path in csv_paths:
        rows = load_csv(csv_path)
        if not rows:
            logger.warning("Empty CSV: %s — skipping.", csv_path.name)
            continue
        all_rows_by_ckpt[csv_path.stem] = rows
        logger.info("  Loaded %4d rows from %s", len(rows), csv_path.name)

    if not all_rows_by_ckpt:
        logger.error("All CSVs were empty or unreadable. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate & score
    # ------------------------------------------------------------------
    stats = [
        aggregate_checkpoint(name, rows)
        for name, rows in all_rows_by_ckpt.items()
    ]

    ranked = compute_composite_scores(stats, args.w_ssim, args.w_lpips, args.w_fid)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print_report(ranked, args.w_ssim, args.w_lpips, args.w_fid)

    if args.style_breakdown:
        print_per_style_breakdown(all_rows_by_ckpt)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    write_rankings_csv(ranked, args.output_csv)


if __name__ == "__main__":
    main()