import os
import argparse
import subprocess
from pathlib import Path


def get_git_ignored(root: Path) -> set:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-i", "--exclude-standard"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        paths = [p.strip() for p in proc.stdout.splitlines() if p.strip()]
        return set(paths)
    except Exception:
        return set()


def tree_lines(root: Path, ignore_git=False, exts=None, max_depth=None):
    root = root.resolve()
    ignored = get_git_ignored(root) if ignore_git else set()
    for dirpath, dirnames, filenames in os.walk(root):
        depth = Path(dirpath).resolve().relative_to(root).parts
        cur_depth = 0 if str(dirpath) == str(root) else len(depth)
        if max_depth is not None and cur_depth > max_depth:
            dirnames[:] = []
            continue

        indent = " " * (cur_depth * 2)
        rel_dir = str(Path(dirpath).resolve()).replace(str(root), "").lstrip(os.sep)
        yield (
            f"{indent}{Path(dirpath).name}/"
            if rel_dir != ""
            else f"{Path(dirpath).name}/"
        )

        filenames = sorted(filenames)
        for fn in filenames:
            rel_path = os.path.join(os.path.relpath(dirpath, root), fn).replace(
                os.sep, "/"
            )
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            if ignore_git and rel_path in ignored:
                continue
            if exts and not any(fn.endswith(e) for e in exts):
                continue
            yield f"{indent}  {fn}"


def main():
    p = argparse.ArgumentParser(
        description="Print tree structure of a workspace to stdout"
    )
    p.add_argument(
        "--root", "-r", default=".", help="Workspace root (default: current dir)"
    )
    p.add_argument(
        "--ext",
        "-e",
        default="*.jpg,*.png",
        help="Comma separated extensions to include (e.g. .py,.md)",
    )
    p.add_argument(
        "--ignore-git",
        action="store_true",
        default=False,
        help="Exclude files listed in .gitignore (requires git)",
    )
    p.add_argument("--max-depth", type=int, default=None, help="Max depth to traverse")
    args = p.parse_args()

    root = Path(args.root)
    exts = [s.strip() for s in args.ext.split(",")] if args.ext else None

    for line in tree_lines(
        root, ignore_git=args.ignore_git, exts=exts, max_depth=args.max_depth
    ):
        print(line)


if __name__ == "__main__":
    main()
