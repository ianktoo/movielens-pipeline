"""Download and unpack the MovieLens 32M dataset, then build the samples.

One command gets a teammate from a fresh clone to a ready-to-run notebook:

    uv run python scripts/download_data.py

It downloads ml-32m.zip (~240 MB) into ``data/raw/``, unzips it, and then runs
the sampling step so ``data/samples/`` is populated. Uses only the Python
standard library (urllib + zipfile), so it works the same on Windows, macOS, and
Linux with no extra tools (no curl/wget needed).

Dataset homepage: https://grouplens.org/datasets/movielens/32m/

Flags:
    --skip-samples   download + unzip only; don't build sample sets.
    --force          re-download even if the data is already present.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile

from movielens import config


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Simple percentage progress bar for urlretrieve."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    sys.stdout.write(f"\r  downloading... {pct:5.1f}%  ({mb:6.1f} / {total_mb:.1f} MB)")
    sys.stdout.flush()


def download_and_unzip(force: bool = False) -> None:
    config.ensure_dirs()
    zip_path = config.RAW_DIR / "ml-32m.zip"

    from movielens import data
    if data.raw_available() and not force:
        print(f"Dataset already present at {config.RAW_MOVIELENS_DIR} - skipping download.")
        print("  (use --force to re-download)")
        return

    print(f"Downloading MovieLens 32M from:\n  {config.DATASET_URL}")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path, reporthook=_progress)
    print(f"\n  saved -> {zip_path}")

    print("Unzipping (this expands to ~1 GB)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.RAW_DIR)
    print(f"  unzipped -> {config.RAW_MOVIELENS_DIR}")

    if data.raw_available():
        print("Done — real dataset is ready.")
    else:
        print("Warning: expected csv files not found after unzip; check the archive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + prepare MovieLens 32M.")
    parser.add_argument("--skip-samples", action="store_true",
                        help="download + unzip only; don't build sample sets")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if data already exists")
    args = parser.parse_args()

    download_and_unzip(force=args.force)

    if args.skip_samples:
        print("\nSkipping sample build (--skip-samples). "
              "Run `uv run python scripts/prepare_samples.py` when ready.")
        return

    print("\nBuilding swappable sample sets...")
    import prepare_samples  # sibling script; scripts/ dir is added to sys.path below
    prepare_samples.main()


if __name__ == "__main__":
    # Make the sibling prepare_samples.py importable regardless of CWD.
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
