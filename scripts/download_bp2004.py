from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004"
DEFAULT_RAW_DIR = "data/raw/bp2004"
REQUIRED_FILES = ("vel_z6.25m_x12.5m_exact.segy.gz", "2004_Benchmark_READMES.pdf")
OPTIONAL_FILES = ("vel_z6.25m_x12.5m_lw.segy.gz", "vel_z6.25m_x12.5m_nosalt.segy.gz", "shots0601_0800.segy.gz")


def file_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"[skip] {output_path} ({file_size(output_path)})")
        return

    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with output_path.open("wb") as f:
                shutil.copyfileobj(response, f)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    print(f"[done] {output_path} ({file_size(output_path)})")


def decompress_gzip(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    output_path = path.with_suffix("")
    if output_path.exists():
        print(f"[skip] decompressed file exists: {output_path} ({file_size(output_path)})")
        return output_path

    print(f"[decompress] {path}")
    with gzip.open(path, "rb") as src:
        with output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"[done] {output_path} ({file_size(output_path)})")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the SEG/BP 2004 velocity benchmark files.")
    parser.add_argument("--output-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--all", action="store_true", help="Download optional BP2004 files as well.")
    parser.add_argument("--decompress", action="store_true", help="Decompress downloaded .gz files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = list(REQUIRED_FILES)
    if args.all:
        targets.extend(OPTIONAL_FILES)

    for filename in targets:
        path = output_dir / filename
        download(f"{BASE_URL}/{filename}", path)
        if args.decompress:
            decompress_gzip(path)


if __name__ == "__main__":
    main()
