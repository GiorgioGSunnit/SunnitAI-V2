# Parses batches of legal document markdown files into structured JSON outputs.
# Offers a simple CLI to process a limited number of documents concurrently.

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .parser import parse_file as parse_markdown_file


def find_markdown_files(data_root: Path) -> List[Path]:
    """Collect markdown files from the data directory in sorted order."""
    files_dir = data_root / "files"
    search_dir = files_dir if files_dir.exists() else data_root
    return sorted([p for p in search_dir.glob("*.md") if p.is_file()])


def ensure_output_dir(app_dir: Path) -> Path:
    """Create (if missing) and return the directory for parsed JSON files."""
    out_dir = app_dir / "parsed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_and_write(source_path_str: str, output_dir_str: str) -> str:
    """Parse a markdown file and write its JSON representation to disk."""
    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str)
    result = parse_markdown_file(str(source_path))
    out_path = output_dir / f"{source_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out_path)


def parse_in_parallel(paths: List[Path], output_dir: Path) -> None:
    """Parse markdown files concurrently and log their output locations."""
    with ProcessPoolExecutor() as executor:
        future_to_path = {
            executor.submit(_parse_and_write, str(p), str(output_dir)): p for p in paths
        }
        for future in as_completed(future_to_path):
            src_path = future_to_path[future]
            try:
                out_path = future.result()
                print(f"Parsed: {src_path.name} -> {Path(out_path).name}")
            except Exception as exc:
                print(f"Failed: {src_path} ({exc})")


def main() -> None:
    """CLI entry point for parsing a limited number of markdown files."""
    parser = argparse.ArgumentParser(
        description="Parse first N markdown documents from app/data into JSON in app/parsed_data."
    )
    parser.add_argument("n", type=int, help="Number of documents to parse")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parents[2]
    data_root = app_dir / "data"
    if not data_root.exists():
        print(f"Data directory not found: {data_root}")
        return

    output_dir = ensure_output_dir(app_dir)
    md_files = find_markdown_files(data_root)
    if not md_files:
        print(f"No markdown files found under {data_root}")
        return

    selected = md_files[: args.n]
    print(f"Parsing {len(selected)} of {len(md_files)} files...")
    parse_in_parallel(selected, output_dir)
    print(f"Done. JSON files saved to: {output_dir}")


if __name__ == "__main__":
    main()
