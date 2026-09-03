"""Verify URL-backed parameter citations and stamp verified YAML entries.

This script intentionally checks ParameterValue citations only. Calculator
definition citations describe formula authority; parameter citations support
date-versioned values and are the ones stamped with last_verified_at.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timezone, datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


FetchResult = tuple[bool, str]
Fetcher = Callable[[str, float], FetchResult]


@dataclass
class VerificationStats:
    checked_entries: int = 0
    checked_urls: int = 0
    verified_entries: int = 0
    skipped_entries: int = 0
    failed_entries: int = 0
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def default_parameters_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "parameters"


def fetch_url(url: str, timeout: float = 10.0) -> FetchResult:
    headers = {"User-Agent": "SunnitAI-CalculationPlatform-CitationVerifier/0.1"}
    for method in ("HEAD", "GET"):
        request = Request(url, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                if 200 <= status < 400:
                    return True, f"{status} {method}"
                return False, f"{status} {method}"
        except HTTPError as exc:
            if method == "HEAD" and exc.code in {403, 405, 501}:
                continue
            return False, f"HTTP {exc.code} {method}"
        except URLError as exc:
            return False, str(exc.reason)
        except TimeoutError:
            return False, "timeout"
    return False, "not reachable"


def citation_urls(entry: dict) -> list[str]:
    return [
        citation["url"]
        for citation in entry.get("citations", [])
        if isinstance(citation, dict) and citation.get("url")
    ]


def iter_parameter_files(parameters_dir: Path) -> Iterable[Path]:
    yield from sorted(parameters_dir.rglob("*.yml"))


def load_entries(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("values", [])


def verify_parameters(
    parameters_dir: Path,
    *,
    verified_at: str | None = None,
    timeout: float = 10.0,
    dry_run: bool = False,
    fetcher: Fetcher = fetch_url,
) -> VerificationStats:
    verified_at = verified_at or datetime.now(timezone.utc).date().isoformat()
    stats = VerificationStats()
    verified_by_file: dict[Path, list[int]] = {}

    for path in iter_parameter_files(parameters_dir):
        entries = load_entries(path)
        for index, entry in enumerate(entries):
            parameter_id = entry.get("parameter_id", "<unknown>")
            effective_from = entry.get("effective_from", "<unknown>")
            label = f"{path.relative_to(parameters_dir)}[{index}] {parameter_id} from {effective_from}"
            urls = citation_urls(entry)
            if not urls:
                stats.skipped_entries += 1
                stats.skipped.append(f"{label}: no citation URL")
                continue

            stats.checked_entries += 1
            entry_ok = True
            for url in urls:
                stats.checked_urls += 1
                ok, detail = fetcher(url, timeout)
                if not ok:
                    entry_ok = False
                    stats.failures.append(f"{label}: {url} -> {detail}")
            if entry_ok:
                stats.verified_entries += 1
                verified_by_file.setdefault(path, []).append(index)
            else:
                stats.failed_entries += 1

    if not dry_run:
        for path, indices in verified_by_file.items():
            stamp_verified_entries(path, indices, verified_at)

    return stats


def stamp_verified_entries(path: Path, entry_indices: Iterable[int], verified_at: str) -> None:
    target_indices = set(entry_indices)
    if not target_indices:
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("  - parameter_id:")]
    if not starts:
        return
    starts.append(len(lines))

    for entry_index in sorted(target_indices, reverse=True):
        start = starts[entry_index]
        end = starts[entry_index + 1]
        block = lines[start:end]
        replacement = _stamp_block(block, verified_at)
        lines[start:end] = replacement

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stamp_block(block: list[str], verified_at: str) -> list[str]:
    for i, line in enumerate(block):
        if line.startswith("    last_verified_at:"):
            block[i] = f"    last_verified_at: {verified_at}"
            return block

    for i, line in enumerate(block):
        if line.startswith("    official:"):
            block.insert(i + 1, f"    last_verified_at: {verified_at}")
            return block

    for i, line in enumerate(block):
        if line.startswith("    citations:"):
            block.insert(i, f"    last_verified_at: {verified_at}")
            return block

    block.append(f"    last_verified_at: {verified_at}")
    return block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify URL-backed parameter citations.")
    parser.add_argument(
        "--parameters-dir",
        type=Path,
        default=default_parameters_dir(),
        help="Directory containing parameter YAML files.",
    )
    parser.add_argument("--verified-at", default=date.today().isoformat(), help="ISO date to stamp.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch URLs but do not modify YAML files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stats = verify_parameters(
        args.parameters_dir,
        verified_at=args.verified_at,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    print(f"Checked entries: {stats.checked_entries}")
    print(f"Checked URLs: {stats.checked_urls}")
    print(f"Verified entries: {stats.verified_entries}")
    print(f"Skipped entries: {stats.skipped_entries}")
    print(f"Failed entries: {stats.failed_entries}")
    for skipped in stats.skipped:
        print(f"SKIP {skipped}")
    for failure in stats.failures:
        print(f"FAIL {failure}")
    return 1 if stats.failed_entries else 0


if __name__ == "__main__":
    raise SystemExit(main())
