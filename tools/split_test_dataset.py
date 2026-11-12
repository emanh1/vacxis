"""Split the golden test datasets into review parts.

Each source dataset is split into three consecutive, near-equal,
order-preserving parts written under ``test_data_review/``::

    test_dataset.json        ->  test_dataset.part{1,2,3}.json
    test_dataset_masked.json ->  test_dataset_masked.part{1,2,3}.json

The split is deterministic, so a part is always the same slice of the same
source file. Running the script again simply rewrites the parts in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "test_data_review"

SOURCE_DATASETS: tuple[tuple[str, str], ...] = (
    ("test_dataset.json", "test_dataset"),
    ("test_dataset_masked.json", "test_dataset_masked"),
)


def split_records(records: list, parts: int) -> list[list]:
    """Split into ``parts`` consecutive, near-equal, order-preserving slices."""
    total = len(records)
    if parts <= 0:
        raise ValueError("parts must be positive")
    slices: list[list] = []
    start = 0
    for index in range(parts):
        end = total * (index + 1) // parts
        slices.append(records[start:end])
        start = end
    return slices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--source-dir", default=str(REPO_ROOT))
    parser.add_argument("--parts", type=int, default=3)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    written = 0
    for source_name, stem in SOURCE_DATASETS:
        source = Path(args.source_dir) / source_name
        if not source.exists():
            print(f"[ERROR] Missing source dataset: {source}", file=sys.stderr)
            return 1
        with source.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            print(f"[ERROR] {source} must contain a JSON list", file=sys.stderr)
            return 1
        for index, part in enumerate(split_records(records, args.parts), start=1):
            target = out_dir / f"{stem}.part{index}.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(part, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(f"{target}: {len(part)} records")
            written += 1
    print(f"Wrote {written} part files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
