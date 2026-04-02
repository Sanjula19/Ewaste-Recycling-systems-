#!/usr/bin/env python3
"""Remap YOLO class IDs inside ewaste-dataset label files.

The script matches each merged label file in ``ewaste-dataset`` back to its
origin inside ``datasets-source`` using the same split-relative label path
(``train/labels/...``, ``valid/labels/...``, ``test/labels/...``).

Supported remaps:
    smartphone -> 0
    laptop     -> 1
    battery    -> 2
    pcb        -> 3
    charger    -> 4
    keyboard   -> 5
    mouse      -> 6

Notes:
    * All PCB source IDs (0, 1, 2, 3) collapse into final class 3.
    * ``ewaste-mixed`` is multi-class. Only charger ID 0 is mapped here.
      Any other ID from ``ewaste-mixed`` will stop the script so we do not
      silently produce labels outside the requested 7-class taxonomy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "ewaste-dataset"
SOURCES_ROOT = ROOT / "datasets-source"
SPLITS = ("train", "valid", "test")

FINAL_CLASS_NAMES = {
    0: "smartphone",
    1: "laptop",
    2: "battery",
    3: "pcb",
    4: "charger",
    5: "keyboard",
    6: "mouse",
}

SINGLE_CLASS_SOURCE_MAP = {
    "mobile-phone": 0,
    "laptop": 1,
    "battery": 2,
    "keyboard": 5,
    "mouse": 6,
}

PCB_SOURCE_NAME = "pcb"
PCB_SOURCE_IDS = {"0", "1", "2", "3"}

EWASTE_MIXED_SOURCE_NAME = "ewaste-mixed"
EWASTE_MIXED_ID_MAP = {
    "0": 4,  # Confirmed charger/power adapter label in ewaste-mixed.
}


def build_source_index() -> dict[Path, str]:
    """Map each merged split-relative label path to its source dataset name."""
    index: dict[Path, str] = {}

    if not SOURCES_ROOT.is_dir():
        raise FileNotFoundError(f"Missing sources directory: {SOURCES_ROOT}")

    for source_dir in sorted(path for path in SOURCES_ROOT.iterdir() if path.is_dir()):
        for split in SPLITS:
            labels_dir = source_dir / split / "labels"
            if not labels_dir.is_dir():
                continue

            for label_path in labels_dir.rglob("*.txt"):
                rel_path = label_path.relative_to(source_dir)
                if rel_path in index:
                    raise ValueError(
                        f"Duplicate source mapping for {rel_path}: "
                        f"{index[rel_path]!r} and {source_dir.name!r}"
                    )
                index[rel_path] = source_dir.name

    if not index:
        raise ValueError(f"No source label files found under {SOURCES_ROOT}")

    return index


def remap_class_id(source_name: str, original_id: str, rel_path: Path) -> int:
    """Return the final class ID for a single YOLO annotation line."""
    if source_name in SINGLE_CLASS_SOURCE_MAP:
        if original_id != "0":
            raise ValueError(
                f"Unexpected class ID {original_id!r} in single-class source "
                f"{source_name!r} for {rel_path}"
            )
        return SINGLE_CLASS_SOURCE_MAP[source_name]

    if source_name == PCB_SOURCE_NAME:
        if original_id not in PCB_SOURCE_IDS:
            raise ValueError(
                f"Unexpected PCB class ID {original_id!r} for {rel_path}; "
                f"expected one of {sorted(PCB_SOURCE_IDS)}"
            )
        return 3

    if source_name == EWASTE_MIXED_SOURCE_NAME:
        if original_id not in EWASTE_MIXED_ID_MAP:
            raise ValueError(
                f"Unsupported ewaste-mixed class ID {original_id!r} in {rel_path}. "
                "Only charger ID '0' is mapped by this script."
            )
        return EWASTE_MIXED_ID_MAP[original_id]

    raise ValueError(f"Unknown source dataset {source_name!r} for {rel_path}")


def remap_file(label_path: Path, source_name: str, rel_path: Path) -> list[str]:
    """Return remapped label lines for a file."""
    remapped_lines: list[str] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            remapped_lines.append("")
            continue

        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(
                f"Invalid YOLO line in {rel_path}:{line_number}: {line!r}"
            )

        parts[0] = str(remap_class_id(source_name, parts[0], rel_path))
        remapped_lines.append(" ".join(parts))

    return remapped_lines


def find_unsupported_mixed_ids(
    dataset_files: list[tuple[Path, Path, str]]
) -> tuple[Counter, dict[str, Path]]:
    """Return unsupported ewaste-mixed class counts and one example path each."""
    unsupported_counts: Counter = Counter()
    examples: dict[str, Path] = {}

    for label_path, rel_path, source_name in dataset_files:
        if source_name != EWASTE_MIXED_SOURCE_NAME:
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            original_id = stripped.split(maxsplit=1)[0]
            if original_id in EWASTE_MIXED_ID_MAP:
                continue

            unsupported_counts[original_id] += 1
            examples.setdefault(original_id, rel_path)

    return unsupported_counts, examples


def main() -> int:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_ROOT}")

    source_index = build_source_index()
    dataset_files: list[tuple[Path, Path, str]] = []
    file_updates: list[tuple[Path, list[str]]] = []
    summary = Counter()
    processed_files = 0

    for split in SPLITS:
        labels_dir = DATASET_ROOT / split / "labels"
        if not labels_dir.is_dir():
            continue

        for label_path in sorted(labels_dir.rglob("*.txt")):
            rel_path = label_path.relative_to(DATASET_ROOT)
            source_name = source_index.get(rel_path)
            if source_name is None:
                raise ValueError(
                    f"No source dataset match found for merged label file {rel_path}"
                )

            dataset_files.append((label_path, rel_path, source_name))

    unsupported_counts, examples = find_unsupported_mixed_ids(dataset_files)
    if unsupported_counts:
        print(
            "Error: ewaste-mixed contains class IDs outside the requested 7-class mapping.",
            file=sys.stderr,
        )
        print(
            "Add those IDs to EWASTE_MIXED_ID_MAP or remove those annotations first.",
            file=sys.stderr,
        )
        for original_id in sorted(unsupported_counts, key=int):
            print(
                f"  id {original_id}: {unsupported_counts[original_id]} annotations "
                f"(example: {examples[original_id]})",
                file=sys.stderr,
            )
        return 1

    for label_path, rel_path, source_name in dataset_files:
        remapped_lines = remap_file(label_path, source_name, rel_path)
        for line in remapped_lines:
            if not line:
                continue
            final_id = int(line.split(maxsplit=1)[0])
            summary[final_id] += 1

        file_updates.append((label_path, remapped_lines))
        processed_files += 1

    for label_path, remapped_lines in file_updates:
        output = "\n".join(remapped_lines)
        if remapped_lines:
            output += "\n"
        label_path.write_text(output, encoding="utf-8")

    print(f"Updated {processed_files} label files in place.")
    print("Summary count per class:")
    for class_id in sorted(FINAL_CLASS_NAMES):
        class_name = FINAL_CLASS_NAMES[class_id]
        print(f"  {class_id} ({class_name}): {summary[class_id]}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - script-level error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
