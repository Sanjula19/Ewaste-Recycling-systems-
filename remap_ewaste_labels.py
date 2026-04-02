#!/usr/bin/env python3
"""Remove YOLO annotations with class IDs greater than 6.

The script scans every label file in:
    ewaste-dataset/train/labels
    ewaste-dataset/valid/labels
    ewaste-dataset/test/labels

Rules:
    * If a label line starts with a class ID greater than 6, that line is removed.
    * If a label file becomes empty afterward, the label file is deleted.
    * When a label file is deleted, the matching image in the sibling ``images``
      directory is deleted too.

The dataset is modified in place.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "ewaste-dataset"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_VALID_CLASS_ID = 6


def find_matching_image(label_path: Path) -> Path | None:
    """Return the sibling image path for a label file, if it exists."""
    images_dir = label_path.parent.parent / "images"
    stem = label_path.stem

    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{extension}"
        if image_path.exists():
            return image_path

    matches = [path for path in images_dir.glob(f"{stem}.*") if path.is_file()]
    if len(matches) == 1:
        return matches[0]

    return None


def filter_label_lines(label_path: Path) -> tuple[list[str], int]:
    """Keep only label lines whose class ID is 0..6."""
    kept_lines: list[str] = []
    removed_count = 0

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        try:
            class_id = int(parts[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid YOLO label line in {label_path}:{line_number}: {line!r}"
            ) from exc

        if class_id > MAX_VALID_CLASS_ID:
            removed_count += 1
            continue

        kept_lines.append(stripped)

    return kept_lines, removed_count


def main() -> int:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_ROOT}")

    total_removed_labels = 0
    total_deleted_images = 0

    for split in SPLITS:
        labels_dir = DATASET_ROOT / split / "labels"
        if not labels_dir.is_dir():
            continue

        for label_path in sorted(labels_dir.rglob("*.txt")):
            kept_lines, removed_count = filter_label_lines(label_path)
            total_removed_labels += removed_count

            if kept_lines:
                output = "\n".join(kept_lines) + "\n"
                label_path.write_text(output, encoding="utf-8")
                continue

            if removed_count == 0:
                continue

            image_path = find_matching_image(label_path)
            label_path.unlink()

            if image_path is not None and image_path.exists():
                image_path.unlink()
                total_deleted_images += 1

    print(f"Labels removed: {total_removed_labels}")
    print(f"Images deleted: {total_deleted_images}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - script-level error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
