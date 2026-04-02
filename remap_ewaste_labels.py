#!/usr/bin/env python3
"""Shuffle and rebuild YOLO train/valid/test splits for ewaste-dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "ewaste-dataset"
TEMP_ROOT = DATASET_ROOT / "_reshuffle_tmp"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetItem:
    stem: str
    image_name: str
    label_name: str | None
    image_path: Path
    label_path: Path | None


def collect_items() -> list[DatasetItem]:
    """Collect all images and matching labels across current split folders."""
    image_index: dict[str, Path] = {}
    label_index: dict[str, Path] = {}

    for split in SPLITS:
        images_dir = DATASET_ROOT / split / "images"
        labels_dir = DATASET_ROOT / split / "labels"

        if images_dir.is_dir():
            for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if image_path.stem in image_index:
                    raise ValueError(f"Duplicate image stem found: {image_path.stem}")
                image_index[image_path.stem] = image_path

        if labels_dir.is_dir():
            for label_path in sorted(labels_dir.glob("*.txt")):
                if label_path.stem in label_index:
                    raise ValueError(f"Duplicate label stem found: {label_path.stem}")
                label_index[label_path.stem] = label_path

    if not image_index:
        raise ValueError(f"No images found under {DATASET_ROOT}")

    orphan_labels = sorted(stem for stem in label_index if stem not in image_index)
    if orphan_labels:
        raise ValueError(
            f"Found labels without matching images, for example: {orphan_labels[0]}"
        )

    items: list[DatasetItem] = []
    for stem in sorted(image_index):
        image_path = image_index[stem]
        label_path = label_index.get(stem)
        items.append(
            DatasetItem(
                stem=stem,
                image_name=image_path.name,
                label_name=label_path.name if label_path else None,
                image_path=image_path,
                label_path=label_path,
            )
        )

    return items


def stage_items(items: list[DatasetItem]) -> list[DatasetItem]:
    """Move images and labels into a temporary holding area."""
    if TEMP_ROOT.exists():
        raise ValueError(
            f"Temporary directory already exists: {TEMP_ROOT}. "
            "Remove it before running this script again."
        )

    temp_images_dir = TEMP_ROOT / "images"
    temp_labels_dir = TEMP_ROOT / "labels"
    temp_images_dir.mkdir(parents=True, exist_ok=False)
    temp_labels_dir.mkdir(parents=True, exist_ok=False)

    staged_items: list[DatasetItem] = []
    for item in items:
        staged_image_path = temp_images_dir / item.image_name
        shutil.move(str(item.image_path), str(staged_image_path))

        staged_label_path: Path | None = None
        if item.label_path is not None and item.label_name is not None:
            staged_label_path = temp_labels_dir / item.label_name
            shutil.move(str(item.label_path), str(staged_label_path))

        staged_items.append(
            DatasetItem(
                stem=item.stem,
                image_name=item.image_name,
                label_name=item.label_name,
                image_path=staged_image_path,
                label_path=staged_label_path,
            )
        )

    return staged_items


def recreate_split_dirs() -> None:
    """Remove old split folders and recreate the expected YOLO structure."""
    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)


def move_items_to_split(items: list[DatasetItem], split: str) -> None:
    """Move staged files into a target split."""
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"

    for item in items:
        shutil.move(str(item.image_path), str(images_dir / item.image_name))
        if item.label_path is not None and item.label_name is not None:
            shutil.move(str(item.label_path), str(labels_dir / item.label_name))


def main() -> int:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_ROOT}")

    items = collect_items()
    random.shuffle(items)

    total = len(items)
    train_count = int(total * 0.7)
    valid_count = int(total * 0.2)
    test_count = total - train_count - valid_count

    train_items = items[:train_count]
    valid_items = items[train_count : train_count + valid_count]
    test_items = items[train_count + valid_count :]

    staged_items = stage_items(items)
    staged_by_stem = {item.stem: item for item in staged_items}

    recreate_split_dirs()

    move_items_to_split([staged_by_stem[item.stem] for item in train_items], "train")
    move_items_to_split([staged_by_stem[item.stem] for item in valid_items], "valid")
    move_items_to_split([staged_by_stem[item.stem] for item in test_items], "test")

    shutil.rmtree(TEMP_ROOT)

    print(f"Train images: {len(train_items)}")
    print(f"Valid images: {len(valid_items)}")
    print(f"Test images: {len(test_items)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - script-level error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
