#!/usr/bin/env python3
"""Train a YOLOv8 model with Ultralytics."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "dataset.yaml"
MODEL_WEIGHTS = "yolov8n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
GPU_BATCH_SIZE = 16
CPU_BATCH_SIZE = -1  # Ultralytics auto batch size.


def main() -> int:
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Missing dataset config: {DATA_CONFIG}")

    use_gpu = torch.cuda.is_available()
    batch_size = GPU_BATCH_SIZE if use_gpu else CPU_BATCH_SIZE
    device = 0 if use_gpu else "cpu"

    print("Starting YOLOv8 training...")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Dataset config: {DATA_CONFIG}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    print("Training progress will be shown below.\n")

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch_size,
        device=device,
        project="runs",
        name="train",
        exist_ok=True,
        save=True,
        verbose=True,
    )

    print("\nTraining finished.")
    print("Best model saved to runs/train/weights/best.pt")
    print(f"Results directory: {results.save_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - script-level error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
