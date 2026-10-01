r"""Fine-tune YOLOv8 for the Assistive Vision Pipeline.

The training data must be in Ultralytics YOLO detection format:

    dataset/
      data.yaml
      images/train, images/val
      labels/train, labels/val

Each label file has one object per line:
    <class_id> <x_center> <y_center> <width> <height>

The four coordinates must be normalised to the range 0..1.

Examples
--------
Quick smoke test using real COCO images downloaded by Ultralytics:
    python train_yolo.py --data coco8.yaml --epochs 3

Fine-tune on an exported Roboflow/Kaggle/custom YOLO dataset:
    python train_yolo.py --data C:\path\to\dataset\data.yaml --epochs 50

Run evaluation only:
    python train_yolo.py --data C:\path\to\dataset\data.yaml --validate-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on labelled real images.")
    parser.add_argument("--data", required=True, help="Dataset YAML path or an Ultralytics dataset name, e.g. coco8.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Starting checkpoint (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50).")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size (default: 640).")
    parser.add_argument("--batch", type=int, default=4, help="Batch size; lower it if memory is insufficient (default: 4).")
    parser.add_argument("--device", default=None, help="cpu, 0, or 0,1. Defaults to CUDA GPU when available, otherwise CPU.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader workers; 0 is reliable on Windows (default: 0).")
    parser.add_argument("--project", default="runs/train", help="Folder for training outputs.")
    parser.add_argument("--name", default="assistive_yolov8n", help="Name of this training run.")
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many non-improving epochs.")
    parser.add_argument("--validate-only", action="store_true", help="Evaluate the checkpoint without training.")
    return parser.parse_args()


def check_local_dataset(data: str) -> None:
    """Give clear early errors for a local YAML file before a long run starts."""
    data_path = Path(data)
    if not data_path.is_file():
        # Dataset names such as coco8.yaml are resolved/downloaded by Ultralytics.
        return

    with data_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    missing = [key for key in ("train", "val", "names") if key not in config]
    if missing:
        raise ValueError(f"{data_path} is missing required field(s): {', '.join(missing)}")

    names = config["names"]
    class_count = len(names) if isinstance(names, (list, dict)) else 0
    if not class_count:
        raise ValueError("'names' must list at least one class, for example names: [person, car]")

    print(f"Dataset configuration: {data_path.resolve()}")
    print(f"Classes ({class_count}): {list(names.values()) if isinstance(names, dict) else names}")


def main() -> None:
    args = parse_args()
    check_local_dataset(args.data)

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device == "cpu":
        print("CPU training selected. Use --epochs 3 for a quick check; full runs may take hours.")

    model = YOLO(args.model)
    common = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }

    if args.validate_only:
        metrics = model.val(**common)
        print(f"Validation complete. mAP50-95: {metrics.box.map:.4f}")
        return

    results = model.train(
        **common,
        epochs=args.epochs,
        patience=args.patience,
        pretrained=True,
        seed=42,
        plots=True,
        verbose=True,
    )
    print("\nTraining complete.")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    print("Use the best.pt path in your inference command or copy it to the project folder.")


if __name__ == "__main__":
    main()
