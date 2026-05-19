"""
YOLOv8 Fine-Tuning Script for PPE Detection
Run on Google Colab with GPU or any CUDA-enabled machine.
Dataset: Roboflow PPE Dataset (4 classes)
"""
import argparse
from ultralytics import YOLO


def train(args):
    # Load pretrained YOLOv8 model
    model = YOLO(args.base_model)

    # Fine-tune on PPE dataset
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Output
        project="runs/train",
        name=args.name,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: runs/train/{args.name}/weights/best.pt")
    print(f"Results: {results.results_dict}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for PPE Detection")
    parser.add_argument("--base_model", default="yolov8n.pt", help="Base model (yolov8n/s/m)")
    parser.add_argument("--data", default="configs/ppe_dataset.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--name", default="ppe_yolov8n", help="Experiment name")
    args = parser.parse_args()

    train(args)
