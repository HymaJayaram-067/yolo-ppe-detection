"""
Model Evaluation — Compute mAP, Precision, Recall on validation set
"""
import argparse
from ultralytics import YOLO
import json


def evaluate(args):
    model = YOLO(args.model)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        plots=True,
        save_json=True,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_AP50": {
            name: float(ap) for name, ap in
            zip(results.names.values(), results.box.ap50)
        } if hasattr(results.box, 'ap50') else {}
    }

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print("=" * 50)

    if metrics["per_class_AP50"]:
        print("\n  Per-Class AP@0.5:")
        for cls_name, ap in metrics["per_class_AP50"].items():
            print(f"    {cls_name}: {ap:.4f}")

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPE Detection Model")
    parser.add_argument("--model", default="runs/train/ppe_yolov8n/weights/best.pt")
    parser.add_argument("--data", default="configs/ppe_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    evaluate(args)
