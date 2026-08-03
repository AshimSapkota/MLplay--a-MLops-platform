"""
train_yolo_object_detector.py

Real (if small) object detection training — YOLOv8n via Ultralytics.
Runs LOCALLY, using the exact same job contract as every other script in
this project (--data-path, --model-name, --job-id, --tracking-uri), so it
plugs into the dashboard's Train tab with compute_target="local" and
requires zero backend changes.

Install first:
  pip install ultralytics mlflow pandas pillow

--data-path: path to a YOLO-format data.yaml (images + label .txt files).
If it doesn't exist, falls back to "coco128" — Ultralytics' own small
128-image sample dataset, auto-downloaded on first use (~7MB). Good enough
to prove the whole pipeline without needing your own dataset ready yet.

What gets logged to MLflow:
  - params: base model, epochs, image size
  - metrics: mAP50, mAP50-95, precision, recall (Ultralytics computes these
    natively during validation — no need to hand-roll detection metrics)
  - the trained weights, wrapped in a custom pyfunc model so it's queryable
    through the dashboard the same way every other model is
"""

import argparse
import os
import time
import shutil

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class YOLODetector(PythonModel):
    """
    Wraps a trained YOLO model for MLflow serving. Input: a DataFrame with
    one column ("image_path") containing a path to an image file on disk.
    Output: a JSON string per row listing detected boxes/classes/scores.

    Why a file path instead of raw image bytes in the DataFrame: keeps the
    signature simple (a single string column) and matches how the dashboard
    already handles file-based inputs (like the CSV batch upload) rather
    than needing a new binary/base64 input path.
    """

    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        import json
        results_out = []
        for image_path in model_input["image_path"]:
            results = self.model(image_path, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox_xyxy": [float(x) for x in box.xyxy[0].tolist()],
                    })
            results_out.append(json.dumps(detections))
        return results_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("object-detection-training")
    client = MlflowClient(tracking_uri=args.tracking_uri)

    # Fall back to Ultralytics' own small sample dataset if no real one given —
    # same "always runnable" pattern as the other training scripts.
    data_yaml = args.data_path if os.path.isfile(args.data_path) else "coco128.yaml"
    if data_yaml == "coco128.yaml":
        log("No data.yaml found at the given path — using Ultralytics' built-in "
            "coco128 sample dataset instead (auto-downloads ~7MB on first use).")

    log(f"Starting YOLO training job {args.job_id} -> registering as '{args.model_name}'")
    log(f"Dataset: {data_yaml} | epochs: {args.epochs} | image size: {args.imgsz}")
    log(f"Training device: {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device == 0 else " (no GPU detected — this will be slow)"))

    with mlflow.start_run(run_name=f"yolo-train-job-{args.job_id}") as run:
        mlflow.log_params({
            "base_model": "yolov8n.pt",
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "dataset": data_yaml,
            "device": str(device),
        })

        model = YOLO("yolov8n.pt")  # pretrained nano checkpoint — small, fast, real weights

        # Ultralytics handles its own training loop, validation, and metric
        # computation (mAP etc.) — no need to hand-roll any of that.
        train_results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project="/tmp/yolo_runs",
            name=f"job_{args.job_id}",
            verbose=True,
            device=device,
        )

        metrics = train_results.results_dict
        log(f"Training complete. mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'n/a')}")

        mlflow.log_metrics({
            "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall": metrics.get("metrics/recall(B)", 0.0),
        })

        # Locate the best checkpoint Ultralytics saved during training
        best_weights = os.path.join("/tmp/yolo_runs", f"job_{args.job_id}", "weights", "best.pt")
        if not os.path.isfile(best_weights):
            log(f"WARNING: expected weights not found at {best_weights}, training may have failed.")
            return

        # Build a tiny example input for the signature — reuse one training image
        example_image = None
        for root, _, files in os.walk(os.path.dirname(data_yaml) if os.path.isdir(os.path.dirname(data_yaml)) else "."):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    example_image = os.path.join(root, f)
                    break
            if example_image:
                break

        example_input = pd.DataFrame({"image_path": [example_image or "example.jpg"]})
        example_output = ["[]"]  # placeholder shape — real output is a JSON string per row
        signature = infer_signature(example_input, example_output)

        model_info = mlflow.pyfunc.log_model(
            python_model=YOLODetector(),
            name="model",
            artifacts={"weights": best_weights},
            signature=signature,
            input_example=example_input,
            registered_model_name=args.model_name,
        )
        version = model_info.registered_model_version

        client.set_model_version_tag(args.model_name, version, "training_job_id", args.job_id)
        client.set_model_version_tag(args.model_name, version, "intended_use",
            "Small-scale object detection demo (YOLOv8n). Not validated for production use.")
        client.set_model_version_tag(args.model_name, version, "known_limitations",
            f"Trained for only {args.epochs} epochs on {data_yaml} — a real deployment "
            f"would need substantially more data/training and a held-out test set.")

        log(f"Registered '{args.model_name}' version {version} (run_id={run.info.run_id})")
        log("DONE")


if __name__ == "__main__":
    main()