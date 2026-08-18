"""
train_dummy_model.py

A STAND-IN for a real object-detection training job. Real object detection
(bounding boxes, e.g. Faster R-CNN / YOLO via torchvision) needs real image
data, GPU time, and heavier dependencies — not appropriate for a local demo.

What this script actually does: generates synthetic "image-like" feature
vectors and trains a small classifier to predict object-present / not-present
— structurally the same kind of binary classification problem, cheap to run
anywhere, but standing in for something that would be a real CV model in
production.

CONTRACT this script follows (required for training_manager.py to work):
  --data-path     path to training data (a CSV here; ignored if it doesn't
                   exist — falls back to synthetic data so the demo always runs)
  --model-name    the registry name to register under
  --job-id        passed in by the API — MUST be set as a tag on the
                   registered version, so the API can find it afterward
  --tracking-uri  where to log to

Any real training script (PyTorch, whatever) just needs to accept these same
four arguments and follow the same tagging convention — the API doesn't care
what's inside.
"""

import argparse
import os
import sys
import time

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tracking-uri", required=True)
    # Real, tunable hyperparameters — this is what a sweep actually searches over.
    # Sensible defaults kept so this script still works standalone, unchanged,
    # for every non-sweep use case elsewhere in this project.
    parser.add_argument("--hidden-size", type=int, default=32,
        help="Size of the first hidden layer (second layer is always half this).")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("training-api-jobs")
    client = MlflowClient(tracking_uri=args.tracking_uri)

    log(f"Starting training job {args.job_id} -> registering as '{args.model_name}'")

    # --- Load real data if a path was given and exists, else synthesize ---
    if os.path.isfile(args.data_path):
        log(f"Loading training data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        if "target" not in df.columns:
            log("ERROR: provided CSV has no 'target' column.")
            sys.exit(1)
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        log(f"No data file found at {args.data_path} — generating synthetic 'object-present' data instead.")
        # Standing in for image feature embeddings (e.g. what a CNN backbone
        # would produce) — 32 synthetic "features" per sample.
        X_arr, y_arr = make_classification(
            n_samples=800, n_features=32, n_informative=15, random_state=7
        )
        feature_cols = [f"embed_{i}" for i in range(X_arr.shape[1])]
        X = pd.DataFrame(X_arr, columns=feature_cols)
        y = pd.Series(y_arr, name="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log(f"Training on {len(X_train)} rows, {X.shape[1]} features...")

    # Simulate a bit of real training time so the status endpoint has
    # something meaningful to poll during a "running" state.
    with mlflow.start_run(run_name=f"train-job-{args.job_id}") as run:
        mlflow.log_param("training_job_id", args.job_id)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("model_type", "MLPClassifier (dummy object-detector stand-in)")
        mlflow.log_params({
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
        })

        model = MLPClassifier(
            hidden_layer_sizes=(args.hidden_size, max(args.hidden_size // 2, 1)),
            learning_rate_init=args.learning_rate,
            max_iter=args.max_iter,
            random_state=42,
        )

        # Fake "epochs" so logs actually show progress over time, like a real job would
        for step in range(3):
            log(f"...training step {step + 1}/3")
            time.sleep(2)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        log(f"Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f}")

        # Explicit, clearly-named metric — this is what the sweep script reads
        # back to know how well a given hyperparameter combination did.
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:2],
            registered_model_name=args.model_name,
            # MLPClassifier carries an AdamOptimizer as part of its fitted state.
            # skops (MLflow's safer default serializer) refuses untrusted types
            # by default — this is the security check from Phase 2 working as
            # intended. We explicitly trust it here because it's our own
            # just-trained model, not a file from an untrusted source.
            skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"],
        )
        version = model_info.registered_model_version

        # --- Required governance tags, same convention as everything else ---
        client.set_model_version_tag(args.model_name, version, "training_job_id", args.job_id)
        client.set_model_version_tag(args.model_name, version, "intended_use",
            "Demo object-presence classifier trained via the training API — placeholder for a real CV model.")
        client.set_model_version_tag(args.model_name, version, "known_limitations",
            "Trained on synthetic embeddings, not real imagery. Not for any real decisioning.")

        log(f"Registered '{args.model_name}' version {version} (run_id={run.info.run_id})")
        log("DONE")


if __name__ == "__main__":
    main()