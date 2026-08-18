"""
sweep_optuna.py

Hyperparameter search layered ON TOP of the existing training infrastructure
— does not bypass it. Each Optuna trial is just another call to
training_manager.submit_job(), the exact same API the dashboard's Train tab
uses. Optuna's only job is picking which hyperparameters to try next and
reading back the resulting metric.

Scope, deliberately, per the agreed priority order:
  - compute_target="local" only (no remote dispatch yet — prove the pattern
    cheaply before adding SSH/AWS complexity on top of it)
  - No pruning yet (would need mid-training metric streaming — a real,
    separate piece of work, not bolted on here)
  - Study persisted to SQLite, independent of whichever MLflow backend is
    active, so a sweep survives a backend restart the same way MLflow's
    own data does

Usage:
  python sweep_optuna.py --n-trials 10

After it finishes, inspect results with:
  pip install optuna-dashboard
  optuna-dashboard sqlite:///optuna_studies.db
"""

import argparse
import os
import sys
import time

import optuna
from mlflow import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_manager import training_manager  # the exact same job manager the dashboard uses

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TRAINING_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training_scripts", "train_dummy_model.py"
)
SWEEP_MODEL_NAME = "internal-tools-sweep-demo"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one_trial(hidden_size: int, learning_rate: float, max_iter: int, sweep_id: str, trial_number: int) -> float:
    """
    Submits one training job with the given hyperparameters, blocks (polling)
    until it finishes, and returns the metric Optuna should optimize.
    Returns a bad score (0.0) on failure rather than raising — one bad trial
    shouldn't kill the whole sweep.
    """
    job_id = training_manager.submit_job(
        script_path=TRAINING_SCRIPT,
        data_path="/tmp/no_such_sweep_data.csv",  # triggers the script's synthetic-data fallback
        model_name=SWEEP_MODEL_NAME,
        compute_target="local",
        extra_args={
            "hidden-size": hidden_size,
            "learning-rate": learning_rate,
            "max-iter": max_iter,
        },
    )
    log(f"Trial {trial_number}: submitted job {job_id} "
        f"(hidden_size={hidden_size}, lr={learning_rate:.5f}, max_iter={max_iter})")

    # Poll until done — this blocks the sweep script itself, which is fine:
    # Optuna runs trials sequentially by default unless you explicitly
    # parallelize, and local jobs finish in seconds anyway.
    while True:
        status = training_manager.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(2)

    if status["status"] == "failed":
        log(f"Trial {trial_number}: job {job_id} FAILED — returning 0.0 for this trial")
        return 0.0

    version = status["registered_version"]
    if not version:
        log(f"Trial {trial_number}: job completed but no registered version found — returning 0.0")
        return 0.0

    # Pull the metric back from MLflow, and tag this version with which
    # sweep/trial produced it — same tagging discipline as everywhere else
    # in this project (training_job_id, intended_use, etc.)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mv = client.get_model_version(SWEEP_MODEL_NAME, version)
    run = client.get_run(mv.run_id)

    test_accuracy = run.data.metrics.get("test_accuracy")
    if test_accuracy is None:
        log(f"Trial {trial_number}: 'test_accuracy' metric not found on run — returning 0.0")
        return 0.0

    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "sweep_id", sweep_id)
    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "sweep_trial_number", str(trial_number))

    log(f"Trial {trial_number}: test_accuracy={test_accuracy:.4f} (version {version})")
    return test_accuracy


def make_objective(sweep_id: str):
    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 500, step=100)

        return run_one_trial(hidden_size, learning_rate, max_iter, sweep_id, trial.number)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", default=None,
        help="Reuse an existing study by name to resume/add more trials to it.")
    args = parser.parse_args()

    study_name = args.study_name or f"sweep-{int(time.time())}"

    # Persisted to SQLite — survives a restart, independent of MLflow's own
    # backend. Same idea as MLflow's own persistence, applied to the sweep.
    storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_studies.db")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        load_if_exists=True,
    )

    log(f"Starting sweep '{study_name}' — {args.n_trials} trials, optimizing test_accuracy")
    study.optimize(make_objective(study_name), n_trials=args.n_trials)

    log("Sweep complete.")
    log(f"Best value: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")

    # Promote the winning trial's model version to a "sweep-winner" alias —
    # a real, promotable model version comes out the other end of this,
    # not just a number. Doesn't touch the champion/challenger aliases your
    # CI gate cares about — this is a separate, clearly-labeled pointer.
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    winning_version = None
    for mv in client.search_model_versions(f"name='{SWEEP_MODEL_NAME}'"):
        tags = mv.tags or {}
        if tags.get("sweep_id") == study_name and tags.get("sweep_trial_number") == str(study.best_trial.number):
            winning_version = mv.version
            break

    if winning_version:
        client.set_registered_model_alias(SWEEP_MODEL_NAME, "sweep-winner", winning_version)
        client.set_model_version_tag(SWEEP_MODEL_NAME, winning_version, "sweep_winner", "true")
        log(f"Tagged version {winning_version} as @sweep-winner")

    log(f"View full results: optuna-dashboard sqlite:///{storage_path}")


if __name__ == "__main__":
    main()