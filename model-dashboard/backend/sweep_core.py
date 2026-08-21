"""
sweep_core.py

The actual sweep logic, extracted so both the standalone CLI
(sweep_optuna.py) and the dashboard-triggered background runner
(sweep_manager.py) share one implementation — no duplicated logic to keep
in sync.
"""

import os
import sys
import time

import optuna
import mlflow
from mlflow import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_manager import training_manager  # same job manager the dashboard uses

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TRAINING_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training_scripts", "train_dummy_model.py"
)
SWEEP_MODEL_NAME = "internal-tools-sweep-demo"
STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_studies.db")
STORAGE_URI = f"sqlite:///{STORAGE_PATH}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one_trial(hidden_size: int, learning_rate: float, max_iter: int, trial: optuna.Trial) -> float:
    """Submits one training job, blocks until it finishes, returns the metric to optimize."""
    job_id = training_manager.submit_job(
        script_path=TRAINING_SCRIPT,
        data_path="/tmp/no_such_sweep_data.csv",
        model_name=SWEEP_MODEL_NAME,
        compute_target="local",
        extra_args={
            "hidden-size": hidden_size,
            "learning-rate": learning_rate,
            "max-iter": max_iter,
            "skip-registry": True,
        },
    )
    log(f"Trial {trial.number}: submitted job {job_id} "
        f"(hidden_size={hidden_size}, lr={learning_rate:.5f}, max_iter={max_iter})")

    while True:
        status = training_manager.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(2)

    if status["status"] == "failed":
        log(f"Trial {trial.number}: job {job_id} FAILED — returning 0.0")
        return 0.0

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name("training-api-jobs")
    if experiment is None:
        log(f"Trial {trial.number}: 'training-api-jobs' experiment not found — returning 0.0")
        return 0.0

    matches = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.training_job_id = '{job_id}'",
        max_results=1,
    )
    if not matches:
        log(f"Trial {trial.number}: no MLflow run found for job {job_id} — returning 0.0")
        return 0.0

    run = matches[0]
    test_accuracy = run.data.metrics.get("test_accuracy")
    if test_accuracy is None:
        log(f"Trial {trial.number}: 'test_accuracy' metric not found — returning 0.0")
        return 0.0

    trial.set_user_attr("run_id", run.info.run_id)
    log(f"Trial {trial.number}: test_accuracy={test_accuracy:.4f} (run_id={run.info.run_id})")
    return test_accuracy


def make_objective():
    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 500, step=100)
        return run_one_trial(hidden_size, learning_rate, max_iter, trial)
    return objective


def register_winner(study: optuna.Study, study_name: str, n_trials: int) -> str | None:
    """
    Registers ONLY the winning trial's model — every other trial exists
    solely as a recoverable MLflow run, never as a registry version.
    Returns the new version number, or None if something went wrong.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    winning_run_id = study.best_trial.user_attrs.get("run_id")
    if not winning_run_id:
        log("WARNING: best trial has no stored run_id — cannot register a winner.")
        return None

    registered = mlflow.register_model(f"runs:/{winning_run_id}/model", SWEEP_MODEL_NAME)
    version = registered.version

    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "sweep_id", study_name)
    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "sweep_trial_number", str(study.best_trial.number))
    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "sweep_winner", "true")
    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "intended_use",
        "Winning configuration from an Optuna hyperparameter sweep — demo purposes.")
    client.set_model_version_tag(SWEEP_MODEL_NAME, version, "known_limitations",
        f"Selected from {n_trials} trials on synthetic data; not validated beyond this sweep's own metric.")
    client.set_registered_model_alias(SWEEP_MODEL_NAME, "sweep-winner", version)

    log(f"Registered and tagged version {version} as @sweep-winner (run_id={winning_run_id})")
    return version


def run_sweep(n_trials: int, study_name: str) -> dict:
    """
    The full sweep lifecycle: create/load study, optimize, register the
    winner. Callable from anywhere — the CLI script, a background thread,
    a future API endpoint — same implementation everywhere.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE_URI,
        direction="maximize",
        load_if_exists=True,
    )

    log(f"Starting sweep '{study_name}' — {n_trials} trials, optimizing test_accuracy")
    study.optimize(make_objective(), n_trials=n_trials)

    log(f"Sweep complete. Best value: {study.best_value:.4f} | Best params: {study.best_params}")
    winning_version = register_winner(study, study_name, n_trials)

    return {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "winning_version": winning_version,
        "n_trials": len(study.trials),
    }