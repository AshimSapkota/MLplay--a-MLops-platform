"""
sweep_optuna.py

CLI entry point for running a sweep standalone (outside the dashboard).
All actual logic lives in sweep_core.py, shared with the dashboard's
background sweep runner (sweep_manager.py) — this file is just argument
parsing.

Usage:
  python sweep_optuna.py --n-trials 10

After it finishes, inspect results either through the dashboard's Sweeps
tab, or standalone:
  pip install optuna-dashboard
  optuna-dashboard sqlite:///optuna_studies.db
"""

import argparse
import time

from sweep_core import run_sweep, STORAGE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", default=None,
        help="Reuse an existing study by name to resume/add more trials to it.")
    args = parser.parse_args()

    study_name = args.study_name or f"sweep-{int(time.time())}"
    result = run_sweep(args.n_trials, study_name)

    print(f"\nView full results: optuna-dashboard sqlite:///{STORAGE_PATH}")
    print(f"Or check the dashboard's Sweeps tab for '{result['study_name']}'")


if __name__ == "__main__":
    main()