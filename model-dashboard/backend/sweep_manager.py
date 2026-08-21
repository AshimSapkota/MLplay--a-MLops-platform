"""
sweep_manager.py

Runs sweeps triggered from the dashboard, in a background thread — not a
subprocess, since sweep_core.run_sweep() mostly just waits on
training_manager.get_status() polls (I/O-bound), and running it in-thread
means it reuses the exact same TrainingManager singleton as everything
else in this process, no extra plumbing needed.

Progress reporting deliberately does NOT maintain its own duplicate state
for "how many trials have run so far" — it queries the Optuna study
directly (optuna.load_study), since that's already the persisted source
of truth. We only track, ourselves, whether the background thread that
kicked off the sweep is still alive vs finished vs crashed — that's the
one thing the study alone can't tell you.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import optuna

import sweep_core


@dataclass
class SweepJob:
    study_name: str
    n_trials: int
    status: str = "running"       # running | completed | failed
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    winning_version: Optional[str] = None


class SweepManager:
    def __init__(self):
        self._sweeps: dict[str, SweepJob] = {}
        self._lock = threading.Lock()

    def start_sweep(self, n_trials: int, study_name: str | None = None) -> str:
        study_name = study_name or f"sweep-{int(time.time())}"

        job = SweepJob(study_name=study_name, n_trials=n_trials)
        with self._lock:
            self._sweeps[study_name] = job

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()

        return study_name

    def _run(self, job: SweepJob):
        try:
            result = sweep_core.run_sweep(job.n_trials, job.study_name)
            job.winning_version = result["winning_version"]
            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
        finally:
            job.finished_at = time.time()

    def get_status(self, study_name: str) -> dict:
        with self._lock:
            job = self._sweeps.get(study_name)
        if job is None:
            raise KeyError(f"No such sweep: {study_name}")

        # Live progress comes straight from Optuna's own persisted study —
        # accurate even mid-sweep, no duplicated bookkeeping to keep in sync.
        trials_summary = []
        best_value, best_params = None, None
        try:
            study = optuna.load_study(study_name=study_name, storage=sweep_core.STORAGE_URI)
            for t in study.trials:
                trials_summary.append({
                    "number": t.number,
                    "state": str(t.state.name),
                    "value": t.value,
                    "params": t.params,
                })
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                best_value = study.best_value
                best_params = study.best_params
        except Exception:
            pass  # study may not have any trials recorded yet on the very first poll

        return {
            "study_name": job.study_name,
            "status": job.status,
            "error": job.error,
            "n_trials_requested": job.n_trials,
            "n_trials_done": len(trials_summary),
            "best_value": best_value,
            "best_params": best_params,
            "winning_version": job.winning_version,
            "trials": trials_summary,
        }

    def list_sweeps(self) -> list:
        with self._lock:
            names = list(self._sweeps.keys())
        return [self.get_status(n) for n in names]


sweep_manager = SweepManager()