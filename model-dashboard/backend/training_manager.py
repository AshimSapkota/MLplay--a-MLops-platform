"""
training_manager.py

Generic training-job orchestration:
  - submit_job(script_path, data_path, model_name, extra_args) -> job_id
  - get_status(job_id) -> status, log tail, and (once done) the MLflow version it registered

IMPORTANT — about "remote GPU server" support:
Right now every job runs via subprocess.Popen on THIS machine. That's the
`compute_target="local"` path below, and it's the only one implemented.

The `compute_target="remote_gpu"` branch is a stub showing exactly where real
remote dispatch would go — in practice that would mean one of:
  - SSH: scp the script + data to the GPU box, run it over SSH, poll the
    remote process (or a job ID it returns), scp/pull logs back
  - Kubernetes: submit a Job manifest requesting a GPU resource, poll its
    pod status via the K8s API, stream logs via the API
  - Cloud ML platforms: call SageMaker/Vertex AI's training-job API, poll
    THEIR status endpoint instead of a local process

The rest of this file (job bookkeeping, status reporting, MLflow lookup)
would stay identical either way — only how the process actually gets
started and polled changes. That's the whole point of designing it this way.
"""

import os
import uuid
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class TrainingJob:
    job_id: str
    script_path: str
    data_path: str
    model_name: str
    compute_target: str
    status: str = "pending"          # pending | running | completed | failed
    process: Optional[subprocess.Popen] = None
    log_path: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    registered_version: Optional[str] = None


class TrainingManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    def submit_job(self, script_path: str, data_path: str, model_name: str,
                    compute_target: str = "local", extra_args: dict | None = None) -> str:
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Training script not found: {script_path}")

        job_id = str(uuid.uuid4())[:8]
        log_path = os.path.join(LOG_DIR, f"{job_id}.log")

        cmd = [
            "python", script_path,
            "--data-path", data_path,
            "--model-name", model_name,
            "--job-id", job_id,
            "--tracking-uri", MLFLOW_TRACKING_URI,
        ]
        for k, v in (extra_args or {}).items():
            cmd += [f"--{k}", str(v)]

        job = TrainingJob(
            job_id=job_id, script_path=script_path, data_path=data_path,
            model_name=model_name, compute_target=compute_target, log_path=log_path,
        )

        if compute_target == "local":
            log_file = open(log_path, "w")
            job.process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            job.status = "running"
            job.started_at = time.time()
        elif compute_target == "remote_gpu":
            # Not implemented — see module docstring. Left explicit rather than
            # silently falling back to local, so nobody assumes GPU dispatch
            # is happening when it isn't.
            raise NotImplementedError(
                "Remote GPU dispatch isn't wired up yet. Use compute_target='local' for now."
            )
        else:
            raise ValueError(f"Unknown compute_target: {compute_target}")

        with self._lock:
            self._jobs[job_id] = job

        return job_id

    def _refresh_status(self, job: TrainingJob):
        if job.status == "running" and job.process is not None:
            returncode = job.process.poll()
            if returncode is not None:
                job.returncode = returncode
                job.finished_at = time.time()
                job.status = "completed" if returncode == 0 else "failed"
                if job.status == "completed":
                    job.registered_version = self._find_registered_version(job)

    def _find_registered_version(self, job: TrainingJob) -> Optional[str]:
        """
        The training script is expected to tag its registered model version
        with training_job_id=<job_id> (see train_dummy_model.py). We look
        that up here rather than parsing stdout, so it's robust either way.
        """
        try:
            versions = self._client.search_model_versions(f"name='{job.model_name}'")
            for v in versions:
                if (v.tags or {}).get("training_job_id") == job.job_id:
                    return v.version
        except Exception:
            pass
        return None

    def get_status(self, job_id: str, log_tail_lines: int = 40) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No such job: {job_id}")

        self._refresh_status(job)

        log_tail = ""
        if os.path.isfile(job.log_path):
            with open(job.log_path) as f:
                lines = f.readlines()
                log_tail = "".join(lines[-log_tail_lines:])

        return {
            "job_id": job.job_id,
            "status": job.status,
            "model_name": job.model_name,
            "compute_target": job.compute_target,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "returncode": job.returncode,
            "registered_version": job.registered_version,
            "log_tail": log_tail,
        }

    def list_jobs(self) -> list:
        with self._lock:
            job_ids = list(self._jobs.keys())
        return [self.get_status(jid) for jid in job_ids]


# Singleton — imported by main.py
training_manager = TrainingManager()