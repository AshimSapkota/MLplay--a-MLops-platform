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
from remote_gpu import RemoteGPUJob
from manual_remote import ManualRemoteJob

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
    remote_gpu_job: Optional[object] = None  # RemoteGPUJob instance, only for compute_target="remote_gpu"
    manual_remote_job: Optional[object] = None  # ManualRemoteJob instance, only for compute_target="manual_remote"


class TrainingManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    def submit_job(self, script_path: str, data_path: str, model_name: str,
                    compute_target: str = "local", extra_args: dict | None = None,
                    requirements_path: str | None = None,
                    manual_host: str | None = None, manual_username: str | None = None,
                    manual_password: str | None = None, manual_use_gpu: bool = True,
                    manual_key_path: str | None = None, manual_port: int = 22) -> str:
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Training script not found: {script_path}")

        # Auto-detect a requirements.txt sitting next to the script if the
        # caller didn't explicitly provide one — a common convention, and
        # saves having to specify it every time for scripts that have one.
        if requirements_path is None:
            candidate = os.path.join(os.path.dirname(os.path.abspath(script_path)), "requirements.txt")
            if os.path.isfile(candidate):
                requirements_path = candidate
        else:
            # An explicit path WAS given — validate it right now, locally,
            # before spending time/money launching a remote instance for a
            # job that's guaranteed to fail. This also removes any ambiguity
            # from SSH-side timing/ordering when checking this deep inside
            # the remote upload step.
            if not os.path.isfile(requirements_path):
                raise FileNotFoundError(
                    f"requirements_path was set to '{requirements_path}' but that file "
                    f"doesn't exist on this machine (the one running the dashboard backend, "
                    f"not the remote instance). Double check the path and permissions."
                )

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
            # Local jobs run inside your existing backend venv, which is
            # assumed to already have whatever the script needs — no
            # per-job install step here (that's specifically a remote_gpu
            # concern, since a fresh EC2 instance starts with nothing).
            log_file = open(log_path, "w")
            job.process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            job.status = "running"
            job.started_at = time.time()
        elif compute_target == "remote_gpu":
            log_file = open(log_path, "w")
            log_file.write("Launching remote GPU instance — this takes a couple minutes "
                            "(instance boot + SSH readiness), not a bug if it looks idle.\n")
            if requirements_path:
                log_file.write(f"Using requirements file: {requirements_path}\n")
            else:
                log_file.write("No requirements.txt found next to the script — "
                                "will fall back to a bare-minimum install (mlflow only).\n")
            log_file.flush()

            remote_job = RemoteGPUJob()
            try:
                remote_job.launch_and_run(script_path, data_path, model_name, job_id,
                                            requirements_path=requirements_path)
            except Exception as e:
                log_file.write(f"Remote launch failed: {e}\n")
                # CRITICAL: launch_and_run can fail AFTER the instance was
                # already created (e.g. upload or docker step failed after
                # SSH succeeded) — without this, that instance keeps running
                # and billing indefinitely with nothing tracking it anymore.
                # This is exactly what caused multiple instances to pile up.
                try:
                    remote_job.stop_instance()
                    log_file.write("Instance stopped after launch failure.\n")
                except Exception as stop_err:
                    log_file.write(
                        f"WARNING: could not confirm instance was stopped after failure: {stop_err}. "
                        f"Check the AWS console manually — instance_id={remote_job.instance_id}\n"
                    )
                log_file.close()
                job.status = "failed"
                job.returncode = 1
                with self._lock:
                    self._jobs[job_id] = job
                return job_id

            job.remote_gpu_job = remote_job
            job.status = "running"
            job.started_at = time.time()
            log_file.close()
        elif compute_target == "manual_remote":
            if not (manual_host and manual_username and (manual_password or manual_key_path)):
                raise ValueError(
                    "manual_remote requires manual_host, manual_username, and either "
                    "manual_password or manual_key_path."
                )
            log_file = open(log_path, "w")
            log_file.write(f"Connecting to {manual_host} — this is YOUR instance, not one this "
                            f"code launched or will stop.\n")
            log_file.flush()

            manual_job = ManualRemoteJob(manual_host, manual_username, password=manual_password,
                                           key_path=manual_key_path, port=manual_port)
            try:
                manual_job.connect()
                manual_job.upload_dataset_and_script(script_path, data_path, requirements_path)
                manual_job.ensure_docker()
                manual_job.start_training(model_name, job_id, MLFLOW_TRACKING_URI, data_path,
                                            use_gpu=manual_use_gpu)
            except Exception as e:
                log_file.write(f"Manual remote setup failed: {e}\n")
                log_file.close()
                manual_job.close()
                job.status = "failed"
                job.returncode = 1
                with self._lock:
                    self._jobs[job_id] = job
                return job_id

            job.manual_remote_job = manual_job
            job.status = "running"
            job.started_at = time.time()
            log_file.close()
        else:
            raise ValueError(f"Unknown compute_target: {compute_target}")

        with self._lock:
            self._jobs[job_id] = job

        return job_id

    def _refresh_status(self, job: TrainingJob):
        if job.status != "running":
            return

        if job.compute_target == "local" and job.process is not None:
            returncode = job.process.poll()
            if returncode is not None:
                job.returncode = returncode
                job.finished_at = time.time()
                job.status = "completed" if returncode == 0 else "failed"
                if job.status == "completed":
                    job.registered_version = self._find_registered_version(job)

        elif job.compute_target == "remote_gpu" and job.remote_gpu_job is not None:
            result = job.remote_gpu_job.poll()
            # Overwrite the local log file with the latest remote tail, so
            # get_status can read it the same way regardless of compute_target.
            with open(job.log_path, "w") as f:
                f.write(result["log_tail"])

            if not result["running"]:
                job.returncode = result["returncode"]
                job.finished_at = time.time()
                job.status = "completed" if result["returncode"] == 0 else "failed"
                if job.status == "completed":
                    job.registered_version = self._find_registered_version(job)
                # COST CONTROL: stop the instance the moment the job is done,
                # success or failure. This is the single most important line
                # in this file for anyone paying for the instance.
                job.remote_gpu_job.stop_instance()

        elif job.compute_target == "manual_remote" and job.manual_remote_job is not None:
            result = job.manual_remote_job.poll()
            with open(job.log_path, "w") as f:
                f.write(result["log_tail"])

            if not result["running"]:
                job.returncode = result["returncode"]
                job.finished_at = time.time()
                job.status = "completed" if result["returncode"] == 0 else "failed"
                if job.status == "completed":
                    job.registered_version = self._find_registered_version(job)
                # NOT stopping anything here — this is your instance, not
                # ours to manage. Just close our SSH session cleanly.
                job.manual_remote_job.close()

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