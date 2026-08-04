"""
manual_remote.py

"Bring your own server" compute mode — for when YOU launch and manage the
EC2 (or any) instance yourself, and just hand this code an IP + credentials
to run a training job on it.

Key differences from remote_gpu.py's auto-provisioned mode:
  - No instance launch, no instance stop. This code NEVER touches the
    instance's lifecycle — that's entirely your responsibility. A job
    finishing or failing here does not shut anything down.
  - Much higher tolerance for transient SSH issues. The auto-provisioned
    mode treats any connection hiccup as "stop billing now, fail safe" —
    that logic doesn't apply here since there's no billing meter tied to
    job success/failure. Instead, this reconnects and retries before ever
    giving up, appropriate for a job that might run for hours.
  - Dataset transfer assumes a real image dataset, not a small CSV — data_path
    can be a local DIRECTORY, which gets zipped, transferred as one file,
    then unzipped remotely (much faster than SCP-ing thousands of small
    files individually).

Auth: username/password (not a key file). Two things worth knowing:
  1. Many Ubuntu AMIs disable password auth by default — you may need to
     manually set `PasswordAuthentication yes` in /etc/ssh/sshd_config and
     restart sshd on your instance first.
  2. Password auth is less secure than key-based. Reasonable for a
     short-lived personal dev box, not recommended for anything longer-lived
     or shared.
"""

import os
import time
import socket
import zipfile
import tempfile

import paramiko
from scp import SCPClient

SSH_CMD_TIMEOUT = 20
RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_SECONDS = 10

REMOTE_WORKDIR_TEMPLATE = "/home/{username}/training_job"


class ManualRemoteJob:
    def __init__(self, host: str, username: str, password: str = None,
                 key_path: str = None, port: int = 22):
        if not password and not key_path:
            raise ValueError("Provide either a password or a key_path — at least one is required.")
        self.host = host
        self.username = username
        self.password = password
        self.key_path = key_path
        self.port = port
        self._ssh = None
        # Workdir depends on the actual connecting user — Lightning AI Studios,
        # for instance, won't necessarily use "ubuntu" the way a fresh EC2
        # Ubuntu AMI does.
        self.remote_workdir = REMOTE_WORKDIR_TEMPLATE.format(username=username)

    def _do_connect(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.key_path:
            ssh.connect(self.host, port=self.port, username=self.username,
                        key_filename=self.key_path, timeout=10)
        else:
            ssh.connect(self.host, port=self.port, username=self.username,
                        password=self.password, timeout=10)
        return ssh

    def connect(self, timeout=60):
        """Initial connection, with retries in case the instance is still booting."""
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                self._ssh = self._do_connect()
                return
            except Exception as e:
                last_err = e
                time.sleep(5)
        raise TimeoutError(f"Could not SSH to {self.host}: {last_err}")

    def _reconnect(self) -> bool:
        """
        Used mid-job if a connection drops. Unlike the auto-provisioned
        mode, we retry several times with backoff before giving up — a
        multi-hour job is worth a bit of patience on transient network blips.
        """
        for attempt in range(RECONNECT_ATTEMPTS):
            try:
                self._ssh = self._do_connect()
                return True
            except Exception:
                time.sleep(RECONNECT_BACKOFF_SECONDS * (attempt + 1))
        return False

    def _exec(self, cmd: str, timeout=SSH_CMD_TIMEOUT):
        """
        A single exec_command wrapper that transparently reconnects on
        failure, instead of the caller having to handle that everywhere.
        Raises only if reconnect attempts are fully exhausted.
        """
        try:
            return self._ssh.exec_command(cmd, timeout=timeout)
        except (socket.timeout, paramiko.SSHException, EOFError, OSError):
            if self._reconnect():
                return self._ssh.exec_command(cmd, timeout=timeout)
            raise ConnectionError(f"Lost connection to {self.host} and could not reconnect.")

    def upload_dataset_and_script(self, script_path: str, data_path: str, requirements_path: str = None):
        """
        If data_path is a directory, zip it locally, upload the single zip,
        unzip remotely — much faster than many small SCP transfers.
        If data_path is a single file (e.g. a CSV, for backward compat),
        upload it directly like before.
        """
        stdin, stdout, stderr = self._exec(f"mkdir -p {self.remote_workdir}")
        stdout.channel.recv_exit_status()

        with SCPClient(self._ssh.get_transport()) as scp:
            scp.put(script_path, f"{self.remote_workdir}/train_script.py")

            if requirements_path and os.path.isfile(requirements_path):
                scp.put(requirements_path, f"{self.remote_workdir}/requirements.txt")

            if os.path.isdir(data_path):
                zip_path = self._zip_directory(data_path)
                scp.put(zip_path, f"{self.remote_workdir}/dataset.zip")
                os.remove(zip_path)  # clean up the local temp zip

                stdin, stdout, stderr = self._exec(
                    f"cd {self.remote_workdir} && unzip -q -o dataset.zip -d dataset && rm dataset.zip",
                    timeout=120,  # unzipping a real dataset can take a bit longer than a normal command
                )
                stdout.channel.recv_exit_status()
            elif os.path.isfile(data_path):
                scp.put(data_path, f"{self.remote_workdir}/data.csv")

        stdin, stdout, stderr = self._exec(
            f"echo '--- Files present in {self.remote_workdir} after upload: ---' >> {self.remote_workdir}/job.log && "
            f"ls -la {self.remote_workdir} >> {self.remote_workdir}/job.log 2>&1"
        )
        stdout.channel.recv_exit_status()

    def _zip_directory(self, directory: str) -> str:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(tmp_fd)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(directory):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    zf.write(full_path, rel_path)
        return tmp_path

    def ensure_docker(self):
        _, stdout, _ = self._exec("which docker")
        has_docker = stdout.read().decode().strip() != ""
        if not has_docker:
            stdin, stdout, stderr = self._exec(
                f"echo '--- Installing Docker (one-time) ---' >> {self.remote_workdir}/job.log && "
                f"sudo apt-get update -qq >> {self.remote_workdir}/job.log 2>&1 && "
                f"sudo apt-get install -y -qq docker.io >> {self.remote_workdir}/job.log 2>&1 && "
                f"sudo usermod -aG docker {self.username}",
                timeout=180,
            )
            stdout.channel.recv_exit_status()

    def start_training(self, model_name: str, job_id: str, tracking_uri: str, data_path: str,
                         use_gpu: bool = True):
        """
        Runs training inside a Docker container, same pattern as the
        auto-provisioned mode. use_gpu=True adds --gpus all, assuming
        nvidia-container-toolkit is present on your instance (standard on
        AWS Deep Learning AMIs) — set False if your instance has no GPU
        runtime configured.
        """
        remote_data_path = (
            f"/workspace/dataset" if os.path.isdir(data_path)
            else (f"/workspace/data.csv" if os.path.isfile(data_path) else data_path)
        )

        gpu_flag = "--gpus all" if use_gpu else ""

        inner_cmd = (
            "cd /workspace && "
            "(test -f requirements.txt && pip install --quiet -r requirements.txt "
            "|| (echo '--- no requirements.txt, falling back to bare mlflow ---' && pip install --quiet mlflow)) && "
            f"python3 train_script.py --data-path {remote_data_path} --model-name {model_name} "
            f"--job-id {job_id} --tracking-uri {tracking_uri}"
        )

        docker_cmd = (
            f"docker run --rm {gpu_flag} -v {self.remote_workdir}:/workspace -w /workspace "
            f"pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime bash -c \"{inner_cmd}\""
        )

        cmd = f"cd {self.remote_workdir} && nohup {docker_cmd} >> job.log 2>&1 & echo $! > job.pid"
        self._exec(cmd)

    def poll(self) -> dict:
        """
        Same shape as RemoteGPUJob.poll(), but never signals "stop the
        instance" on connection loss — there's nothing this code is allowed
        to stop. A connection failure here just means "still unknown,
        retrying," reported clearly, rather than a hard failure — since a
        long job is worth being patient about network blips on.
        """
        try:
            _, stdout, _ = self._exec(f"cat {self.remote_workdir}/job.pid 2>/dev/null")
            pid = stdout.read().decode().strip()

            still_running = False
            if pid:
                _, stdout, _ = self._exec(f"kill -0 {pid} 2>/dev/null && echo alive || echo dead")
                still_running = stdout.read().decode().strip() == "alive"

            _, stdout, _ = self._exec(f"tail -n 60 {self.remote_workdir}/job.log 2>/dev/null")
            log_tail = stdout.read().decode()

        except ConnectionError as e:
            return {
                "running": True,  # NOT marking failed — just temporarily unknown
                "returncode": None,
                "log_tail": f"(Temporarily unreachable: {e}. Will keep retrying on next poll — "
                             f"the job itself is untouched by this, it's still on your instance.)",
            }

        if not still_running and pid:
            returncode = 0 if "DONE" in log_tail else 1
            return {"running": False, "returncode": returncode, "log_tail": log_tail}

        return {"running": True, "returncode": None, "log_tail": log_tail}

    def close(self):
        """Just closes our SSH session — does NOT touch the instance itself."""
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass