"""
remote_gpu.py

Handles the "remote_gpu" compute target: launches a GPU EC2 instance on demand,
copies the training script + data to it over SSH, runs the script remotely,
lets training_manager.py poll its status, and stops the instance afterward.

REQUIRED one-time setup (see the walkthrough) before this will work:
  - AWS credentials configured (`aws configure`)
  - An SSH key pair created, .pem file saved locally
  - A security group allowing SSH from your IP
  - A GPU-ready AMI ID (Deep Learning AMI recommended)
  - Your MLflow server reachable from AWS (ngrok tunnel, same as the Kaggle setup)

Configure via environment variables (set these before starting the FastAPI backend):
  AWS_REGION            e.g. us-east-1
  GPU_AMI_ID            from the ssm get-parameters command in the setup steps
  GPU_INSTANCE_TYPE     e.g. g4dn.xlarge
  GPU_KEY_NAME          e.g. training-gpu-key   (matches the .pem file name, no extension)
  GPU_KEY_PATH          e.g. /home/you/.ssh/training-gpu-key.pem
  GPU_SECURITY_GROUP    e.g. training-gpu-sg (name — kept for reference/tags only)
  GPU_SECURITY_GROUP_ID e.g. sg-0a1b2c3d4e5f6g7h8 (REQUIRED — VPC launches need the ID, not the name)
  REMOTE_MLFLOW_URI     your ngrok tunnel URL, e.g. https://xxxx.ngrok-free.dev

COST WARNING: every job launches a real, billed EC2 instance. This module
stops (not terminates — cheaper to restart, still billed for storage) the
instance as soon as the job completes OR fails. If your backend process
itself crashes mid-job, the instance can be left running — check the AWS
console periodically, especially during development of this feature.
"""

import os
import time
import io
import socket

import boto3
import paramiko
from scp import SCPClient

SSH_CMD_TIMEOUT = 15  # seconds — every exec_command below uses this, so a
                       # flaky/dead connection can only ever block this long,
                       # never indefinitely (this is what caused the earlier hang).

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
GPU_AMI_ID = os.environ.get("GPU_AMI_ID")
GPU_INSTANCE_TYPE = os.environ.get("GPU_INSTANCE_TYPE", "g4dn.xlarge")
GPU_KEY_NAME = os.environ.get("GPU_KEY_NAME")
GPU_KEY_PATH = os.environ.get("GPU_KEY_PATH")
GPU_SECURITY_GROUP = os.environ.get("GPU_SECURITY_GROUP")
GPU_SECURITY_GROUP_ID = os.environ.get("GPU_SECURITY_GROUP_ID")
REMOTE_MLFLOW_URI = os.environ.get("REMOTE_MLFLOW_URI")

REMOTE_WORKDIR = "/home/ubuntu/training_job"


def _validate_config():
    missing = [k for k, v in {
        "GPU_AMI_ID": GPU_AMI_ID, "GPU_KEY_NAME": GPU_KEY_NAME,
        "GPU_KEY_PATH": GPU_KEY_PATH, "GPU_SECURITY_GROUP_ID": GPU_SECURITY_GROUP_ID,
        "REMOTE_MLFLOW_URI": REMOTE_MLFLOW_URI,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required remote GPU config env vars: {missing}. "
            f"See remote_gpu.py's module docstring for what to set."
        )


class RemoteGPUJob:
    """
    One instance of this per training job. Mirrors the same lifecycle as a
    local subprocess.Popen job (launch -> poll -> read logs -> done), just
    implemented over SSH against a freshly-launched EC2 instance instead.
    """

    def __init__(self):
        _validate_config()
        self.ec2 = boto3.client("ec2", region_name=AWS_REGION)
        self.instance_id = None
        self.public_ip = None
        self._ssh = None
        self._remote_pid = None
        self._finished = False
        self._returncode = None

    def launch_and_run(self, script_path: str, data_path: str, model_name: str, job_id: str,
                        requirements_path: str = None):
        """Launches the instance, waits for SSH, uploads files, and starts training inside a container."""
        self.instance_id = self._launch_instance()
        self._wait_for_running()
        self._wait_for_ssh()
        self._upload_files(script_path, data_path, requirements_path)
        self._ensure_docker()
        self._start_remote_training(script_path, data_path, model_name, job_id)

    def _launch_instance(self) -> str:
        resp = self.ec2.run_instances(
            ImageId=GPU_AMI_ID,
            InstanceType=GPU_INSTANCE_TYPE,
            KeyName=GPU_KEY_NAME,
            SecurityGroupIds=[GPU_SECURITY_GROUP_ID],
            MinCount=1, MaxCount=1,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "training-job-ondemand"}],
            }],
        )
        return resp["Instances"][0]["InstanceId"]

    def _wait_for_running(self):
        waiter = self.ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[self.instance_id])
        desc = self.ec2.describe_instances(InstanceIds=[self.instance_id])
        self.public_ip = desc["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    def _wait_for_ssh(self, timeout=180):
        """EC2 instances take a bit after 'running' before SSH actually accepts connections."""
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.public_ip, username="ubuntu", key_filename=GPU_KEY_PATH, timeout=10)
                self._ssh = ssh
                return
            except Exception as e:
                last_err = e
                time.sleep(5)
        raise TimeoutError(f"SSH never became reachable on {self.public_ip}: {last_err}")

    def _upload_files(self, script_path: str, data_path: str, requirements_path: str = None):
        # Block until mkdir actually finishes — exec_command() alone doesn't wait,
        # and starting the SCP transfer before the directory exists is a real
        # race condition that can silently drop the upload.
        stdin, stdout, stderr = self._ssh.exec_command(f"mkdir -p {REMOTE_WORKDIR}", timeout=SSH_CMD_TIMEOUT)
        stdout.channel.recv_exit_status()

        with SCPClient(self._ssh.get_transport()) as scp:
            scp.put(script_path, f"{REMOTE_WORKDIR}/train_script.py")
            if os.path.isfile(data_path):
                scp.put(data_path, f"{REMOTE_WORKDIR}/data.csv")
            if requirements_path and os.path.isfile(requirements_path):
                scp.put(requirements_path, f"{REMOTE_WORKDIR}/requirements.txt")
            elif requirements_path and not os.path.isfile(requirements_path):
                # This is a LOCAL path problem, not a remote one — worth
                # surfacing loudly rather than silently uploading nothing.
                self._ssh.exec_command(
                    f"echo '--- WARNING: requirements_path was set to {requirements_path} "
                    f"but that file does not exist on the machine running the dashboard. "
                    f"Nothing was uploaded for it. ---' >> {REMOTE_WORKDIR}/job.log"
                )

        # Explicit, visible proof of what actually landed remotely — no more
        # inferring this indirectly from downstream symptoms.
        stdin, stdout, stderr = self._ssh.exec_command(
            f"echo '--- Files present in {REMOTE_WORKDIR} after upload: ---' >> {REMOTE_WORKDIR}/job.log && "
            f"ls -la {REMOTE_WORKDIR} >> {REMOTE_WORKDIR}/job.log 2>&1"
        )
        stdout.channel.recv_exit_status()

    def _ensure_docker(self):
        """
        Deep Learning AMIs ship with Docker preinstalled — this is just a
        safety check + auto-install fallback for other AMIs (e.g. if you
        swap to a plain Ubuntu image for a non-GPU test).
        """
        _, stdout, _ = self._ssh.exec_command("which docker")
        has_docker = stdout.read().decode().strip() != ""

        if not has_docker:
            install_cmd = (
                f"echo '--- Docker not found, installing (one-time, adds ~30s) ---' >> {REMOTE_WORKDIR}/job.log && "
                f"sudo apt-get update -qq >> {REMOTE_WORKDIR}/job.log 2>&1 && "
                f"sudo apt-get install -y -qq docker.io >> {REMOTE_WORKDIR}/job.log 2>&1 && "
                f"sudo usermod -aG docker ubuntu"
            )
            stdin, stdout, stderr = self._ssh.exec_command(install_cmd)
            stdout.channel.recv_exit_status()

    def _start_remote_training(self, script_path, data_path, model_name, job_id):
        """
        Runs the training script inside a clean, isolated python:3.11-slim
        container instead of directly on the host. This avoids the class of
        bugs we hit installing straight onto the AMI's base environment
        (system package version conflicts, PATH warnings, partial state left
        over from a previous job on a reused instance, etc.) — every run
        starts from an identical, clean environment.

        Trade-off: no GPU passthrough in this version (no --gpus flag, and
        python:3.11-slim has no CUDA runtime) — fine for CPU-friendly demo
        scripts like the tiny GPT-2 finetune, NOT fine for anything that
        actually needs real GPU acceleration. That would need an official
        CUDA-enabled base image (e.g. pytorch/pytorch:*-cuda*-runtime) plus
        `--gpus all`, and the AMI's nvidia-container-toolkit wired through —
        a reasonable next step, not implemented here.
        """
        remote_data_path = "/workspace/data.csv" if os.path.isfile(data_path) else data_path

        inner_cmd = (
            "cd /workspace && "
            "(test -f requirements.txt && pip install --quiet -r requirements.txt "
            "|| (echo '--- no requirements.txt in container, falling back to bare mlflow ---' && pip install --quiet mlflow)) && "
            f"python3 train_script.py --data-path {remote_data_path} --model-name {model_name} "
            f"--job-id {job_id} --tracking-uri {REMOTE_MLFLOW_URI}"
        )

        docker_cmd = (
            f"docker run --rm -v {REMOTE_WORKDIR}:/workspace -w /workspace "
            f"python:3.11-slim bash -c \"{inner_cmd}\""
        )

        cmd = (
            f"cd {REMOTE_WORKDIR} && "
            f"nohup {docker_cmd} >> job.log 2>&1 & "
            f"echo $! > job.pid"
        )
        self._ssh.exec_command(cmd)

    def poll(self) -> dict:
        """
        Returns {'running': bool, 'returncode': int|None, 'log_tail': str}.

        IMPORTANT: any SSH failure here (timeout, dropped connection, host
        unreachable) is treated as "not running" with a failure returncode —
        NOT as "still running, try again later". This is a deliberate safety
        choice: training_manager._refresh_status() calls stop_instance()
        whenever poll() reports not-running, so a lost connection now
        reliably triggers an instance stop instead of leaving a possibly-dead
        job's instance billing forever with nothing left to check on it.
        A previous version had no timeout here at all, which let a single
        hung SSH call block status checks indefinitely.
        """
        if self._ssh is None:
            return {"running": True, "returncode": None, "log_tail": "(waiting for SSH...)"}

        try:
            _, stdout, _ = self._ssh.exec_command(
                f"cat {REMOTE_WORKDIR}/job.pid 2>/dev/null", timeout=SSH_CMD_TIMEOUT
            )
            pid = stdout.read().decode().strip()

            still_running = False
            if pid:
                _, stdout, _ = self._ssh.exec_command(
                    f"kill -0 {pid} 2>/dev/null && echo alive || echo dead", timeout=SSH_CMD_TIMEOUT
                )
                still_running = stdout.read().decode().strip() == "alive"

            _, stdout, _ = self._ssh.exec_command(
                f"tail -n 40 {REMOTE_WORKDIR}/job.log 2>/dev/null", timeout=SSH_CMD_TIMEOUT
            )
            log_tail = stdout.read().decode()

        except (socket.timeout, paramiko.SSHException, EOFError, OSError) as e:
            return {
                "running": False,
                "returncode": 1,
                "log_tail": f"SSH connection lost/unreachable ({e}). "
                             f"Marking job as failed and stopping the instance for safety — "
                             f"check the AWS console to confirm.",
            }

        if not still_running and pid:
            returncode = 0 if "DONE" in log_tail else 1
            return {"running": False, "returncode": returncode, "log_tail": log_tail}

        return {"running": True, "returncode": None, "log_tail": log_tail}

    def stop_instance(self):
        """
        Stops (not terminates) the instance via the AWS API directly — this
        works regardless of SSH state, which matters: if SSH is what's
        broken, this is still the reliable way to actually stop billing.
        """
        if self.instance_id:
            self.ec2.stop_instances(InstanceIds=[self.instance_id])
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass  # don't let a failure closing SSH prevent the stop above from having happened