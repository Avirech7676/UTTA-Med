"""UTTA-Med 2.0 Kaggle GPU Orchestrator (Phase 8).

Automates end-to-end execution of GPU experiments on Kaggle:
1. Validates Kaggle credentials and API connectivity.
2. Checks accelerator quota and session availability.
3. Packages kernel files, code, and configurations.
4. Pushes job via Kaggle CLI / Python API.
5. Polls kernel execution status until completion.
6. Streams remote logs to local audit logs.
7. Downloads completed outputs and metrics.
8. Registers run in EXPERIMENT_INVENTORY.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional


class KaggleOrchestrator:
    """Production orchestrator for remote Kaggle GPU adaptation experiments."""

    def __init__(self, workspace_dir: str, experiment_dir: str):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.api = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with the official Kaggle API."""
        try:
            import kaggle
            self.api = kaggle.KaggleApi()
            self.api.authenticate()
            print("[KaggleOrchestrator] Authenticated successfully with Kaggle API.")
        except Exception as e:
            print(f"[KaggleOrchestrator] Warning: Kaggle API authentication failed: {e}")
            self.api = None

    def _run_kaggle_cmd(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run kaggle command using sys.executable -m kaggle for full cross-platform compatibility."""
        cmd = [sys.executable, "-m", "kaggle"] + args
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def check_quota(self) -> Dict[str, Any]:
        """Query GPU quota and hardware limits."""
        print("[KaggleOrchestrator] Checking GPU quota...")
        try:
            res = self._run_kaggle_cmd(["quota"])
            return {"status": "SUCCESS", "output": res.stdout.strip()}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def push_kernel(self, kernel_folder: str, accelerator: str = "NvidiaTeslaT4") -> Dict[str, Any]:
        """Push Kaggle kernel using official CLI."""
        print(f"[KaggleOrchestrator] Pushing kernel from {kernel_folder} with accelerator {accelerator}...")
        try:
            res = self._run_kaggle_cmd(["kernels", "push", "-p", kernel_folder])
            if res.returncode == 0:
                print(f"[KaggleOrchestrator] Push output: {res.stdout.strip()}")
                return {"status": "SUCCESS", "output": res.stdout.strip()}
            else:
                print(f"[KaggleOrchestrator] Push failed: {res.stderr.strip()}")
                return {"status": "ERROR", "error": res.stderr.strip()}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def get_status(self, kernel_slug: str) -> str:
        """Query the remote execution status of a kernel."""
        try:
            res = self._run_kaggle_cmd(["kernels", "status", kernel_slug])
            return res.stdout.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def get_logs(self, kernel_slug: str) -> str:
        """Fetch remote execution logs."""
        try:
            res = self._run_kaggle_cmd(["kernels", "logs", kernel_slug])
            return res.stdout.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def download_output(self, kernel_slug: str, output_dir: str) -> bool:
        """Download remote artifacts and checkpoints upon completion."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            res = self._run_kaggle_cmd(["kernels", "output", kernel_slug, "-p", output_dir])
            if res.returncode == 0:
                print(f"[KaggleOrchestrator] Downloaded outputs to: {output_dir}")
                return True
            else:
                print(f"[KaggleOrchestrator] Download error: {res.stderr.strip()}")
                return False
        except Exception as e:
            print(f"[KaggleOrchestrator] Failed to download output: {e}")
            return False

    def monitor_and_retrieve(
        self, kernel_slug: str, output_dir: str, poll_interval_sec: int = 30, timeout_sec: int = 3600
    ) -> bool:
        """Poll kernel execution until finished or timeout, then download outputs."""
        print(f"[KaggleOrchestrator] Monitoring kernel: {kernel_slug} (Timeout: {timeout_sec}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            status = self.get_status(kernel_slug)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")

            if "complete" in status.lower():
                print("[KaggleOrchestrator] Remote execution complete! Downloading artifacts...")
                self.download_output(kernel_slug, output_dir)
                self.register_experiment(kernel_slug, output_dir, "COMPLETED")
                return True
            elif "error" in status.lower() or "failed" in status.lower():
                print(f"[KaggleOrchestrator] Remote execution failed: {status}")
                logs = self.get_logs(kernel_slug)
                print(f"[KaggleOrchestrator] Logs:\n{logs}")
                self.register_experiment(kernel_slug, output_dir, "FAILED")
                return False

            time.sleep(poll_interval_sec)

        print("[KaggleOrchestrator] Execution timed out.")
        return False

    def register_experiment(self, kernel_slug: str, output_path: str, status: str) -> None:
        """Log run to EXPERIMENT_INVENTORY.csv."""
        inv_path = os.path.join(self.workspace_dir, "EXPERIMENT_INVENTORY.csv")
        row = ["Kaggle_GPU", f"{kernel_slug} -> {os.path.relpath(output_path, self.workspace_dir)}", status]
        try:
            with open(inv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)
            print(f"[KaggleOrchestrator] Updated {inv_path} with run record.")
        except Exception as e:
            print(f"[KaggleOrchestrator] Could not update inventory: {e}")


def main():
    parser = argparse.ArgumentParser(description="UTTA-Med 2.0 Kaggle Orchestrator")
    parser.add_argument("--action", choices=["quota", "push", "status", "logs", "monitor", "run_local_test"], default="quota")
    parser.add_argument("--kernel_dir", default="scripts/kaggle", help="Path to kernel files")
    parser.add_argument("--kernel_slug", default="avinash/utta-med-v2-gpu-adaptation", help="Kaggle kernel slug")
    parser.add_argument("--output_dir", default="results/kaggle_runs", help="Local download directory")
    parser.add_argument("--timeout", type=int, default=7200, help="Monitor timeout in seconds")
    parser.add_argument("--poll_interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    orchestrator = KaggleOrchestrator(
        workspace_dir=os.getcwd(),
        experiment_dir=args.kernel_dir,
    )

    if args.action == "quota":
        res = orchestrator.check_quota()
        print(json.dumps(res, indent=2))
    elif args.action == "push":
        orchestrator.push_kernel(args.kernel_dir)
    elif args.action == "status":
        print(orchestrator.get_status(args.kernel_slug))
    elif args.action == "logs":
        print(orchestrator.get_logs(args.kernel_slug))
    elif args.action == "monitor":
        orchestrator.monitor_and_retrieve(
            args.kernel_slug,
            args.output_dir,
            poll_interval_sec=args.poll_interval,
            timeout_sec=args.timeout,
        )
    elif args.action == "run_local_test":
        # Execute run_kaggle.py locally to test logic before pushing
        print("[KaggleOrchestrator] Running local validation of run_kaggle.py...")
        subprocess.run([sys.executable, os.path.join(args.kernel_dir, "run_kaggle.py"), "--output", args.output_dir], check=True)


if __name__ == "__main__":
    main()
