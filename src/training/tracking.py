"""Experiment tracking utilities for UTTA-Med.

Every run must record: experiment_id, dataset, split, model, method, seed,
hyperparameters, τ, MC passes, metrics, Git commit, checkpoint, runtime, GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_git_commit(short: bool = True) -> str:
    """Return current Git commit hash, or 'unknown' if not in a repo."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_experiment_id(
    dataset: str,
    model: str,
    method: str,
    seed: int,
    extra: str = "",
) -> str:
    """Standard naming: dataset_model_method_seed[_extra]."""
    parts = [dataset, model, method, f"s{seed}"]
    if extra:
        parts.append(extra)
    return "_".join(parts)


class ExperimentTracker:
    """Minimal file-based experiment tracker.

    Creates an experiment directory under results/run_logs/ or experiments/
    and writes config, metrics, and metadata.
    """

    def __init__(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        base_dir: str = "experiments",
    ):
        self.experiment_id = experiment_id
        self.config = config
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
        self.metadata = {
            "experiment_id": experiment_id,
            "git_commit": get_git_commit(),
            "start_time": datetime.utcnow().isoformat() + "Z",
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        }

        # Persist initial artifacts
        self._write_yaml(self.run_dir / "config.yaml", config)
        self._write_text(self.run_dir / "git_commit.txt", self.metadata["git_commit"])
        self._write_json(self.run_dir / "metadata.json", self.metadata)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is not None:
            self.metrics[f"step_{step}"] = metrics
        else:
            self.metrics.update(metrics)
        self._write_json(self.run_dir / "metrics.json", self.metrics)

    def log_command(self, command: str) -> None:
        self._write_text(self.run_dir / "command.txt", command)

    def finish(self, extra_metrics: Optional[Dict[str, Any]] = None) -> Path:
        if extra_metrics:
            self.metrics.update(extra_metrics)
        elapsed = time.time() - self.start_time
        self.metadata["end_time"] = datetime.utcnow().isoformat() + "Z"
        self.metadata["runtime_seconds"] = round(elapsed, 2)
        self._write_json(self.run_dir / "metrics.json", self.metrics)
        self._write_json(self.run_dir / "metadata.json", self.metadata)
        return self.run_dir

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _write_yaml(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        with open(path, "w") as f:
            f.write(text + "\n")
