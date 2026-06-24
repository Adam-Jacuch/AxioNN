# src/axionn/training/logging.py

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class LocalRunLogger:
    """
    A lightweight, structured, and extremely fast local run logger for training.
    Automatically logs hyperparameters config, metrics (JSON Lines), and step throughput.
    """
    def __init__(
        self,
        run_name: str,
        log_dir: str = "runs",
        config: Optional[Dict[str, Any]] = None,
        console_verbose: bool = True
    ):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.run_dir = self.log_dir / f"{run_name}_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"
        
        self.console_verbose = console_verbose
        self.start_time = time.time()
        self.last_step_time = time.time()

        # Save config if provided
        if config is not None:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            if self.console_verbose:
                print(f"[LocalRunLogger] Initialized run '{run_name}'. Config saved to {self.config_file}")
        elif self.console_verbose:
            print(f"[LocalRunLogger] Initialized run '{run_name}'. Logging to {self.log_file}")

    def log(self, step: int, metrics: Dict[str, Any], tokens_processed: Optional[int] = None):
        """
        Logs metrics for a specific step. Calculates step duration and optional throughput.
        """
        current_time = time.time()
        step_duration = current_time - self.last_step_time
        elapsed = current_time - self.start_time
        self.last_step_time = current_time

        # Calculate throughput (tokens/sec) if token count is provided
        throughput_metrics = {}
        if tokens_processed is not None and step_duration > 0:
            throughput_metrics["tokens_per_sec"] = float(tokens_processed / step_duration)

        # Safely convert jax arrays/Axiom Tensors to floats
        processed_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "unwrap"):
                v = v.unwrap()
            if hasattr(v, "item") and callable(v.item):
                processed_metrics[k] = float(v.item())
            elif isinstance(v, (int, float)):
                processed_metrics[k] = float(v)
            else:
                processed_metrics[k] = v

        log_entry = {
            "step": step,
            "timestamp": current_time,
            "elapsed_seconds": elapsed,
            "step_duration": step_duration,
            **throughput_metrics,
            **processed_metrics
        }

        # Append structured metrics line
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Beautiful console print
        if self.console_verbose:
            metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in processed_metrics.items())
            throughput_str = f" | {throughput_metrics['tokens_per_sec']:.1f} tok/s" if "tokens_per_sec" in throughput_metrics else ""
            print(f"Step {step:05d} | Time: {step_duration:.3f}s{throughput_str} | {metrics_str}")
