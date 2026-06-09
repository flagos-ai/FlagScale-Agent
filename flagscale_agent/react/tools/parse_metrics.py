"""Parse Megatron training metrics from log files."""

import math
import os
import re
import subprocess

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_FS
from flagscale_agent.react.tools.find_log import _parse_megatron_metrics, _health_check


class ParseTrainingMetricsTool(Tool):
    name = "parse_training_metrics"
    effects = EFFECT_READ_FS
    description = (
        "Parse training metrics from a Megatron log file or experiment directory. "
        "Returns structured metrics: iterations, losses, grad norms, throughput, anomalies. "
        "Automatically checks training health (random output, zero gradients, stalled loss)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "log_path": {
                "type": "string",
                "description": "Path to stdout.log file or experiment directory",
            },
            "vocab_size": {
                "type": "integer",
                "description": "Model vocab size for health check (e.g. 32000, 128256)",
            },
            "last_n": {
                "type": "integer",
                "description": "Parse only the last N lines. Default: 200",
            },
        },
        "required": ["log_path"],
    }

    def execute(self, **kwargs) -> str:
        log_path = kwargs["log_path"]
        vocab_size = kwargs.get("vocab_size", 0)
        last_n = kwargs.get("last_n", 200)

        if os.path.isdir(log_path):
            log_path = self._find_loss_log(log_path)
            if log_path.startswith("ERROR"):
                return log_path

        if not os.path.isfile(log_path):
            return f"ERROR: File not found: {log_path}"

        try:
            out = subprocess.run(
                ["tail", f"-{last_n}", log_path],
                capture_output=True, text=True, timeout=10,
            )
            content = out.stdout
        except Exception as e:
            return f"ERROR reading {log_path}: {e}"

        metrics = _parse_megatron_metrics(content)

        parts = [f"Log: {log_path}"]

        if metrics["last_iter"] is not None:
            parts.append(f"Latest iteration: {metrics['last_iter']}")
            parts.append(f"Total iterations seen: {len(metrics['iterations'])}")
            for k, v in metrics["last_loss"].items():
                parts.append(f"  {k}: {v}")
        else:
            parts.append("No training iterations found in the log.")

        if metrics["iterations"] and len(metrics["iterations"]) >= 2:
            first_iter_line = self._get_first_iter_metrics(content)
            if first_iter_line:
                parts.append(f"\nFirst iteration metrics: {first_iter_line}")

        health_warnings = _health_check(metrics, vocab_size)
        if health_warnings:
            parts.append("\n--- Health warnings ---")
            parts.extend(health_warnings)
        elif metrics["last_iter"] is not None:
            parts.append("\nHealth: OK (no anomalies detected)")

        return "\n".join(parts)

    def _find_loss_log(self, exp_dir: str) -> str:
        """Find the stdout.log with training metrics in an experiment directory."""
        details_dir = os.path.join(exp_dir, "logs", "details")
        if not os.path.isdir(details_dir):
            for root, dirs, files in os.walk(exp_dir):
                if "stdout.log" in files:
                    return os.path.join(root, "stdout.log")
            return f"ERROR: No stdout.log found in {exp_dir}"

        for root, dirs, files in os.walk(details_dir):
            dirs.sort(key=lambda d: int(re.search(r'\d+$', d).group()) if re.search(r'\d+$', d) else 0, reverse=True)
            if "stdout.log" in files:
                path = os.path.join(root, "stdout.log")
                try:
                    out = subprocess.run(
                        ["grep", "-c", "-i", "iteration", path],
                        capture_output=True, text=True, timeout=5,
                    )
                    if int(out.stdout.strip() or "0") > 0:
                        return path
                except Exception:
                    pass

        return f"ERROR: No log with training metrics found in {exp_dir}"

    def _get_first_iter_metrics(self, content: str) -> str:
        for line in content.splitlines():
            if re.search(r'iteration\s+\d+', line, re.IGNORECASE):
                return line.strip()
        return ""
