"""Experiment manager — manages per-experiment YAML files."""

import os
import re
import time
from typing import Dict, List, Optional

import yaml



class ExperimentManager:
    """Manages experiment records under a session-specific directory.

    Schema:
        Experiment (top-level):
            name, purpose, hypothesis, status, created, attempts[],
            root_cause, learnings[], finalized_at

        Attempt:
            timestamp, change, hardware{gpus, gpu_type, driver?, cuda?},
            config{model, tp, dp, pp?, global_batch_size, seq_length,
                   train_iters, precision, ...},
            output_dir, result
    """

    _CONFIG_REQUIRED_KEYS = ("model", "tp", "dp")

    def __init__(self, experiments_dir: str):
        self._dir = experiments_dir

    def _path(self, name: str) -> str:
        safe = name.replace("/", "_").replace(" ", "_")
        return os.path.join(self._dir, f"{safe}.yaml")

    def create(self, name: str, purpose: str, hypothesis: str = "") -> str:
        if os.path.isfile(self._path(name)):
            return f"ERROR: Experiment '{name}' already exists."
        os.makedirs(self._dir, exist_ok=True)
        exp = {
            "name": name,
            "purpose": purpose,
            "hypothesis": hypothesis,
            "status": "running",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attempts": [],
            "root_cause": "",
            "learnings": [],
            "finalized_at": "",
        }
        self._save(name, exp)
        return f"Experiment '{name}' created."

    def read(self, name: str) -> Optional[Dict]:
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or None

    def _save(self, name: str, exp: Dict):
        with open(self._path(name), "w", encoding="utf-8") as f:
            yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def add_attempt(self, name: str, change: str, hardware: Dict = None,
                    config: Dict = None, output_dir: str = "") -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        config = config or {}
        warnings = self._validate_attempt_config(config)
        attempt = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "change": change,
            "hardware": hardware or {},
            "config": config,
            "output_dir": output_dir,
            "result": "(pending)",
        }
        exp.setdefault("attempts", []).append(attempt)
        exp["status"] = "running"
        self._save(name, exp)
        msg = f"Attempt #{len(exp['attempts'])} added to '{name}'."
        if warnings:
            msg += f"\nWARNING: {warnings}"
        return msg

    def _validate_attempt_config(self, config: Dict) -> str:
        """Warn if attempt config is missing key training parameters."""
        if not config:
            return "config is empty — should contain model, tp, dp, global_batch_size, etc."
        missing = [k for k in self._CONFIG_REQUIRED_KEYS if k not in config]
        if missing:
            return f"config missing recommended fields: {', '.join(missing)}"
        # Reject non-training keys mixed into config
        non_config_keys = {"reason", "fix", "note", "description", "change"}
        bad_keys = non_config_keys & set(config.keys())
        if bad_keys:
            return (f"config contains non-config fields: {', '.join(bad_keys)}. "
                    "Use the 'change' parameter for descriptions, keep config for training parameters only.")
        return ""

    def update_last_attempt(self, name: str, result: str) -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        attempts = exp.get("attempts", [])
        if not attempts:
            return f"ERROR: No attempts in '{name}'."
        attempts[-1]["result"] = result
        self._save(name, exp)
        return f"Updated last attempt result for '{name}'."

    def finalize(self, name: str, status: str, root_cause: str = "",
                 learnings: List[str] = None) -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        exp["status"] = status
        exp["root_cause"] = root_cause
        exp["learnings"] = learnings or []
        exp["finalized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save(name, exp)
        return f"Experiment '{name}' finalized as '{status}'."

    def list(self) -> List[Dict]:
        if not os.path.isdir(self._dir):
            return []
        results = []
        for f in sorted(os.listdir(self._dir)):
            if not f.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    exp = yaml.safe_load(fh) or {}
                results.append({
                    "name": exp.get("name", f.replace(".yaml", "")),
                    "status": exp.get("status", "unknown"),
                    "attempts": len(exp.get("attempts", [])),
                    "created": exp.get("created", ""),
                })
            except Exception:
                pass
        return results

    def get_current_experiment(self) -> str:
        """Return the name of the most recent running experiment, or ''."""
        for exp_info in reversed(self.list()):
            if exp_info.get("status") == "running":
                return exp_info["name"]
        return ""

    def compare(self, name1: str, name2: str) -> dict:
        """Compare the latest attempts of two experiments.

        Returns {diffs: [...], summary: str, regression: bool}.
        Diffs are human-readable per-key comparisons. Regression flag
        is set when throughput or loss regressed significantly.
        """
        exp1 = self.read(name1)
        exp2 = self.read(name2)
        if not exp1 or not exp2:
            missing = name1 if not exp1 else name2
            return {"diffs": [], "summary": f"Experiment '{missing}' not found.", "regression": False}

        a1 = exp1.get("attempts", [])
        a2 = exp2.get("attempts", [])
        if not a1 or not a2:
            return {"diffs": [], "summary": "One or both experiments have no attempts to compare.", "regression": False}

        last1 = a1[-1]
        last2 = a2[-1]
        c1 = last1.get("config", {})
        c2 = last2.get("config", {})

        diffs = []
        all_keys = set(c1.keys()) | set(c2.keys())
        for k in sorted(all_keys):
            v1 = c1.get(k, "(not set)")
            v2 = c2.get(k, "(not set)")
            if v1 != v2:
                diffs.append({"key": k, "from": v1, "to": v2})

        # Check for regression indicators in results
        regression = False
        regression_reasons = []
        r1 = last1.get("result", "")
        r2 = last2.get("result", "")

        # Loss regression: new attempt has higher loss
        loss_vals = []
        for label, result in [(name1, r1), (name2, r2)]:
            loss_match = re.search(r'loss[:=\s]+([\d.]+)', str(result))
            if loss_match:
                loss_vals.append((label, float(loss_match.group(1))))
        if len(loss_vals) == 2:
            if loss_vals[0][1] < loss_vals[1][1] * 0.95:
                regression = True
                regression_reasons.append(
                    f"Loss regression: {loss_vals[0][0]}={loss_vals[0][1]:.4f} → "
                    f"{loss_vals[1][0]}={loss_vals[1][1]:.4f}"
                )

        # Throughput regression
        tput_vals = []
        for label, result in [(name1, r1), (name2, r2)]:
            for pat in [r'throughput[:=\s]+([\d.]+)', r'tokens/s[:=\s]+([\d.]+)',
                        r'samples/s[:=\s]+([\d.]+)']:
                m = re.search(pat, str(result), re.IGNORECASE)
                if m:
                    tput_vals.append((label, float(m.group(1))))
                    break
        if len(tput_vals) == 2:
            if tput_vals[0][1] * 0.95 > tput_vals[1][1]:
                regression = True
                regression_reasons.append(
                    f"Throughput regression: {tput_vals[0][0]}={tput_vals[0][1]:.1f} → "
                    f"{tput_vals[1][0]}={tput_vals[1][1]:.1f}"
                )

        summary_parts = []
        if diffs:
            diff_strs = [f"  {d['key']}: {d['from']} → {d['to']}" for d in diffs]
            summary_parts.append(f"Config diffs ({len(diffs)}):\n" + "\n".join(diff_strs))
        else:
            summary_parts.append("Configs are identical — no parameter-level diffs.")

        if regression_reasons:
            summary_parts.append("REGRESSION: " + "; ".join(regression_reasons))
        else:
            summary_parts.append("No significant regression detected.")

        return {
            "diffs": diffs,
            "summary": "\n".join(summary_parts),
            "regression": regression,
        }

    def diff_last_attempts(self, name: str) -> dict:
        """Compare the last two attempts within the same experiment.

        Returns {diffs: [...], summary: str}. Useful for understanding
        what changed between consecutive runs of the same experiment.
        """
        exp = self.read(name)
        if not exp:
            return {"diffs": [], "summary": f"Experiment '{name}' not found."}

        attempts = exp.get("attempts", [])
        if len(attempts) < 2:
            return {"diffs": [], "summary": "Need at least 2 attempts to diff."}

        prev = attempts[-2]
        curr = attempts[-1]
        pc = prev.get("config", {})
        cc = curr.get("config", {})

        diffs = []
        all_keys = set(pc.keys()) | set(cc.keys())
        for k in sorted(all_keys):
            v1 = pc.get(k, "(not set)")
            v2 = cc.get(k, "(not set)")
            if v1 != v2:
                diffs.append({"key": k, "from": v1, "to": v2})

        changes = prev.get("change", "") + " → " + curr.get("change", "")

        if diffs:
            diff_strs = [f"  {d['key']}: {d['from']} → {d['to']}" for d in diffs]
            summary = f"Changes ({changes}):\n" + "\n".join(diff_strs)
        else:
            summary = f"Config unchanged between attempts. Change description: {changes}"

        p_result = prev.get("result", "(pending)")
        c_result = curr.get("result", "(pending)")
        summary += f"\n\nPrevious result: {p_result[:200]}"
        summary += f"\nCurrent result: {c_result[:200]}"

        return {"diffs": diffs, "summary": summary}
