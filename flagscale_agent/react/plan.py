"""Task plan — structured multi-step planning with persistence."""

import os
import re
import tempfile
import threading
import time
import uuid

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml




@dataclass
class StepCheckpoint:
    """Checkpoint recorded when a plan step completes."""

    step_id: int
    timestamp: float
    files_modified: List[str] = field(default_factory=list)
    memory_keys: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "files_modified": self.files_modified,
            "memory_keys": self.memory_keys,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepCheckpoint":
        return cls(
            step_id=data.get("step_id", 0),
            timestamp=data.get("timestamp", 0.0),
            files_modified=data.get("files_modified", []),
            memory_keys=data.get("memory_keys", []),
            summary=data.get("summary", ""),
        )

VALID_STEP_STATUSES = ("pending", "doing", "done", "skipped", "blocked")
VALID_PLAN_STATUSES = ("active", "paused", "completed", "abandoned")

# Auto-sync thresholds
_STEP_STALE_TURNS = 10  # A "doing" step with no update for this many turns is stale
_PLAN_REBUILD_FAILURES = 3  # Consecutive failures before suggesting rebuild

# Plan ID must only contain safe characters (prevent path traversal)
_PLAN_ID_RE = re.compile(r'^plan_[a-zA-Z0-9_-]+$')

STATUS_ICONS = {
    "pending": " ",
    "doing": "→",
    "done": "✓",
    "skipped": "-",
    "blocked": "!",
}


class TaskPlan:
    """Manages structured task plans with YAML persistence."""

    def __init__(self, plan_dir: str):
        self._dir = plan_dir
        self._lock = threading.RLock()
        self._checkpoints: Dict[str, Dict[int, StepCheckpoint]] = {}  # plan_id → {step_id → checkpoint}

    def _plan_path(self, plan_id: str) -> str:
        # Prevent path traversal
        if not _PLAN_ID_RE.match(plan_id):
            raise ValueError(f"Invalid plan_id: {plan_id} — must match {_PLAN_ID_RE.pattern}")
        return os.path.join(self._dir, f"{plan_id}.yaml")

    def _active_path(self) -> str:
        return os.path.join(self._dir, "active.yaml")

    def _save(self, plan: dict):
        os.makedirs(self._dir, exist_ok=True)
        plan["updated"] = time.time()
        path = self._plan_path(plan["id"])
        # Atomic write: write to tmp then rename
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, prefix=".tmp_plan_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(plan, f, allow_unicode=True, default_flow_style=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        if plan["status"] == "active":
            active_path = self._active_path()
            fd2, tmp_active = tempfile.mkstemp(dir=self._dir, prefix=".tmp_active_", suffix=".yaml")
            try:
                with os.fdopen(fd2, "w", encoding="utf-8") as f:
                    yaml.dump({"active_id": plan["id"]}, f)
                os.replace(tmp_active, active_path)
            except Exception:
                try:
                    os.unlink(tmp_active)
                except OSError:
                    pass
                raise

    def _clear_active(self):
        self._set_active(None)

    def _set_active(self, plan_id: Optional[str]):
        """Set active plan id, or None to deactivate."""
        os.makedirs(self._dir, exist_ok=True)
        active_path = self._active_path()
        with open(active_path, "w", encoding="utf-8") as f:
            yaml.dump({"active_id": plan_id}, f)

    def create(self, title: str, steps: List[str], session_id: str = "") -> dict:
        with self._lock:
            # Pause any existing active plan (check both active.yaml and scan files)
            old = self.get_active()
            if not old:
                # active.yaml might be stale — scan for any plan with status=active
                old = self._find_active_plan_by_scan()
            if old:
                old["status"] = "paused"
                old["updated"] = time.time()
                self._save(old)
                self._set_active(None)

            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            step_list = []
            for i, desc in enumerate(steps, 1):
                step_list.append({
                    "id": i,
                    "title": desc,
                    "status": "pending",
                    "notes": "",
                    "experiments": [],
                    "depends_on": [i - 1] if i > 1 else [],
                })

            plan = {
                "id": plan_id,
                "title": title,
                "status": "active",
                "created": time.time(),
                "updated": time.time(),
                "session_id": session_id,
                "steps": step_list,
            }
            self._save(plan)
            return plan

    def get_active(self) -> Optional[dict]:
        with self._lock:
            active_path = self._active_path()
            if os.path.isfile(active_path):
                try:
                    with open(active_path, "r", encoding="utf-8") as f:
                        ref = yaml.safe_load(f)
                    active_id = ref.get("active_id")
                    if active_id:
                        plan = self._load(active_id)
                        if plan:
                            return plan
                except Exception:
                    pass
            # Fallback: scan for any plan file with status=active
            return self._find_active_plan_by_scan()

    def _find_active_plan_by_scan(self) -> Optional[dict]:
        """Scan plan files for any with status=active (fallback when active.yaml is stale)."""
        if not os.path.isdir(self._dir):
            return None
        for fname in os.listdir(self._dir):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            try:
                with open(os.path.join(self._dir, fname), "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                if plan and plan.get("status") == "active":
                    return plan
            except Exception:
                continue
        return None

    def _load(self, plan_id: str) -> Optional[dict]:
        path = self._plan_path(plan_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def update_step(self, step_id: int, status: str, notes: str = "") -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            if status not in VALID_STEP_STATUSES:
                raise ValueError(f"Invalid status: {status}")

            step = self._find_step(plan, step_id)

            # Plan Phase Gate: block completion if notes contain deferred work markers
            if status == "done":
                check_notes = notes or step.get("notes", "")
                _DEFERRED_MARKERS = ("todo", "deferred", "skipped", "pending", "later", "not yet", "tbd")
                found = [m for m in _DEFERRED_MARKERS if m in check_notes.lower()]
                if found:
                    raise ValueError(
                        f"Step {step_id} has unfinished markers ({', '.join(found)}) in notes. "
                        f"Move deferred work to a new step with add_steps, or remove the markers."
                    )

            step["status"] = status
            if notes:
                step["notes"] = notes

            if status in ("done", "skipped"):
                for s in plan["steps"]:
                    if s["status"] == "pending":
                        deps = s.get("depends_on", [])
                        if not deps or all(
                            self._find_step(plan, d)["status"] in ("done", "skipped")
                            for d in deps
                        ):
                            s["status"] = "doing"
                            break

            self._save(plan)
            return plan

    def add_steps(self, steps: List[str], after_step_id: Optional[int] = None) -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")

            existing_ids = [s["id"] for s in plan["steps"]]
            next_id = max(existing_ids) + 1 if existing_ids else 1

            new_steps = []
            for i, desc in enumerate(steps):
                sid = next_id + i
                new_steps.append({
                    "id": sid,
                    "title": desc,
                    "status": "pending",
                    "notes": "",
                    "depends_on": [],
                })

            if after_step_id is not None:
                idx = next(
                    (i for i, s in enumerate(plan["steps"]) if s["id"] == after_step_id),
                    None,
                )
                if idx is None:
                    raise ValueError(f"Step {after_step_id} not found")
                for ns in new_steps:
                    ns["depends_on"] = [after_step_id]
                insert_pos = idx + 1
                plan["steps"] = plan["steps"][:insert_pos] + new_steps + plan["steps"][insert_pos:]
            else:
                if plan["steps"]:
                    last_id = plan["steps"][-1]["id"]
                    for ns in new_steps:
                        ns["depends_on"] = [last_id]
                        last_id = ns["id"]
                plan["steps"].extend(new_steps)

            self._save(plan)
            return plan

    def link_experiment(self, step_id: int, experiment_name: str) -> dict:
        """Link an experiment to a plan step.

        If the experiment is already linked to other steps, returns the plan
        with a warning annotation so the agent knows coordination is needed.
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            step = self._find_step(plan, step_id)

            # Check if this experiment is already linked to other steps
            other_steps = []
            for s in plan["steps"]:
                if s["id"] != step_id and experiment_name in s.get("experiments", []):
                    other_steps.append(s["id"])

            if "experiments" not in step or not isinstance(step.get("experiments"), list):
                step["experiments"] = []
            if experiment_name not in step["experiments"]:
                step["experiments"].append(experiment_name)

            if other_steps:
                step.setdefault("_shared_experiment_note", "")
                step["_shared_experiment_note"] = (
                    f"Experiment '{experiment_name}' is shared with step(s) {other_steps}. "
                    f"Coordinate changes — modifications affect all linked steps."
                )

            self._save(plan)
            return plan

    def skip_step(self, step_id: int, reason: str = "") -> dict:
        return self.update_step(step_id, "skipped", notes=reason or "skipped")

    def complete(self) -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")

            # Check all steps are done or skipped
            incomplete = [
                s for s in plan["steps"]
                if s["status"] not in ("done", "skipped")
            ]
            if incomplete:
                incomplete_ids = [s["id"] for s in incomplete]
                incomplete_statuses = [f"step {s['id']} ({s['status']})" for s in incomplete]
                raise ValueError(
                    f"Cannot complete plan: {len(incomplete)} step(s) not finished: "
                    f"{', '.join(incomplete_statuses)}. "
                    f"Mark them as done/skipped first, or use abandon() if giving up."
                )

            plan["status"] = "completed"
            self._save(plan)
            self._clear_active()
            return plan

    def abandon(self, reason: str = "") -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            plan["status"] = "abandoned"
            if reason:
                plan["abandon_reason"] = reason
            self._save(plan)
            self._clear_active()
            return plan

    def deactivate(self) -> Optional[dict]:
        """Pause the active plan without abandoning it. Returns the plan or None."""
        with self._lock:
            plan = self.get_active()
            if not plan:
                return None
            plan["status"] = "paused"
            plan["updated"] = time.time()
            self._save(plan)
            self._set_active(None)
            return plan

    def reactivate(self, plan_id: str) -> Optional[dict]:
        """Re-activate a paused plan by id. Returns the plan or None."""
        with self._lock:
            plan = self._load(plan_id)
            if not plan:
                return None
            if plan.get("status") not in ("paused", "abandoned"):
                return None
            # Deactivate current active plan first
            current = self.get_active()
            if current:
                current["status"] = "paused"
                current["updated"] = time.time()
                self._save(current)
            plan["status"] = "active"
            plan["updated"] = time.time()
            self._save(plan)
            self._set_active(plan["id"])
            return plan

    def _list_plans_from_disk(self) -> List[dict]:
        """Common helper: read all plan files from disk."""
        if not os.path.isdir(self._dir):
            return []
        results = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            if fname.startswith(".tmp_"):
                continue  # Skip temp files from atomic writes
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                results.append(plan)
            except Exception:
                continue
        return results

    def list_titles(self) -> List[Dict]:
        """Return [{id, title, status}] for all plans. Used for semantic matching."""
        results = []
        for plan in self._list_plans_from_disk():
            results.append({
                "id": plan.get("id", "?"),
                "title": plan.get("title", ""),
                "status": plan.get("status", "?"),
            })
        return results

    def summary(self) -> str:
        plan = self.get_active()
        if not plan:
            return "No active plan."
        return self._format_plan(plan)

    def context_for_prompt(self) -> str:
        plan = self.get_active()
        if not plan:
            return ""
        lines = []
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"{s['id']}. [{icon}] {s['title']}"
            if s.get("notes"):
                line += f" — {s['notes']}"
            exps = s.get("experiments", [])
            if exps:
                line += f" [exp: {', '.join(exps)}]"
            shared_note = s.get("_shared_experiment_note", "")
            if shared_note:
                line += f" ⚠️ {shared_note}"
            lines.append(line)
        return (
            f'<active-plan title="{plan["title"]}">\n'
            + "\n".join(lines)
            + "\n</active-plan>"
        )

    def _format_plan(self, plan: dict) -> str:
        lines = [f"Plan: {plan['title']} [{plan['status']}]"]
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"  {s['id']}. [{icon}] {s['title']}"
            if s.get("notes"):
                line += f" — {s['notes']}"
            exps = s.get("experiments", [])
            if exps:
                line += f" [exp: {', '.join(exps)}]"
            lines.append(line)
        done = sum(1 for s in plan["steps"] if s["status"] in ("done", "skipped"))
        lines.append(f"Progress: {done}/{len(plan['steps'])}")
        return "\n".join(lines)

    @staticmethod
    def _find_step(plan: dict, step_id: int) -> dict:
        for s in plan["steps"]:
            if s["id"] == step_id:
                return s
        raise ValueError(f"Step {step_id} not found")

    def list_plans(self) -> List[dict]:
        plans = []
        for plan in self._list_plans_from_disk():
            steps = plan.get("steps", [])
            plans.append({
                "id": plan.get("id", "?"),
                "title": plan.get("title", ""),
                "status": plan.get("status", "?"),
                "done": sum(1 for s in steps if s.get("status") in ("done", "skipped")),
                "total": len(steps),
                "created": plan.get("created", 0),
            })
        return plans

    def clear_completed(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                if plan.get("status") in ("completed", "abandoned"):
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count

    # ── Auto-sync: tool results → plan step updates ──────────────────────

    def auto_sync_step(self, tool_name: str, success: bool, result_summary: str = "", turn: int = 0):
        """Auto-update the current 'doing' step based on tool execution results.

        Called after productive tool execution to keep plan in sync with reality.
        - success=True on a productive tool → mark step done via update_step
          (which enforces deferred work validation)
        - success=False → append failure note to step
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                return

            doing_steps = [s for s in plan["steps"] if s["status"] == "doing"]
            if not doing_steps:
                return

            step = doing_steps[0]

            # Track turn for staleness detection
            step["_last_activity_turn"] = turn

            if success and tool_name in ("write_file", "edit_file"):
                # Delegate to update_step so deferred work validation applies
                notes = result_summary if result_summary else ""
                try:
                    plan = self.update_step(step["id"], "done", notes=notes)
                except ValueError as e:
                    # Deferred work found — append note instead
                    step["notes"] = (step.get("notes", "") + f" ⚠ auto-done blocked: {e}").strip()
                    self._save(plan)
            elif not success:
                # Failure — append to notes
                failures = step.get("_failure_count", 0) + 1
                step["_failure_count"] = failures
                note = f" [fail #{failures}: {result_summary[:80]}]" if result_summary else f" [fail #{failures}]"
                step["notes"] = (step.get("notes", "") + note).strip()
                self._save(plan)

    def _auto_advance(self, plan: dict):
        """Advance the next pending step to 'doing' after current step completes."""
        for s in plan["steps"]:
            if s["status"] == "pending":
                deps = s.get("depends_on", [])
                if not deps or all(
                    self._find_step(plan, d)["status"] in ("done", "skipped")
                    for d in deps
                ):
                    s["status"] = "doing"
                    break

    # ── Consistency check ────────────────────────────────────────────────

    def check_consistency(self, current_turn: int) -> Optional[str]:
        """Check plan consistency and return a warning message if issues found.

        Called periodically (every N turns) to detect:
        - Stale "doing" steps (no activity for too many turns)
        - Steps with multiple failures still marked "doing"
        - Plan overall progress stalled
        """
        plan = self.get_active()
        if not plan:
            return None

        issues = []
        doing_steps = [s for s in plan["steps"] if s["status"] == "doing"]

        for step in doing_steps:
            last_activity = step.get("_last_activity_turn", 0)
            turns_stale = current_turn - last_activity if last_activity else current_turn

            # Stale step detection
            if turns_stale >= _STEP_STALE_TURNS:
                issues.append(
                    f"Step {step['id']} ('{step['title'][:40]}') has been 'doing' for "
                    f"{turns_stale} turns without progress. Consider: is it still relevant?"
                )

            # Repeated failure detection
            failures = step.get("_failure_count", 0)
            if failures >= 3:
                issues.append(
                    f"Step {step['id']} ('{step['title'][:40]}') has {failures} failures. "
                    f"Consider skipping it or replanning."
                )

        if not issues:
            return None

        return (
            "\n[PLAN CONSISTENCY CHECK]\n"
            + "\n".join(f"  ⚠ {issue}" for issue in issues)
            + "\nConsider: plan_update(action='abandon') + plan_create to replan, "
            "or plan_update(action='step_skip') for blocked steps."
        )

    def should_rebuild(self, consecutive_failures: int) -> bool:
        """Return True if the plan should be rebuilt due to repeated failures."""
        if consecutive_failures < _PLAN_REBUILD_FAILURES:
            return False
        plan = self.get_active()
        if not plan:
            return False
        doing_steps = [s for s in plan["steps"] if s["status"] == "doing"]
        if doing_steps:
            failures = doing_steps[0].get("_failure_count", 0)
            return failures >= _PLAN_REBUILD_FAILURES
        return False

    def record_turn_activity(self, turn: int):
        """Record that the current turn had activity on the active plan's doing step."""
        with self._lock:
            plan = self.get_active()
            if not plan:
                return
            doing_steps = [s for s in plan["steps"] if s["status"] == "doing"]
            if doing_steps:
                doing_steps[0]["_last_activity_turn"] = turn
                self._save(plan)

    # ── Checkpoint & Rollback ────────────────────────────────────────────────

    def checkpoint(self, step_id: int, files: List[str] = None,
                   memory_keys: List[str] = None, summary: str = "") -> Optional[StepCheckpoint]:
        """Record a checkpoint when a plan step completes.

        Called automatically when update_step marks a step as 'done'.
        Stores which files were modified and which memory keys were written.
        """
        plan = self.get_active()
        if not plan:
            return None

        plan_id = plan.get("id", "")
        cp = StepCheckpoint(
            step_id=step_id,
            timestamp=time.time(),
            files_modified=files or [],
            memory_keys=memory_keys or [],
            summary=summary,
        )

        if plan_id not in self._checkpoints:
            self._checkpoints[plan_id] = {}
        self._checkpoints[plan_id][step_id] = cp

        return cp

    def get_checkpoint(self, step_id: int) -> Optional[StepCheckpoint]:
        """Get checkpoint for a specific step in the active plan."""
        plan = self.get_active()
        if not plan:
            return None
        plan_id = plan.get("id", "")
        plan_cps = self._checkpoints.get(plan_id, {})
        return plan_cps.get(step_id)

    def get_rollback_info(self, step_id: int) -> str:
        """Get rollback information for a step — what was done and how to undo.

        Does NOT perform actual rollback (too dangerous). Returns information
        so the agent knows where to restart from.
        """
        plan = self.get_active()
        if not plan:
            return "No active plan."

        plan_id = plan.get("id", "")
        plan_cps = self._checkpoints.get(plan_id, {})

        # Find all checkpoints from step_id onward
        steps_to_rollback = sorted(
            [sid for sid in plan_cps if sid >= step_id]
        )

        if not steps_to_rollback:
            return f"No checkpoints found from step {step_id} onward."

        lines = [f"Rollback info from step {step_id}:"]
        for sid in steps_to_rollback:
            cp = plan_cps[sid]
            lines.append(f"  Step {sid}: {cp.summary}")
            if cp.files_modified:
                lines.append(f"    Files modified: {', '.join(cp.files_modified)}")
            if cp.memory_keys:
                lines.append(f"    Memory keys written: {', '.join(cp.memory_keys)}")

        lines.append("")
        lines.append("To rollback: revert the listed files and re-execute from step "
                     f"{step_id}. Memory keys may need to be updated.")
        return "\n".join(lines)

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints for the active plan."""
        plan = self.get_active()
        if not plan:
            return []
        plan_id = plan.get("id", "")
        plan_cps = self._checkpoints.get(plan_id, {})
        return [cp.to_dict() for cp in sorted(plan_cps.values(), key=lambda c: c.step_id)]
