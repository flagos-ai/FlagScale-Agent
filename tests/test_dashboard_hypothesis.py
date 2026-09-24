
# ── Hypothesis block (plan-level thinking, permanently resident) ────────

import pytest
from unittest.mock import MagicMock, patch

from tests.test_prompt_builder import make_builder


# ── Hypothesis block (plan-level thinking, permanently resident) ────────

class TestDashboardHypothesis:
    """Hypothesis block: full passthrough, no truncation, no step-count pollution."""

    def test_hypothesis_rendered_in_full(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            "<current-hypothesis>\n"
            "bottleneck: A2A overlap\n"
            "prediction: if X changes, metric reaches Y\n"
            "</current-hypothesis>\n"
            "[🔄] Step 1: work\n"
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Hypothesis (current problem model):" in result
        assert "bottleneck: A2A overlap" in result
        assert "prediction: if X changes, metric reaches Y" in result
        # full text, no truncation marker
        assert "…" not in result.split("Hypothesis (current problem model):")[1]

    def test_no_hypothesis_no_block(self):
        """Absent or empty thinking → byte-identical to pre-hypothesis behavior."""
        b = make_builder()
        b._turn_count = 1
        plan_ctx = '<active-plan title="T">\n[🔄] Step 1: work\n'
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Hypothesis" not in result

        b2 = make_builder()
        b2._turn_count = 1
        plan_ctx_empty = (
            '<active-plan title="T">\n'
            "<current-hypothesis>\n"
            "</current-hypothesis>\n"
            "[🔄] Step 1: work\n"
        )
        with patch.object(b2, "_build_memory_keys_summary", return_value=""):
            result2 = b2._build_dashboard(plan_ctx_empty, session_dir="")
        assert "Hypothesis" not in result2

    def test_hypothesis_step_pattern_does_not_pollute_count(self):
        """thinking text containing '[🔄] Step N:' must not corrupt Step counting."""
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            "<current-hypothesis>\n"
            "old notes mention [🔄] Step 9: stale and [⬜] Step 4: gone\n"
            "</current-hypothesis>\n"
            "[✅] Step 1: done\n"
            "[🔄] Step 2: in progress\n"
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        # Real steps only: 2 steps, current = 2 (the polluted 9/4 ignored)
        assert "Step: 2/2" in result
        # Hypothesis text itself still shown in full
        assert "[🔄] Step 9: stale" in result
