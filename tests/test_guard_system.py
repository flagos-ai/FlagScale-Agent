# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the new guard system."""
import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.training_attempt import TrainingAttemptGuard
from flagscale_agent.react.guard.experiment_tracking import ExperimentTrackingGuard
from flagscale_agent.react.guard.output_dir_reuse import OutputDirReuseGuard
from flagscale_agent.react.guard.debug_discipline import DebugDisciplineGuard
from flagscale_agent.react.guard.file_tool import FileToolGuard
from flagscale_agent.react.guard.megatron_path import MegatronPathGuard
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard


def make_ctx(tool_name="", tool_args=None, tool_result=""):
    """Helper to create a GuardContext."""
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = tool_result
    ctx.classify_fn = None
    ctx.current_experiment_name = ""
    ctx.experiment_diff_fn = None
    return ctx


class TestTrainingAttemptGuard:
    """Test the 2-Strike rule at attempt granularity."""

    def test_initial_state_no_block(self):
        guard = TrainingAttemptGuard()
        ctx = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        result = guard.check_pre(ctx)
        assert result is None  # No block on first launch

    def test_two_same_category_failures_block(self):
        guard = TrainingAttemptGuard()

        # First attempt: edit + launch + fail with AttributeError
        ctx_edit = make_ctx("edit_file", {"path": "model.py", "new_string": "fix"})
        guard.check_post(ctx_edit)

        ctx_launch = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        guard.check_post(ctx_launch)

        ctx_fail = make_ctx("monitor", {"output_dir": "/tmp/test"},
                           tool_result="TRAINING CRASHED\nAttributeError: 'X' has no attribute 'y'")
        guard.check_post(ctx_fail)

        # Second attempt: edit + launch + fail with same category
        ctx_edit2 = make_ctx("edit_file", {"path": "model.py", "new_string": "fix2"})
        guard.check_post(ctx_edit2)

        ctx_launch2 = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        guard.check_post(ctx_launch2)

        ctx_fail2 = make_ctx("monitor", {"output_dir": "/tmp/test"},
                            tool_result="TRAINING CRASHED\nAttributeError: 'Z' has no attribute 'w'")
        guard.check_post(ctx_fail2)

        # Now trying to launch again should be BLOCKED
        ctx_launch3 = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        result = guard.check_pre(ctx_launch3)
        assert result is not None
        assert result.action == "block"

    def test_different_categories_no_block(self):
        guard = TrainingAttemptGuard()

        # First attempt: fail with AttributeError
        ctx_launch = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        guard.check_post(ctx_launch)
        ctx_fail = make_ctx("monitor", tool_result="TRAINING CRASHED\nAttributeError: missing")
        guard.check_post(ctx_fail)

        # Second attempt: fail with shape error (different category)
        ctx_edit = make_ctx("edit_file", {"path": "model.py", "new_string": "x"})
        guard.check_post(ctx_edit)
        ctx_launch2 = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        guard.check_post(ctx_launch2)
        ctx_fail2 = make_ctx("monitor", tool_result="TRAINING CRASHED\nRuntimeError: mat1 size mismatch")
        guard.check_post(ctx_fail2)

        # Should NOT be blocked (different categories)
        ctx_launch3 = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        result = guard.check_pre(ctx_launch3)
        assert result is None

    def test_source_reading_unblocks(self):
        guard = TrainingAttemptGuard()
        guard._is_blocked = True
        guard._blocked_category = "attribute"
        guard._source_reads_since_block = 0

        # Reading source files should count toward unblock
        for i in range(3):
            ctx = make_ctx("read_file", {"path": f"/src/model_{i}.py"}, tool_result="class Model:...")
            guard.check_post(ctx)

        assert guard._source_reads_since_block >= 3

        # Also need hypothesis to fully unblock
        guard._hypothesis_declared = True

        # Now launch should be unblocked
        ctx_launch = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        result = guard.check_pre(ctx_launch)
        assert result is None


class TestExperimentTrackingGuard:
    """Test experiment recording enforcement."""

    def test_warns_on_first_unrecorded_launch(self):
        guard = ExperimentTrackingGuard()
        ctx = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_blocks_after_three_unrecorded(self):
        guard = ExperimentTrackingGuard()
        ctx = make_ctx("shell", {"command": "python run.py --config-path conf action=run"})
        guard.check_pre(ctx)  # warn 1
        guard._attempt_recorded = False
        guard.check_pre(ctx)  # warn 2
        guard._attempt_recorded = False
        result = guard.check_pre(ctx)  # block 3
        assert result is not None
        assert result.action == "block"

    def test_recording_resets_count(self):
        guard = ExperimentTrackingGuard()
        guard._unrecorded_launches = 2

        # Record an attempt
        ctx = make_ctx("workspace_experiment", {"action": "add_attempt", "name": "test"})
        guard.check_post(ctx)

        assert guard._attempt_recorded is True
        assert guard._unrecorded_launches == 0


class TestDebugDisciplineGuard:
    """Test hypothesis enforcement."""

    def test_no_warning_without_failure(self):
        guard = DebugDisciplineGuard()
        ctx = make_ctx("edit_file", {"path": "model.py", "new_string": "fix"})
        result = guard.check_pre(ctx)
        assert result is None

    def test_warns_after_failure_without_hypothesis(self):
        guard = DebugDisciplineGuard()

        # Observe failure
        ctx_fail = make_ctx("monitor", tool_result="TRAINING CRASHED\nRuntimeError: bad")
        guard.check_post(ctx_fail)

        # First edit is fine
        ctx_edit1 = make_ctx("edit_file", {"path": "model.py", "new_string": "fix1"})
        guard.check_pre(ctx_edit1)

        # Second edit triggers warning
        ctx_edit2 = make_ctx("edit_file", {"path": "model.py", "new_string": "fix2"})
        result = guard.check_pre(ctx_edit2)
        assert result is not None

    def test_debug_print_reminder(self):
        guard = DebugDisciplineGuard()
        ctx = make_ctx("edit_file", {"path": "model.py", "new_string": 'print(f"[DBG] value={x}")'})
        result = guard.check_post(ctx)
        assert result is not None  # Should get maximization reminder


class TestFileToolGuard:
    """Test file truncation detection."""

    def test_detects_truncated_content(self):
        guard = FileToolGuard()
        # Content with unbalanced brackets (looks truncated)
        content = "def foo():\n" + "    x = {\n" * 10 + "    'key': 'value',\n" * 200
        ctx = make_ctx("write_file", {"path": "test.py", "content": content, "mode": "write"})
        result = guard.check_pre(ctx)
        # Should detect unbalanced brackets
        if len(content) > 4000:
            assert result is not None

    def test_no_warning_for_balanced_content(self):
        guard = FileToolGuard()
        content = "x = 1\ny = 2\n" * 400  # Long but balanced
        ctx = make_ctx("write_file", {"path": "test.py", "content": content, "mode": "write"})
        result = guard.check_pre(ctx)
        # Balanced content shouldn't trigger truncation warning
        assert result is None


class TestLLMFallback:
    """Test that guards fall back to LLM classify when regex doesn't match."""

    def test_error_classify_falls_through_to_llm(self):
        """When regex can't classify, try LLM."""
        guard = TrainingAttemptGuard()
        # Create a classify_fn that returns a known category
        def mock_classify(category, context, default=None):
            if category == "training_error_category":
                return {"category": "config", "confidence": 0.9}
            return default

        ctx = make_ctx("monitor", tool_result="TRAINING CRASHED\nSome weird custom error nobody expected")
        ctx.classify_fn = mock_classify

        # This error doesn't match any regex pattern
        result = guard._classify_training_error(ctx.tool_result, ctx)
        assert result == "config"  # LLM said config with high confidence

    def test_error_classify_ignores_low_confidence_llm(self):
        """LLM results below confidence threshold are ignored."""
        guard = TrainingAttemptGuard()

        def mock_classify(category, context, default=None):
            if category == "training_error_category":
                return {"category": "nccl", "confidence": 0.3}  # Low confidence
            return default

        ctx = make_ctx("monitor", tool_result="TRAINING CRASHED\nVague error message")
        ctx.classify_fn = mock_classify

        result = guard._classify_training_error(ctx.tool_result, ctx)
        assert result == "general"  # Falls back to general

    def test_error_classify_regex_takes_priority(self):
        """Regex fast-path should win even if LLM would disagree."""
        guard = TrainingAttemptGuard()

        def mock_classify(category, context, default=None):
            return {"category": "data", "confidence": 1.0}  # Would return data

        ctx = make_ctx("monitor", tool_result="TRAINING CRASHED\nAttributeError: 'X' has no attr")
        ctx.classify_fn = mock_classify

        # Regex should catch AttributeError → "attribute"
        result = guard._classify_training_error(ctx.tool_result, ctx)
        assert result == "attribute"

    def test_memory_discipline_reminder_threshold(self):
        """Memory discipline reminds every 10 non-memory tool calls."""
        guard = MemoryDisciplineGuard()

        # 9 calls — no reminder
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None, f"Unexpected reminder on call {i+1}: {result}"

        # 10th call — triggers reminder, counter resets
        ctx = make_ctx("shell", {"command": "echo 10"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "10 tool calls" in result.message
        assert guard._calls_since_memory == 0  # Reset after firing

        # Next 9 calls — no reminder again
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None

        # 20th total call (10th since last reminder) — triggers again
        ctx = make_ctx("shell", {"command": "echo again"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"

        # memory_read resets counter
        ctx = make_ctx("memory_read", {"key": "test"}, tool_result="value")
        result = guard.check_pre(ctx)
        assert result is None
        assert guard._calls_since_memory == 0

    def test_debug_residue_llm_detection(self):
        """LLM can detect non-obvious debug prints."""
        guard = DebugDisciplineGuard()
        guard._modified_files.add("/tmp/test_debug_llm.py")

        # Write a file with ambiguous print statement
        import tempfile, os
        test_file = "/tmp/test_debug_llm.py"
        with open(test_file, "w") as f:
            f.write("""\
import torch

def forward(self, x):
    out = self.attn(x)
    print(f"shape after attn: {out.shape}")  # This is debug!
    return self.mlp(out)
""")

        def mock_classify(category, context, default=None):
            if category == "is_debug_residue":
                return {"is_residue": True, "reason": "Temporary shape print for debugging"}
            return default

        guard._modified_files = {test_file}
        residues = guard.check_clean_diff(classify_fn=mock_classify)
        assert len(residues) >= 1
        assert "LLM" in residues[0] or "shape after attn" in residues[0]

        # Cleanup
        os.unlink(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
