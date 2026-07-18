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

"""Tests for guard lifecycle v3: override, decay, and inject suppression.

Tests the fix for the death spiral where:
1. ExperimentTrackingGuard's _unrecorded_launches only increased, never reset
2. Guards blocked permanently with no escape hatch
3. Inject messages polluted context indefinitely
"""

import pytest
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict, GuardRegistry
from flagscale_agent.react.guard.experiment_tracking import ExperimentTrackingGuard


class TestExperimentTrackingLifecycle:
    """Test that ExperimentTrackingGuard properly resets and doesn't death-spiral."""

    def test_false_positive_regex_not_triggered(self):
        """Commands that merely reference train.py should NOT trigger the guard."""
        guard = ExperimentTrackingGuard()
        false_positives = [
            "grep train.py *.yaml",
            "find . -name train.py",
            "cat examples/train.py",
            "head -20 train.py",
            "ls -la *train*",
            "wc -l pretrain.py",
        ]
        for cmd in false_positives:
            ctx = GuardContext(tool_name="shell", tool_args={"command": cmd})
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"False positive for: {cmd}"

    def test_actual_launch_detected(self):
        """Actual training launches should be detected."""
        actual_launches = [
            "python train.py --config config.yaml",
            "torchrun --nproc_per_node=8 train.py",
            "python run.py action=run",
            "flagscale train --config test",
            "python3 pretrain.py --model-path /path",
        ]
        for cmd in actual_launches:
            # Fresh guard for each test — avoids state leaking between commands
            guard = ExperimentTrackingGuard()
            ctx = GuardContext(tool_name="shell", tool_args={"command": cmd})
            verdict = guard.check_pre(ctx)
            assert verdict is not None, f"Missed actual launch: {cmd}"

    def test_counter_resets_on_add_attempt(self):
        """_unrecorded_launches resets when add_attempt is called."""
        guard = ExperimentTrackingGuard()

        # Simulate 2 unrecorded launches
        for _ in range(2):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "python train.py"})
            guard.check_pre(ctx)
        assert guard._unrecorded_launches == 2

        # Now record an attempt
        ctx = GuardContext(
            tool_name="workspace_experiment",
            tool_args={"action": "add_attempt", "name": "test_exp"},
        )
        guard.check_pre(ctx)
        assert guard._unrecorded_launches == 0
        assert guard._attempt_recorded is True

    def test_override_resets_state(self):
        """Overriding the guard should reset its state, preventing death spiral."""
        guard = ExperimentTrackingGuard()

        # Simulate 3 unrecorded launches to trigger block
        for _ in range(3):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "python train.py"})
            guard.check_pre(ctx)
        assert guard._unrecorded_launches == 3

        # Override should reset state
        ctx = GuardContext(tool_name="shell", tool_args={"command": "python train.py"})
        accepted = guard.accept_override("This is a test run, no experiment needed", ctx)
        assert accepted is True
        assert guard._unrecorded_launches == 0

    def test_decay_resets_after_idle(self):
        """Guard state should decay after N idle iterations."""
        guard = ExperimentTrackingGuard()
        guard._unrecorded_launches = 5  # Simulate accumulated state

        # Simulate idle iterations (guard not firing)
        for _ in range(guard.decay_after_idle):
            guard._tick_idle()

        # After decay, state should be reset
        assert guard._unrecorded_launches == 0
        assert guard._attempt_recorded is False

    def test_inject_suppression_after_max_repeats(self):
        """Inject warnings should stop after max_inject_repeats."""
        guard = ExperimentTrackingGuard()

        # Fire inject warnings up to the limit
        for i in range(guard.max_inject_repeats):
            guard._record_trigger("launch_without_recording")

        # Now it should be suppressed
        assert guard._should_suppress_inject("launch_without_recording") is True

    def test_satisfaction_check(self):
        """Guard should report satisfied when attempt is recorded and no pending results."""
        guard = ExperimentTrackingGuard()
        ctx = GuardContext(tool_name="", tool_args={})

        # Not satisfied initially
        assert guard.is_satisfied(ctx) is False

        # Record an attempt
        guard._attempt_recorded = True
        guard._result_pending = False
        assert guard.is_satisfied(ctx) is True

    def test_full_death_spiral_scenario(self):
        """Reproduce the exact death spiral from the conversation and verify fix."""
        reg = GuardRegistry()
        guard = ExperimentTrackingGuard()
        reg.register(guard)

        # Simulate the death spiral: many commands matching train keyword
        commands = [
            "python train.py --config test",
            "python train.py --config test2",
            "python train.py --config test3",
            "python train.py --config test4",
            "python train.py --config test5",
        ]

        block_count = 0
        for cmd in commands:
            from flagscale_agent.react.guard import AgentState
            ctx = GuardContext(
                tool_name="shell",
                tool_args={"command": cmd},
                current_state=AgentState.EXECUTING,
            )
            verdict = reg.check_pre(ctx)
            if verdict and verdict.action == "block":
                block_count += 1
                # The LLM overrides — this should RESET the guard
                ctx_override = GuardContext(
                    tool_name="shell",
                    tool_args={"command": cmd},
                    override_reason="I understand, proceeding with test",
                    current_state=AgentState.EXECUTING,
                )
                verdict2 = reg.check_pre(ctx_override)
                # After override, should not block (state was reset)
                assert verdict2 is None or verdict2.action != "block"
                break

        # After override, subsequent launches should start fresh (warn, not block)
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "python train.py --config next"},
            current_state=AgentState.EXECUTING,
        )
        verdict = reg.check_pre(ctx)
        # Should be inject (warning) not block, since counter was reset
        if verdict:
            assert verdict.action != "block", "Death spiral! Guard still blocking after override"


class TestGuardLifecycleBase:
    """Test base class lifecycle mechanics work for all guards."""

    def test_subclass_gets_lifecycle_attrs(self):
        """Guards that don't call super().__init__() still get lifecycle attrs."""

        class MyGuard(Guard):
            name = "test_guard"

            def __init__(self):
                # Deliberately NOT calling super().__init__()
                self.my_state = 42

        g = MyGuard()
        assert hasattr(g, '_inject_counts')
        assert hasattr(g, '_iterations_since_trigger')
        assert hasattr(g, '_is_suppressed')

    def test_decay_resets_state(self):
        """After decay_after_idle iterations without firing, state resets."""

        class StatefulGuard(Guard):
            name = "stateful"
            decay_after_idle = 5

            def __init__(self):
                super().__init__()
                self.counter = 10

            def reset_state(self):
                super().reset_state()
                self.counter = 0

        g = StatefulGuard()
        assert g.counter == 10

        # Tick idle for decay_after_idle iterations
        for _ in range(5):
            g._tick_idle()

        assert g.counter == 0

    def test_override_resets_state(self):
        """Override acceptance resets guard state."""

        class BlockingGuard(Guard):
            name = "blocker"
            overridable = True

            def __init__(self):
                super().__init__()
                self.block_count = 5

            def reset_state(self):
                super().reset_state()
                self.block_count = 0

        g = BlockingGuard()
        ctx = GuardContext(tool_name="test", tool_args={})
        g.accept_override("good reason", ctx)
        assert g.block_count == 0

    def test_inject_suppression(self):
        """Inject messages are suppressed after max_inject_repeats."""

        class InjectGuard(Guard):
            name = "injector"
            max_inject_repeats = 3

        g = InjectGuard()
        for _ in range(3):
            g._record_trigger("my_category")

        assert g._should_suppress_inject("my_category") is True
        assert g._should_suppress_inject("other_category") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
