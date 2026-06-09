"""Tests for Phase 2 Judge tiered architecture: fast path, batch classify."""

import pytest
from flagscale_agent.react.judge import (
    Judge, JudgeBudget,
    SOURCE_FAST, SOURCE_LLM, SOURCE_CACHE, SOURCE_DEFAULT, SOURCE_UNAVAILABLE,
)
from flagscale_agent.react.judge_fast import FastParser, FastClassifier


# ── FastParser tests ─────────────────────────────────────────────────────


class TestFastParser:
    def test_knowledge_confirm_yes(self):
        text = "[PIPELINE_KNOWLEDGE_CONFIRMED: YES]"
        assert FastParser.parse_knowledge_confirm(text) is True

    def test_knowledge_confirm_no(self):
        text = "[PIPELINE_KNOWLEDGE_CONFIRMED: NO]"
        assert FastParser.parse_knowledge_confirm(text) is False

    def test_knowledge_confirm_missing(self):
        text = "No confirmation here."
        assert FastParser.parse_knowledge_confirm(text) is None


# ── FastClassifier tests ─────────────────────────────────────────────────


class TestFastClassifier:
    # is_read_only_shell
    def test_read_only_ls(self):
        assert FastClassifier.is_read_only_shell("ls -la /tmp") is True

    def test_read_only_nvidia_smi(self):
        assert FastClassifier.is_read_only_shell("nvidia-smi") is True

    def test_read_only_grep(self):
        assert FastClassifier.is_read_only_shell("grep -r 'pattern' .") is True

    def test_not_read_only_pip(self):
        assert FastClassifier.is_read_only_shell("pip install torch") is False

    def test_not_read_only_python(self):
        assert FastClassifier.is_read_only_shell("python train.py") is False

    def test_read_only_with_redirect_is_false(self):
        assert FastClassifier.is_read_only_shell("ls > output.txt") is False

    def test_uncertain_returns_none(self):
        # Complex command that needs LLM judgment
        assert FastClassifier.is_read_only_shell("some_custom_script --check") is None

    # is_dangerous
    def test_dangerous_rm_rf_root(self):
        assert FastClassifier.is_dangerous("rm -rf /") is True

    def test_dangerous_rm_rf_home(self):
        assert FastClassifier.is_dangerous("rm -rf ~") is True

    def test_dangerous_fork_bomb(self):
        assert FastClassifier.is_dangerous(":(){ :|:& };:") is True

    def test_not_obviously_dangerous(self):
        # Normal rm on a specific file — needs LLM to judge
        assert FastClassifier.is_dangerous("rm /tmp/test.txt") is None

    # is_training_command
    def test_training_torchrun(self):
        assert FastClassifier.is_training_command("torchrun --nproc_per_node=8 train.py") is True

    def test_training_deepspeed(self):
        assert FastClassifier.is_training_command("deepspeed train.py --config ds.json") is True

    def test_training_help_is_not(self):
        assert FastClassifier.is_training_command("torchrun --help") is False

    def test_not_training_grep(self):
        assert FastClassifier.is_training_command("grep torchrun logs.txt") is False

    def test_uncertain_python_script(self):
        assert FastClassifier.is_training_command("python some_script.py") is None

    # is_kill_command
    def test_kill_command(self):
        assert FastClassifier.is_kill_command("kill -9 12345") is True

    def test_pkill_command(self):
        assert FastClassifier.is_kill_command("pkill python") is True

    def test_not_kill_ps(self):
        assert FastClassifier.is_kill_command("ps aux") is False

    def test_uncertain_kill(self):
        assert FastClassifier.is_kill_command("python kill_zombies.py") is None


# ── Judge fast-path integration ──────────────────────────────────────────


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:100])
        resp = self.responses.pop(0) if self.responses else '{}'
        return {"content": resp}


class TestJudgeFastPath:
    def test_fast_path_skips_llm(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        # "ls" is definitely read-only — fast path should handle it
        result, source = judge.classify_traced(
            "is_read_only_shell", {"command": "ls -la"}
        )
        assert result is True
        assert source == SOURCE_FAST
        assert len(provider.calls) == 0  # No LLM call made

    def test_fast_path_escalates_to_llm(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        # "some_custom_script" is uncertain — should escalate to LLM
        result, source = judge.classify_traced(
            "is_read_only_shell", {"command": "some_custom_script --check"}
        )
        assert source == SOURCE_LLM
        assert len(provider.calls) == 1

    def test_dangerous_fast_path(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        result, source = judge.classify_traced(
            "is_dangerous", {"command": "rm -rf /"}
        )
        assert result is True
        assert source == SOURCE_FAST
        assert len(provider.calls) == 0

    def test_training_fast_path(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        result, source = judge.classify_traced(
            "is_training_command", {"command": "torchrun --nproc_per_node=8 train.py"}
        )
        assert result is True
        assert source == SOURCE_FAST

    def test_kill_fast_path(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        result, source = judge.classify_traced(
            "is_kill_command", {"command": "kill -9 1234"}
        )
        assert result is True
        assert source == SOURCE_FAST

    def test_non_fast_category_goes_to_llm(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        # is_error has no fast path
        result, source = judge.classify_traced(
            "is_error", {"command": "python x.py", "output": "RuntimeError"}
        )
        assert source == SOURCE_LLM


# ── Judge batch classify ─────────────────────────────────────────────────


class TestJudgeBatchClassify:
    def test_batch_all_fast(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        items = [
            ("is_read_only_shell", {"command": "ls"}, False),
            ("is_kill_command", {"command": "kill -9 123"}, False),
            ("is_dangerous", {"command": "rm -rf /"}, False),
        ]
        results = judge.classify_batch(items)
        assert results[0] == (True, SOURCE_FAST)
        assert results[1] == (True, SOURCE_FAST)
        assert results[2] == (True, SOURCE_FAST)
        assert len(provider.calls) == 0

    def test_batch_mixed_fast_and_llm(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        items = [
            ("is_read_only_shell", {"command": "ls"}, False),  # fast
            ("is_error", {"command": "x", "output": "Error"}, False),  # LLM
        ]
        results = judge.classify_batch(items)
        assert results[0] == (True, SOURCE_FAST)
        assert results[1][1] == SOURCE_LLM
        assert len(provider.calls) == 1

    def test_batch_with_cache(self):
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        # First call populates cache
        judge.classify("is_error", {"command": "x", "output": "Error"})
        # Batch should hit cache
        items = [
            ("is_error", {"command": "x", "output": "Error"}, False),
        ]
        results = judge.classify_batch(items)
        assert results[0][1] == SOURCE_CACHE

    def test_batch_no_provider(self):
        judge = Judge(None)
        items = [
            ("is_read_only_shell", {"command": "ls"}, False),
            ("is_error", {"command": "x", "output": "Error"}, None),
        ]
        results = judge.classify_batch(items)
        assert results[0] == (False, SOURCE_UNAVAILABLE)
        assert results[1] == (None, SOURCE_UNAVAILABLE)

    def test_batch_budget_exhausted(self):
        provider = MockProvider(responses=[])
        budget = JudgeBudget(max_calls_per_turn=0)
        judge = Judge(provider, budget=budget)
        items = [
            ("is_error", {"command": "x", "output": "Error"}, "default_val"),
        ]
        results = judge.classify_batch(items)
        assert results[0] == ("default_val", SOURCE_DEFAULT)


# ── SOURCE_FAST in ClassifyTrace ─────────────────────────────────────────


class TestClassifyTraceFast:
    def test_trace_records_fast(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        judge.classify("is_read_only_shell", {"command": "ls"})
        assert judge.trace.source_of("is_read_only_shell") == SOURCE_FAST

    def test_trace_any_from_fast(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        judge.classify("is_kill_command", {"command": "kill 123"})
        assert judge.trace.any_from(SOURCE_FAST) is True
        assert judge.trace.any_from(SOURCE_LLM) is False
