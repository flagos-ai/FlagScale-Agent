"""Tests for Judge — classify, budget, caching, multi-round interaction."""

import pytest

from flagscale_agent.react.judge import Judge, JudgeBudget, _CLASSIFY_PROMPTS


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:100])
        resp = self.responses.pop(0) if self.responses else "{}"
        return {"content": resp}


# ── JudgeBudget ────────────────────────────────────────────────────────


class TestJudgeBudget:
    def test_initial_state(self):
        budget = JudgeBudget(max_calls_per_turn=8)
        assert budget.calls_this_turn == 0
        assert budget.exhausted is False

    def test_consume_and_exhaust(self):
        budget = JudgeBudget(max_calls_per_turn=3)
        assert budget.consume() is True
        assert budget.consume() is True
        assert budget.consume() is True
        assert budget.calls_this_turn == 3
        assert budget.exhausted is True
        assert budget.consume() is False  # exhausted, can't consume more

    def test_reset_turn(self):
        budget = JudgeBudget(max_calls_per_turn=3)
        budget.consume()
        budget.consume()
        budget.reset_turn()
        assert budget.calls_this_turn == 0
        assert budget.exhausted is False

    def test_total_calls_tracks_across_resets(self):
        budget = JudgeBudget(max_calls_per_turn=3)
        budget.consume()
        budget.consume()
        assert budget.total_calls == 2
        budget.reset_turn()
        assert budget.total_calls == 2  # total is NOT reset


# ── Judge.classify ─────────────────────────────────────────────────────


class TestJudgeClassify:
    def test_is_error_false(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_error", {
            "command": "pip install torch",
            "result": "Successfully installed"
        })
        assert result is False

    def test_is_error_true(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_error", {
            "command": "python train.py",
            "result": "RuntimeError: CUDA out of memory"
        })
        assert result is True

    def test_is_dangerous_true(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_dangerous", {"command": "rm -rf /"})
        assert result is True

    def test_is_dangerous_false(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_dangerous", {"command": "ls -la"})
        assert result is False

    def test_is_read_only_shell(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_read_only_shell", {"command": "nvidia-smi"})
        assert result is True

    def test_is_training_command(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_training_command",
            {"command": "torchrun --nproc_per_node=8 train.py"})
        assert result is True

    def test_is_kill_command(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        result = judge.classify("is_kill_command", {"command": "kill -9 12345"})
        assert result is True

    def test_is_user_porting_confirm_mode_b(self):
        provider = MockProvider(responses=['{"decision": "mode_b"}'])
        judge = Judge(provider)
        result = judge.classify("is_user_porting_confirm",
            {"user_input": "mode_b"})
        assert result == "mode_b"

    def test_is_user_porting_confirm_empty(self):
        provider = MockProvider(responses=['{"decision": ""}'])
        judge = Judge(provider)
        result = judge.classify("is_user_porting_confirm",
            {"user_input": "hello"})
        assert result == ""

    def test_checklist_rule_match(self):
        provider = MockProvider(responses=[
            '{"match": true, "reason": "contains TODO"}'])
        judge = Judge(provider)
        result = judge.classify("checklist_rule", {
            "rule": '{"type": "contains", "pattern": "TODO"}',
            "description": "No TODOs",
            "tool_name": "write_file",
            "content": "# TODO: implement",
        })
        assert result == {"match": True, "reason": "contains TODO"}

    def test_unknown_category_returns_default(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        result = judge.classify("nonexistent", {}, default=False)
        assert result is False
        assert len(provider.calls) == 0


# ── Multi-round classify ────────────────────────────────────────────────


class TestMultiRoundClassify:
    def test_need_more_triggers_second_call(self):
        responses = [
            '{"need_more": ["result"]}',
            '{"real": true, "need_more": null}',
        ]
        provider = MockProvider(responses=responses)
        judge = Judge(provider)
        result = judge.classify("is_error", {
            "command": "python train.py",
            "result": "Long output ... RuntimeError at the end"
        })
        assert result is True
        assert len(provider.calls) == 2

    def test_max_3_rounds(self):
        """After 3 rounds, should return default regardless of need_more."""
        responses = [
            '{"need_more": ["result"]}',
            '{"need_more": ["result"]}',
            '{"need_more": ["result"]}',
        ]
        provider = MockProvider(responses=responses)
        judge = Judge(provider)
        result = judge.classify("is_error", {
            "command": "python train.py",
            "result": "still not enough context..."
        })
        assert result is False  # default
        assert len(provider.calls) == 3


# ── Caching ─────────────────────────────────────────────────────────────


class TestJudgeCaching:
    def test_same_context_hits_cache(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        r1 = judge.classify("is_error",
            {"command": "echo hello", "result": "hello"})
        r2 = judge.classify("is_error",
            {"command": "echo hello", "result": "hello"})
        assert r1 == r2 == False
        assert len(provider.calls) == 1
        assert judge.budget.total_saved_by_cache == 1

    def test_different_context_misses_cache(self):
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        r1 = judge.classify("is_error",
            {"command": "echo hello", "result": "hello"})
        r2 = judge.classify("is_error",
            {"command": "python fail.py", "result": "RuntimeError"})
        assert len(provider.calls) == 2

    def test_cache_is_per_category(self):
        """Same context different categories = different cache keys."""
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": false, "need_more": null}',
        ])
        judge = Judge(provider)
        judge.classify("is_error",
            {"command": "some_custom_cmd", "result": "ok"})
        judge.classify("is_success",
            {"command": "some_custom_cmd", "result": "ok"})
        assert len(provider.calls) == 2


# ── Budget exhaustion ───────────────────────────────────────────────────


class TestBudgetExhaustion:
    def test_returns_default_when_exhausted(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'] * 10)
        judge = Judge(provider)
        judge.budget.max_calls_per_turn = 3
        results = []
        for i in range(5):
            r = judge.classify("is_error", {f"key_{i}": f"val_{i}"})
            results.append(r)
        # First 3 use LLM, last 2 return default
        assert results[0] is False
        assert results[1] is False
        assert results[2] is False
        assert results[3] is None  # default
        assert results[4] is None  # default
        assert judge.budget.calls_this_turn == 3
        assert judge.budget.exhausted is True
        assert len(provider.calls) == 3


# ── Context truncation ──────────────────────────────────────────────────


class TestContextTruncation:
    def test_truncate_one_short(self):
        text = "short"
        result = Judge._truncate_one(text, max_chars=800)
        assert result == text

    def test_truncate_one_long(self):
        text = "X" * 2000
        result = Judge._truncate_one(text, max_chars=800)
        assert len(result) <= 850  # 800 + overhead for omission marker + tail chars
        assert "omitted" in result.lower() or "..." in result
        # Must preserve head and tail
        assert result.startswith("X" * 10)
        assert result.rstrip().endswith("X" * 10)

    def test_truncate_context_preserves_keys(self):
        ctx = {"cmd": "echo hello", "result": "world"}
        result = Judge._truncate_context(ctx, max_chars=800)
        assert set(result.keys()) == set(ctx.keys())


# ── JSON parsing ────────────────────────────────────────────────────────


class TestParseJson:
    def test_clean_json(self):
        result = Judge._parse_json('{"real": true}')
        assert result == {"real": True}

    def test_json_with_surrounding_text(self):
        result = Judge._parse_json('Some text {"real": false} more text')
        assert result == {"real": False}

    def test_empty_string(self):
        result = Judge._parse_json("")
        assert result == {}

    def test_no_braces(self):
        result = Judge._parse_json("no json here")
        assert result == {}


# ── Provider error handling ─────────────────────────────────────────────


class TestJudgeProviderErrors:
    def test_provider_exception_returns_default(self):
        class FailingProvider:
            def chat(self, messages, tools=None):
                raise RuntimeError("API unavailable")
        judge = Judge(FailingProvider())
        result = judge.classify("is_error",
            {"command": "test", "result": "test"})
        assert result is False  # default for is_error

    def test_none_provider_returns_default(self):
        judge = Judge(None)
        result = judge.classify("is_error",
            {"command": "test", "result": "test"}, default=False)
        assert result is False


# ── Classify prompts completeness ───────────────────────────────────────


class TestClassifyPrompts:
    EXPECTED_CATEGORIES = {
        "is_error", "is_success", "is_dangerous", "is_read_only_shell",
        "is_training_command", "is_kill_command", "is_training_failure",
        "is_zombie_gpu", "is_stuck_in_loop", "is_user_porting_confirm",
        "checklist_rule", "checklist_rule_batch", "extract_constraints",
        "route_intent", "is_constraint_violated", "skill_suggest",
        "skill_suggest_by_context", "is_continuation", "is_warning_triggered",
    }

    def test_all_categories_present(self):
        assert set(_CLASSIFY_PROMPTS.keys()) == self.EXPECTED_CATEGORIES

    def test_all_prompts_non_empty(self):
        for category, prompt in _CLASSIFY_PROMPTS.items():
            assert prompt.strip(), f"Prompt for '{category}' is empty"

    def test_all_boolean_prompts_have_need_more(self):
        """All boolean classify prompts should support multi-round via need_more."""
        boolean_categories = self.EXPECTED_CATEGORIES - {
            "is_user_porting_confirm", "checklist_rule", "checklist_rule_batch",
            "extract_constraints", "route_intent", "skill_suggest",
            "skill_suggest_by_context"}
        for category in boolean_categories:
            prompt = _CLASSIFY_PROMPTS[category]
            assert "need_more" in prompt.lower(), (
                f"'{category}' prompt missing need_more support")
