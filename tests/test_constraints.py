"""Tests for Phase 3 constraint system — schema, extractor, ConstraintGuard."""

import pytest

from flagscale_agent.react.constraint import (
    Constraint, ConstraintTrigger, ConstraintViolation,
)
from flagscale_agent.react.constraint.extractor import extract_constraints, _compile_one
from flagscale_agent.react.guard.constraint import ConstraintGuard
from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState
from flagscale_agent.react.judge import Judge


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:200])
        resp = self.responses.pop(0) if self.responses else "{}"
        return {"content": resp}


# ── ConstraintTrigger ────────────────────────────────────────────────────


class TestConstraintTrigger:
    def test_empty_trigger_matches_everything(self):
        t = ConstraintTrigger()
        assert t.matches("shell", {"command": "ls"}) is True
        assert t.matches("write_file", {"path": "/tmp/x"}) is True

    def test_tool_name_filter(self):
        t = ConstraintTrigger(tool_names={"shell"})
        assert t.matches("shell", {"command": "rm -rf /"}) is True
        assert t.matches("write_file", {"path": "/tmp/x"}) is False

    def test_keyword_filter(self):
        # OR logic: any one keyword match triggers
        t = ConstraintTrigger(keywords=["rm", "delete"])
        assert t.matches("shell", {"command": "rm -rf /tmp"}) is True   # matches "rm"
        assert t.matches("shell", {"command": "delete /tmp"}) is True   # matches "delete"
        assert t.matches("shell", {"command": "ls -la"}) is False       # no match

    def test_keyword_normalization(self):
        # pip ↔ pip3
        t1 = ConstraintTrigger(keywords=["pip install"])
        assert t1.matches("shell", {"command": "pip3 install torch"}) is True
        # underscore ↔ hyphen
        t2 = ConstraintTrigger(keywords=["transformer_engine"])
        assert t2.matches("shell", {"command": "pip install transformer-engine"}) is True
        # python ↔ python3
        t3 = ConstraintTrigger(keywords=["python"])
        assert t3.matches("shell", {"command": "python3 --version"}) is True

    def test_keyword_case_insensitive(self):
        t = ConstraintTrigger(keywords=["DELETE"])
        assert t.matches("shell", {"command": "delete /tmp/x"}) is True

    def test_keyword_in_tool_result(self):
        t = ConstraintTrigger(keywords=["error"])
        assert t.matches("shell", {"command": "run"}, "RuntimeError occurred") is True
        assert t.matches("shell", {"command": "run"}, "Success") is False

    def test_combined_tool_and_keyword(self):
        t = ConstraintTrigger(tool_names={"shell"}, keywords=["rm"])
        assert t.matches("shell", {"command": "rm -rf /tmp"}) is True
        assert t.matches("write_file", {"command": "rm stuff"}) is False
        assert t.matches("shell", {"command": "ls -la"}) is False


# ── Extractor (_compile_one) ─────────────────────────────────────────────


class TestCompileOne:
    def test_valid_item(self):
        item = {
            "description": "Never delete output dirs",
            "tool_names": ["shell"],
            "keywords": ["rm", "rmdir"],
            "severity": "error",
            "prompt": "Does this delete output?",
            "correction": "Don't delete output dirs.",
            "check_phase": "pre",
        }
        c = _compile_one(item, "train-run", 0)
        assert c is not None
        assert c.id == "train-run_0"
        assert c.description == "Never delete output dirs"
        assert c.trigger.tool_names == {"shell"}
        assert c.trigger.keywords == ["rm", "rmdir"]

    def test_missing_description_returns_none(self):
        item = {"tool_names": ["shell"], "keywords": ["rm"]}
        assert _compile_one(item, "test", 0) is None

    def test_empty_tool_names_and_keywords(self):
        item = {"description": "test constraint"}
        c = _compile_one(item, "test", 0)
        assert c.trigger.tool_names == set()
        assert c.trigger.keywords == []


# ── extract_constraints (full pipeline) ──────────────────────────────────


class TestExtractConstraints:
    def test_empty_content_returns_empty(self):
        result = extract_constraints("", lambda *a, **kw: [], "test")
        assert result == []

    def test_successful_extraction(self):
        def mock_classify(category, context, **kwargs):
            assert category == "extract_constraints"
            return [
                {"description": "No rm -rf", "tool_names": ["shell"],
                 "keywords": ["rm -rf"], "severity": "error",
                 "prompt": "Does this rm?", "correction": "Don't rm.",
                 "check_phase": "pre"},
            ]

        result = extract_constraints("skill content here", mock_classify, "my-skill")
        assert len(result) == 1
        assert result[0].id == "my-skill_0"
        assert result[0].description == "No rm -rf"

    def test_classify_returns_non_list(self):
        result = extract_constraints("content", lambda *a, **kw: "bad", "test")
        assert result == []

    def test_classify_raises_exception(self):
        def failing_classify(*a, **kw):
            raise RuntimeError("API down")

        result = extract_constraints("content", failing_classify, "test")
        assert result == []

    def test_malformed_items_skipped(self):
        def mock_classify(category, context, **kwargs):
            return [
                {"description": "valid", "tool_names": ["shell"]},
                "not a dict",
                {"no_description": True},
                {"description": "also valid", "tool_names": ["shell"]},
            ]

        result = extract_constraints("content", mock_classify, "test")
        assert len(result) == 2


# ── ConstraintGuard ──────────────────────────────────────────────────────


def _make_constraint(
    id="c1", description="test", tool_names=None, keywords=None,
    prompt="violated?", correction="fix it", **kwargs
):
    return Constraint(
        id=id,
        description=description,
        trigger=ConstraintTrigger(
            tool_names=set(tool_names or []),
            keywords=keywords or [],
        ),
        prompt=prompt,
        correction=correction,
    )


def _ctx(tool_name="", tool_args=None, tool_result=None,
         classify_fn=None, state=AgentState.EXECUTING):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=state,
        classify_fn=classify_fn,
    )


class TestConstraintGuard:
    def test_blocks_on_pre_violation(self):
        """Constraint with check_phase=pre blocks before tool execution."""
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        c = _make_constraint(tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("shell", {"command": "rm -rf /output"}, classify_fn=judge.classify)
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "fix it" in result.message

    def test_allows_when_not_violated(self):
        """LLM says not violated → no block."""
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        c = _make_constraint(tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("shell", {"command": "rm -rf /tmp/cache"}, classify_fn=judge.classify)
        result = guard.check_pre(ctx)
        assert result is None

    def test_skips_when_trigger_not_matched(self):
        """Trigger doesn't match → no LLM call."""
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        c = _make_constraint(tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("shell", {"command": "ls -la"}, classify_fn=judge.classify)
        result = guard.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_skips_wrong_tool(self):
        """Wrong tool name → no trigger."""
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        c = _make_constraint(tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("read_file", {"path": "/tmp/rm_log.txt"}, classify_fn=judge.classify)
        result = guard.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_blocks_without_classify_fn(self):
        """No classify_fn → conservative block."""
        c = _make_constraint(tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("shell", {"command": "rm -rf /data"})
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_violation_counter_increments(self):
        """Each violation increments the counter."""
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        c = _make_constraint(id="no_rm", tool_names=["shell"], keywords=["rm"])
        guard = ConstraintGuard(constraints=[c])
        ctx = _ctx("shell", {"command": "rm file1"}, classify_fn=judge.classify)
        guard.check_pre(ctx)
        guard.check_pre(ctx)
        assert guard.violations["no_rm"] == 2

    def test_add_constraints(self):
        """add_constraints appends to existing list."""
        guard = ConstraintGuard(constraints=[_make_constraint(id="c1")])
        guard.add_constraints([_make_constraint(id="c2")])
        assert len(guard.constraints) == 2

    def test_multiple_constraints_first_violation_wins(self):
        """First triggered+violated constraint produces the verdict."""
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        c1 = _make_constraint(id="c1", keywords=["rm"], correction="c1 fix")
        c2 = _make_constraint(id="c2", keywords=["rm"], correction="c2 fix")
        guard = ConstraintGuard(constraints=[c1, c2])
        ctx = _ctx("shell", {"command": "rm file"}, classify_fn=judge.classify)
        result = guard.check_pre(ctx)
        assert result.message == "c1 fix"
