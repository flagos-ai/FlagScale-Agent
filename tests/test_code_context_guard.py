"""Tests for CodeContextGuard — code read/write tracking across compaction."""

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.code_context import (
    CodeContextGuard, FileContext, _MEMORY_KEY, _RESTORE_GRACE_PERIOD,
)
from flagscale_agent.react.state_machine import AgentState


def _ctx(tool_name="", tool_args=None, tool_result=None,
         assistant_text="", context_pressure=0.0,
         recent_tool_names=None, recent_tool_history=None, **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=AgentState.EXECUTING,
        assistant_text=assistant_text,
        context_pressure=context_pressure,
        recent_tool_names=recent_tool_names or [],
        recent_tool_history=recent_tool_history or [],
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Basic Properties
# ══════════════════════════════════════════════════════════════════════════════


class TestBasicProperties:
    def test_overridable(self):
        g = CodeContextGuard()
        assert g.overridable is True

    def test_accept_override_always(self):
        g = CodeContextGuard()
        assert g.accept_override("any reason", _ctx()) is True

    def test_name(self):
        g = CodeContextGuard()
        assert g.name == "code_context"


# ══════════════════════════════════════════════════════════════════════════════
# Recording Reads
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordRead:
    def test_records_file_path(self):
        g = CodeContextGuard()
        ctx = _ctx("read_file", {"path": "/src/model.py"},
                   tool_result="class MyModel:\n    def forward(self):\n        pass\n")
        g.check_post(ctx)
        assert "/src/model.py" in g._file_map
        assert g._file_map["/src/model.py"].read_count == 1

    def test_increments_read_count(self):
        g = CodeContextGuard()
        for _ in range(3):
            ctx = _ctx("read_file", {"path": "/src/model.py"},
                       tool_result="class MyModel:\n    pass\n")
            g.check_post(ctx)
        assert g._file_map["/src/model.py"].read_count == 3

    def test_extracts_python_classes(self):
        g = CodeContextGuard()
        content = "class Transformer:\n    pass\nclass Attention:\n    pass\ndef forward():\n    pass\n"
        ctx = _ctx("read_file", {"path": "/src/model.py"}, tool_result=content)
        g.check_post(ctx)
        entry = g._file_map["/src/model.py"]
        assert "class Transformer" in entry.key_elements
        assert "class Attention" in entry.key_elements
        assert "def forward" in entry.key_elements

    def test_extracts_yaml_keys(self):
        g = CodeContextGuard()
        content = "experiment:\n  name: test\nsystem:\n  gpus: 8\nmodel:\n  name: qwen\n"
        ctx = _ctx("read_file", {"path": "/conf/train.yaml"}, tool_result=content)
        g.check_post(ctx)
        entry = g._file_map["/conf/train.yaml"]
        assert "experiment" in entry.key_elements
        assert "system" in entry.key_elements
        assert "model" in entry.key_elements

    def test_lru_eviction(self):
        g = CodeContextGuard()
        # Fill beyond max
        for i in range(55):
            ctx = _ctx("read_file", {"path": f"/src/file{i}.py"},
                       tool_result=f"class File{i}:\n    pass\n")
            g.check_post(ctx)
        assert len(g._file_map) <= 50

    def test_empty_path_ignored(self):
        g = CodeContextGuard()
        ctx = _ctx("read_file", {"path": ""}, tool_result="content")
        g.check_post(ctx)
        assert len(g._file_map) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Recording Writes
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordWrite:
    def test_records_edit_file(self):
        g = CodeContextGuard()
        ctx = _ctx("edit_file", {
            "path": "/src/guard.py",
            "old_string": "priority = 30",
            "new_string": "priority = 30\n    overridable = True",
        })
        g.check_post(ctx)
        entry = g._file_map["/src/guard.py"]
        assert len(entry.modifications) == 1
        assert "edited:" in entry.modifications[0]

    def test_records_write_file(self):
        g = CodeContextGuard()
        ctx = _ctx("write_file", {
            "path": "/src/new_file.py",
            "content": "class NewGuard:\n    pass\n",
            "mode": "write",
        })
        g.check_post(ctx)
        entry = g._file_map["/src/new_file.py"]
        assert len(entry.modifications) == 1
        assert "write:" in entry.modifications[0]

    def test_keeps_last_5_modifications(self):
        g = CodeContextGuard()
        for i in range(8):
            ctx = _ctx("edit_file", {
                "path": "/src/guard.py",
                "old_string": f"old_{i}",
                "new_string": f"new_{i}",
            })
            g.check_post(ctx)
        entry = g._file_map["/src/guard.py"]
        assert len(entry.modifications) == 5


# ══════════════════════════════════════════════════════════════════════════════
# Pre-check: Inject Prior Context
# ══════════════════════════════════════════════════════════════════════════════


class TestPreCheckInject:
    def test_no_inject_on_first_read(self):
        """First read of a file should not inject."""
        g = CodeContextGuard()
        ctx = _ctx("read_file", {"path": "/src/model.py"})
        result = g.check_pre(ctx)
        assert result is None

    def test_injects_on_re_read(self):
        """Re-reading a file should inject prior context."""
        g = CodeContextGuard()
        # First read
        post_ctx = _ctx("read_file", {"path": "/src/model.py"},
                        tool_result="class Transformer:\n    def forward(self):\n        pass\n")
        g.check_post(post_ctx)

        # Second read — pre-check should inject
        pre_ctx = _ctx("read_file", {"path": "/src/model.py"})
        result = g.check_pre(pre_ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "CodeContext" in result.message
        assert "model.py" in result.message

    def test_no_inject_for_non_read(self):
        """Non-read tools don't trigger inject."""
        g = CodeContextGuard()
        # Record a read
        g.check_post(_ctx("read_file", {"path": "/src/x.py"}, tool_result="class X:\n    pass\n"))
        # edit_file doesn't trigger pre inject
        result = g.check_pre(_ctx("edit_file", {"path": "/src/x.py"}))
        assert result is None

    def test_no_inject_for_continuation_read(self):
        """Continuation reads (same file, drilling deeper) should not inject."""
        g = CodeContextGuard()
        g.check_post(_ctx("read_file", {"path": "/src/big.py"},
                          tool_result="class Big:\n    pass\n"))

        # Re-read same file as continuation
        ctx = _ctx("read_file", {"path": "/src/big.py"},
                   recent_tool_names=["read_file"],
                   recent_tool_history=[{"args_summary": "/src/big.py lines 1-50"}])
        result = g.check_pre(ctx)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Context Pressure Dump
# ══════════════════════════════════════════════════════════════════════════════


class TestContextPressureDump:
    def test_no_dump_below_threshold(self):
        """No dump suggested below pressure threshold."""
        g = CodeContextGuard()
        for i in range(10):
            g.check_post(_ctx("read_file", {"path": f"/src/f{i}.py"},
                              tool_result="pass\n", context_pressure=0.3))
        # _should_suggest_dump checks context_pressure from ctx
        # But check_post only triggers if pressure >= 0.70
        assert g._dump_suggested is False

    def test_dump_at_high_pressure(self):
        """Dump suggested when pressure is high and enough files tracked."""
        g = CodeContextGuard()
        # Record enough ops
        for i in range(9):
            g.check_post(_ctx("read_file", {"path": f"/src/f{i}.py"},
                              tool_result=f"class F{i}:\n    pass\n",
                              context_pressure=0.5))

        # Now high pressure triggers dump
        result = g.check_post(_ctx("read_file", {"path": "/src/last.py"},
                                   tool_result="class Last:\n    pass\n",
                                   context_pressure=0.75))
        assert result is not None
        assert result.action == "inject_msg"
        assert "memory_write" in result.message
        assert g._dump_suggested is True

    def test_dump_only_once(self):
        """Dump only suggested once per session."""
        g = CodeContextGuard()
        for i in range(10):
            g.check_post(_ctx("read_file", {"path": f"/src/f{i}.py"},
                              tool_result="pass\n", context_pressure=0.75))

        # Second high-pressure read — no duplicate
        result = g.check_post(_ctx("read_file", {"path": "/src/extra.py"},
                                   tool_result="pass\n", context_pressure=0.80))
        assert result is None

    def test_mark_dump_done_resets(self):
        """mark_dump_done allows fresh tracking."""
        g = CodeContextGuard()
        g._dump_suggested = True
        g._ops_since_dump = 20
        g.mark_dump_done()
        assert g._dump_done is True
        assert g._ops_since_dump == 0


# ══════════════════════════════════════════════════════════════════════════════
# Reset Turn
# ══════════════════════════════════════════════════════════════════════════════


class TestResetTurn:
    def test_file_map_persists(self):
        """File map should survive reset_turn — that's the point."""
        g = CodeContextGuard()
        g.check_post(_ctx("read_file", {"path": "/src/x.py"}, tool_result="class X:\n    pass\n"))
        g.reset_turn()
        assert "/src/x.py" in g._file_map

    def test_dump_suggested_resets(self):
        """dump_suggested resets so it can fire again next turn."""
        g = CodeContextGuard()
        g._dump_suggested = True
        g.reset_turn()
        assert g._dump_suggested is False


# ══════════════════════════════════════════════════════════════════════════════
# get_context_map API
# ══════════════════════════════════════════════════════════════════════════════


class TestGetContextMap:
    def test_returns_tracked_files(self):
        g = CodeContextGuard()
        g.check_post(_ctx("read_file", {"path": "/src/a.py"},
                          tool_result="class A:\n    pass\n"))
        g.check_post(_ctx("edit_file", {"path": "/src/b.py",
                                         "old_string": "x", "new_string": "y"}))
        ctx_map = g.get_context_map()
        assert "/src/a.py" in ctx_map
        assert "/src/b.py" in ctx_map

    def test_empty_entries_excluded(self):
        """Files with no useful context are excluded from map."""
        g = CodeContextGuard()
        g.check_post(_ctx("read_file", {"path": "/src/empty.py"}, tool_result=""))
        ctx_map = g.get_context_map()
        # Empty file → no elements, no summary → excluded
        assert "/src/empty.py" not in ctx_map


# ══════════════════════════════════════════════════════════════════════════════
# FileContext compact_repr
# ══════════════════════════════════════════════════════════════════════════════


class TestFileContextRepr:
    def test_with_summary(self):
        fc = FileContext(path="/x.py", summary="Implements TP layer communication")
        assert "Implements TP layer" in fc.compact_repr()

    def test_with_modifications(self):
        fc = FileContext(path="/x.py", modifications=["added overridable=True"])
        assert "Modified:" in fc.compact_repr()

    def test_with_elements(self):
        fc = FileContext(path="/x.py", key_elements=["class Guard", "def check_pre"])
        assert "Contains:" in fc.compact_repr()
        assert "class Guard" in fc.compact_repr()

    def test_truncation(self):
        fc = FileContext(path="/x.py", summary="A" * 300)
        assert len(fc.compact_repr()) <= 200



# ══════════════════════════════════════════════════════════════════════════════
# Restore from Memory
# ══════════════════════════════════════════════════════════════════════════════


class TestRestoreSuggestion:
    """Guard suggests memory_read when file_map is empty after grace period."""

    def test_no_suggest_before_grace_period(self):
        """Don't suggest restore in the first few tool calls."""
        g = CodeContextGuard()
        ctx = _ctx("read_file", {"path": "/src/model.py"})
        result = g.check_pre(ctx)
        assert result is None  # Too early

    def test_suggests_restore_after_grace_period(self):
        """After grace period, suggests restore if file_map is empty."""
        g = CodeContextGuard()
        # Simulate tool calls to pass grace period
        for i in range(_RESTORE_GRACE_PERIOD):
            g.check_post(_ctx("shell", {"command": "ls"}))

        # Now a read_file should trigger restore suggestion
        ctx = _ctx("read_file", {"path": "/src/model.py"})
        result = g.check_pre(ctx)
        assert result is not None
        assert _MEMORY_KEY in result.message
        assert "restore" in result.reason

    def test_suggests_only_once(self):
        """Restore suggestion fires at most once."""
        g = CodeContextGuard()
        for i in range(_RESTORE_GRACE_PERIOD):
            g.check_post(_ctx("shell", {"command": "ls"}))

        # First suggestion
        result1 = g.check_pre(_ctx("read_file", {"path": "/a.py"}))
        assert result1 is not None

        # Second call — no suggestion
        result2 = g.check_pre(_ctx("read_file", {"path": "/b.py"}))
        assert result2 is None

    def test_no_suggest_if_file_map_nonempty(self):
        """If file_map already has entries, no restore needed."""
        g = CodeContextGuard()
        # Build some context
        g.check_post(_ctx("read_file", {"path": "/src/a.py"},
                          tool_result="class Foo:\n    pass\n"))
        g.check_post(_ctx("shell", {"command": "ls"}))
        g.check_post(_ctx("shell", {"command": "ls"}))

        # Should not suggest — already have context
        result = g.check_pre(_ctx("read_file", {"path": "/src/b.py"}))
        assert result is None

    def test_no_suggest_after_restored(self):
        """After successful restore, no more suggestions."""
        g = CodeContextGuard()
        for i in range(_RESTORE_GRACE_PERIOD):
            g.check_post(_ctx("shell", {"command": "ls"}))

        # Simulate restore
        g._restored = True
        result = g.check_pre(_ctx("read_file", {"path": "/src/a.py"}))
        assert result is None


class TestRestoreFromMemory:
    """Guard parses memory_read result to rebuild file_map."""

    def test_parses_simple_format(self):
        """Parse 'path: summary' lines."""
        g = CodeContextGuard()
        content = (
            "/src/model.py: Implements Transformer with multi-head attention\n"
            "/src/config.yaml: Contains: experiment, model, system\n"
            "/src/guard.py: Modified: added overridable=True\n"
        )
        ctx = _ctx("memory_read", {"key": _MEMORY_KEY}, tool_result=content)
        g.check_post(ctx)

        assert g._restored is True
        assert "/src/model.py" in g._file_map
        assert "/src/config.yaml" in g._file_map
        assert "/src/guard.py" in g._file_map

        # Check parsed content
        assert g._file_map["/src/model.py"].summary == "Implements Transformer with multi-head attention"
        assert "experiment" in g._file_map["/src/config.yaml"].key_elements
        assert len(g._file_map["/src/guard.py"].modifications) > 0

    def test_skips_headers_and_empty_lines(self):
        """Ignore [SECTION] headers and blank lines."""
        g = CodeContextGuard()
        content = (
            "[FILES_READ]\n"
            "\n"
            "/src/model.py: Main model class\n"
            "# comment line\n"
            "/src/util.py: Helper utilities\n"
        )
        ctx = _ctx("memory_read", {"key": _MEMORY_KEY}, tool_result=content)
        g.check_post(ctx)

        assert g._restored is True
        assert "/src/model.py" in g._file_map
        assert "/src/util.py" in g._file_map
        assert len(g._file_map) == 2  # No header or comment entries

    def test_no_restore_for_other_memory_keys(self):
        """Don't parse memory_read results for other keys."""
        g = CodeContextGuard()
        content = "/src/model.py: something"
        ctx = _ctx("memory_read", {"key": "some_other_key"}, tool_result=content)
        g.check_post(ctx)

        assert g._restored is False
        assert len(g._file_map) == 0

    def test_empty_content_no_crash(self):
        """Empty memory_read result doesn't crash."""
        g = CodeContextGuard()
        ctx = _ctx("memory_read", {"key": _MEMORY_KEY}, tool_result="")
        g.check_post(ctx)
        assert g._restored is False

    def test_marks_entries_as_previously_seen(self):
        """Restored entries have read_count >= 1."""
        g = CodeContextGuard()
        content = "/src/model.py: Main model class\n"
        ctx = _ctx("memory_read", {"key": _MEMORY_KEY}, tool_result=content)
        g.check_post(ctx)

        assert g._file_map["/src/model.py"].read_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Full Dump → Restore Cycle
# ══════════════════════════════════════════════════════════════════════════════


class TestDumpRestoreCycle:
    """End-to-end: build context → dump format → parse back."""

    def test_dump_format_is_parseable(self):
        """What _build_dump_summary produces must be parseable by _restore_from_memory."""
        g = CodeContextGuard()

        # Build some context
        g.check_post(_ctx("read_file", {"path": "/src/model.py"},
                          tool_result="class Transformer:\n    pass\nclass Attention:\n    pass\n"))
        g.check_post(_ctx("edit_file", {
            "path": "/src/guard.py",
            "old_string": "priority = 30",
            "new_string": "priority = 30\n    overridable = True",
        }))
        g.check_post(_ctx("read_file", {"path": "/conf/train.yaml"},
                          tool_result="experiment:\n  name: test\nmodel:\n  name: qwen\n"))

        # Get dump summary
        dump = g._build_dump_summary()
        assert len(dump) > 0

        # Create fresh guard and restore
        g2 = CodeContextGuard()
        g2._restore_from_memory(dump)

        # Verify restored entries match original
        assert g2._restored is True
        assert len(g2._file_map) >= 2  # At least model.py and train.yaml

    def test_memory_write_detection(self):
        """Detect when agent writes our key and mark dump done."""
        g = CodeContextGuard()
        # Simulate agent writing code_context_map
        ctx = _ctx("memory_write", {"key": _MEMORY_KEY, "content": "...", "type": "context"})
        g.check_post(ctx)
        assert g._dump_done is True
        assert g._ops_since_dump == 0

    def test_full_lifecycle(self):
        """Simulate: build context → dump → new session → restore → inject on re-read."""
        # Session 1: build context
        g1 = CodeContextGuard()
        g1.check_post(_ctx("read_file", {"path": "/src/layers.py"},
                           tool_result="class ColumnParallelLinear:\n    def forward(self):\n        pass\n"))
        g1.check_post(_ctx("read_file", {"path": "/src/layers.py"},
                           tool_result="class ColumnParallelLinear:\n    def forward(self):\n        pass\n"))

        dump_content = g1._build_dump_summary()

        # Session 2: fresh guard, restore, then read same file
        g2 = CodeContextGuard()
        # Pass grace period
        for _ in range(_RESTORE_GRACE_PERIOD):
            g2.check_post(_ctx("shell", {"command": "ls"}))

        # Restore
        g2.check_post(_ctx("memory_read", {"key": _MEMORY_KEY}, tool_result=dump_content))
        assert g2._restored is True

        # Now reading the same file should inject prior context
        result = g2.check_pre(_ctx("read_file", {"path": "/src/layers.py"}))
        assert result is not None
        assert "previously read" in result.message
