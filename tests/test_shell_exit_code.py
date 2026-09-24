"""Tests: foreground shell surfaces the RAW process exit code (no interpretation).

Contract: when a synchronous command exits non-zero, the returned output ends
with a bare `[exit code: N]` line so the LLM can judge what N means. No mapping
of specific codes to causes (e.g. 137 -> OOM) lives in the tool or the prompt —
the same code means different things across programs and platforms. A clean
exit (0) adds no line.
"""

from flagscale_agent.react.tools.shell import ShellTool


def test_nonzero_exit_surfaces_raw_code():
    sh = ShellTool()
    out = sh.execute(command="echo hi; exit 137")
    assert "hi" in out
    assert "[exit code: 137]" in out
    # RAW code only — no interpretation of what 137 means.
    low = out.lower()
    assert "oom" not in low
    assert "out of memory" not in low
    assert "killed" not in low


def test_clean_exit_adds_no_line():
    sh = ShellTool()
    out = sh.execute(command="echo ok")
    assert "ok" in out
    assert "exit code" not in out.lower()


def test_generic_failure_code_surfaced():
    sh = ShellTool()
    out = sh.execute(command="exit 2")
    assert "[exit code: 2]" in out


def test_exit_code_line_is_last():
    sh = ShellTool()
    out = sh.execute(command="echo before; exit 3")
    assert out.rstrip().endswith("[exit code: 3]")
