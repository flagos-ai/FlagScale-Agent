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

"""Regression tests: ShellTool bounds returned output (head+tail truncation).

A chatty build can emit hundreds of MB in seconds; returning it all as one
observation floods the LLM context and grows memory unbounded. execute() must
keep the head and tail and drop the middle, while leaving small outputs intact.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flagscale_agent.react.tools.shell import (
    ShellTool,
    _truncate_output,
    _MAX_OUTPUT_CHARS,
    _OUTPUT_HEAD_CHARS,
    _OUTPUT_TAIL_CHARS,
)


class TestTruncateOutputHelper:
    def test_short_output_unchanged(self):
        s = "hello\nworld\n"
        assert _truncate_output(s) == s

    def test_exactly_at_limit_unchanged(self):
        s = "x" * _MAX_OUTPUT_CHARS
        assert _truncate_output(s) == s

    def test_over_limit_keeps_head_and_tail(self):
        # Distinct head/tail sentinels so we can prove both survive.
        head = "HEAD_SENTINEL\n" + ("a" * 100_000)
        tail = ("z" * 200_000) + "\nTAIL_SENTINEL"
        s = head + tail
        out = _truncate_output(s)
        assert "HEAD_SENTINEL" in out, "head lost during truncation"
        assert "TAIL_SENTINEL" in out, "tail lost during truncation"
        assert "output truncated" in out, "no truncation marker"
        # Result is bounded: head budget + tail budget + a small marker.
        assert len(out) <= _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS + 500

    def test_marker_reports_omitted_count(self):
        s = "b" * (_MAX_OUTPUT_CHARS * 3)
        out = _truncate_output(s)
        omitted = len(s) - _OUTPUT_HEAD_CHARS - _OUTPUT_TAIL_CHARS
        # No newlines to snap to, so head/tail keep the raw slice sizes.
        assert f"{omitted:,}" in out

    def test_cut_points_snap_to_line_boundaries(self):
        # Build line-structured text well over the limit. Every line is short,
        # so head/tail cuts must land on newlines, never mid-line.
        line = "LOGLINE_" + ("x" * 40) + "\n"
        s = line * (_MAX_OUTPUT_CHARS // len(line) * 3)
        out = _truncate_output(s)
        head, _, tail = out.partition("... [output truncated")
        # The kept head must end exactly on a newline (no dangling half-line).
        assert head.endswith("\n"), "head cut mid-line"
        # The kept tail (after the marker line) must be whole lines only:
        # every non-empty line matches the original template.
        tail_body = tail.split("] ...\n\n", 1)[-1]
        for ln in tail_body.split("\n"):
            if ln:
                assert ln == line.rstrip("\n"), f"tail has partial line: {ln!r}"

    def test_bound_honored_when_no_newline_to_snap(self):
        # Pathological single giant line: no newline anywhere. Must still be
        # bounded (raw slices kept) rather than blowing the budget.
        s = "q" * (_MAX_OUTPUT_CHARS * 3)
        out = _truncate_output(s)
        assert len(out) <= _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS + 500


class TestShellExecuteBounded:
    def test_chatty_command_output_bounded(self):
        tool = ShellTool()
        # ~200k short lines emitted fast — simulates a chatty compile.
        cmd = ("for i in $(seq 1 200000); do "
               "echo \"line $i building object file xyz.o\"; done")
        result = tool.execute(command=cmd)
        # Bounded well under the raw ~7.7 MB.
        assert len(result) <= _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS + 500, (
            f"output not bounded: {len(result):,} chars"
        )
        # Head (first line) and tail (last line) both survive.
        assert "line 1 building" in result
        assert "line 200000 building" in result
        assert "output truncated" in result

    def test_small_command_output_intact(self):
        tool = ShellTool()
        result = tool.execute(command="echo SMALL_OK")
        assert "SMALL_OK" in result
        assert "output truncated" not in result
