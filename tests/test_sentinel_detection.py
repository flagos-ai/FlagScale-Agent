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

"""Regression tests for sentinel mention-vs-use detection in kernel.

Root cause: The kernel used `"[TASK_COMPLETE]" in assistant_text` (simple
substring match) to detect the completion signal. If the agent mentioned
"[TASK_COMPLETE]" anywhere in its text — e.g. while explaining guard
logic — the kernel treated it as a real completion signal and invoked
completion-path guard consultation, blocking the turn.

Fix: End-anchored detection — only treat [TASK_COMPLETE] / [NEED_USER_INPUT]
as a sentinel when it appears at the END of the text (optionally followed
by an _override_reason line), not when it appears in the middle as a quote.
"""

import re


def _is_completion_signal(text: str) -> bool:
    """Replicate the kernel's end-anchored sentinel detection logic."""
    stripped = text.rstrip()
    return (
        stripped.endswith("[TASK_COMPLETE]")
        or re.search(
            r'\[TASK_COMPLETE\]\s*\n?_override_reason\s*[:=]',
            stripped,
        ) is not None
    )


def _is_need_input_signal(text: str) -> bool:
    """Replicate the kernel's [NEED_USER_INPUT] detection."""
    return text.rstrip().endswith("[NEED_USER_INPUT]")


class TestSentinelMentionVsUse:
    """[TASK_COMPLETE] mentioned in the middle of text must NOT trigger
    the completion path. Only a trailing [TASK_COMPLETE] counts."""

    def test_trailing_task_complete(self):
        """[TASK_COMPLETE] at end of text IS a completion signal."""
        text = "All done with the task.\n[TASK_COMPLETE]"
        assert _is_completion_signal(text)

    def test_trailing_task_complete_with_override(self):
        """[TASK_COMPLETE] followed by _override_reason IS a completion signal."""
        text = (
            "[TASK_COMPLETE]\n"
            "_override_reason: Verified all tests pass, delivery path confirmed."
        )
        assert _is_completion_signal(text)

    def test_trailing_task_complete_with_override_same_line(self):
        """[TASK_COMPLETE] with _override_reason on same line IS completion."""
        text = "[TASK_COMPLETE] _override_reason: all good"
        assert _is_completion_signal(text)

    def test_mention_in_middle_not_completion(self):
        """'[TASK_COMPLETE]' mentioned in explanatory text is NOT completion."""
        text = (
            "The guard fires when the agent emits [TASK_COMPLETE] with an "
            "active plan. This is the expected behavior."
        )
        assert not _is_completion_signal(text)

    def test_mention_in_code_block_not_completion(self):
        """'[TASK_COMPLETE]' inside a code block is NOT completion."""
        text = (
            "```python\n"
            'if "[TASK_COMPLETE]" in assistant_text:\n'
            "    print('done')\n"
            "```\n"
            "That's how the detection works."
        )
        assert not _is_completion_signal(text)

    def test_trailing_need_user_input(self):
        """[NEED_USER_INPUT] at end IS a need-input signal."""
        text = "I need more info to proceed.\n[NEED_USER_INPUT]"
        assert _is_need_input_signal(text)

    def test_mention_need_user_input_not_signal(self):
        """[NEED_USER_INPUT] mentioned in text is NOT a signal."""
        text = (
            "You should end with [NEED_USER_INPUT] if you need user input."
        )
        assert not _is_need_input_signal(text)

    def test_whitespace_after_sentinel(self):
        """Trailing whitespace after [TASK_COMPLETE] is still completion."""
        text = "[TASK_COMPLETE]   \n  \n"
        assert _is_completion_signal(text)

    def test_empty_text(self):
        """Empty text is not a completion signal."""
        assert not _is_completion_signal("")
        assert not _is_need_input_signal("")

    def test_both_sentinels_mentioned_neither_trailing(self):
        """Both sentinels mentioned but neither at end = not completion."""
        text = (
            "The agent can end with [TASK_COMPLETE] or [NEED_USER_INPUT]. "
            "Choose based on whether the task is done."
        )
        assert not _is_completion_signal(text)
        assert not _is_need_input_signal(text)
