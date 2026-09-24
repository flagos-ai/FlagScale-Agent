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

"""Tests for KnowledgeIndexGuard — reminds to rebuild the knowledge index
after a knowledge doc is edited (its cached line numbers go stale)."""

import pytest

from flagscale_agent.react.guard.knowledge_index import KnowledgeIndexGuard
from flagscale_agent.react.guard import GuardContext


@pytest.fixture
def guard():
    return KnowledgeIndexGuard()


@pytest.fixture
def make_ctx():
    """Factory for GuardContext with tool_name and path."""
    def _make(tool_name="edit_file", path="", tool_result="ok"):
        ctx = GuardContext()
        ctx.tool_name = tool_name
        ctx.tool_args = {"path": path}
        ctx.tool_result = tool_result
        return ctx
    return _make


DOC = "flagscale_agent/knowledge/docs/moe_training/01_moe_pretrain_perf_survey.md"


class TestKnowledgeIndexDocDetection:
    def test_knowledge_doc_detected(self):
        assert KnowledgeIndexGuard._is_knowledge_doc(DOC)
        assert KnowledgeIndexGuard._is_knowledge_doc(
            "/workspace/FlagScale-Agent/" + DOC)

    def test_non_markdown_excluded(self):
        assert not KnowledgeIndexGuard._is_knowledge_doc(
            "flagscale_agent/knowledge/docs/moe_training/config.yaml")

    def test_non_docs_paths_not_detected(self):
        # Agent source / other dirs must never trigger
        assert not KnowledgeIndexGuard._is_knowledge_doc(
            "flagscale_agent/react/guard/knowledge_index.py")
        assert not KnowledgeIndexGuard._is_knowledge_doc(
            "flagscale_agent/knowledge/indexes/know-moe-training.idx")
        assert not KnowledgeIndexGuard._is_knowledge_doc("")
        assert not KnowledgeIndexGuard._is_knowledge_doc(
            "/workspace/notes/readme.md")


class TestKnowledgeIndexGuardBehavior:
    def test_inject_on_knowledge_doc_edit(self, guard, make_ctx):
        verdict = guard.check_post(make_ctx("edit_file", DOC))
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "knowledge_index_rebuild"
        assert "generate_index" in verdict.message

    def test_fires_on_write_file_too(self, guard, make_ctx):
        verdict = guard.check_post(make_ctx("write_file", DOC))
        assert verdict is not None
        assert verdict.action == "inject"

    def test_deduplicated_within_turn(self, guard, make_ctx):
        """Second edit in the same turn must NOT re-inject."""
        first = guard.check_post(make_ctx("edit_file", DOC))
        second = guard.check_post(make_ctx("edit_file",
                                           DOC.replace("01_", "02_")))
        assert first is not None
        assert second is None

    def test_reinjects_after_reset_turn(self, guard, make_ctx):
        guard.check_post(make_ctx("edit_file", DOC))
        guard.reset_turn()
        verdict = guard.check_post(make_ctx("edit_file", DOC))
        assert verdict is not None

    def test_ignores_non_write_tools(self, guard, make_ctx):
        ctx1 = make_ctx("read_file", DOC)
        assert guard.check_post(ctx1) is None
        ctx2 = make_ctx("shell", "cat " + DOC)
        assert guard.check_post(ctx2) is None
        assert len(guard._pending_docs) == 0

    def test_no_inject_for_non_knowledge_paths(self, guard, make_ctx):
        assert guard.check_post(
            make_ctx("edit_file", "flagscale_agent/react/agent.py")) is None
        assert guard.check_post(
            make_ctx("edit_file", "docs/readme.md")) is None

    def test_reset_turn_clears_state(self, guard, make_ctx):
        guard.check_post(make_ctx("edit_file", DOC))
        guard.reset_turn()
        assert not guard._injected_this_turn
        assert len(guard._pending_docs) == 0

    def test_check_pre_is_none(self, guard, make_ctx):
        assert guard.check_pre(make_ctx("edit_file", DOC)) is None
