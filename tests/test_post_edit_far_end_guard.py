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

"""Tests for PostEditFarEndGuard — the every-successful-edit far-end reminder."""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_edit_far_end import PostEditFarEndGuard


def _ctx(tool_name="edit_file", path="a.py", result="Successfully edited a.py"):
    return GuardContext(
        tool_name=tool_name,
        tool_args={"path": path},
        tool_result=result,
    )


@pytest.fixture
def guard():
    return PostEditFarEndGuard()


class TestFiresOnSuccessOnly:
    def test_py_edit_success_fires_with_py_compile_hint(self, guard):
        v = guard.check_post(
            _ctx(path="flagscale_agent/react/guard/unit_test.py",
                 result="Successfully edited flagscale_agent/react/guard/unit_test.py")
        )
        assert v is not None
        assert v.action == "inject"
        assert "py_compile" in v.message
        assert "/reload" in v.message  # agent-source reload branch
        assert "flagscale_agent/" in v.message

    def test_yaml_write_success_fires_with_parse_hint_no_reload(self, guard):
        v = guard.check_post(
            _ctx(tool_name="write_file", path="cfg/exp.yaml",
                 result="Wrote 100 chars to cfg/exp.yaml (total file size: 100 bytes)")
        )
        assert v is not None and v.action == "inject"
        assert "yaml.safe_load" in v.message
        assert "/reload" not in v.message  # not agent source

    def test_sh_edit_success_fires_with_bash_n(self, guard):
        v = guard.check_post(
            _ctx(path="scripts/run.sh", result="Successfully edited scripts/run.sh")
        )
        assert v is not None and "bash -n" in v.message

    def test_unknown_ext_falls_back_to_re_read(self, guard):
        v = guard.check_post(_ctx(path="README.md", result="Successfully edited README.md"))
        assert v is not None and "re-read" in v.message

    def test_uppercase_extension_matched(self, guard):
        v = guard.check_post(_ctx(path="DATA.JSON", result="Successfully edited DATA.JSON"))
        assert v is not None and "json.load" in v.message


class TestSilentCases:
    @pytest.mark.parametrize("result", [
        "ERROR: old_string not found in a.py",
        "ERROR: Cannot write to protected system path: /etc/hosts",
        "Error executing tool: boom",
        "",
        None,
    ])
    def test_error_or_empty_results_stay_silent(self, guard, result):
        assert guard.check_post(_ctx(result=result)) is None

    def test_non_write_tool_stays_silent(self, guard):
        assert guard.check_post(_ctx(tool_name="shell", path="x.py")) is None
        assert guard.check_post(_ctx(tool_name="shell")) is None

    def test_missing_path_stays_silent(self, guard):
        ctx = GuardContext(tool_name="write_file", tool_args={},
                           tool_result="Wrote 5 chars to x (total file size: 5 bytes)")
        assert guard.check_post(ctx) is None

    def test_pre_check_never_blocks(self, guard):
        # Inject-only contract: check_pre returns None regardless of input.
        assert guard.check_pre(_ctx()) is None
        assert guard.check_pre(GuardContext(tool_name="write_file",
                                            tool_args={"path": "/etc/passwd"},
                                            tool_result="x")) is None


class TestCategoryIndependence:
    def test_category_is_post_edit_far_end(self, guard):
        v = guard.check_post(_ctx())
        assert v is not None
        assert v.category == "post_edit_far_end"
        # Not colliding with the other post-edit guard's category
        assert v.category != "unit_test_reminder"

    def test_fires_every_edit_no_latch(self, guard):
        # User decision: fire on EVERY successful edit — no per-path latch,
        # no suppression across repeats of the same file.
        for _ in range(3):
            v = guard.check_post(_ctx(path="same.py"))
            assert v is not None, "must re-fire on every successful edit"

    def test_reset_turn_is_safe_noop(self, guard):
        guard.reset_turn()
        assert guard.check_post(_ctx()) is not None


class TestRegistryWiring:
    def test_registered_in_agent_kernel(self):
        # The agent's kernel must wire the guard so it actually runs in prod.
        from flagscale_agent.react import agent as agent_mod
        import inspect
        src = inspect.getsource(agent_mod)
        assert "PostEditFarEndGuard" in src
        # both import and registration present
        assert "from flagscale_agent.react.guard.post_edit_far_end import" in src
        assert "guard_registry.register(PostEditFarEndGuard())" in src

    def test_registry_dedup_keeps_distinct_categories(self):
        from flagscale_agent.react.guard import GuardRegistry
        from flagscale_agent.react.guard.unit_test import UnitTestGuard
        reg = GuardRegistry()
        reg.register(PostEditFarEndGuard())
        reg.register(UnitTestGuard())
        # Both guards survive registration
        names = [g.name for g in reg.guards]
        assert "post_edit_far_end" in names and "unit_test_reminder" in names
        v = reg.check_post(_ctx(path="flagscale_agent/react/guard/unit_test.py",
                                result="Successfully edited unit_test.py"))
        # merged inject from both guards (each has its own category); the
        # unit-test reminder needs >=2 sources so only the far-end one fires here
        assert v is not None
        assert "FAR end" in v.message

