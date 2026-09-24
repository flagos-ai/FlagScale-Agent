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

"""Guards the hard-reset summary against dropping task constraints.

Root cause (pov-ray rerun 2026-08-23): auto-compaction summarized TASK/
PROGRESS/STATE/NEXT/CONTEXT but NOT the negative constraints (approaches
already ruled out). After compaction the agent re-inferred the task from
leftover files and reverted to a degraded approach it had already rejected
(building v3.7 when the task required v2.2). These tests assert the summary
prompt and the programmatic fallback both preserve ruled-out approaches.
"""

import inspect

from flagscale_agent.react.agent import WorkerAgent as Agent


class TestHardResetSummaryConstraints:
    def test_llm_summary_prompt_requires_ruled_out_approaches(self):
        """The LLM summary prompt must ask for constraints + ruled-out approaches."""
        src = inspect.getsource(Agent._generate_hard_reset_summary)
        assert "RULED-OUT" in src, "summary prompt must request ruled-out approaches"
        assert "CONSTRAINTS" in src, "summary prompt must request constraints section"
        # Must explain the failure mode it defends against
        assert "re-infer" in src, "prompt must warn about re-inferring task after compaction"

    def test_llm_summary_prompt_requires_verbatim_constraints(self):
        """Task hard constraints must be preserved verbatim through compaction."""
        src = inspect.getsource(Agent._generate_hard_reset_summary)
        assert "VERBATIM" in src, "task constraints must be preserved verbatim"

    def test_programmatic_fallback_has_constraints_warning(self):
        """The LLM-free fallback summary must warn to recover constraints."""
        src = inspect.getsource(Agent._build_programmatic_summary)
        assert "CONSTRAINTS WARNING" in src, "fallback must carry a constraints warning"
        assert "ruled out" in src.lower(), "fallback must mention ruled-out approaches"
        assert "memory" in src.lower(), "fallback must point agent to memory for recovery"

    def test_no_cjk_in_summary_methods(self):
        """Prompt wording must stay ASCII (no CJK corruption)."""
        for fn in (Agent._generate_hard_reset_summary, Agent._build_programmatic_summary):
            src = inspect.getsource(fn)
            for ch in src:
                assert not ("\u4e00" <= ch <= "\u9fff"), f"CJK char found: {ch!r}"
