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

"""Integration test: the activity string handed to the LLM health judge must
report WHOLE-TREE CPU% and memory, not the parent-only snapshot.

Regression for insight/flagscale_agent/health_parent_only_cpu_mem_rootcause:
a launcher/wrapper parent (bash sleep, a training driver forking workers) reads
~0% CPU and ~2 MB RSS on its own while children saturate cores and use GBs. The
old activity string surfaced those parent-only numbers, so the judge repeatedly
reported "CPU is 0% with only 2 MB memory" and false-flagged healthy jobs.
"""

from flagscale_agent.react.tools.shell import ShellTool


def test_activity_string_reports_tree_cpu_and_memory():
    """
    A bash wrapper that itself does nothing (sleep) but forks a CPU-burning,
    memory-holding Python child. The parent is ~0% CPU / ~2 MB RSS; the tree is
    busy. Capture what the judge receives.
    """
    captured = []

    def fake_judge(command, recent_text, time_str, **kwargs):
        captured.append(kwargs.get("activity", ""))
        return {"kill": False}

    # remind_interval=1 forces the monitor loop to check every ~1s and the
    # sampler to sample fast, so we get fresh delta-based samples in-window.
    tool = ShellTool(remind_interval=1, health_judge_fn=fake_judge)

    # Wrapper parent is idle (sleep); child burns CPU and holds ~30MB for 6s.
    child = (
        "python3 -c \""
        "import time\n"
        "buf = bytearray(30*1024*1024)\n"
        "t=time.time()\n"
        "while time.time()-t < 6:\n"
        "    x=0\n"
        "    for i in range(200000): x+=i\""
    )
    command = f"{child} & sleep 7"

    tool.execute(command=command, _quiet=True)

    assert captured, "judge never received an activity string"

    tree_cpu_values = []
    mem_values = []
    for act in captured:
        assert "whole process tree" in act, f"activity not tree-labeled: {act}"
        assert "whole tree" in act
        try:
            tree_cpu_values.append(float(act.split("CPU")[1].split("%")[0].strip()))
        except (IndexError, ValueError):
            pass
        mem_values.append(float(act.split("memory")[1].split("MB")[0].strip()))

    # The busy child drives tree CPU well above 0% on at least one sample.
    assert max(tree_cpu_values) > 20, (
        f"tree CPU% never reflected the busy child: {tree_cpu_values}"
    )
    # Tree memory should exceed the parent-only ~2MB floor on some sample.
    assert max(mem_values) > 15, (
        f"tree memory never reflected the child's ~30MB: {mem_values}"
    )
