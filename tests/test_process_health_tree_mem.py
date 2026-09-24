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

"""Test process health tree-wide memory accounting.

Regression test for insight/flagscale_agent/health_parent_only_cpu_mem_rootcause:
get_process_health used to report only the PARENT process RSS (e.g. ~2MB for a
bash wrapper) even while its children used gigabytes, causing the LLM judge to
repeatedly (mis)report "CPU is 0% with only 2 MB memory" and false-flag healthy
launcher/wrapper jobs as hung. memory_mb now sums RSS across the walked tree.
"""

import os
import subprocess
import time

import pytest

from flagscale_agent.react.tools.process_health import get_process_health


def test_tree_memory_sums_children():
    """
    Launch a parent that spawns child workers, verify memory_mb reflects the
    TREE footprint, not just the parent shell (~2MB).
    """
    # Parent shell script spawns two Python children that each allocate ~10MB.
    script = """
import time
data = bytearray(10 * 1024 * 1024)  # 10MB
time.sleep(10)
"""
    wrapper = f"""
python3 -c '{script}' &
python3 -c '{script}' &
sleep 15
"""
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give children time to allocate memory
        time.sleep(2)
        
        health = get_process_health(proc.pid)
        
        # Parent bash shell RSS should be tiny (~2MB)
        assert health['memory_mb_parent'] < 10, (
            f"Parent-only RSS {health['memory_mb_parent']:.1f} MB unexpectedly high"
        )
        
        # Tree RSS should include the two 10MB children (~20MB + parent overhead).
        # Allow margin for interpreter overhead and OS variance.
        assert health['memory_mb'] > 15, (
            f"Tree RSS {health['memory_mb']:.1f} MB did not sum children "
            f"(parent {health['memory_mb_parent']:.1f} MB, "
            f"{health['num_children']} children)"
        )
        
        # Verify we actually walked children
        assert health['num_children'] >= 2, "Should have spawned 2+ children"
        
    finally:
        proc.kill()
        proc.wait()


def test_tree_memory_single_process():
    """Single process (no children): memory_mb should equal memory_mb_parent."""
    # A simple sleep has no children.
    proc = subprocess.Popen(
        ["sleep", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.5)
        health = get_process_health(proc.pid)
        
        # No children: tree RSS == parent RSS
        assert health['num_children'] == 0
        assert health['memory_mb'] == health['memory_mb_parent']
        
    finally:
        proc.kill()
        proc.wait()


def test_tree_memory_dead_process():
    """Dead process: memory_mb_parent and memory_mb both 0.0."""
    proc = subprocess.Popen(["sleep", "0.01"])
    proc.wait()  # let it exit
    
    health = get_process_health(proc.pid)
    assert health['alive'] is False
    assert health['memory_mb'] == 0.0
    assert health['memory_mb_parent'] == 0.0
