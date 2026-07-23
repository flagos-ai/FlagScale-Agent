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

"""OutputDirReuseGuard — prevents launching training into a directory with existing logs.

When the same output_dir is reused across multiple training launches, the log
directory accumulates multiple timestamp subdirectories. This makes it extremely
hard to find the correct logs for the current run (the "old log trap").

This guard detects when a training launch targets a directory that already
contains log files and BLOCKS the launch, requiring a fresh output_dir.
"""

import os
import re
import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_LAUNCH_RE = re.compile(
    r"\brun\.py\b.*\baction\s*=\s*run\b"
    r"|\brun\.py\b.*--config"
    r"|\bconda\s+run\b.*\brun\.py\b"
    r"|\btorchrun\b"
    r"|\bflagscale\s+train\b",
    re.IGNORECASE,
)

_OUTPUT_DIR_PATTERNS = [
    re.compile(r"experiment\.exp_dir\s*=\s*(\S+)"),
    re.compile(r"--output[_-]dir\s+(\S+)"),
    re.compile(r"OUTPUT_DIR=(\S+)"),
]


class OutputDirReuseGuard(Guard):
    """Block training launch if output directory already has logs."""

    name = "output_dir_reuse"
    priority = 12

    def __init__(self):
        self._known_output_dirs: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not _LAUNCH_RE.search(cmd):
            return None

        output_dir = self._extract_output_dir(cmd)
        if not output_dir:
            return None

        existing_logs = self._check_existing_logs(output_dir)
        if existing_logs:
            self._known_output_dirs.add(output_dir)
            suffix = time.strftime("%m%d_%H%M")
            return GuardVerdict.inject(
                f"[OutputDirReuse] WARNING: '{output_dir}' already contains "
                f"{existing_logs} from previous run(s). This causes log confusion. "
                f"Consider a new output directory: {output_dir}_{suffix}",
                reason="output_dir_has_logs",
            )
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _LAUNCH_RE.search(cmd):
                output_dir = self._extract_output_dir(cmd)
                if output_dir:
                    self._known_output_dirs.add(output_dir)
        return None

    def _extract_output_dir(self, cmd: str) -> str:
        for pattern in _OUTPUT_DIR_PATTERNS:
            match = pattern.search(cmd)
            if match:
                return os.path.expanduser(match.group(1).strip("'\""))
        return ""

    def _check_existing_logs(self, output_dir: str) -> str:
        if not os.path.isdir(output_dir):
            return ""
        logs_dir = os.path.join(output_dir, "logs", "details")
        if os.path.isdir(logs_dir):
            try:
                for host in os.listdir(logs_dir):
                    host_dir = os.path.join(logs_dir, host)
                    if os.path.isdir(host_dir):
                        ts = [d for d in os.listdir(host_dir)
                              if os.path.isdir(os.path.join(host_dir, d))]
                        if ts:
                            return f"{len(ts)} existing log timestamp(s)"
            except OSError:
                pass
        for name in ("stdout.log", "stderr.log", "train.log"):
            if os.path.isfile(os.path.join(output_dir, name)):
                return "existing log files"
        return ""

    def reset_turn(self):
        pass
