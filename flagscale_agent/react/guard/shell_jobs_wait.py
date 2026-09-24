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

"""ShellJobsWaitGuard — block long blocking waits on background jobs.

`shell_jobs(action="wait", timeout=N)` blocks the agent until the job exits or
N seconds elapse. The whole point of backgrounding a long job is to OVERLAP it
with other useful work (read files, edit code, prepare the next step, verify a
different assumption). A long single `wait` throws that away: the agent sits idle
for minutes, doing nothing, and just re-polls — synchronous blocking wearing a
costume. The default timeout is 60s precisely so the agent stays in the loop and
re-evaluates frequently.

This guard blocks any `wait` with timeout > 60s and points the agent back at the
right pattern: a SHORT bounded wait (<=60s) in a loop with real work between
checks, or a non-blocking `poll`. It is overridable: when every dependent step is
already written and staged AND the output contract is confirmed, a single longer
join can be justified with an _override_reason.

Only `action="wait"` is affected. `poll` (non-blocking), `list`, and `kill` are
never touched.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Default/allowed maximum bounded wait. A wait at or below this keeps the agent in
# the supervision loop; above it, the agent is effectively leaving the room.
_MAX_WAIT_SEC = 60


def _coerce_int(value) -> int | None:
    """Best-effort int coercion for a timeout arg that may arrive as int/float/str."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


_MESSAGE = (
    "[ShellJobsWaitGuard] A `wait` with timeout > {max}s blocks you idle for the "
    "whole duration — that is synchronous blocking wearing a costume, not "
    "concurrency. The point of backgrounding a job is to OVERLAP it with other "
    "useful work.\n"
    "\n"
    "Prefer, in order:\n"
    "  1. `poll` (non-blocking) — check incremental output and return immediately, "
    "then keep doing real work.\n"
    "  2. A SHORT bounded `wait` (timeout <= {max}) in a loop — check in, "
    "re-evaluate, do other work between checks. This keeps you supervising the job "
    "so you can react to the first sign of trouble.\n"
    "\n"
    "While the job runs, PREPARE the dependent steps now: write the post-processing "
    "/ eval / conversion script, re-read the task to re-list every output CONSTRAINT "
    "(size, format, path, metric threshold), validate the expected output will "
    "satisfy them. That is the highest-value use of the wait.\n"
    "\n"
    "Only when every dependent step is written and staged AND the output contract is "
    "confirmed is a longer join justified. If that is genuinely the case, override "
    "with _override_reason stating what downstream work is already staged and why no "
    "shorter wait works."
)


class ShellJobsWaitGuard(Guard):
    """Block `shell_jobs(action="wait", timeout>60)`; overridable when justified."""

    name = "shell_jobs_wait"
    priority = 25

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell_jobs":
            return None

        action = ctx.tool_args.get("action")
        if action != "wait":
            return None

        timeout = _coerce_int(ctx.tool_args.get("timeout"))
        # Missing/unparseable timeout defaults to 60s (the tool default) — allow it.
        if timeout is None or timeout <= _MAX_WAIT_SEC:
            return None

        return GuardVerdict.block(
            _MESSAGE.format(max=_MAX_WAIT_SEC),
            reason="long_blocking_wait",
            category="shell_jobs_wait",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None
