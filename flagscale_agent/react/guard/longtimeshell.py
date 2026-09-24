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

"""LongTimeShellGuard — pre-check block on long foreground sleep/timeout.

The agent has repeatedly issued multi-minute foreground sleeps (`sleep 180`,
`sleep 60`) and long `timeout N ...` waits instead of backgrounding the job and
polling with shell_jobs — burning wall-clock while doing nothing, the exact
anti-pattern the Tool Guide's background doctrine forbids. It is also the one
cost lever a cheap intervention can move: a foreground `sleep 300` is five
minutes of pure idle burn.

This guard fires BEFORE execution on foreground (non-background) shell calls
containing `sleep N...` or `timeout [opts] N <cmd>` whose duration exceeds
THRESHOLD seconds (strictly greater). `background=true` is always allowed —
the doctrine's answer to long waits is a job handle + polling, not avoidance.
Short sleeps (<= threshold) stay legal: brief settle/poll delays are a
legitimate pattern.

Fires on: sleep 31 / sleep 1m / sleep 1.5h / timeout 45 cmd / timeout -k 5 45 cmd
Never fires on: sleep 30 / sleep 30.0 / sleep 0.5s / timeout 30 cmd /
  background=true / `grep timeout file` / `echo sleep 999` / `alias sleep 5`
  (word is not in command position there — position is determined by the
  preceding delimiter, see _split_segments).
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Foreground sleep/timeout longer than this is blocked (strictly greater).
THRESHOLD_SECONDS = 30.0

# Quoted spans are data, not command syntax: strip them BEFORE segment
# splitting, or a documentation string like echo "run && timeout 120 cmd"
# yields a fake command position after the quoted &&. Unterminated quotes
# leave the text intact (regex requires a closing quote) — conservative.
_QUOTED_SPAN_RE = re.compile(r'"[^"]*"|\'[^\']*\'')

# Duration token: number with optional s/m/h/d suffix (coreutils timeout form).
_DURATION_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)?)(?P<suffix>[smhd]?)$", re.IGNORECASE
)

_SUFFIX_TO_SEC = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

# timeout options that CONSUME a following value — these values are not the
# duration. -k/--kill-after take a duration; -s/--signal take a signal spec.
_TIMEOUT_VALUE_FLAGS = (
    "-k", "--kill-after", "-s", "--signal",
)

_LONG_MESSAGE = """[LongTimeShellGuard] Foreground {kind} of {duration:.0f}s (> {thresh:.0f}s) blocked — this burns wall-clock while doing nothing and cannot be interrupted productively.

DO: run it as a background job and poll:
  shell(command="{command}", background=true)   # returns a handle immediately
  shell_jobs(action="wait", job_id="jobN", timeout=30)   # short bounded waits

If the foreground wait is genuinely correct here (e.g. a rate-limit cooldown the
NEXT command depends on, a health-check settle window), proceed with:
  _override_reason: <your concrete reason — any non-trivial explanation is accepted>

A wait is the right foreground tool only when nothing else can proceed in
parallel and the consumer cannot poll a job handle."""


def _parse_duration(token: str) -> float | None:
    """Return seconds for a duration token, else None.

    Accepts `31`, `30.5`, `1m`, `45S`, `1.5h`, `2d` (case-insensitive suffix).
    """
    m = _DURATION_RE.match(token)
    if not m:
        return None
    return float(m.group("num")) * _SUFFIX_TO_SEC[m.group("suffix").lower()]


def _split_segments(command: str) -> list[list[str]]:
    """Split a command line into token lists at shell delimiters.

    Splits on `;`, `&&`, `||`, `|` and newlines so that a word's command
    position is judged per segment: in `grep timeout file | echo sleep 999`
    neither word is in command position of its own segment.
    """
    segments: list[list[str]] = []
    for seg in re.split(r"(?:&&|\|\||[;|\n])", _QUOTED_SPAN_RE.sub("", command)):
        toks = seg.split()
        if toks:
            segments.append(toks)
    return segments


def _scan_timeout(toks: list[str]) -> float | None:
    """Duration of a timeout invocation if `timeout` is in command position.

    Command position = first token of the segment (env-var assignments like
    `FOO=1 timeout 45 cmd` are handled by skipping leading VAR=VAL tokens).
    Returns the duration in seconds, or None when absent/not-a-duration.
    """
    i = 0
    # Skip leading VAR=VAL env assignments.
    while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
        i += 1
    if i >= len(toks) or toks[i] != "timeout":
        return None
    i += 1
    # Walk options; values of value-taking flags are not the duration.
    while i < len(toks):
        tok = toks[i]
        if tok.startswith("-"):
            if tok in _TIMEOUT_VALUE_FLAGS:
                i += 2  # flag + its value
            elif tok.startswith("--") and "=" in tok and tok.split("=", 1)[0] in (
                "--kill-after", "--signal",
            ):
                i += 1  # --flag=value form
            else:
                i += 1  # boolean flag (e.g. --preserve-status, -v)
            continue
        return _parse_duration(tok)  # first bare token after flags = duration
    return None


def _scan_sleep(toks: list[str]) -> float | None:
    """Max duration of a sleep invocation in command position.

    `sleep N [N...]` — any argument pushing the total over the threshold is
    enough to fire (coreutils sleep sums its arguments).
    """
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
        i += 1
    if i >= len(toks) or toks[i] != "sleep":
        return None
    total = 0.0
    seen = False
    for tok in toks[i + 1:]:
        d = _parse_duration(tok)
        if d is None:
            break
        total += d
        seen = True
    return total if seen else None


def _find_excessive(command: str,
                    threshold: float = THRESHOLD_SECONDS) -> tuple[str, float] | None:
    """Scan all segments; return (kind, duration) of the first offense.

    Offense = duration strictly greater than `threshold`.
    """
    for toks in _split_segments(command):
        t = _scan_sleep(toks)
        if t is not None and t > threshold:
            return ("sleep", t)
        t = _scan_timeout(toks)
        if t is not None and t > threshold:
            return ("timeout", t)
    return None


class LongTimeShellGuard(Guard):
    """Block long foreground sleep/timeout; background=true passes freely."""

    name = "longtimeshell"
    priority = 16  # next to VcsBackupGuard(15); before generic shell safety

    def __init__(self, threshold: float = THRESHOLD_SECONDS):
        self.threshold = threshold
        # Offense acknowledged via a valid override this turn (kind, duration).
        self._acked: tuple | None = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        # background=true is the doctrine's answer to long waits — never block.
        if bool(ctx.tool_args.get("background", False)):
            return None
        command = str(ctx.tool_args.get("command", ""))
        if not command:
            return None
        found = _find_excessive(command, self.threshold)
        if found is None or found == self._acked:
            return None
        kind, duration = found
        example = command.strip().split("\n")[0][:120]
        return GuardVerdict.block(
            message=_LONG_MESSAGE.format(
                kind=kind, duration=duration, thresh=self.threshold,
                command=example.replace('"', "'"),
            ),
            reason=f"long_foreground_{kind}_{duration:.0f}s",
            category="longtimeshell",
            overridable=True,
        )

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Trust the LLM's override reason — no keyword validation (base-class
        default: any non-trivial reason releases). One offense per override:
        a different command re-blocks."""
        if not reason or len(reason.strip()) <= 5:
            return False
        found = _find_excessive(
            str(ctx.tool_args.get("command", "")) if ctx else "", self.threshold
        )
        if found is not None:
            self._acked = found
        return True

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        self._acked = None
