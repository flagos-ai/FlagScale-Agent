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

"""Watchdog for a wedged interactive prompt.

Long interactive sessions occasionally end up sitting at the prompt (``>``)
while the input loop no longer consumes keystrokes: the process is alive and
the pane still renders, but typing does nothing (``tmux send-keys`` is
ignored). From the outside this looks like a frozen terminal; from the inside
the ``prompt_toolkit`` event loop is wedged on its input fd (observed as a
main thread parked in ``ep_poll`` with bytes available on the tty that were
never read).

This watchdog runs on a daemon thread and watches for exactly that condition:
input is waiting but unconsumed while a prompt is supposed to be showing, and
that state persists for ``threshold`` seconds. "Waiting" is detected two ways,
OR-ed: bytes still readable on stdin (the kernel-buffer wedge), and a
non-empty userspace backlog via an injected ``pending_probe`` (the second wedge
class, where prompt_toolkit's reader thread already consumed the keystroke — so
``select`` on the fd is blind to it — yet the loop stalled before dispatching
it). On detection it first nudges the terminal with ``SIGWINCH`` (a cheap,
harmless re-arm); if the input is *still* unconsumed after ``winch_grace`` more
seconds it escalates to ``SIGINT``, which breaks the wedged ``prompt()`` so the
REPL can rebuild a fresh :class:`PromptSession` and accept input again.

Properties that make this safe:
  * It never fires during a turn (guarded by ``is_at_prompt``), so a long
    model call or tool run cannot trip it.
  * It never steals input: it only *queries* readability with ``select``;
    ``prompt_toolkit`` performs the actual read. When the loop is healthy it
    consumes each keystroke immediately, so the pending window resets and the
    watchdog stays silent.
  * All effects are injectable, so the decision logic is unit-testable without
    a real tty (see ``tests/test_prompt_watchdog.py``).
"""

import os
import select
import signal
import threading
import time


class PromptWatchdog:
    """Detects an interactive prompt whose input loop has stopped consuming
    keystrokes, and escalates (SIGWINCH -> SIGINT) to unstick it."""

    def __init__(
        self,
        is_at_prompt,
        on_sigint=None,
        fd=None,
        *,
        pending_probe=None,
        interval=2.0,
        threshold=60.0,
        winch_grace=15.0,
        select_fn=select.select,
        monotonic=time.monotonic,
        kill_fn=os.kill,
        getpid=os.getpid,
        logger=None,
    ):
        self._is_at_prompt = is_at_prompt
        self._on_sigint = on_sigint
        self._pending_probe = pending_probe
        self._fd = fd if fd is not None else self._default_fd()
        self._interval = interval
        self._threshold = threshold
        self._winch_grace = winch_grace
        self._select = select_fn
        self._monotonic = monotonic
        self._kill = kill_fn
        self._getpid = getpid
        self._logger = logger or (lambda msg: None)
        self._pending_since = None
        self._nudged_at = None
        self._fired_winch = False
        self._fired_sigint = False
        self._thread = None

    @staticmethod
    def _default_fd():
        try:
            import sys

            return sys.stdin.fileno()
        except Exception:
            return None

    def _input_pending(self) -> bool:
        """True iff input is known to be waiting but not consumed.

        Two independent signals, OR-ed:

        * **Kernel bytes** — ``select([fd])`` reports the tty readable. Covers
          the classic wedge where bytes sit in the kernel buffer and the loop
          never reads them.
        * **Userspace backlog** — ``pending_probe()`` reports keys that were
          *already read off the fd* but not yet processed (prompt_toolkit's
          ``KeyProcessor.input_queue`` / ``Vt100Input._buffer``). Covers the
          second wedge class where the reader thread consumed the keystroke, so
          the fd is no longer readable, yet the application stalled before
          dispatching it — the exact case ``select`` alone is blind to.
        """
        if self._fd is not None:
            try:
                r, _, _ = self._select([self._fd], [], [], 0)
                if r:
                    return True
            except Exception:
                pass
        if self._pending_probe is not None:
            try:
                if self._pending_probe():
                    return True
            except Exception:
                pass
        return False

    def tick(self):
        """Run one evaluation.

        Returns the action taken: ``None``, ``"winch"`` or ``"sigint"``.
        Kept side-effect-injectable so tests can drive it deterministically.
        """
        # Only act while we believe a prompt is showing.
        if not self._is_at_prompt():
            self._reset()
            return None

        now = self._monotonic()

        if not self._input_pending():
            # Healthy: nothing waiting, or the loop already consumed it.
            self._reset()
            return None

        if self._pending_since is None:
            self._pending_since = now
            return None

        # Stage 1 — gentle re-arm. Some prompt_toolkit wedges (e.g. after a
        # terminal resize desync) clear on a fresh SIGWINCH.
        if not self._fired_winch and (now - self._pending_since) >= self._threshold:
            self._fired_winch = True
            self._nudged_at = now
            self._kill(self._getpid(), signal.SIGWINCH)
            self._logger(
                "prompt watchdog: stdin readable but unconsumed for "
                f"{now - self._pending_since:.0f}s — sent SIGWINCH to re-arm"
            )
            return "winch"

        # Stage 2 — escalate. The nudge did not help, so break the wedged
        # prompt(); the REPL catches KeyboardInterrupt and rebuilds.
        if (
            self._fired_winch
            and not self._fired_sigint
            and self._nudged_at is not None
            and (now - self._nudged_at) >= self._winch_grace
        ):
            self._fired_sigint = True
            if self._on_sigint is not None:
                try:
                    self._on_sigint()
                except Exception:
                    pass
            self._kill(self._getpid(), signal.SIGINT)
            self._logger(
                "prompt watchdog: input still unconsumed — sent SIGINT to break "
                "the wedged prompt and rebuild the session"
            )
            return "sigint"

        return None

    def _reset(self):
        self._pending_since = None
        self._nudged_at = None
        self._fired_winch = False
        self._fired_sigint = False

    def _loop(self):
        while True:
            time.sleep(self._interval)
            try:
                self.tick()
            except Exception:
                pass

    def start(self):
        """Start the background watcher (idempotent)."""
        if (self._fd is None and self._pending_probe is None) or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="prompt-watchdog"
        )
        self._thread.start()
