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

"""Shared utilities for guards."""

from __future__ import annotations

import re
import shlex


# Tools that only read state and never modify anything.
# Used by guards to distinguish exploratory actions from mutations.
READ_ONLY_TOOLS = frozenset({
    "read_file", "memory_read", "memory_list", "recall",
    "load_skill", "load_knowledge", "plan_status",
    "inspect_checkpoint", "web_fetch", "flagscale_train_monitor",
})


# ---------------------------------------------------------------------------
# Launch command detection
# ---------------------------------------------------------------------------

# Sub-commands / flags that mean "not a real (long-running) launch".
_NON_RUN_FLAGS = ("--stop", "--dryrun", "--test", "--query", "--tune")
_RUN_NON_ACTIONS = ("dryrun", "stop", "test", "query", "auto_tune")

# ssh options that consume the following token as their argument.
_SSH_OPTIONS_WITH_ARG = frozenset({
    "-b", "-c", "-D", "-e", "-F", "-i", "-J", "-l", "-L", "-m",
    "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
})

# How many nested exec wrappers (ssh -> bash -c -> ...) to descend into.
_MAX_WRAPPER_DEPTH = 3


def _basename(tok: str) -> str:
    """Return the program basename of a path-like token, lowercased.

    ``$ENV/bin/flagscale`` -> ``flagscale``; ``dev_flagscale`` -> ``dev_flagscale``
    (so a mere substring/word like ``dev_flagscale`` is NOT mistaken for the
    ``flagscale`` program).
    """
    return tok.rsplit("/", 1)[-1].lower()


def _tokens_indicate_launch(tokens: list[str]) -> bool:
    """Decide whether an already-tokenized command performs a FlagScale launch.

    Token-based, not substring-based: a quoted ``"flagscale train"`` argument
    (grep/echo) stays ONE token and therefore never matches, which removes the
    old false-positive class without any fragile quote-stripping.
    """
    # Pattern 3: python[3] run.py ... [--config-name|--config-path] ... action=run
    if any(_basename(t) in ("python", "python3") for t in tokens):
        has_run_py = any(_basename(t) == "run.py" for t in tokens)
        has_cfg = any(
            t in ("--config-name", "--config-path")
            or t.startswith("--config-name=")
            or t.startswith("--config-path=")
            for t in tokens
        )
        if has_run_py and has_cfg and "action=run" in tokens:
            return True

    # Patterns 1 & 2: <path/>flagscale train|run ...
    for i, tok in enumerate(tokens):
        if _basename(tok) != "flagscale" or i + 1 >= len(tokens):
            continue
        sub = tokens[i + 1]
        rest = tokens[i + 2:]
        if sub == "train":
            if any(t == f or t.startswith(f + "=") for t in rest for f in _NON_RUN_FLAGS):
                return False
            return True
        if sub == "run":
            if any(a in rest for a in _RUN_NON_ACTIONS):
                return False
            joined = " ".join(rest)
            if any(
                f"--action {a}" in joined or f"-a {a}" in joined
                or f"--action={a}" in joined or f"-a={a}" in joined
                for a in _RUN_NON_ACTIONS
            ):
                return False
            return True
    return False


def _inner_command(tokens: list[str]) -> str | None:
    """Return the inner command string of a known exec wrapper, else None.

    Handles the production launch shape ``ssh [opts] host '<remote cmd>'`` and
    ``sh|bash|zsh|dash -c '<script>'`` — the wrapper body is what actually runs
    the launch and must be inspected (the previous quote-stripping erased it).
    """
    if not tokens:
        return None
    prog = _basename(tokens[0])

    if prog == "ssh":
        i = 1
        while i < len(tokens):
            t = tokens[i]
            if t == "--":
                i += 1
                break
            if t.startswith("-") and t != "-":
                if t in _SSH_OPTIONS_WITH_ARG and i + 1 < len(tokens):
                    i += 2
                else:
                    i += 1
                continue
            break
        if i < len(tokens):
            inner = " ".join(tokens[i + 1:])  # everything after the host
            if inner:
                return inner
        return None

    if prog in ("sh", "bash", "zsh", "dash") and "-c" in tokens:
        j = tokens.index("-c")
        if j + 1 < len(tokens):
            return " ".join(tokens[j + 1:])

    return None


def _is_flagscale_launch_command(cmd: str, _depth: int = 0) -> bool:
    """Detect FlagScale training launch commands.

    Supports compound commands (cd xxx && flagscale train ...).

    Detection is shell-AWARE, not substring-based:
      * Heredoc bodies are stripped first, so a script written with
        ``cat > x <<EOF ... flagscale train ... EOF`` is not misread as a launch
        (the launch token lives in the body text, not the executed command line).
      * The remaining command is tokenized with ``shlex``; a quoted
        ``"flagscale train"`` argument (grep/echo) stays a single token and does
        not match — there is no fragile quote-stripping to get wrong.
      * Known exec wrappers (``ssh host '<cmd>'``, ``bash -c '<cmd>'``) are
        descended into recursively, so the production launch shape
        ``ssh ... '... && flagscale train ...'`` IS detected (previously the
        single-quoted remote command was erased and the launch was missed).
    """
    if not isinstance(cmd, str):
        return False

    cmd_lower = cmd.lower()

    # Remove heredoc bodies first (<<[-]TAG ... line-with-TAG). Must run before
    # tokenizing: the body may contain a literal launch line that is data, not
    # an executed command.
    cmd_lower = re.sub(
        r"<<-?\s*['\"]?(\w+)['\"]?\b.*?^\1[ \t]*$",
        "",
        cmd_lower,
        flags=re.MULTILINE | re.DOTALL,
    )

    try:
        tokens = shlex.split(cmd_lower)
    except ValueError:
        # Unbalanced quotes — fall back to a whitespace split. Basename checks
        # still avoid the classic substring false positive.
        tokens = cmd_lower.split()

    if _tokens_indicate_launch(tokens):
        return True

    if _depth < _MAX_WRAPPER_DEPTH:
        inner = _inner_command(tokens)
        if inner:
            return _is_flagscale_launch_command(inner, _depth + 1)

    return False
