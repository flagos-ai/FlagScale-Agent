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

"""FindGuard — block `find` invocations in favor of internal retrieval.

`find` over a large directory tree (especially / or a shared/NFS mount) is slow
and hammers the filesystem. Most path/file lookups are already answered by the
internal information-retrieval order (conversation logs → memory → scoped grep →
ask the user). This guard blocks any shell command that invokes `find` and
points the agent at those cheaper channels first.

The block is overridable: a genuinely necessary, tightly-scoped find can proceed
with an _override_reason.

Detection is shell-aware so it does not over- or under-block:

  * A `find` command word is recognized at the start of the command OR after a
    shell statement separator: `|`, `;`, `&`, `&&`, `||`, a NEWLINE (a command on
    its own line is a real invocation), or the keywords `then`/`do`. Substrings
    like `findutils`, `myfind`, or `--find` are NOT matched.
  * Quoted regions ('...' / "...") and heredoc bodies (<<EOF ... EOF) are
    stripped BEFORE matching, because a `find` token appearing there is data
    (an echo string, a python docstring piped via a heredoc), not a command
    invocation. Without this, `echo 'a | find b'` would be a false positive.
  * Newlines are statement boundaries (re.MULTILINE), so a bare
    `\\nfind ...` on its own line is blocked — previously such a find escaped
    the guard entirely.
  * A RECURSIVE `grep` whose walk starts at a broad root (`/`, `/mnt`,
    `/public-nvme`, ...) OR at an unbounded cwd-relative target (`.`, `..`,
    `~`, a bare `*` glob, or no target at all) is blocked the same way — a
    whole-tree recursive grep is as slow and filesystem-heavy as a bare find.
    Scoped greps (`grep -rn PATTERN ./src`), non-recursive greps, and greps
    into a specifically named subdirectory pass freely; they are what the
    guard recommends instead.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Match `find` as a command word: at start of the command/line (re.MULTILINE so
# `^` also matches after a newline), or after a shell separator/pipe/keyword
# (so it catches `find ...`, `cd x && find ...`, `... | find`, `x\nfind ...`),
# but NOT substrings like `findutils`, `myfind`, or `--find`.
_FIND_RE = re.compile(
    r"(?:^|[|;&]|&&|\|\||\n|\bthen\b|\bdo\b)\s*find(?:\s|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Heredoc introducer: `<<WORD`, `<<-WORD`, `<<'WORD'`, `<<"WORD"`. The word must
# start with a letter/underscore so a bitshift like `2 << 3` is not mistaken for
# a heredoc.
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?")

# Statement separators inside a shell command (newline included, since a command
# on its own line is a separate statement).
_STMT_SPLIT_RE = re.compile(r"\|\||&&|[;|&\n]")

# Recursive-grep flags. `-r`/`-R` and the long forms, plus combined short flags
# (e.g. `-rn`, `-Rl`) whose letters include r/R.
_GREP_RECURSIVE_LONG = {"--recursive", "--dereference-recursive"}

# A grep is "broad" when its recursive walk starts at one of these roots — a
# whole system tree or the top of a shared/NFS mount. Scanning one of these can
# take minutes and hammer the filesystem. A specific SUBdirectory (e.g.
# /public-nvme/proj/src) is scoped and allowed.
_GREP_BROAD_ROOTS = {
    "/", "/bin", "/boot", "/data", "/etc", "/home", "/lib", "/mnt", "/opt",
    "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
    "/clistorage", "/public-nvme", "/public-mixed",
}

# Targets that make a recursive grep UNBOUNDED relative to the shell's cwd.
# The guard cannot statically know the cwd, so these are conservatively broad:
# a recursive walk from `.`, `..`, `~`, a bare `*` glob, or NO target at all
# (grep defaults to `.`) can cover an entire tree/mount depending on where the
# command runs. A NAMED component (`./src`, `src`, `/path/to/src`) is scoped and
# still passes. One rule per target semantics — the round-2 lesson.
_GREP_CWD_RELATIVE_TARGETS = {
    ".", "..", "~", "*", ".*",
    # always-unbounded env vars — never a scoped subdirectory.
    "$HOME", "${HOME}", "$PWD", "${PWD}", "$OLDPWD", "${OLDPWD}",
}


def _grep_target_is_broad(tok: str) -> bool:
    """True if a grep TARGET token is a broad root or a cwd-relative anchor.

    Broad = an explicit system/shared-mount root (see _GREP_BROAD_ROOTS) OR an
    unbounded cwd-relative anchor: `.`/`..`/`~` (any chain of `..` components),
    a bare `*` glob, `$HOME`-like env vars, and the whole-tree glob forms
    (`./*`, `../*`, `~/*`). A target naming a real component (`./src`, `src`,
    `/public-nvme/proj/src`) is scoped -> False.
    """
    t = tok.rstrip("/") or "/"
    if t in _GREP_BROAD_ROOTS or t in _GREP_CWD_RELATIVE_TARGETS:
        return True
    # `.`/`..`/`~` optionally followed by more `.`/`..`/`*` components, with any
    # number of slashes between them (`..//..` is the same path as `../..`):
    # `.`, `./`, `..`, `../..`, `./.`, `./*`, `../*`, `~/*`.
    if re.fullmatch(r"(?:\.\.?|~)(?:/+(?:\.\.?|\*+))*/*", t):
        return True
    return False


# Prefix words that may precede the real command word (`sudo grep ...`).
_CMD_PREFIXES = {"sudo", "env", "command", "nohup", "time", "nice", "stdbuf"}


# ── Hidden-find detection: payloads handed to remote/shell EXECUTORS ──
#
# The sanitize pass strips quoted regions to avoid false-positives on string
# data (`echo 'a | find b'`). But a quoted region passed to an EXECUTOR is not
# data — it is a payload the executor will run (possibly on a remote host or
# inside a container, i.e. on NFS trees even slower than local ones):
#
#   ssh host "docker exec c bash -lc 'find / -name x'"
#
# After stripping quotes, the guard sees only `ssh` and `head`. This block
# closes that gap: when a statement's command word is an executor, its quoted
# arguments and heredoc body are re-scanned recursively for the same
# violations (find / broad recursive grep), bounded depth.

_EXECUTORS = {
    "ssh", "scp", "sftp", "mosh", "kubectl", "docker", "podman", "nerdctl",
    "ctr", "crictl", "nsenter", "su", "sudo", "doas", "setsid", "stdbuf",
    "nohup", "xargs", "parallel", "env",
}
_SHELL_WRAPPERS = {"bash", "sh", "zsh", "fish", "csh", "tcsh", "ksh", "dash"}

_MAX_PAYLOAD_DEPTH = 6


def _extract_payloads(sanitized: str) -> list[str]:
    """Collect quoted regions from `sanitized` (heredocs already blanked).

    Returns each quoted region's raw contents, so inner layers of nested
    quoting remain visible to the re-scanner (ssh "docker ... 'find ...'").
    """
    payloads: list[str] = []
    i, n = 0, len(sanitized)
    quote = None
    buf: list[str] = []
    while i < n:
        c = sanitized[i]
        if quote is None:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c in ("'", '"'):
                quote = c
                i += 1
                continue
            i += 1
        elif c == "\\" and quote == '"' and i + 1 < n:
            buf.append(c)
            buf.append(sanitized[i + 1])
            i += 2
        elif c == quote:
            payloads.append("".join(buf))
            buf = []
            quote = None
            i += 1
        else:
            buf.append(c)
            i += 1
    if quote is not None and buf:
        payloads.append("".join(buf))
    return payloads


def _statement_command_word(toks: list[str]) -> str | None:
    """First real command word of a token list, skipping VAR=val and prefixes."""
    idx = 0
    while idx < len(toks) and ("=" in toks[idx] or toks[idx] in _CMD_PREFIXES):
        idx += 1
    return toks[idx] if idx < len(toks) else None


def _cmd_base(word: str | None) -> str:
    """Basename of a command word, obfuscation-stripped.

    `/usr/bin/find` -> `find`; `find"` (closing quote glued to the token
    after quote-region splitting) -> `find`; and the round-2 word-splitting
    escapes — quote/backslash/backtick characters are removed from ANYWHERE
    in the word: `f'in'd` -> `find`, `f\\ind` -> `find`, `` `find `` -> `find`.
    `myfind` / `findutils` / `--find` still do NOT normalize to `find`.
    """
    if not word:
        return ""
    # $'...' / $"..." (ANSI-C / locale quoting) — bash expands to the bare word.
    word = word[1:] if word[:1] == "$" else word
    return re.sub(r"['\"\\`]", "", word).rsplit("/", 1)[-1]


def _statement_find_violation(text: str) -> GuardVerdict | None:
    """find as a statement's command word.

    Covers forms the anchored _FIND_RE misses: absolute paths
    (`/usr/bin/find ...`) and prefix forms (`sudo find ...`, `nohup find
    ...`, `VAR=val find ...`). Used at both top level and inside payloads.
    """
    for stmt in _STMT_SPLIT_RE.split(text):
        if _cmd_base(_statement_command_word(stmt.split())) == "find":
            return GuardVerdict.block(
                _FIND_MESSAGE, reason="find_invocation", category="find_guard",
            )
    return None


def _violation_in_shell_text(text: str) -> GuardVerdict | None:
    """Scan an already-sanitized shell fragment for find / broad grep."""
    v = _statement_find_violation(text)
    if v is not None:
        return v
    if _FIND_RE.search(text):
        return GuardVerdict.block(
            _FIND_MESSAGE, reason="find_invocation", category="find_guard",
        )
    if _grep_is_broad(text):
        return GuardVerdict.block(
            _GREP_MESSAGE, reason="broad_recursive_grep", category="find_guard",
        )
    return None


def _cmd_subst_violation(sanitized: str) -> GuardVerdict | None:
    """find inside `$( ... )` or backticks (command substitution executes)."""
    for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", sanitized):
        inner = m.group(1) or m.group(2) or ""
        # Prefix with a dummy separator so a leading find is a command word.
        v = _violation_in_shell_text("dummy_sep; " + inner)
        if v is not None:
            return v
    return None


def _xargs_find_violation(toks: list[str]) -> GuardVerdict | None:
    """find as a bare token after an executor: `xargs find` (stdin-driven
    walk) and `kubectl exec pod -- find ...` (no quoting layer to carry the
    payload, the find token rides bare in the statement)."""
    head = _cmd_base(_statement_command_word(toks))
    if head not in _EXECUTORS and head != "xargs":
        return None
    for tok in toks[1:]:
        if _cmd_base(tok) == "find":
            return GuardVerdict.block(
                _FIND_MESSAGE, reason="find_invocation", category="find_guard",
            )
    return None


def _hidden_violation(cmd: str, depth: int = 0) -> GuardVerdict | None:
    """Detect find / broad grep hidden inside executor payloads.

    `cmd` is the RAW command; this function blanks heredoc bodies itself (a
    heredoc is stdin data, never executed) but KEEPS quoted regions — they
    are payloads an executor (ssh, docker, kubectl, bash -lc, ...) will run,
    often against remote hosts or NFS trees where a stray recursive find is
    the slowest of all.

    Recursion: a payload's own statement may itself be an executor (ssh ->
    docker exec -> bash -lc), so payloads are re-scanned with the same rule,
    bounded by _MAX_PAYLOAD_DEPTH. Pure data payloads (echo / python -c) are
    never re-scanned — their command word is not an executor.
    """
    if depth >= _MAX_PAYLOAD_DEPTH:
        return None
    blanked = _strip_heredocs(cmd)
    for stmt in _STMT_SPLIT_RE.split(blanked):
        toks = _lex_tokens(stmt)
        v = _xargs_find_violation(toks)
        if v is not None:
            return v
        word = _cmd_base(_statement_command_word(toks))
        if word in _EXECUTORS or word in _SHELL_WRAPPERS:
            for payload in _extract_payloads(stmt):
                v = _violation_in_shell_text(payload)
                if v is None:
                    v = _hidden_violation(payload, depth + 1)
                if v is not None:
                    return v
    return _cmd_subst_violation(blanked)


def _grep_is_broad(sanitized: str) -> bool:
    """True if `sanitized` invokes a RECURSIVE grep over a broad root.

    Only recursive greps targeting a whole system tree / shared-mount root are
    flagged. Scoped recursive greps (`grep -rn PATTERN ./src`), non-recursive
    greps (`grep PATTERN /etc/hosts`), and subdirectory targets all pass — the
    guard message itself recommends scoped `grep -rn <pattern> <dir>`.

    `grep` must be the COMMAND word of its statement (optionally after a prefix
    like `sudo`, and with any absolute-path form via _cmd_base:
    `/usr/bin/grep -rn p /data`), so `echo grep -rn foo /` is not mistaken
    for a real grep.
    """
    for stmt in _STMT_SPLIT_RE.split(sanitized):
        toks = stmt.split()
        if not toks:
            continue
        # Locate the command word: skip leading VAR=val assignments and prefixes.
        idx = 0
        while idx < len(toks) and (
            "=" in toks[idx] or toks[idx] in _CMD_PREFIXES
        ):
            idx += 1
        if idx >= len(toks) or _cmd_base(toks[idx]) != "grep":
            continue

        recursive = False
        positionals = []
        for t in toks[idx + 1:]:
            if t in _GREP_RECURSIVE_LONG:
                recursive = True
            elif t.startswith("-") and not t.startswith("--") and re.search(r"[rR]", t[1:]):
                recursive = True
            elif t.startswith("-"):
                continue  # other option (--include=..., -e, --regexp=..., --)
            else:
                positionals.append(t)
        if not recursive:
            continue
        # This layer sees SANITIZED text: a quoted pattern was blanked away, so
        # the first positional is AMBIGUOUS (it may be the pattern). Treat it as
        # a target only when it stands alone; otherwise targets are the
        # positionals after it. The no-target case (grep defaults to `.`) is
        # caught by the quote-preserving lexer path `_toks_recursive_broad`.
        targets = positionals[1:] if len(positionals) >= 2 else positionals
        if any(_grep_target_is_broad(t) for t in targets):
            return True
    return False


def _blank_heredocs(cmd: str) -> tuple[str, list[str]]:
    """Core heredoc pass: blank bodies, return them for re-scanning.

    Returns `(blanked_text, bodies)` where `bodies` are the raw heredoc body
    texts (in order). A heredoc body is DATA for most readers (cat, python),
    but a SCRIPT for a shell/executor reading stdin (`bash <<EOF`,
    `ssh host <<EOF`) — callers re-scan bodies when such a consumer is
    present.
    """
    bodies: list[str] = []
    lines = cmd.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        delims = _HEREDOC_RE.findall(lines[i])
        if not delims:
            i += 1
            continue
        pending = list(delims)
        end = None
        j = i + 1
        while j < n and pending:
            tok = lines[j].strip()
            if tok in pending:
                pending.remove(tok)
                if not pending:
                    end = j
            j += 1
        if end is not None:
            bodies.append("\n".join(lines[i + 1 : end]))
            for k in range(i + 1, end + 1):
                lines[k] = ""
            i = end + 1
        else:
            i += 1
    return "\n".join(lines), bodies


def _strip_heredocs(cmd: str) -> str:
    """Blank out heredoc bodies so `find` inside them is not seen as a command.

    A heredoc body is data fed to a program's stdin, not a shell command, so any
    `find` token there is not an invocation. Only blank lines up to a matching
    terminator; if no terminator is found, leave the text untouched (avoids
    mangling a line that merely contains `<<`).
    """
    return _blank_heredocs(cmd)[0]


def _strip_quoted(cmd: str) -> str:
    """Blank out single/double-quoted regions (respecting backslash escapes).

    A `find` token inside a quoted string is string data, not a command word.
    Newlines inside quotes are blanked too, so an unterminated/spanning quote
    cannot accidentally manufacture a statement boundary.
    """
    out: list[str] = []
    i = 0
    n = len(cmd)
    quote: str | None = None
    while i < n:
        c = cmd[i]
        if quote is None:
            if c == "\\" and i + 1 < n:
                out.append(c)
                out.append(cmd[i + 1])
                i += 2
                continue
            if c in ("'", '"'):
                quote = c
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
        else:
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == quote:
                quote = None
                out.append(" ")
                i += 1
                continue
            out.append(" ")
            i += 1
    return "".join(out)


def _sanitize(cmd: str) -> str:
    """Remove heredoc bodies and quoted regions, leaving only executable text."""
    return _strip_quoted(_strip_heredocs(cmd))


# ── Round-2 hardening: systematic escape-family coverage ──
#
# The layers above (sanitize + anchored regex + payload recursion) miss
# shell-semantics obfuscations. This block adds a lexical scanner over the
# RAW command (quotes preserved) that models how the shell actually splits
# and executes words. Families closed, one rule each:
#
#   word-splitting  f'in'd / f"i"nd / f\ind   -> _lex_tokens drops quote and
#                                                backslash chars INSIDE a word
#   line continuation  fi\<newline>nd         -> _join_continuations upstream
#   subshell/brace/proc-subst                 -> ( ) { } < > are separators
#   variable indirection  X=find; $X ...      -> assignment taint + $VAR use
#   wrapper executors                         -> a bare find/grep token after
#                                                ANY execution wrapper
#                                                (timeout 10 find /, eval $X,
#                                                stdbuf -oL find, kubectl
#                                                exec pod -- grep ...)
#   pipe-to-shell  echo 'find /' | sh         -> tail-shell pipeline scan
#   nested substitution  $(ssh h "find /x")   -> paren separators + payload
#                                                recursion; backtick spans
#                                                extracted and re-scanned
#   heredoc fed to a shell  bash <<EOF        -> bodies re-scanned when any
#                                                stage is a shell/executor
#
# Data-only contexts stay legal (echo / python -c payloads, `cat find`,
# `ls find`, `man find`, scoped greps) — each rule fires only when the
# executed text actually contains a find / broad-recursive-grep invocation,
# and the negative tests pin every data path.

# Execution wrappers: a bare find/grep token after any of these words IS the
# executed program. Scoped to real executors so `cat find`, `ls find`,
# `echo find` (data / file names) stay legal.
_EXEC_WRAPPERS = _EXECUTORS | _SHELL_WRAPPERS | _CMD_PREFIXES | {
    "timeout", "eval", "watch", "strace", "ltrace", "ionice", "taskset",
    "setpriv", "chroot", "unshare", "bwrap", "busybox", "screen", "tmux",
    "script", "expect", "mpirun", "mpiexec", "torchrun", "srun", "deepspeed",
}

# Shell keywords that may precede the real command word (`if ..; then find ..`,
# `for x in ..; do find ..`).
_KEYWORDS = {"then", "do", "else", "elif"}

# `NAME=value` assignment (leading identifier; `--opt=val` does NOT match).
_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)=(.*)$", re.DOTALL)

# A bare `$VAR` / `${VAR}` use.
_VAR_USE_RE = re.compile(r"^\$\{?(\w+)\}?$")

# Pipeline separators: statement breaks AND grouping/redirect chars. Splitting
# on ( ) { } < > makes `(find /)`, `{ find /; }`, `cat <(find /)`,
# `X=$(find /)` all start a sub-statement with `find` as its command word.
_PIPELINE_SEPS = ";&\n(){}"

_BACKTICK_RE = re.compile(r"`([^`]*)`")


def _join_continuations(cmd: str) -> str:
    """Join backslash line continuations: `fi\\<newline>nd` -> `find`."""
    return cmd.replace("\\\n", "")


def _split_outside_quotes(text: str, seps: str) -> list[str]:
    """Split on every char in `seps` that sits OUTSIDE quotes.

    Backslash escapes are carried through untouched (the lexer resolves
    them), so `echo 'a | find b' | wc` keeps its quoted pipe intact.
    """
    parts: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote is not None:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(c)
            cur.append(text[i + 1])
            i += 2
            continue
        if c in seps:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _lex_tokens(text: str) -> list[str]:
    """Shell word lexer: quote-aware, escape-resolving, quote-merging.

    Quoted spans merge into their token (`'find /x'` stays ONE token — quoted
    payloads are handled by payload extraction, not bare-token matching), and
    quote/backslash chars are dropped from the word, so `f'in'd` -> `find`.
    """
    toks: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote is not None:
            if c == "\\" and quote == '"' and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
                i += 1
                continue
            cur.append(c)
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(text[i + 1])
            i += 2
            continue
        if c.isspace():
            if cur:
                toks.append("".join(cur))
                cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        toks.append("".join(cur))
    return toks


def _resolve_var(tok: str, tainted: dict[str, str]) -> str:
    """`$VAR` / `${VAR}` -> the tainted assignment value, else the token."""
    m = _VAR_USE_RE.match(tok)
    if m and m.group(1) in tainted:
        return tainted[m.group(1)]
    return tok


def _toks_recursive_broad(toks: list[str]) -> bool:
    """True if a grep token tail is a RECURSIVE grep over a broad/unbounded root.

    The tokens come from the quote-MERGING lexer, so the pattern is intact: the
    FIRST positional is the PATTERN (grep grammar) and targets are the
    positionals AFTER it. Broad when any target is broad (explicit root OR
    cwd-relative anchor) OR when there is NO target at all — a recursive grep
    with no file operand defaults to `.`, i.e. the cwd, and is unbounded.
    """
    recursive = False
    pattern_supplied = False  # pattern given by -e/--regexp/-f/--file ...
    positionals: list[str] = []
    for t in toks:
        if t in _GREP_RECURSIVE_LONG:
            recursive = True
        elif t.startswith("-") and not t.startswith("--") and re.search(r"[rR]", t[1:]):
            recursive = True
        elif t.startswith("-"):
            # Inline pattern/file forms (`--regexp=PAT`, `-ePAT`, `--file=F`)
            # embed their argument in the option token, so the FIRST positional
            # is then a TARGET, not the pattern. A bare `-e`/`--regexp`/`-f`
            # leaves its argument as the next positional (already dropped by the
            # positionals[1:] rule below), so it does not set this flag.
            if t.startswith("--regexp=") or t.startswith("--file="):
                pattern_supplied = True
            elif len(t) > 2 and t[1] in "ef":
                pattern_supplied = True
            continue  # other option (--include=..., -e, --, ...)
        else:
            positionals.append(t)
    if not recursive:
        return False
    # When the pattern came from an inline option, every positional is a target;
    # otherwise the first positional is the pattern (grep grammar) and targets
    # are those after it.
    targets = positionals if pattern_supplied else positionals[1:]
    if not targets:
        return True
    return any(_grep_target_is_broad(t) for t in targets)


def _shell_tail_reads_stdin(toks: list[str]) -> bool:
    """True for a bare/flag-only shell tail (`| sh`, `| bash -s`) — its stdin
    script executes. A `-c` payload shell runs its own script instead, which
    the executor payload path already scans."""
    for t in toks[1:]:
        b = _cmd_base(t)
        if b in ("--command", "--c") or re.fullmatch(r"-[a-zA-Z]*c", b):
            return False
    return True


def _stage_violation(
    st_text: str, toks: list[str], tainted: dict[str, str], depth: int,
) -> GuardVerdict | None:
    """One pipeline segment: taint, head find/grep, executor bare tokens,
    quoted payload recursion."""
    if not toks:
        return None
    i = 0
    # VAR=val assignments: record taint; a $()/backtick value EXECUTES now.
    while i < len(toks):
        m = _ASSIGN_RE.match(toks[i])
        if not m:
            break
        val = m.group(2)
        if val and ("$(" in val or "`" in val):
            v = _violation_in_shell_text(val) or _scan(val, tainted, depth + 1)
            if v is not None:
                return v
        tainted[m.group(1)] = val
        i += 1
    # Shell keywords before the command word (`; do find $x`).
    while i < len(toks) and _cmd_base(toks[i]) in _KEYWORDS:
        i += 1
    if i >= len(toks):
        return None
    head_tok = toks[i]
    resolved = _resolve_var(head_tok, tainted)
    if resolved != head_tok:
        # The head IS a tainted value: `X=find; $X /x` executes it.
        v = _violation_in_shell_text(resolved) or _scan(resolved, tainted, depth + 1)
        if v is not None:
            return v
    head = _cmd_base(resolved)
    if head == "find":
        return GuardVerdict.block(
            _FIND_MESSAGE, reason="find_invocation", category="find_guard",
        )
    if head == "grep" and _toks_recursive_broad(toks[i + 1 :]):
        return GuardVerdict.block(
            _GREP_MESSAGE, reason="broad_recursive_grep", category="find_guard",
        )
    if head in _EXEC_WRAPPERS:
        # A bare find/grep token after the wrapper IS the executed program:
        # `timeout 10 find /`, `kubectl exec pod -- grep -rn p /data`.
        for j in range(i + 1, len(toks)):
            rtok = _resolve_var(toks[j], tainted)
            if rtok != toks[j]:
                v = _violation_in_shell_text(rtok) or _scan(rtok, tainted, depth + 1)
                if v is not None:
                    return v
            rbase = _cmd_base(rtok)
            if rbase == "find":
                return GuardVerdict.block(
                    _FIND_MESSAGE, reason="find_invocation",
                    category="find_guard",
                )
            if rbase == "grep" and _toks_recursive_broad(toks[j + 1 :]):
                return GuardVerdict.block(
                    _GREP_MESSAGE, reason="broad_recursive_grep",
                    category="find_guard",
                )
        # Quoted payloads the wrapper will execute (ssh / docker / bash -c).
        for payload in _extract_payloads(st_text):
            v = _violation_in_shell_text(payload) or _scan(payload, tainted, depth + 1)
            if v is not None:
                return v
    # Prefix-skipped statement find/grep: `sudo find /x`, `nohup grep -rn p /`.
    k = i
    while k < len(toks) and _cmd_base(toks[k]) in _CMD_PREFIXES:
        k += 1
    if k < len(toks):
        rtok = _resolve_var(toks[k], tainted)
        if _cmd_base(rtok) == "find":
            return GuardVerdict.block(
                _FIND_MESSAGE, reason="find_invocation", category="find_guard",
            )
        if _cmd_base(toks[k]) == "grep" and _toks_recursive_broad(toks[k + 1 :]):
            return GuardVerdict.block(
                _GREP_MESSAGE, reason="broad_recursive_grep", category="find_guard",
            )
    return None


def _scan(text: str, tainted: dict[str, str] | None = None, depth: int = 0) -> GuardVerdict | None:
    """Lexical scanner over the raw command (quotes preserved) — the round-2
    escape-family net. See the block comment above `_EXEC_WRAPPERS` for the
    family -> rule map. Recurses into payloads/backticks with depth bound;
    `tainted` threads assignment values across statements and into payloads.
    """
    if depth >= _MAX_PAYLOAD_DEPTH:
        return None
    if tainted is None:
        tainted = {}
    joined = _join_continuations(text)
    blanked, bodies = _blank_heredocs(joined)

    any_executor_stage = False
    for pipeline in _split_outside_quotes(blanked, _PIPELINE_SEPS):
        stage_texts = _split_outside_quotes(pipeline, "|")
        stage_pairs = [(st, _lex_tokens(st)) for st in stage_texts]
        for st_text, toks in stage_pairs:
            v = _stage_violation(st_text, toks, tainted, depth)
            if v is not None:
                return v
            if toks and _cmd_base(toks[0]) in _EXEC_WRAPPERS:
                any_executor_stage = True
        # Pipe-to-shell: `echo 'find / -name y' | sh` — the tail shell runs
        # the piped text as a script, so earlier stages' quoted payloads are
        # code. Tails with their own -c payload are scanned by that path.
        if len(stage_pairs) > 1:
            tail_toks = stage_pairs[-1][1]
            if (
                tail_toks
                and _cmd_base(tail_toks[0]) in _SHELL_WRAPPERS
                and _shell_tail_reads_stdin(tail_toks)
            ):
                for st_text, _ in stage_pairs[:-1]:
                    for payload in _extract_payloads(st_text):
                        v = _violation_in_shell_text(payload) or _scan(
                            payload, tainted, depth + 1,
                        )
                        if v is not None:
                            return v
    # Backtick command substitution executes its inner text.
    for m in _BACKTICK_RE.finditer(blanked):
        v = _violation_in_shell_text(m.group(1)) or _scan(m.group(1), tainted, depth + 1)
        if v is not None:
            return v
    # Heredoc bodies are SCRIPTS when a shell/executor consumes them
    # (`bash <<EOF`, `ssh host <<EOF`) — data otherwise (cat/python).
    if any_executor_stage:
        for body in bodies:
            v = _violation_in_shell_text(body) or _scan(body, tainted, depth + 1)
            if v is not None:
                return v
    return None


_FIND_MESSAGE = (
    "[FindGuard] `find` on a large directory tree is slow and often the wrong "
    "tool. Before running find, follow the internal information-retrieval order:\n"
    "\n"
    "  1. conversation_full.json / conversation.json (session dir) — grep for the "
    "path/file you're chasing. Near-zero cost, it may already be recorded.\n"
    "  2. memory — memory_list(keyword=...) or memory_read(key='fact/<domain>/'). "
    "A path you discovered before is likely already saved.\n"
    "  3. scoped search tools — locate files by name with a fast indexed search; "
    "search contents with `grep -rn <pattern> <specific_dir>`. Both beat a bare "
    "`find /` walk.\n"
    "  4. ask the user for the path if it's a package/source location.\n"
    "\n"
    "A broad `find /` or a find over a big shared/NFS tree can take minutes and "
    "hammer the filesystem — that is what this guard blocks.\n"
    "\n"
    "If your find is genuinely necessary AND tightly scoped (bounded root, "
    "-maxdepth / -name filters keeping the walk cheap), override with "
    "_override_reason explaining the root is bounded and the cheaper channels "
    "above don't apply."
)


_GREP_MESSAGE = (
    "[FindGuard] a RECURSIVE `grep` over a broad root (`/`, `/mnt`, "
    "`/public-nvme`, ...) or from an unbounded cwd-relative target (`.`, `..`, "
    "`~`, a bare `*` glob, or no target at all) is slow and hammers the "
    "filesystem — same cost class as a bare `find`. Prefer the internal "
    "information-retrieval order:\n"
    "\n"
    "  1. conversation_full.json / conversation.json (session dir) — the earlier "
    "hit may already be recorded. Near-zero cost.\n"
    "  2. memory — memory_list(keyword=...) or memory_read(key='fact/<domain>/').\n"
    "  3. scope the search — `grep -rn <pattern> <specific_subdir>` (NOT a whole "
    "tree); locate files by name with a fast indexed search.\n"
    "  4. ask the user for the path if it's a package/source location.\n"
    "\n"
    "If the broad recursive grep is genuinely necessary, override with "
    "_override_reason explaining why no scoped root applies."
)


class FindGuard(Guard):
    """Block shell commands that invoke `find`; overridable when scoped."""

    name = "find_guard"
    priority = 25

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None

        command = ctx.tool_args.get("command", "")
        if not command:
            return None

        sanitized = _sanitize(command)

        # Block on EVERY find invocation (not once-per-turn): each new find that
        # lacks an override should be stopped. The registry's override mechanism
        # releases a single call when _override_reason is supplied for it.
        # _statement_find_violation adds token-anchored coverage for absolute
        # paths (/usr/bin/find) and prefix forms (sudo/nohup/VAR=val find).
        if _FIND_RE.search(sanitized) or _statement_find_violation(sanitized):
            return GuardVerdict.block(
                _FIND_MESSAGE,
                reason="find_invocation",
                category="find_guard",
            )

        # Broad recursive grep: same cost class as a bare find.
        if _grep_is_broad(sanitized):
            return GuardVerdict.block(
                _GREP_MESSAGE,
                reason="broad_recursive_grep",
                category="find_guard",
            )

        # Hidden invocations: find / broad grep inside payloads handed to
        # remote/shell executors (ssh, docker, kubectl, bash -lc, ...) or
        # command substitution. Quote-stripping above makes these invisible;
        # they still execute (often on remote NFS trees, where a stray
        # recursive find is the slowest of all). Pass the RAW command —
        # _hidden_violation re-derives its own quote-preserving view.
        hidden = _hidden_violation(command)
        if hidden is not None:
            return hidden

        # Round-2 lexical scanner: shell-semantics escape families the
        # string-regex layers above cannot see (word-splitting, line
        # continuations, subshells, variable indirection, wrapper executors,
        # pipe-to-shell, nested substitution, shell-fed heredocs).
        scanned = _scan(command)
        if scanned is not None:
            return scanned

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None
