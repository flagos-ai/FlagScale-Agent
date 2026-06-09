"""Shell command tool with safety checks and user confirmation."""

import os
import re
import subprocess
import sys
import threading
import time

from flagscale_agent.react.tools.base import Tool, EFFECT_SHELL

FATAL_PATTERNS = [
    r"rm\s+-[^\s]*r[^\s]*f\s+/\s*$",
    r"rm\s+-[^\s]*f[^\s]*r\s+/\s*$",
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\{\s*:\|:&\s*\};:",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+777\s+/\s*$",
]

CONFIRM_PATTERNS = [
    r"\brm\s+",
    r"\bkill\b(?!\s+-(0|s\s+0)\b)",
    r"\bkillall\b",
    r"\bpkill\b",
    r"\breboot\b",
    r"\bshutdown\b",
    r"\bsystemctl\s+(stop|restart|disable)",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean\s+-[^\s]*f",
    r"\bchmod\s+(-[^\s]*\s+)*[0-7]{3,4}\b",
    r"\bchmod\s+-[^\s]*R",
    r"\bchown\b",
    r"\bmv\s+/",
    r"\bcp\s+.*\s+/",
    r"\bpip\s+install\b",
    r"\bpip\s+uninstall\b",
    r"\bconda\s+install\b",
    r"\bconda\s+remove\b",
    r"\bapt\s+(install|remove|purge)",
    r"\byum\s+(install|remove|erase)",
    r"\bcurl\s+.*\|\s*(ba)?sh",
    r"\bwget\s+.*\|\s*(ba)?sh",
]

_CONFIRM_REASONS = {
    r"\brm\s+": "delete files",
    r"\bkill\b(?!\s+-(0|s\s+0)\b)": "kill process",
    r"\bkillall\b": "kill processes",
    r"\bpkill\b": "kill processes",
    r"\breboot\b": "reboot system",
    r"\bshutdown\b": "shutdown system",
    r"\bsystemctl\s+(stop|restart|disable)": "modify system service",
    r"\bgit\s+push\b": "push to remote",
    r"\bgit\s+reset\s+--hard": "discard commits",
    r"\bgit\s+clean\s+-[^\s]*f": "delete untracked files",
    r"\bchmod\s+(-[^\s]*\s+)*[0-7]{3,4}\b": "change file permissions",
    r"\bchmod\s+-[^\s]*R": "recursive permission change",
    r"\bchown\b": "change file ownership",
    r"\bmv\s+/": "move system files",
    r"\bcp\s+.*\s+/": "copy to system path",
    r"\bpip\s+install\b": "install Python packages",
    r"\bpip\s+uninstall\b": "uninstall Python packages",
    r"\bconda\s+install\b": "install conda packages",
    r"\bconda\s+remove\b": "remove conda packages",
    r"\bapt\s+(install|remove|purge)": "modify system packages",
    r"\byum\s+(install|remove|erase)": "modify system packages",
    r"\bcurl\s+.*\|\s*(ba)?sh": "pipe remote script to shell",
    r"\bwget\s+.*\|\s*(ba)?sh": "pipe remote script to shell",
}

_FATAL_RE = re.compile("|".join(FATAL_PATTERNS))
_CONFIRM_RE = re.compile("|".join(CONFIRM_PATTERNS))

# Patterns inside grep/awk/ps arguments are not real commands — strip them
# before matching confirm patterns to avoid false positives like:
#   ps aux | grep "pip install" | grep -v grep
_GREP_PATTERN_RE = re.compile(
    r"""\bgrep\s+(?:-[^\s]*\s+)*(?:"[^"]*"|'[^']*')"""
    r"""|"""
    r"""\bgrep\s+(?:-[^\s]*\s+)*\S+"""
)


# wget without -c/--continue flag
_WGET_NO_CONTINUE_RE = re.compile(r'\bwget\b(?!.*\s-c\b)(?!.*--continue\b)')

# Download commands for result annotation
_DOWNLOAD_CMD_RE = re.compile(r'\b(wget|curl)\b')

# Training-specific guardrail patterns
_RAW_TORCHRUN_RE = re.compile(
    r'\btorchrun\b|\bpython\s+-m\s+torch\.distributed\.launch\b'
)
_FLAGSCALE_TRAIN_RE = re.compile(r'\bflagscale\s+train\b')
_PIP_NO_DEPS_RISKY_RE = re.compile(
    r'\bpip\s+install\b(?!.*--no-deps).*\b(flash.attn|deepspeed|apex)\b'
)
_UNFILTERED_FIND_RE = re.compile(
    r'\bfind\s+(/workspace|/home|\.\s)'
    r'(?!.*-not\s+-path)(?!.*--exclude)(?!.*-prune)'
    r'.*-name\s+["\']?\*\.py'
)


def _training_guardrails(command: str) -> list:
    """Return warning strings for training-specific anti-patterns."""
    warnings = []
    cleaned = _strip_grep_patterns(command)

    if _RAW_TORCHRUN_RE.search(cleaned) and not _FLAGSCALE_TRAIN_RE.search(cleaned):
        warnings.append(
            "[GUARDRAIL] Using raw torchrun instead of FlagScale Launcher. "
            "This loses per-rank logs, experiment directory structure, config validation, "
            "and clean shutdown. Use `flagscale train <model> --config <config>` instead."
        )

    if _PIP_NO_DEPS_RISKY_RE.search(cleaned):
        warnings.append(
            "[GUARDRAIL] Installing flash-attn/deepspeed/apex without --no-deps. "
            "This may silently upgrade PyTorch and break CUDA compatibility. "
            "Use `pip install --no-deps <package>`."
        )

    if _UNFILTERED_FIND_RE.search(cleaned):
        warnings.append(
            "[GUARDRAIL] find for *.py without excluding envs/site-packages. "
            "Add: -not -path '*/envs/*' -not -path '*site-packages*'"
        )

    # GPU pre-check for training launches
    is_dryrun = any(p in cleaned.lower() for p in ['--dryrun', '--dry-run', '--dry_run', 'action=dryrun', 'action=dry_run', 'action=dry-run'])
    if _FLAGSCALE_TRAIN_RE.search(cleaned) and not is_dryrun and '--stop' not in cleaned:
        try:
            gpu_check = subprocess.run(
                "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if gpu_check.returncode == 0 and gpu_check.stdout.strip():
                busy_gpus = []
                for line in gpu_check.stdout.strip().splitlines():
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        idx, mem = parts[0].strip(), int(parts[1].strip())
                        if mem > 1000:
                            busy_gpus.append(f"GPU {idx}: {mem}MB")
                if busy_gpus:
                    warnings.append(
                        f"[GUARDRAIL] GPUs have significant memory usage before training launch: "
                        f"{', '.join(busy_gpus)}. Old processes may still be running. "
                        f"Check with `pgrep -fa torchrun` and kill before launching."
                    )
        except Exception:
            pass

    # Check for stale processes before training launch
    if _FLAGSCALE_TRAIN_RE.search(cleaned) and '--stop' not in cleaned and not is_dryrun:
        try:
            proc_check = subprocess.run(
                "pgrep -c -f 'torchrun|train_' 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            count = int(proc_check.stdout.strip()) if proc_check.stdout.strip() else 0
            if count > 0:
                warnings.append(
                    f"[GUARDRAIL] {count} training-related processes already running. "
                    f"Kill them first: `pkill -9 -f 'torchrun|train_'` then wait for GPU memory release."
                )
        except Exception:
            pass

    return warnings


def _inject_proxy_exports(command: str, env: dict) -> str:
    exports = []
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        val = env.get(var)
        if val:
            # Use shlex.quote to prevent shell injection via proxy values
            import shlex
            exports.append(f'export {var}={shlex.quote(val)}')
    if not exports:
        return command
    return " && ".join(exports) + " && " + command


def _ensure_wget_continue(command: str) -> str:
    return _WGET_NO_CONTINUE_RE.sub("wget -c", command)


def _inject_git_timeout(command: str) -> str:
    """Inject network timeout env vars for git commands.

    Without these, git will hang indefinitely when a remote is unreachable
    (e.g., behind a proxy that accepts TCP but never responds to HTTPS).
    GIT_HTTP_LOW_SPEED_LIMIT=1000 + GIT_HTTP_LOW_SPEED_TIME=60 means:
    kill if transfer drops below 1KB/s for 60 seconds.
    """
    if not re.search(r'\bgit\s+(clone|fetch|pull|push|submodule)\b', command):
        return command
    # Don't inject if user already set these
    if 'GIT_HTTP_LOW_SPEED' in command:
        return command
    timeout_vars = (
        'GIT_HTTP_LOW_SPEED_LIMIT=1000 '
        'GIT_HTTP_LOW_SPEED_TIME=60'
    )
    return f'{timeout_vars} {command}'


def _strip_grep_patterns(command: str) -> str:
    """Remove grep/search patterns from command so they don't trigger confirm."""
    return _GREP_PATTERN_RE.sub("grep __PATTERN__", command)


def _strip_heredoc_bodies(command: str) -> str:
    """Replace heredoc bodies with a placeholder so their content doesn't
    trigger confirm patterns.  Handles both quoted and unquoted delimiters."""
    import re
    return re.sub(
        r"<<-?\s*['\"]?(\w+)['\"]?.*?\n.*?\n\1\b",
        "<<HEREDOC_STRIPPED",
        command,
        flags=re.DOTALL,
    )


def _summarize_for_confirm(command: str) -> str:
    """Produce a short display string for the confirmation prompt.

    For short commands (≤3 lines) show as-is.
    For long commands, show the first meaningful line, an ellipsis with the
    total line count, and every line that actually triggered a CONFIRM_PATTERN.
    """
    lines = command.split("\n")
    if len(lines) <= 3:
        return command

    # Match against the command with heredoc bodies stripped so that
    # file content inside heredocs doesn't produce false trigger lines.
    stripped_cmd = _strip_heredoc_bodies(command)
    stripped_lines = stripped_cmd.split("\n")

    trigger_lines = []
    for line in stripped_lines:
        s = line.strip()
        if not s:
            continue
        cleaned_line = _strip_grep_patterns(s)
        if _CONFIRM_RE.search(cleaned_line):
            trigger_lines.append(s)

    first = next((l.strip() for l in lines if l.strip()), lines[0])
    parts = [first, f"  ... ({len(lines)} lines total)"]
    for tl in trigger_lines:
        if tl != first:
            parts.append(f"  \033[33m→ {tl}\033[0m")
    return "\n".join(parts)


def _default_confirm(command: str, matched_patterns: list = None) -> bool:
    """Ask user to confirm a potentially risky command."""
    from flagscale_agent.react import display
    if display._active_spinner:
        display._active_spinner.stop()
        display._active_spinner = None
    summary = _summarize_for_confirm(command)
    # Build reason string from matched patterns
    reasons = []
    if matched_patterns:
        for p in matched_patterns:
            r = _CONFIRM_REASONS.get(p)
            if r and r not in reasons:
                reasons.append(r)
    reason_str = f" ({', '.join(reasons)})" if reasons else ""
    print(f"\n\033[33m⚠  Confirm{reason_str}:\033[0m\n{summary}")
    try:
        answer = input("\033[33m   Allow? [y/N/a(llow all similar)]: \033[0m").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    result = "allow_pattern" if answer in ("a", "allow") else (answer in ("y", "yes"))
    if result:
        display._active_spinner = display._Spinner()
        print()
        display._active_spinner.start()
    return result


_SELF_KILL_RE = re.compile(
    r"\bkill\b.*\b(flagscale|agent\.py|react/agent)\b"
    r"|\bgrep\b.*\b(flagscale|agent\.py)\b.*\bkill\b"
    r"|\bpkill\b.*\b(flagscale|agent)\b"
    r"|\bkillall\b.*\b(flagscale|agent)\b",
)


def _get_agent_pids():
    """Get PIDs of the agent process tree that must not be killed."""
    agent_pid = os.getpid()
    ppid = os.getppid()
    exclude = {agent_pid, ppid}
    try:
        with open(f"/proc/{ppid}/stat") as f:
            pppid = int(f.read().split()[3])
            exclude.add(pppid)
    except (OSError, ValueError, IndexError):
        pass
    # Also find child processes
    try:
        result = subprocess.run(
            f"pgrep -P {agent_pid}", shell=True,
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            try:
                exclude.add(int(line.strip()))
            except ValueError:
                pass
    except Exception:
        pass
    return exclude


def _protect_self_kill(command: str) -> str:
    """Rewrite kill pipelines to exclude the agent's own process tree."""
    exclude = _get_agent_pids()
    pids_str = "|".join(str(p) for p in sorted(exclude))

    # pkill/killall with flagscale/agent pattern — rewrite to ps | grep | filter | kill
    pkill_re = re.compile(r"\b(pkill|killall)\s+(-\S+\s+)*(flagscale\S*|agent\S*)")
    m = pkill_re.search(command)
    if m:
        signal_flag = m.group(2) or ""
        pattern = m.group(3)
        kill_sig = "-9" if "-9" in signal_flag else ""
        replacement = (
            f"ps aux | grep '{pattern}' | grep -v grep"
            f" | awk '{{print $2}}'"
            f" | grep -Ev '\\b({pids_str})\\b'"
            f" | xargs -r kill {kill_sig}"
        )
        command = command[:m.start()] + replacement + command[m.end():]
        return command

    # xargs kill pipelines — inject PID filter before xargs
    if "xargs" in command and "kill" in command:
        pid_filter = f"grep -Ev '\\b({pids_str})\\b' | "
        command = re.sub(
            r'\|\s*xargs\s+(-r\s+)?kill',
            lambda m: f"| {pid_filter}xargs {m.group(1) or ''}kill",
            command,
        )

    return command


class ShellTool(Tool):
    name = "shell"
    effects = EFFECT_SHELL
    description = "Execute a shell command and return its output (stdout + stderr)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, remind_interval: int = 120, check_dangerous: bool = True,
                 confirm_fn=None, require_confirm: bool = True, env: dict = None,
                 timeout: int = None, network_judge_fn=None,
                 stall_judge_fn=None, stall_threshold: int = 2,
                 health_judge_fn=None):
        # Support legacy 'timeout' kwarg
        self._remind_interval = timeout if timeout is not None else remind_interval
        self._check_dangerous = check_dangerous
        self._require_confirm = require_confirm
        self._confirm_fn = confirm_fn or _default_confirm
        self._env = env or {}
        self._approved_patterns = set()
        self._network_judge_fn = network_judge_fn
        self._stall_judge_fn = stall_judge_fn
        self._stall_threshold = stall_threshold
        self._health_judge_fn = health_judge_fn

    def _match_confirm_patterns(self, command: str):
        """Return the list of CONFIRM_PATTERNS that match this command."""
        cleaned = _strip_heredoc_bodies(_strip_grep_patterns(command))
        return [p for p in CONFIRM_PATTERNS if re.search(p, cleaned)]

    def needs_confirm(self, command: str) -> bool:
        """Check if a command would require user confirmation (without prompting)."""
        if not self._require_confirm:
            return False
        cleaned = _strip_heredoc_bodies(_strip_grep_patterns(command))
        if not _CONFIRM_RE.search(cleaned):
            return False
        matched = self._match_confirm_patterns(command)
        unapproved = [p for p in matched if p not in self._approved_patterns]
        return bool(unapproved)

    def _call_confirm(self, command: str, matched_patterns: list):
        """Call confirm function, falling back to single-arg for custom fns."""
        try:
            return self._confirm_fn(command, matched_patterns)
        except TypeError:
            return self._confirm_fn(command)

    def pre_confirm(self, command: str) -> bool:
        """Run the confirmation prompt for a command. Returns True if approved."""
        matched = self._match_confirm_patterns(command)
        unapproved = [p for p in matched if p not in self._approved_patterns]
        if not unapproved:
            return True
        result = self._call_confirm(command, unapproved)
        if result == "allow_pattern":
            self._approved_patterns.update(matched)
            return True
        return bool(result)

    def execute(self, **kwargs) -> str:
        command = kwargs["command"]
        skip_confirm = kwargs.pop("_skip_confirm", False)
        parallel_index = kwargs.pop("_parallel_index", None)
        quiet = skip_confirm  # suppress dots/progress in parallel mode

        if self._check_dangerous and self._require_confirm and _FATAL_RE.search(command):
            return f"FATAL: Refused to execute potentially dangerous command: {command}"

        # Training-specific guardrails — prepend warnings to result
        guardrail_warnings = _training_guardrails(command)

        if not skip_confirm:
            cleaned = _strip_heredoc_bodies(_strip_grep_patterns(command))
            if self._require_confirm and _CONFIRM_RE.search(cleaned):
                matched = self._match_confirm_patterns(command)
                unapproved = [p for p in matched if p not in self._approved_patterns]
                if unapproved:
                    result = self._call_confirm(command, unapproved)
                    if result == "allow_pattern":
                        self._approved_patterns.update(matched)
                    elif not result:
                        return "DENIED: User declined to execute this command."

        if _SELF_KILL_RE.search(command):
            command = _protect_self_kill(command)

        command, post_fn = _strip_trailing_pipe(command)

        # Ensure conda run streams output in real-time
        command = re.sub(
            r'\bconda\s+run\b(?!\s+--live-stream)',
            'conda run --live-stream',
            command,
        )

        # Auto-add -c to wget for resume support
        command = _ensure_wget_continue(command)

        # Inject git network timeout to prevent indefinite hangs
        command = _inject_git_timeout(command)

        # Inject proxy exports for network commands
        command = _inject_proxy_exports(command, self._env)

        try:
            run_env = {**os.environ, **self._env} if self._env else None
            proc = None
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env,
            )

            stdout_chunks: list = []
            stderr_chunks: list = []

            def _read_stream(stream, buf):
                for line in stream:
                    buf.append(line)

            t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True)
            t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True)
            t_out.start()
            t_err.start()

            # Show wait hint for commands with embedded sleep or known long-running patterns
            _sleep_m = re.search(r"\bsleep\s+(\d+)", command)
            if _sleep_m and not quiet:
                secs = _sleep_m.group(1)
                from flagscale_agent.react import display
                if display._active_spinner:
                    display._active_spinner.set_hint(f"⏳ Waiting {secs}s")

            start = time.time()
            # First health check at 15s, then LLM decides (default 60s)
            next_check = min(15, self._remind_interval)
            long_run_approved = True
            last_output_snapshot = ""
            stall_count = 0
            health_reason = ""  # last health judge verdict for display
            while proc.poll() is None:
                elapsed = time.time() - start
                if elapsed > next_check:
                    next_check = elapsed + self._remind_interval
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}m{secs}s" if mins > 0 else f"{secs}s"
                    recent_text = "".join(stdout_chunks[-20:] + stderr_chunks[-20:])
                    current_snapshot = "".join(stdout_chunks[-10:] + stderr_chunks[-10:])

                    # Track output changes for context
                    output_changed = not current_snapshot or current_snapshot != last_output_snapshot
                    if not output_changed:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_output_snapshot = current_snapshot

                    # Unified LLM health judge — every interval, LLM sees everything
                    if self._health_judge_fn:
                        decision = self._health_judge_fn(
                            command, recent_text, time_str,
                            output_changed=output_changed,
                            stall_count=stall_count,
                        )
                        if decision.get("kill"):
                            proc.kill()
                            t_out.join(timeout=2)
                            t_err.join(timeout=2)
                            partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                            reason = decision.get("reason", "Unhealthy command")
                            hint = _network_error_hint(partial, self._env) or ""
                            return (
                                f"TERMINATED: {reason} (after {time_str}).{hint}\n"
                                f"Output:\n{partial}"
                            )
                        else:
                            reason = decision.get("reason", "")
                            health_reason = reason
                            if reason:
                                if quiet and parallel_index is not None:
                                    from flagscale_agent.react import display
                                    display.parallel_tool_hint(parallel_index, reason)
                                elif not quiet:
                                    from flagscale_agent.react import display
                                    if display._active_spinner:
                                        display._active_spinner.set_hint(f"🩺 {reason}")
                            else:
                                if not quiet:
                                    from flagscale_agent.react import display
                                    if display._active_spinner:
                                        display._active_spinner.set_hint("")
                            # LLM decides next check interval
                            ncs = decision.get("next_check_seconds")
                            if isinstance(ncs, (int, float)) and 10 <= ncs <= 300:
                                next_check = elapsed + ncs
                            else:
                                next_check = elapsed + self._remind_interval
                    else:
                        # Fallback: legacy network error check
                        if self._network_judge_fn and _NETWORK_ERROR_PATTERNS.search(recent_text):
                            decision = self._network_judge_fn(command, recent_text, time_str)
                            if decision.get("kill"):
                                proc.kill()
                                t_out.join(timeout=2)
                                t_err.join(timeout=2)
                                partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                                reason = decision.get("reason", "Network error")
                                hint = _network_error_hint(partial, self._env) or ""
                                return (
                                    f"TERMINATED: {reason} (after {time_str}).{hint}\n"
                                    f"Output:\n{partial}"
                                )

                        # Fallback: legacy stall detection
                        if not output_changed and stall_count >= self._stall_threshold:
                            stall_secs = stall_count * self._remind_interval
                            stall_dur = f"{stall_secs // 60}m{stall_secs % 60}s" if stall_secs >= 60 else f"{stall_secs}s"
                            if self._stall_judge_fn:
                                decision = self._stall_judge_fn(command, current_snapshot, time_str, stall_dur)
                                if decision.get("kill"):
                                    proc.kill()
                                    t_out.join(timeout=2)
                                    t_err.join(timeout=2)
                                    partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                                    reason = decision.get("reason", "Output stalled")
                                    return (
                                        f"STALLED: {reason} (output unchanged for {stall_dur}, total {time_str}).\n"
                                        f"Output:\n{partial}"
                                    )
                            else:
                                proc.kill()
                                t_out.join(timeout=2)
                                t_err.join(timeout=2)
                                partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                                return (
                                    f"STALLED: Output unchanged for {stall_dur} (total {time_str}). "
                                    f"Command appears stuck.\n"
                                    f"Output:\n{partial}"
                                )

                    if long_run_approved:
                        recent = stdout_chunks[-5:] + stderr_chunks[-5:]
                        if recent and not quiet:
                            from flagscale_agent.react import display
                            if display._active_spinner:
                                display._active_spinner.stop()
                            # Include health judge verdict in the output line
                            health_note = f"\n   🩺 {health_reason}\n" if health_reason else ""
                            lines_out = [f"\033[2m   ⏳ [{time_str}]{health_note}   Recent output:\033[0m"]
                            for line in recent[-5:]:
                                lines_out.append(f"\033[2m   │ {line.rstrip()}\033[0m")
                            with display._stdout_lock:
                                sys.stdout.write("\n".join(lines_out) + "\n")
                                sys.stdout.flush()
                            if display._active_spinner:
                                display._active_spinner.start()
                    else:
                        from flagscale_agent.react import display
                        display._stop_all_spinners()
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        recent = stdout_chunks[-10:] + stderr_chunks[-10:]
                        if recent:
                            print("\033[2m   Recent output:\033[0m")
                            for line in recent[-10:]:
                                print(f"\033[2m   │ {line.rstrip()}\033[0m")
                        else:
                            print("\033[2m   (no output yet — command may be buffering due to pipes)\033[0m")
                        print(f"\033[33m⏳ Command still running ({time_str} elapsed)\033[0m")
                        try:
                            answer = input("   Continue? [y/N/a(lways)]: ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            answer = "n"
                        if answer in ("a", "always"):
                            long_run_approved = True
                            print("\033[2m   Will not ask again for this command.\033[0m")
                        elif answer in ("y", "yes"):
                            print("\033[2m   Continuing...\033[0m")
                        else:
                            proc.kill()
                            t_out.join(timeout=2)
                            t_err.join(timeout=2)
                            partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                            if partial:
                                return f"TERMINATED by user after {int(elapsed)}s. Partial output:\n{partial}"
                            return f"TERMINATED by user after {int(elapsed)}s."
                time.sleep(0.2)

            t_out.join(timeout=5)
            t_err.join(timeout=5)

            output = ""
            if stdout_chunks:
                output += "".join(stdout_chunks)
            if stderr_chunks:
                output += "".join(stderr_chunks)
            if not output:
                output = "(no output)"
            if post_fn and output != "(no output)":
                output = post_fn(output)
            if proc.returncode != 0:
                hint = _network_error_hint(output, self._env)
                if hint:
                    output += hint
                if _DOWNLOAD_CMD_RE.search(command):
                    output += "\n[NOTICE: Download exited with non-zero status. File may be incomplete. Verify with `ls -lh` and retry with `wget -c` or `curl -C -`.]"
            if guardrail_warnings:
                output = "\n".join(guardrail_warnings) + "\n" + output
            return output
        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
            partial = "".join(stdout_chunks) + "".join(stderr_chunks) if (
                "stdout_chunks" in locals() and "stderr_chunks" in locals()
            ) else ""
            raise
        except Exception as e:
            return f"ERROR: {e}"


_TRAILING_PIPE_RE = re.compile(
    r"\|\s*(tail|head)\s+-n\s*(\d+)\s*$"
    r"|\|\s*(tail|head)\s+-(\d+)\s*$"
    r"|\|\s*(tail|head)\s*$"
)


def _strip_trailing_pipe(command: str):
    """Strip trailing | tail -N / | head -N and return (new_cmd, post_fn).

    post_fn(output) applies the equivalent truncation in Python so we get
    real-time output from the main command instead of buffering in tail/head.
    """
    m = _TRAILING_PIPE_RE.search(command)
    if not m:
        return command, None

    cmd_name = m.group(1) or m.group(3) or m.group(5)
    count_str = m.group(2) or m.group(4)
    count = int(count_str) if count_str else 10

    stripped = command[:m.start()].rstrip()
    # Remove trailing 2>&1 since Popen captures both streams separately
    stripped = re.sub(r'\s*2>&1\s*$', '', stripped)

    if cmd_name == "tail":
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[-count:]) if len(lines) > count else output
    else:  # head
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[:count]) if len(lines) > count else output

    return stripped, post_fn


_NETWORK_ERROR_PATTERNS = re.compile(
    r"Could not resolve host|Connection refused|Connection timed out|"
    r"Network is unreachable|No route to host|"
    r"Failed to connect|Connection reset by peer|"
    r"unable to access|SSL connection timeout|"
    r"Failed to establish a new connection|"
    r"Temporary failure in name resolution|"
    r"WARNING:\s*Connection timed out|"
    r"Attempting to resume incomplete download",
    re.IGNORECASE,
)

_PROXY_HINT = (
    "\n\n💡 Network error detected and no proxy configured. "
    "Set proxy in ~/.flagscale/agent.yaml:\n"
    "  shell_env:\n"
    '    HTTP_PROXY: "http://host:port"\n'
    '    HTTPS_PROXY: "http://host:port"\n'
    "Then use /reload to apply."
)


def _network_error_hint(output: str, env: dict) -> str | None:
    if not _NETWORK_ERROR_PATTERNS.search(output):
        return None
    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}
    if proxy_keys & set(env):
        return None
    return _PROXY_HINT
