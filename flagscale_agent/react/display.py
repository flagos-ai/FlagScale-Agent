"""Display utilities for agent interactive output."""

import os
import re
import sys
import threading
import time


# ── Thread-safe stdout ─────────────────────────────────────────────────

_stdout_lock = threading.Lock()


def _print(*args, **kwargs):
    """Thread-safe print."""
    with _stdout_lock:
        print(*args, **kwargs)


def _write(text):
    """Thread-safe sys.stdout.write + flush."""
    with _stdout_lock:
        sys.stdout.write(text)
        sys.stdout.flush()


def _use_color():
    return os.environ.get("NO_COLOR") is None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    if _use_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def _c256(n, text):
    """256-color foreground."""
    if _use_color():
        return f"\033[38;5;{n}m{text}\033[0m"
    return text


def dim(text):
    return _c256(245, text)


def green(text):
    return _c256(114, text)


def yellow(text):
    return _c256(214, text)


def cyan(text):
    return _c256(80, text)


def red(text):
    return _c256(203, text)


def bold(text):
    return _c("1", text)


def magenta(text):
    return _c256(141, text)


def blue(text):
    return _c256(117, text)


def _term_width():
    """Get terminal width, default 120."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        return 120


def _char_width(c):
    """Terminal display width of a single character."""
    import unicodedata
    w = unicodedata.east_asian_width(c)
    return 2 if w in ('W', 'F') else 1


def _visible_width(text):
    """Display width of text excluding ANSI escape sequences."""
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    return sum(_char_width(c) for c in plain)


def _truncate_to_width(text, max_width):
    """Truncate text so display width fits within max_width."""
    if _visible_width(text) <= max_width:
        return text
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    width = 0
    cut = 0
    for i, c in enumerate(plain):
        cw = _char_width(c)
        if width + cw > max_width - 3:
            cut = i
            break
        width += cw
        cut = i + 1
    return plain[:cut] + "..." + "\033[0m"


def _fmt_tokens(n):
    if n is None:
        return "?"
    if n >= 100000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n:,}"
    return str(n)


# ── Tool icons ──────────────────────────────────────────────────────────

_TOOL_ICONS = {
    "shell": "⚡",
    "write_file": "📝",
    "read_file": "📖",
    "edit_file": "✏️",
    "web_fetch": "🌐",
    "web_search": "🔍",
    "memory_write": "💾",
    "memory_read": "🧠",
    "plan_create": "📋",
    "plan_update": "📋",
    "plan_status": "📋",
    "find_latest_log": "📄",
}


def _tool_icon(name):
    return _TOOL_ICONS.get(name, "⚙️")


# ── Spinner for long-running tools ──────────────────────────────────────

class _Spinner:
    """Inline spinner that updates on the same line."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, prefix=""):
        self._prefix = prefix
        self._stop = threading.Event()
        self._thread = None
        self._hint = ""
        self._hint_lock = threading.Lock()
        self._t0 = time.time()

    def set_hint(self, hint):
        """Update the hint text shown alongside the spinner."""
        with self._hint_lock:
            self._hint = hint

    def start(self):
        if not _use_color():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            frame = self._FRAMES[i % len(self._FRAMES)]
            with self._hint_lock:
                hint = self._hint
            parts = [f"  {dim(frame)}"]
            if self._prefix:
                parts.append(f" {dim(self._prefix)}")
            parts.append(f" {dim(f'{elapsed:.0f}s')}")
            if hint:
                parts.append(f" {dim(hint)}")
            line = _truncate_to_width("".join(parts), _term_width())
            with _stdout_lock:
                sys.stdout.write(f"\r\033[K{line}")
                sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        with _stdout_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


_active_spinner = None


def _stop_all_spinners():
    global _active_spinner, _parallel_display, _poll_anim
    if _active_spinner:
        _active_spinner.stop()
        _active_spinner = None
    if _parallel_display:
        _parallel_display.finish()
        _parallel_display = None
    if _poll_anim:
        _poll_anim.stop()
        _poll_anim = None


# ── Paste display ──────────────────────────────────────────────────────

def pasted_input(text: str):
    """Display a multi-line pasted input in collapsed form.

    Shows first line, a "... (N lines)" indicator, and last line.
    Uses terminal scroll region awareness to handle long pastes.
    """
    lines = text.splitlines()
    if len(lines) <= 3:
        return  # Short enough, prompt_toolkit already echoed it
    # Get terminal height to cap how many lines we can erase
    try:
        term_height = os.get_terminal_size().lines
    except (AttributeError, ValueError, OSError):
        term_height = 24
    # We can only erase lines still visible in the terminal (not scrolled off)
    erase_count = min(len(lines) - 1, term_height - 2)
    for _ in range(erase_count):
        sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()
    first = lines[0][:100]
    last = lines[-1][:100]
    _print(f"  {first}")
    _print(dim(f"  ... ({len(lines)} lines)"))
    _print(f"  {last}")


# ── Banner ──────────────────────────────────────────────────────────────

def banner(provider, model, mode=None, context_window=None, extra_lines=None):
    from flagscale_agent import __version__

    # ASCII logo
    logo = [
        " _____ _             ____            _           _                    _   ",
        "|  ___| | __ _  __ _/ ___|  ___ __ _| | ___     / \\   __ _  ___ _ __ | |_ ",
        "| |_  | |/ _` |/ _` \\___ \\ / __/ _` | |/ _ \\   / _ \\ / _` |/ _ \\ '_ \\| __|",
        "|  _| | | (_| | (_| |___) | (_| (_| | |  __/  / ___ \\ (_| |  __/ | | | |_ ",
        "|_|   |_|\\__,_|\\__, |____/ \\___\\__,_|_|\\___| /_/   \\_\\__, |\\___|_| |_|\\__|",
        "               |___/                                 |___/                  ",
    ]
    for line in logo:
        _print(cyan(line))
    _print()

    title = f"FlagScale Agent v{__version__}"
    mode_str = f" | Mode: {mode}" if mode else ""
    ctx_str = f" | Context: {context_window // 1000}k" if context_window else ""
    info = f"Provider: {provider} | Model: {model}{mode_str}{ctx_str}"
    cmds = "Commands: /skill  /file  /plan  /save  /load  /resume  /export  /memory  /mode  /reload  /quit"
    lines = [info, cmds]
    if extra_lines:
        lines.extend(extra_lines)
    width = max(len(title), *(len(l) for l in lines)) + 4
    _print(cyan(f"╭─ {title} {'─' * (width - len(title) - 3)}╮"))
    for l in lines:
        _print(cyan(f"│  {l}{' ' * (width - len(l) - 2)}│"))
    _print(cyan(f"╰{'─' * width}╯"))
    _print()


# ── Thinking ────────────────────────────────────────────────────────────

_thinking_anim = None


class _ThinkingAnim:
    """Animated '⏳ Thinking...' indicator with cycling dots."""
    _DOTS = [".", "..", "..."]

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._timestamp = time.strftime("%H:%M:%S")

    def start(self):
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self):
        i = 0
        while not self._stop.is_set():
            dots = self._DOTS[i % len(self._DOTS)]
            line = f"[{self._timestamp}] ⏳ Thinking{dots}"
            with _stdout_lock:
                sys.stdout.write(f"\r\033[K{dim(line)}")
                sys.stdout.flush()
            i += 1
            self._stop.wait(0.4)

    def done(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        line = f"[{self._timestamp}] ⏳ Thinking ✓"
        with _stdout_lock:
            sys.stdout.write(f"\r\033[K{dim(line)}\n")
            sys.stdout.flush()

    def cancel(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        with _stdout_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def thinking():
    global _thinking_anim
    _stop_all_spinners()
    _thinking_anim = _ThinkingAnim()
    _thinking_anim.start()


def thinking_done():
    """Mark thinking as done — clear the line (info merged into llm_done)."""
    global _thinking_anim
    _stop_all_spinners()
    if _thinking_anim:
        _thinking_anim.cancel()
        _thinking_anim = None


def thinking_clear():
    """Cancel thinking — removes the line entirely. Alias for thinking_done."""
    thinking_done()


# ── Context compaction ─────────────────────────────────────────────────

_last_compaction_to = 0
_compaction_suppressed = 0


def context_compacted(from_tokens, to_tokens, compaction_num=None, ratio=None):
    global _last_compaction_to, _compaction_suppressed
    # Suppress if change from last notification is < 5k tokens
    if abs(to_tokens - _last_compaction_to) < 5000 and _last_compaction_to > 0:
        _compaction_suppressed += 1
        if _compaction_suppressed >= 5:
            from_k = from_tokens // 1000
            to_k = to_tokens // 1000
            _print(dim(f"📦 Context compacted ×{_compaction_suppressed + 1}: now {to_k}k tokens"))
            _compaction_suppressed = 0
            _last_compaction_to = to_tokens
        return
    _compaction_suppressed = 0
    _last_compaction_to = to_tokens
    from_k = from_tokens // 1000
    to_k = to_tokens // 1000
    detail = f"{from_k}k → {to_k}k"
    if compaction_num is not None and ratio is not None:
        detail += f" (#{compaction_num}, target {int(ratio * 100)}%)"
    _print(dim(f"📦 Context compacted: {detail}"))


# ── LLM done ────────────────────────────────────────────────────────────

def llm_done(elapsed, input_tokens=None, output_tokens=None):
    ts = time.strftime("%H:%M:%S")
    parts = [f"[{ts}]", green("✓"), f"{elapsed:.1f}s"]
    if input_tokens is not None:
        parts.append(f"↑{_fmt_tokens(input_tokens)}")
    if output_tokens is not None:
        parts.append(f"↓{_fmt_tokens(output_tokens)}")
    _print(dim(" | ".join(parts)))


# ── Tool start / done (compact single-line) ─────────────────────────────

# Track whether tool_start printed a newline (spinner) or stayed on same line
_tool_inline = False


def tool_start(name, args_summary=""):
    """Show tool invocation and start spinner for long-running tools."""
    global _active_spinner, _tool_inline
    icon = _tool_icon(name)
    label = f"  {icon} {name}"
    if args_summary:
        label += f" {args_summary}"
    tw = _term_width()
    _print(_truncate_to_width(dim(label), tw), end="", flush=True)
    if name == "shell":
        _active_spinner = _Spinner()
        _print()
        _active_spinner.start()
        _tool_inline = False
    else:
        _tool_inline = True


def tool_done(name, elapsed, detail="", error=False):
    """Show tool completion — inline if fast, new line if spinner was used."""
    global _active_spinner, _tool_inline
    if _active_spinner:
        _active_spinner.stop()
        _active_spinner = None
    if error:
        status = red(f"✖ {elapsed:.1f}s")
    elif elapsed > 5:
        status = yellow(f"✓ {elapsed:.1f}s")
    else:
        status = dim(f"✓ {elapsed:.1f}s")

    if _tool_inline:
        suffix = f" {status}"
        if detail:
            suffix += f" {dim(detail)}"
        _print(suffix)
    else:
        line = f"    {status}"
        if detail:
            line += f" {dim(detail)}"
        _print(line)
    _tool_inline = False


# ── Parallel tool display ──────────────────────────────────────────────

class _ParallelDisplay:
    """In-place updating display for parallel tool execution."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _MAX_DISPLAY_LINES = 10  # Show at most this many tool lines

    def __init__(self, tool_summaries):
        """tool_summaries: list of (name, args_summary) tuples."""
        self._tools = tool_summaries
        self._n = len(tool_summaries)
        self._collapsed = self._n > self._MAX_DISPLAY_LINES
        self._display_n = min(self._n, self._MAX_DISPLAY_LINES)
        self._results = {}  # index -> (elapsed, error)
        self._hints = {}    # index -> hint text for running tools
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = 0
        self._extra_lines = 0  # lines printed below display area

    def start(self):
        if not _use_color() or self._n == 0:
            for name, args in self._tools[:self._MAX_DISPLAY_LINES]:
                icon = _tool_icon(name)
                label = f"  {icon} {name}"
                if args:
                    label += f" {args}"
                _print(dim(label))
            if self._collapsed:
                _print(dim(f"  ... and {self._n - self._MAX_DISPLAY_LINES} more"))
            return
        # Print initial lines with pending indicator (capped)
        tw = _term_width()
        with _stdout_lock:
            for name, args in self._tools[:self._display_n]:
                icon = _tool_icon(name)
                label = f"{icon} {name}"
                if args:
                    label += f" {args}"
                frame = self._FRAMES[0]
                line = f"  {dim(label)} {dim(frame)}"
                sys.stdout.write(f"{_truncate_to_width(line, tw)}\n")
            if self._collapsed:
                sys.stdout.write(f"  {dim(f'... and {self._n - self._display_n} more')}\n")
            sys.stdout.flush()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def add_extra_lines(self, count):
        """Track lines printed below the display area by external code."""
        with self._lock:
            self._extra_lines += count

    def mark_done(self, index, elapsed, error=False, detail=""):
        with self._lock:
            self._results[index] = (elapsed, error, detail)

    def update_hint(self, index, hint):
        """Update a running tool's hint text (e.g. health check status)."""
        with self._lock:
            self._hints[index] = hint

    def _animate(self):
        while not self._stop.is_set():
            self._redraw()
            self._frame += 1
            self._stop.wait(0.5)

    def _redraw(self):
        with self._lock:
            results = dict(self._results)
            hints = dict(self._hints)
            extra = self._extra_lines
        tw = _term_width()
        # collapsed summary line counts as 1 extra display line
        display_lines = self._display_n + (1 if self._collapsed else 0)
        with _stdout_lock:
            total_up = display_lines + extra
            if total_up > 0:
                sys.stdout.write(f"\033[{total_up}A")
            for i in range(self._display_n):
                name, args = self._tools[i]
                icon = _tool_icon(name)
                label = f"{icon} {name}"
                if args:
                    label += f" {args}"
                if i in results:
                    elapsed, error, detail = results[i]
                    if error:
                        status = red(f"✖ {elapsed:.1f}s")
                    elif elapsed > 5:
                        status = yellow(f"✓ {elapsed:.1f}s")
                    else:
                        status = dim(f"✓ {elapsed:.1f}s")
                    line = f"  {label} {status}"
                    if detail:
                        line += f" {dim(detail)}"
                else:
                    frame = self._FRAMES[self._frame % len(self._FRAMES)]
                    hint = hints.get(i, "")
                    if hint:
                        line = f"  {dim(label)} {dim(frame)} {dim('🩺 ' + hint)}"
                    else:
                        line = f"  {dim(label)} {dim(frame)}"
                sys.stdout.write(f"\r\033[K{_truncate_to_width(line, tw)}\n")
            if self._collapsed:
                done_hidden = sum(1 for i, _ in results.items() if i >= self._display_n)
                summary = f"  ... and {self._n - self._display_n} more"
                if done_hidden:
                    summary += f" ({done_hidden} done)"
                sys.stdout.write(f"\r\033[K{dim(summary)}\n")
            if extra > 0:
                sys.stdout.write(f"\033[{extra}B")
            sys.stdout.flush()

    def finish(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        if _use_color() and self._n > 0:
            self._redraw()


_parallel_display = None


def parallel_tools_start(tool_summaries):
    """Print all tool names and start in-place updating display.

    tool_summaries: list of (name, args_summary) tuples
    Returns: _ParallelDisplay instance
    """
    global _active_spinner, _parallel_display
    if _active_spinner:
        _active_spinner.stop()
        _active_spinner = None
    _parallel_display = _ParallelDisplay(tool_summaries)
    _parallel_display.start()
    return _parallel_display


def parallel_tool_update(index, elapsed, error=False, detail=""):
    """Update a specific tool line with completion status."""
    global _parallel_display
    if _parallel_display:
        _parallel_display.mark_done(index, elapsed, error, detail)


def parallel_tool_hint(index, hint):
    """Update a running tool's hint (e.g. health check diagnosis)."""
    if _parallel_display and not _parallel_display._stop.is_set():
        _parallel_display.update_hint(index, hint)


def parallel_extra_lines(count):
    """Notify parallel display that external code printed extra lines below it."""
    if _parallel_display and not _parallel_display._stop.is_set():
        _parallel_display.add_extra_lines(count)


def parallel_tools_finish():
    """Stop parallel display and do final redraw."""
    global _parallel_display
    if _parallel_display:
        _parallel_display.finish()
        _parallel_display = None


# ── Poll mode display ─────────────────────────────────────────────────

_poll_anim = None


class _PollAnim:
    """Animated inline display for poll mode checks."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._status = ""
        self._lock = threading.Lock()

    def set_status(self, text):
        with self._lock:
            self._status = text

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            with self._lock:
                status = self._status
            line = f"  {dim(frame)} {dim(status)}"
            tw = _term_width()
            with _stdout_lock:
                sys.stdout.write(f"\r\033[K{_truncate_to_width(line, tw)}")
                sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        with _stdout_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def poll_mode_start(command_summary, interval):
    """Print poll mode activation banner."""
    global _poll_anim
    _stop_all_spinners()
    _print()
    _print(cyan(f"  🔄 Poll mode: re-running every {interval}s until interesting change"))
    _print(dim(f"     cmd: {command_summary}"))
    _print(dim(f"     Ctrl+C to exit poll mode"))
    _poll_anim = _PollAnim()
    _poll_anim.start()


def poll_check(n, elapsed, changed=False, routine_change=False):
    """Update poll status on the same line."""
    global _poll_anim
    if _poll_anim:
        if routine_change:
            _poll_anim.set_status(f"poll #{n} — routine change ({elapsed:.0f}s)")
        else:
            _poll_anim.set_status(f"poll #{n} — no change ({elapsed:.0f}s)")


def poll_mode_end(reason, poll_count, total_elapsed, routine_changes=0):
    """Print poll mode exit summary."""
    global _poll_anim
    if _poll_anim:
        _poll_anim.stop()
        _poll_anim = None
    icon = green("✓") if reason == "changed" else yellow("⏱")
    reasons = {
        "changed": "interesting change detected",
        "timeout": "max duration reached",
        "interrupted": "interrupted by user",
    }
    reason_text = reasons.get(reason, reason)
    extra = f", {routine_changes} routine changes absorbed" if routine_changes else ""
    _print(f"  {icon} Poll ended: {reason_text} ({poll_count} checks, {total_elapsed:.0f}s{extra})")
    _print()


# ── Turn / session summary ──────────────────────────────────────────────

def warn(message):
    """Display a warning message to the user."""
    _print(f"  {yellow('⚠')} {yellow(message)}")


def gate_triggered(name, description, reason, is_hard=True):
    """Display structured gate trigger info to terminal."""
    icon = "🚫" if is_hard else "⚠"
    label = "blocked" if is_hard else "warning"
    _print(f"  {icon} Gate {label}: {bold(name)}")
    if description:
        _print(f"     {dim(description)}")
    if reason:
        _print(f"     Reason: {dim(reason)}")


def turn_summary(turn_num, elapsed, input_tokens, output_tokens):
    _stop_all_spinners()
    _print()
    parts = [f"Turn {turn_num}", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    _print(dim(f"── {' | '.join(parts)} ──"))
    _print()


def session_summary(turns, elapsed, input_tokens, output_tokens):
    _print()
    parts = [f"Session: {turns} turns", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    _print(dim(" | ".join(parts)))


# ── File / session ──────────────────────────────────────────────────────

def file_injected(path, chars):
    _print(dim(f"  📎 {path} ({chars:,} chars)"))


def session_saved(path):
    _print(green(f"  ✓ Session saved: {path}"))


def resume_found(session_id, last_msg, timestamp):
    import datetime
    ts = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    _print(dim(f"  Resumable session: {session_id} ({ts})"))
    if last_msg:
        _print(dim(f"    Last: {last_msg[:80]}"))


def session_resumed(session_id):
    _print(green(f"  ✓ Resumed session: {session_id}"))


def session_loaded(path, turns):
    _print(green(f"  ✓ Session loaded: {path} ({turns} user turns)"))


def session_list(sessions):
    if not sessions:
        _print(dim("  No saved sessions."))
        return
    import datetime
    for s in sessions[:10]:
        ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
        _print(dim(f"    {s['id']}  {ts}  ({s['turns']} turns)"))
        _print(dim(f"      {s['path']}"))


# ── Skill / plan ────────────────────────────────────────────────────────

def plan_created(title, step_count):
    _print(green(f"  📋 Plan created: {title} ({step_count} steps)"))


def plan_step_updated(step_id, title, status):
    icons = {"done": green("✓"), "doing": yellow("→"), "skipped": dim("-"), "blocked": red("!")}
    icon = icons.get(status, " ")
    _print(f"    [{icon}] Step {step_id}: {title}")


def plan_completed(title):
    _print(green(f"  📋 Plan completed: {title}"))


def plan_abandoned(title):
    _print(yellow(f"  📋 Plan abandoned: {title}"))


def plan_summary(text):
    for line in text.split("\n"):
        if line.startswith("Plan:"):
            _print(cyan(line))
        elif line.strip().startswith("[✓]"):
            _print(dim(line))
        elif line.strip().startswith("[→]"):
            _print(yellow(line))
        elif line.startswith("Progress:"):
            _print(dim(line))
        else:
            _print(line)


def complexity_hint():
    _print(magenta("  📋 Complex task detected — suggesting plan creation."))


# ── Autosave ────────────────────────────────────────────────────────────

def autosave_found(turn_count, user_turns, last_user_msg, timestamp):
    import datetime
    ts = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    _print(yellow("╭─ Unfinished session detected ──────────────────╮"))
    _print(yellow(f"│  Time: {ts}"))
    _print(yellow(f"│  {turn_count} turns, {user_turns} user messages"))
    if last_user_msg:
        preview = last_user_msg[:60] + ("..." if len(last_user_msg) > 60 else "")
        _print(yellow(f"│  Last message: {preview}"))
    _print(yellow("╰─────────────────────────────────────────────────╯"))


def autosave_resumed(turn_count):
    _print(green(f"  ✓ Resumed previous session ({turn_count} turns). You can continue."))
    _print()


def interrupted():
    _stop_all_spinners()
    _print(yellow("\n  ⚠  Interrupted. Back to prompt."))


def goodbye():
    _stop_all_spinners()
    _print(green("\n  I'll remember where we left off. See you next time. 🚀\n"))


def skill_auto_loaded(name):
    _print(magenta(f"  🔧 Auto-loaded skill: {name}"))


# ── Markdown rendering ──────────────────────────────────────────────────

def render_markdown(text):
    """Render markdown with basic syntax highlighting for terminal output."""
    if not _use_color():
        return text

    lines = text.split("\n")
    output = []
    in_code_block = False
    code_lang = ""

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                code_lang = line[3:].strip()
                output.append(dim(f"┌─ {code_lang}" if code_lang else "┌─"))
            else:
                output.append(dim("└─"))
                code_lang = ""
            continue

        if in_code_block:
            output.append(f"  {_c('36', line)}")
            continue

        if line.startswith("# "):
            output.append(bold(line))
        elif line.startswith("## "):
            output.append(bold(line))
        elif line.startswith("### "):
            output.append(bold(line))
        elif line.startswith("- ") or line.startswith("* "):
            output.append(f"  {line}")
        elif re.match(r"^\d+\.\s", line):
            output.append(f"  {line}")
        else:
            line = re.sub(r"`([^`]+)`", lambda m: _c("36", m.group(1)), line)
            line = re.sub(r"\*\*([^*]+)\*\*", lambda m: _c("1", m.group(1)), line)
            output.append(line)

    return "\n".join(output)
