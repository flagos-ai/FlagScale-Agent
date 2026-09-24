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


# Completion sentinels the kernel gates on. They must NOT be streamed to the
# terminal as the model types them, because the completion gate runs only after
# the full response is received — streaming the raw token makes a *blocked*
# completion appear on screen with authority ("[TASK_COMPLETE]" then a 🚫 block),
# looking as if the gate fired after the fact. The kernel prints the authoritative
# marker itself, only once completion is actually accepted.
COMPLETION_SENTINELS = ("[TASK_COMPLETE]", "[NEED_USER_INPUT]")


class SentinelStripper:
    """Streaming filter that removes completion sentinels from displayed text.

    Handles sentinels split across arbitrary stream-delta boundaries by holding
    back only the trailing substring that could still be the start of a sentinel.
    The RAW text is kept by the caller (content_parts) for the kernel's gate;
    this only affects what reaches the screen.
    """

    def __init__(self, sentinels=COMPLETION_SENTINELS):
        self._sentinels = tuple(sentinels)
        self._buf = ""

    @staticmethod
    def _held_suffix_len(buf, sentinels):
        """Longest k such that buf[-k:] is a proper prefix of some sentinel."""
        best = 0
        for s in sentinels:
            maxk = min(len(buf), len(s) - 1)
            for k in range(maxk, 0, -1):
                if s.startswith(buf[-k:]):
                    if k > best:
                        best = k
                    break
        return best

    def feed(self, text):
        """Consume a text delta; return the portion safe to display now."""
        self._buf += text
        for s in self._sentinels:
            self._buf = self._buf.replace(s, "")
        hold = self._held_suffix_len(self._buf, self._sentinels)
        emit = self._buf[:len(self._buf) - hold] if hold else self._buf
        self._buf = self._buf[len(self._buf) - hold:] if hold else ""
        return emit

    def flush(self):
        """Return any residual buffered text at end of stream.

        A leftover here is a partial that never completed into a sentinel (e.g. a
        truncated stream), so it is genuine text and should be shown.
        """
        out, self._buf = self._buf, ""
        return out


def _enable_windows_ansi() -> bool:
    """Enable ANSI escape processing on Windows via SetConsoleMode.

    Returns True if ANSI is supported/enabled, False otherwise.
    Only has effect on Windows; no-op on other platforms.
    """
    import sys as _sys
    if _sys.platform != "win32":
        return True
    try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Get handle to stdout
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle == -1:
            return False
        # Get current console mode
        mode = ctypes.wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True  # already enabled
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False


# Try to enable ANSI on Windows at import time
_WINDOWS_ANSI_ENABLED = _enable_windows_ansi()


def _use_color():
    import sys as _sys
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if _sys.platform == "win32":
        return _WINDOWS_ANSI_ENABLED
    return True


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
    """Truncate text so display width fits within max_width, preserving ANSI codes."""
    if _visible_width(text) <= max_width:
        return text
    # Walk through text keeping ANSI sequences intact
    width = 0
    result = []
    i = 0
    while i < len(text):
        # Skip ANSI escape sequences (don't count toward width)
        if text[i] == '\033' and i + 1 < len(text) and text[i + 1] == '[':
            j = i + 2
            while j < len(text) and text[j] not in 'mGHJK':
                j += 1
            result.append(text[i:j + 1])
            i = j + 1
            continue
        cw = _char_width(text[i])
        if width + cw > max_width - 3:
            break
        result.append(text[i])
        width += cw
        i += 1
    return ''.join(result) + "...\033[0m"


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
    "flagscale_train_monitor": "🚂",
    "evict": "🗑️",
    "recall": "↩️",
    "shell_jobs": "🧵",

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
_poll_anim = None


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
    first = lines[0]
    last = lines[-1]
    _print(f"  {first}")
    _print(dim(f"  ... ({len(lines)} lines)"))
    _print(f"  {last}")


# ── Banner ──────────────────────────────────────────────────────────────

def banner(provider, model, context_window=None, extra_lines=None):
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
    ctx_str = f" | Context: {context_window // 1000}k" if context_window else ""
    info = f"Provider: {provider} | Model: {model}{ctx_str}"
    cmds = "Commands: /resume  /reload  /session  /quit"
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



# ── LLM done ────────────────────────────────────────────────────────────

def llm_done(elapsed, input_tokens=None, output_tokens=None,
             cache_read_tokens=None, cache_creation_tokens=None):
    ts = time.strftime("%H:%M:%S")
    parts = [f"[{ts}]", green("✓"), f"{elapsed:.1f}s"]
    if input_tokens is not None:
        token_str = f"↑{_fmt_tokens(input_tokens)}"
        # Show cache breakdown when caching is active
        if cache_read_tokens or cache_creation_tokens:
            cache_parts = []
            if cache_read_tokens:
                cache_parts.append(f"cache:{_fmt_tokens(cache_read_tokens)}")
            if cache_creation_tokens:
                cache_parts.append(f"new:{_fmt_tokens(cache_creation_tokens)}")
            token_str += f"({'+'.join(cache_parts)})"
        parts.append(token_str)
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
    # Display full label without truncation for reviewer visibility
    _print(dim(label), end="", flush=True)
    # shell always spins (blocks for the whole run). shell_jobs only spins on
    # 'wait', which blocks up to `timeout`s — list/poll/kill return instantly
    # and read better inline.
    spins = name == "shell" or (name == "shell_jobs" and args_summary.startswith("wait"))
    if spins:
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

    def mark_done(self, index, elapsed, error=False, detail=""):
        with self._lock:
            self._results[index] = (elapsed, error, detail)

    def _animate(self):
        while not self._stop.is_set():
            self._redraw()
            self._frame += 1
            self._stop.wait(0.5)

    def _redraw(self):
        # Skip in-place animation if ANSI cursor control is not available
        # (e.g. Windows CMD without virtual terminal processing enabled).
        if not _use_color():
            return
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



def parallel_tools_finish():
    """Stop parallel display and do final redraw."""
    global _parallel_display
    if _parallel_display:
        _parallel_display.finish()
        _parallel_display = None


# ── Turn / session summary ──────────────────────────────────────────────

def warn(message):
    """Display a warning message to the user."""
    _print(f"  {yellow('⚠')} {yellow(message)}")


def guard_overridden(guard_name, reason):
    """Display when a guard block is overridden by LLM-provided reason."""
    _print(f"  {yellow('⚡')} Guard override: {bold(guard_name)}")
    _print(f"     {dim(reason)}")


def guard_inject(message):
    """Display a guard inject message to the terminal (visible to user).
    
    Shows all lines with icon, subsequent lines indented to align with text.
    No truncation for reviewer visibility.
    """
    lines = [l for l in message.strip().split('\n') if l.strip()]
    if not lines:
        return
    _print(f"  🛡 {dim(lines[0].strip())}")
    for line in lines[1:]:
        _print(f"     {dim(line.strip())}")


def completion_signal(sentinel):
    """Display the authoritative completion marker AFTER the gate has passed.

    The raw sentinel is stripped from the live stream (SentinelStripper), so this
    is the single place the marker reaches the screen — only once the kernel's
    completion gate accepted the completion. This guarantees the marker never
    appears before, or alongside, a gate block.
    """
    _print()
    _print(green(bold(sentinel)) if _use_color() else sentinel)


def guard_block(message):
    """Display a guard block message to the terminal (visible to user).
    
    First line: red icon + summary. Subsequent lines indented.
    No truncation for reviewer visibility.
    """
    lines = [l for l in message.strip().split('\n') if l.strip()]
    if not lines:
        return
    _print(f"  {red('🚫')} {lines[0].strip()}")
    for line in lines[1:]:
        _print(f"     {line.strip()}")


def guard_escalate(message):
    """Display a guard escalate message to the terminal (visible to user).
    
    Escalate is stronger than inject but not a block — it demands attention
    but doesn't prevent execution. No truncation for reviewer visibility.
    """
    lines = [l for l in message.strip().split('\n') if l.strip()]
    if not lines:
        return
    _print(f"  {yellow('⚠️')} {lines[0].strip()}")
    for line in lines[1:]:
        _print(f"     {line.strip()}")


def turn_summary(turn_num, elapsed, input_tokens, output_tokens,
                 session_input_tokens=None, session_output_tokens=None):
    _stop_all_spinners()
    _print()
    parts = [f"Turn {turn_num}", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    if session_input_tokens is not None:
        parts.append(f"Σ↑{_fmt_tokens(session_input_tokens)} ↓{_fmt_tokens(session_output_tokens or 0)}")
    _print(dim(f"── {' | '.join(parts)} ──"))
    _print()


def interrupted():
    _stop_all_spinners()
    _print(yellow("\n  ⚠  Interrupted. Back to prompt."))


def goodbye():
    _stop_all_spinners()
    _print(green("\n  I'll remember where we left off. See you next time. 🚀\n"))


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
