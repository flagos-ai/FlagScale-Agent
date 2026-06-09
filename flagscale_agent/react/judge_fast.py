"""Fast-path format parsers for Judge — zero LLM calls for simple patterns.

Phase 2 refactoring: separate format parsing from semantic classification.
Semantic classification still goes through LLM.
"""

from __future__ import annotations

import re
from typing import Any

# ── Format parsing regexes (NOT semantic judgment) ────────────────────────

_KNOWLEDGE_CONFIRM_RE = re.compile(
    r"\[PIPELINE_KNOWLEDGE_CONFIRMED:\s*(YES|NO)\]",
    re.IGNORECASE,
)


class FastParser:
    """Fast-path format parsers that skip LLM calls entirely.

    These handle structured patterns where the format itself is the signal,
    not the semantic content. Examples:
    - [PIPELINE_KNOWLEDGE_CONFIRMED: YES/NO]
    """

    @staticmethod
    def parse_knowledge_confirm(text: str) -> bool | None:
        """Extract pipeline knowledge confirmation.

        Returns True (YES), False (NO), or None (not found).
        """
        match = _KNOWLEDGE_CONFIRM_RE.search(text)
        if match:
            return match.group(1).upper() == "YES"
        return None


# ── Fast-path heuristics for classify categories ──────────────────────────

class FastClassifier:
    """Heuristic classifiers that can answer some questions without LLM.

    These are CONSERVATIVE: when uncertain, return None to escalate to LLM.
    Only return a definitive answer when the pattern is unambiguous.
    """

    @staticmethod
    def is_read_only_shell(command: str) -> bool | None:
        """Fast check if command is read-only diagnostic.

        Returns True/False if confident, None if needs LLM.
        """
        cmd = command.strip().lower()

        # Definitely read-only
        read_only_prefixes = (
            "ls", "cat", "head", "tail", "grep", "find", "wc", "file",
            "stat", "which", "type", "echo", "pwd", "env", "printenv",
            "hostname", "uname", "date", "id", "whoami", "ps", "pgrep",
            "nvidia-smi", "rocminfo", "nvcc", "df", "du", "free", "top",
            "htop", "lscpu", "lspci", "lsblk", "mount", "uptime", "nproc",
            "getconf", "locale",
        )

        first_word = cmd.split()[0] if cmd else ""
        if first_word in read_only_prefixes:
            # Check for output redirection (makes it non-read-only)
            if ">" in command or ">>" in command:
                return False
            return True

        # Definitely NOT read-only
        write_prefixes = (
            "rm", "mv", "cp", "mkdir", "touch", "chmod", "chown",
            "pip", "conda", "apt", "yum", "dnf", "brew",
            "git", "wget", "curl -O", "curl -o",
            "python", "torchrun", "deepspeed", "mpirun",
            "kill", "pkill", "killall",
        )

        if any(cmd.startswith(prefix) for prefix in write_prefixes):
            return False

        # Uncertain — escalate to LLM
        return None

    @staticmethod
    def is_dangerous(command: str) -> bool | None:
        """Fast check if command is dangerous.

        Returns True if definitely dangerous, False if definitely safe, None otherwise (escalate to LLM).
        """
        cmd = command.strip().lower()
        first_word = cmd.split()[0] if cmd else ""

        # Definitely safe — version/help queries and read-only diagnostics
        safe_prefixes = (
            "ls", "cat", "head", "tail", "grep", "find", "wc", "file",
            "stat", "which", "type", "echo", "pwd", "env", "printenv",
            "hostname", "uname", "date", "id", "whoami", "ps", "pgrep",
            "nvidia-smi", "rocminfo", "nvcc", "df", "du", "free", "top",
            "htop", "lscpu", "lspci", "lsblk", "mount", "uptime", "nproc",
            "getconf", "locale", "ulimit", "sysctl",
        )
        if first_word in safe_prefixes:
            return False

        # --version / --help flags are always safe
        if cmd.endswith("--version") or cmd.endswith("-V") or cmd.endswith("--help") or cmd.endswith("-h"):
            return False

        # Definitely dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /*",
            "rm -rf ~/*",
            "chmod 777 /",
            "chmod -r 777 /",
            "mkfs",
            "dd if=/dev/zero of=/dev/sd",
            ":(){ :|:& };:",  # fork bomb
        ]

        for pattern in dangerous_patterns:
            if pattern in cmd:
                return True

        # Not obviously dangerous — escalate to LLM for nuanced judgment
        return None

    @staticmethod
    def is_training_command(command: str) -> bool | None:
        """Fast check if command launches training.

        Returns True if definitely training, False if definitely not, None if uncertain.
        """
        cmd = command.strip().lower()
        first_word = cmd.split()[0] if cmd else ""

        # Definitely NOT training (check first — grep/cat/ls mentioning torchrun is not training)
        not_training_prefixes = ("grep", "cat", "ls", "find", "head", "tail", "echo", "ps", "pgrep")
        if first_word in not_training_prefixes:
            return False

        # Definitely training
        training_launchers = ("torchrun", "deepspeed", "horovodrun", "mpirun")
        if first_word in training_launchers:
            # But not if it's --help or --version
            if "--help" in cmd or "--version" in cmd or cmd.endswith(" -h"):
                return False
            return True

        # Uncertain
        return None

    @staticmethod
    def is_kill_command(command: str) -> bool | None:
        """Fast check if command kills processes.

        Returns True/False if confident, None if uncertain.
        """
        cmd = command.strip().lower()
        first_word = cmd.split()[0] if cmd else ""

        if first_word in ("kill", "pkill", "killall"):
            return True

        if first_word in ("ps", "pgrep", "grep"):
            return False

        return None
