"""TrainingAttemptGuard — 2-Strike Rule at attempt granularity.

An "attempt" is the cycle: edit code → launch training → observe result.
This guard tracks attempts as semantic units, not individual tool calls.

Unlike CircuitBreakerGuard (which counts consecutive tool-level errors),
this guard understands the training workflow:
1. Detect code modification (edit_file/write_file to .py files)
2. Detect training launch (shell with run.py/torchrun/flagscale)
3. Detect training result (monitor/find_latest_log returning crash or success)

When 2 consecutive attempts fail with the SAME error category,
the guard BLOCKS further launches until the agent demonstrates
genuine root cause analysis (reading source code deeply).

Training error categories (more granular than CircuitBreaker):
- shape: tensor size mismatch, dimension errors, broadcast failures
- attribute: AttributeError, missing methods/properties
- import: ModuleNotFoundError, ImportError
- numerical: NaN, Inf, grad_norm explosion, loss divergence
- nccl: NCCL errors, timeout, rank mismatch
- oom: CUDA OOM, memory allocation failures
- config: unknown arguments, invalid config values
- data: data loading errors, wrong batch format
- general: anything else
"""

import re
import time
from dataclasses import dataclass, field

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Error category patterns for training failures
_ERROR_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    "shape": [
        re.compile(r"size\s+mismatch", re.I),
        re.compile(r"shape\s+.*(?:expected|got)", re.I),
        re.compile(r"RuntimeError.*(?:mat1|mat2|matmul).*(?:size|shape|dim)", re.I),
        re.compile(r"cannot\s+(?:reshape|broadcast)", re.I),
        re.compile(r"dimension\s+\d+\s+(?:does\s+not|doesn't)\s+match", re.I),
    ],
    "attribute": [
        re.compile(r"AttributeError:\s+'?\w+", re.I),
        re.compile(r"has\s+no\s+attribute", re.I),
        re.compile(r"'NoneType'\s+object\s+has\s+no\s+attribute", re.I),
    ],
    "import": [
        re.compile(r"ModuleNotFoundError", re.I),
        re.compile(r"ImportError", re.I),
        re.compile(r"No\s+module\s+named", re.I),
    ],
    "numerical": [
        re.compile(r"\bnan\b.*loss", re.I),
        re.compile(r"loss.*\bnan\b", re.I),
        re.compile(r"grad.norm.*(?:inf|nan|\d{4,})", re.I),
        re.compile(r"overflow", re.I),
        re.compile(r"underflow", re.I),
    ],
    "nccl": [
        re.compile(r"NCCL\s+error", re.I),
        re.compile(r"NCCL.*timeout", re.I),
        re.compile(r"ProcessGroupNCCL", re.I),
        re.compile(r"rank\s+\d+.*(?:crash|fail|abort)", re.I),
    ],
    "oom": [
        re.compile(r"CUDA\s+out\s+of\s+memory", re.I),
        re.compile(r"OutOfMemoryError", re.I),
        re.compile(r"OOM", re.I),
        re.compile(r"allocat.*failed", re.I),
    ],
    "config": [
        re.compile(r"(?:Unrecognized|Unknown|Invalid)\s+(?:argument|option|key|config)", re.I),
        re.compile(r"MissingMandatoryValue", re.I),
        re.compile(r"ConfigurationError", re.I),
        re.compile(r"Could\s+not\s+(?:load|find)\s+.*\.yaml", re.I),
    ],
    "data": [
        re.compile(r"(?:data|dataset|dataloader).*(?:error|fail|not found)", re.I),
        re.compile(r"KeyError.*(?:input_ids|labels|attention_mask)", re.I),
        re.compile(r"(?:\.bin|\.idx)\s+(?:not found|missing)", re.I),
    ],
}

# Launch patterns (same as TrainingRuntimeGuard but compiled here for independence)
_LAUNCH_RE = re.compile(
    r"\brun\.py\b.*\baction\s*=\s*run\b"
    r"|\brun\.py\b.*--config"
    r"|\bconda\s+run\b.*\brun\.py\b"
    r"|\btorchrun\b"
    r"|\bflagscale\s+train\b"
    r"|\btrain\.py\b"
    r"|\bpretrain\.py\b",
    re.IGNORECASE,
)

# Monitor failure pattern
_MONITOR_CRASH_RE = re.compile(
    r"TRAINING CRASHED|Traceback\s+\(most\s+recent"
    r"|RuntimeError:|NCCL\s+error|Out\s+of\s+memory"
    r"|CUDA\s+error|AttributeError:|ModuleNotFoundError:"
    r"|exit\s+code\s+[1-9]|Process\s+died",
    re.IGNORECASE,
)


@dataclass
class AttemptRecord:
    """Record of a single training attempt."""
    started_at: float = 0.0
    error_category: str = ""
    error_snippet: str = ""
    files_modified: list[str] = field(default_factory=list)
    hypothesis: str = ""  # Agent's stated hypothesis (if any)
    succeeded: bool = False


class TrainingAttemptGuard(Guard):
    """2-Strike Rule at attempt granularity.
    
    Tracks the edit→launch→observe cycle. When 2 consecutive attempts
    fail with the same error category, BLOCKS further launches until
    the agent reads source code (demonstrating genuine diagnosis).
    """

    name = "training_attempt"
    priority = 15  # Higher priority than CircuitBreaker (25) and TrainingRuntime (20)
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override with a substantive root cause explanation."""
        if reason and len(reason.strip()) > 20:
            return True
        return False

    STRIKE_THRESHOLD = 2  # 2 same-category failures → block
    SOURCE_READ_REQUIREMENT = 3  # Must read 3+ source files to unblock

    def __init__(self, strike_threshold: int = 2, source_read_requirement: int = 3):
        self.STRIKE_THRESHOLD = strike_threshold
        self.SOURCE_READ_REQUIREMENT = source_read_requirement

        # Attempt tracking
        self._attempts: list[AttemptRecord] = []
        self._current_attempt: AttemptRecord | None = None

        # State
        self._is_blocked = False
        self._blocked_category = ""
        self._source_reads_since_block = 0
        self._hypothesis_declared = False

        # File modification tracking
        self._recent_file_edits: list[str] = []

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block training launches if 2-Strike triggered."""
        if not self._is_blocked:
            return None

        # Check if trying to launch training
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _LAUNCH_RE.search(cmd):
                if self._source_reads_since_block < self.SOURCE_READ_REQUIREMENT:
                    return GuardVerdict.block(
                        f"[2-Strike Rule] BLOCKED: {self.STRIKE_THRESHOLD} consecutive "
                        f"attempts failed with category '{self._blocked_category}'. "
                        f"You must read source code to diagnose the root cause before retrying. "
                        f"Progress: {self._source_reads_since_block}/{self.SOURCE_READ_REQUIREMENT} "
                        f"source files read. "
                        f"\n\nPrevious failures:\n"
                        + "\n".join(
                            f"  Attempt {i+1}: [{a.error_category}] {a.error_snippet[:150]}"
                            for i, a in enumerate(self._attempts[-self.STRIKE_THRESHOLD:])
                            if a.error_category
                        )
                        + "\n\nBefore launching again, you MUST:\n"
                        "1. State your hypothesis about the root cause\n"
                        "2. Read relevant source code to verify\n"
                        "3. Explain what was wrong and how the fix addresses it",
                        reason=f"2_strike_{self._blocked_category}",
                    )
                elif not self._hypothesis_declared:
                    return GuardVerdict.block(
                        f"[2-Strike Rule] You've read source code, but haven't stated your "
                        f"hypothesis. Before relaunching, declare:\n"
                        f"  HYPOTHESIS: [what's actually wrong]\n"
                        f"  FIX: [what you changed and why]\n"
                        f"  EXPECTED: [what you expect to see this time]",
                        reason="2_strike_no_hypothesis",
                    )
                else:
                    # Unblock — hypothesis declared, enough source read
                    self._is_blocked = False
                    self._blocked_category = ""
                    self._source_reads_since_block = 0
                    self._hypothesis_declared = False

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track the attempt lifecycle: edit → launch → result."""
        
        # Track file modifications
        if ctx.tool_name in ("edit_file", "write_file"):
            path = ctx.tool_args.get("path", "")
            if path.endswith(".py"):
                self._recent_file_edits.append(path)

        # Track source code reading (for unblocking)
        if self._is_blocked and ctx.tool_name == "read_file":
            path = ctx.tool_args.get("path", "")
            if path.endswith(".py"):
                self._source_reads_since_block += 1

        # Track hypothesis declaration (look for it in LLM text)
        # Note: This is approximated by checking if the agent used inject/text
        # The actual mechanism is in the system prompt enforcement

        # Detect training launch → start new attempt
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _LAUNCH_RE.search(cmd):
                self._current_attempt = AttemptRecord(
                    started_at=time.time(),
                    files_modified=self._recent_file_edits.copy(),
                )
                self._recent_file_edits.clear()

        # Detect training result
        if self._current_attempt and ctx.tool_result:
            if ctx.tool_name in ("monitor", "find_latest_log", "parse_training_metrics"):
                if _MONITOR_CRASH_RE.search(ctx.tool_result):
                    # Training failed
                    category = self._classify_training_error(ctx.tool_result, ctx)
                    self._current_attempt.error_category = category
                    self._current_attempt.error_snippet = self._extract_error_snippet(ctx.tool_result)
                    self._current_attempt.succeeded = False
                    self._attempts.append(self._current_attempt)
                    self._current_attempt = None

                    # Check 2-Strike condition
                    verdict = self._check_strike_condition()
                    if verdict:
                        return verdict
                elif self._is_success_result(ctx.tool_result):
                    # Training succeeded
                    self._current_attempt.succeeded = True
                    self._attempts.append(self._current_attempt)
                    self._current_attempt = None
                    # Reset strike tracking on success
                    self._is_blocked = False
                    self._blocked_category = ""

        return None

    def _check_strike_condition(self) -> GuardVerdict | None:
        """Check if the last N attempts failed with the same category."""
        if len(self._attempts) < self.STRIKE_THRESHOLD:
            return None

        recent = self._attempts[-self.STRIKE_THRESHOLD:]
        categories = [a.error_category for a in recent if a.error_category]

        if len(categories) == self.STRIKE_THRESHOLD and len(set(categories)) == 1:
            # Same category N times in a row!
            self._is_blocked = True
            self._blocked_category = categories[0]
            self._source_reads_since_block = 0
            self._hypothesis_declared = False

            return GuardVerdict.inject(
                f"\n[2-STRIKE RULE TRIGGERED]\n"
                f"Category '{self._blocked_category}' has failed {self.STRIKE_THRESHOLD} "
                f"consecutive times. The approach is fundamentally wrong.\n\n"
                f"STOP making incremental fixes. You MUST:\n"
                f"1. Re-read the relevant source code end-to-end\n"
                f"2. Identify the systemic assumption that is wrong\n"
                f"3. State your root cause hypothesis clearly\n"
                f"4. Propose a fundamentally different approach\n\n"
                f"Further training launches are BLOCKED until you do this.",
                reason=f"2_strike_triggered_{self._blocked_category}",
            )

        return None

    def _classify_training_error(self, text: str, ctx: GuardContext = None) -> str:
        """Classify training error into a fine-grained category.
        
        Tiered approach:
        1. Regex fast-path: instant, zero cost, handles 80% of standard errors
        2. LLM fallback: for ambiguous errors that regex can't classify
        """
        # Tier 1: Regex fast-path
        for category, patterns in _ERROR_CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    return category

        # Tier 2: LLM classify (if available and text is non-trivial)
        if ctx and ctx.classify_fn and len(text) > 30:
            try:
                from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
                result, source = get_judge_result(
                    ctx.classify_fn,
                    "training_error_category",
                    {"error_text": text[:2000]},
                    default={"category": "general", "confidence": 0.0},
                )
                if is_trusted(source) and isinstance(result, dict):
                    category = result.get("category", "general")
                    confidence = result.get("confidence", 0.0)
                    # Only trust LLM if confidence >= 0.6
                    if confidence >= 0.6 and category in _ERROR_CATEGORY_PATTERNS:
                        return category
            except Exception:
                pass

        return "general"

    def _extract_error_snippet(self, text: str) -> str:
        """Extract the most relevant error line from training output."""
        lines = text.split("\n")
        # Look for the actual error line (after "Traceback")
        for i, line in enumerate(lines):
            if "Error:" in line or "Exception:" in line:
                return line.strip()[:200]
        # Fallback: last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:200]
        return text[:200]

    @staticmethod
    def _is_success_result(text: str) -> bool:
        """Check if monitor output indicates training success."""
        success_patterns = [
            r"training\s+completed",
            r"iteration\s+\d+/\s*\d+.*lm\s+loss",  # Normal metrics output
            r"saved\s+checkpoint",
        ]
        for pat in success_patterns:
            if re.search(pat, text, re.I):
                return True
        return False

    def declare_hypothesis(self, hypothesis: str):
        """Called when agent declares a hypothesis (from prompt enforcement)."""
        self._hypothesis_declared = True
        if self._current_attempt:
            self._current_attempt.hypothesis = hypothesis

    def reset_turn(self):
        """Attempt state persists across turns (session-level)."""
        pass

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)

    @property
    def failed_attempts(self) -> list[AttemptRecord]:
        return [a for a in self._attempts if not a.succeeded]
