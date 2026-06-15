"""ErrorClassifierGuard — classifies tool errors and injects recovery suggestions.

Uses two-phase detection:
1. Cheap trigger: regex patterns against error text
2. Precise: classify_fn for ambiguous cases

v2: Expanded categories for training infrastructure:
- nccl_topology: NCCL/collective communication failures
- shape_mismatch: Tensor shape/size errors
- checkpoint: Checkpoint loading/saving errors
- cuda_driver: CUDA/driver version incompatibilities
- oom: Out of memory (GPU or CPU)
- import_error: Missing imports/modules
"""

from __future__ import annotations

import re
from typing import Optional

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState

# Error category patterns — ordered from most specific to most general
_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Training-specific (v2 additions)
    ("nccl_topology", re.compile(
        r"NCCL\s*(error|timeout|warn)|"
        r"ncclSystemError|ncclInternalError|"
        r"ProcessGroupNCCL|all_reduce.*timeout|"
        r"Watchdog caught.*NCCL|"
        r"NCCL_DEBUG|nccl_net|ib_|rdma|"
        r"collective.*timed\s*out",
        re.IGNORECASE,
    )),
    ("oom", re.compile(
        r"CUDA\s*out\s*of\s*memory|"
        r"OutOfMemoryError|"
        r"OOM|"
        r"torch\.cuda\.OutOfMemoryError|"
        r"RuntimeError.*allocat|"
        r"Cannot\s*allocate\s*memory|"
        r"CUBLAS_STATUS_ALLOC_FAILED|"
        r"Killed.*signal\s*9",
        re.IGNORECASE,
    )),
    ("shape_mismatch", re.compile(
        r"size\s*mismatch|"
        r"shape.*mismatch|"
        r"RuntimeError.*expected.*shape|"
        r"RuntimeError.*mat[12].*must.*match|"
        r"expected.*got.*shape|"
        r"cannot\s*reshape|"
        r"invalid.*dimensions|"
        r"incompatible.*tensor.*size",
        re.IGNORECASE,
    )),
    ("checkpoint", re.compile(
        r"Error.*load.*checkpoint|"
        r"Error.*state_dict|"
        r"missing\s*keys|unexpected\s*keys|"
        r"Checkpoint.*not\s*found|"
        r"invalid.*checkpoint|"
        r"DistributedCheckpointing|"
        r"ShardedTensor.*load|"
        r"cannot.*load.*model.*from",
        re.IGNORECASE,
    )),
    ("cuda_driver", re.compile(
        r"CUDA\s*driver\s*version|"
        r"CUDA\s*runtime\s*version|"
        r"CUDA\s*error|"
        r"cudaErrorInsufficientDriver|"
        r"CUDA_HOME|CUDA_PATH|"
        r"nvcc.*not\s*found|"
        r"CUDA.*capability|"
        r"sm_\d+.*not\s*supported|"
        r"libcuda|libcudart|libnccl",
        re.IGNORECASE,
    )),
    ("import_error", re.compile(
        r"ModuleNotFoundError|"
        r"ImportError|"
        r"No module named|"
        r"cannot import name",
        re.IGNORECASE,
    )),
    # Original categories (broadened)
    ("env_missing", re.compile(
        r"command not found|"
        r"No such file or directory|"
        r"FileNotFoundError|"
        r"not recognized|"
        r"executable.*not found|"
        r"which:.*no\s+\w+\s+in",
        re.IGNORECASE,
    )),
    ("permission", re.compile(
        r"Permission denied|"
        r"PermissionError|"
        r"EACCES|Operation not permitted|"
        r"read-only\s*file\s*system",
        re.IGNORECASE,
    )),
    ("network", re.compile(
        r"Connection\s*(refused|timed\s*out|reset)|"
        r"Could not resolve|"
        r"Network\s*unreachable|"
        r"TimeoutError|ConnectionError|"
        r"HTTP\s*Error\s*(4|5)\d\d|"
        r"SSHException",
        re.IGNORECASE,
    )),
    ("config", re.compile(
        r"KeyError|"
        r"ConfigAttribute|"
        r"Hydra.*error|"
        r"omegaconf.*error|"
        r"invalid.*config|"
        r"required.*field|"
        r"YAML.*error|"
        r"MissingMandatoryValue",
        re.IGNORECASE,
    )),
    ("resource", re.compile(
        r"disk\s*(full|space)|"
        r"No\s*space\s*left|"
        r"Disk\s*quota|"
        r"too many open files|"
        r"EMFILE|ENOMEM",
        re.IGNORECASE,
    )),
]

# Recovery suggestions per category
_RECOVERY_SUGGESTIONS: dict[str, str] = {
    "nccl_topology": (
        "NCCL communication error. Common fixes:\n"
        "1. Check NCCL_DEBUG=INFO for detailed topology info\n"
        "2. Verify NCCL_IB_DISABLE / NCCL_P2P_LEVEL settings\n"
        "3. Check if all nodes can reach each other (SSH, IB)\n"
        "4. Try NCCL_TIMEOUT=1800 for slow initialization"
    ),
    "oom": (
        "Out of memory. Options:\n"
        "1. Reduce batch size or sequence length\n"
        "2. Increase TP/PP to distribute model across more GPUs\n"
        "3. Enable activation checkpointing (recompute_granularity=selective)\n"
        "4. Use mixed precision (bf16/fp16)\n"
        "5. Check for memory leaks: torch.cuda.memory_summary()"
    ),
    "shape_mismatch": (
        "Tensor shape mismatch. Common causes:\n"
        "1. Model config (hidden_size, num_heads, etc.) doesn't match checkpoint\n"
        "2. TP/PP degree changed without re-converting checkpoint\n"
        "3. Vocabulary size mismatch (padded vs unpadded)\n"
        "4. Sequence length exceeds model's positional embedding size"
    ),
    "checkpoint": (
        "Checkpoint loading error. Check:\n"
        "1. Checkpoint path exists and has correct TP/PP structure\n"
        "2. Key names match: use inspect_checkpoint to compare\n"
        "3. TP/PP degree matches what checkpoint was saved with\n"
        "4. model_type/architecture matches checkpoint format"
    ),
    "cuda_driver": (
        "CUDA/driver incompatibility. Check:\n"
        "1. nvidia-smi shows the GPU and driver version\n"
        "2. PyTorch CUDA version matches system: python -c 'import torch; print(torch.version.cuda)'\n"
        "3. CUDA_HOME is set correctly: echo $CUDA_HOME\n"
        "4. For sm_90 (H100): need CUDA >= 11.8, PyTorch >= 2.0"
    ),
    "import_error": (
        "Module not found. Check:\n"
        "1. Conda environment is activated correctly\n"
        "2. Package is installed: pip show <package>\n"
        "3. PYTHONPATH includes the project root\n"
        "4. If local package: check __init__.py exists"
    ),
    "env_missing": (
        "File or command not found. Check:\n"
        "1. PATH includes the binary location\n"
        "2. File path is correct (typo?)\n"
        "3. Package is installed in the active environment"
    ),
    "permission": (
        "Permission denied. Options:\n"
        "1. Check file ownership: ls -la <path>\n"
        "2. Use shared storage paths with group write\n"
        "3. Don't use sudo unless absolutely necessary"
    ),
    "network": (
        "Network/connection error. Check:\n"
        "1. SSH connectivity to target hosts\n"
        "2. Proxy/firewall settings\n"
        "3. DNS resolution\n"
        "4. Port availability (especially 29500 for torch.distributed)"
    ),
    "config": (
        "Configuration error. Check:\n"
        "1. YAML syntax (indentation, colons, quotes)\n"
        "2. All required fields are present\n"
        "3. Run validate_config to catch structural issues\n"
        "4. Check Hydra override syntax"
    ),
    "resource": (
        "Disk/resource exhaustion. Check:\n"
        "1. df -h on relevant mount points\n"
        "2. du -sh on large directories (checkpoints, logs)\n"
        "3. Clean up old checkpoints/logs\n"
        "4. Check ulimits: ulimit -a"
    ),
}


class ErrorClassifierGuard(Guard):
    """Classifies errors and suggests recovery actions.

    v2: Expanded from 5 to 11 error categories with training-specific patterns.
    Provides contextual recovery suggestions based on the FlagScale domain.
    """

    name = "error_classifier"
    priority = 25
    activate_on_states = {AgentState.EXECUTING}

    def __init__(self):
        self._error_history: list[str] = []
        self._last_category: str | None = None
        self._consecutive_same_category: int = 0

    def set_shared_state(self, shared_state):
        """Accept shared state (unused in this guard but required by interface)."""
        pass

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_result:
            return None

        # Only classify if the result looks like an error
        result_lower = ctx.tool_result.lower() if ctx.tool_result else ""
        if not self._looks_like_error(result_lower):
            # Clear consecutive counter on success
            if self._last_category:
                self._consecutive_same_category = 0
            return None

        # Phase 1: Pattern matching
        category = self._classify_by_pattern(ctx.tool_result)

        if not category:
            # Phase 2: Try LLM judge for ambiguous cases (if available)
            if ctx.classify_fn:
                result, source = get_judge_result(
                    ctx.classify_fn,
                    "error_category",
                    {"error_text": ctx.tool_result[:500], "tool": ctx.tool_name},
                    default="unknown",
                )
                if is_trusted(source) and result in _RECOVERY_SUGGESTIONS:
                    category = result
            if not category:
                return None

        # Track for circuit breaker integration
        self._error_history.append(category)
        if len(self._error_history) > 20:
            self._error_history = self._error_history[-20:]

        # Track consecutive same-category errors
        if category == self._last_category:
            self._consecutive_same_category += 1
        else:
            self._consecutive_same_category = 1
        self._last_category = category

        # Build recovery message
        suggestion = _RECOVERY_SUGGESTIONS.get(category, self._generic_suggestion(category))

        # Escalate if same error repeated
        prefix = ""
        if self._consecutive_same_category >= 3:
            prefix = (
                f"⚠️ Same error category '{category}' hit {self._consecutive_same_category} times. "
                f"Stop and diagnose the root cause before retrying.\n\n"
            )

        return GuardVerdict.inject(
            f"{prefix}[ErrorClassifier: {category}]\n{suggestion}",
            reason=f"error_classified_{category}",
            category=f"error_{category}",
        )

    @staticmethod
    def _looks_like_error(text_lower: str) -> bool:
        """Quick check — does this look like an error output?"""
        error_indicators = (
            "error", "traceback", "exception", "failed", "fatal",
            "denied", "not found", "no such", "cannot", "killed",
            "timeout", "refused", "mismatch",
        )
        return any(ind in text_lower for ind in error_indicators)

    @staticmethod
    def _classify_by_pattern(text: str) -> Optional[str]:
        """Classify error text using regex patterns."""
        for category, pattern in _ERROR_PATTERNS:
            if pattern.search(text):
                return category
        return None

    @staticmethod
    def _generic_suggestion(category: str) -> str:
        """Fallback suggestion for unknown categories."""
        return f"Unknown error ({category})"

    def reset_turn(self):
        pass
