"""Unified Judge + JudgeBudget — Phase 2 tiered architecture.

Three-tier classification:
1. Fast path: Heuristic classifiers (zero LLM cost, instant)
2. Cache path: MD5-keyed per-category cache (zero LLM cost)
3. Deep path: LLM calls with multi-round support

Additional features:
- classify_batch(): merge multiple classify calls into one LLM request
- health(), result(), skill(), complexity(): domain-specific judges
- Per-turn call budget (max 64/turn)
- Source tracking: SOURCE_FAST / SOURCE_CACHE / SOURCE_LLM / SOURCE_DEFAULT / SOURCE_UNAVAILABLE
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

from flagscale_agent.react.judge_fast import FastClassifier


# ── Classify prompts (replaces ALL regex/keyword matching) ─────────────────

_CLASSIFY_PROMPTS: Dict[str, str] = {
    "is_error": """\
Determine if this is a REAL execution error that needs attention.

Context: {context}

Answer NO for: warnings, deprecation notices, informational logs,
HTTP error pages returned by web_fetch (these are the content, not an error in the tool itself),
expected cleanup messages, compile messages, pkg_resources deprecation notices.

Answer YES for: crashes, exceptions, OOM, NCCL failures, missing modules,
assertion errors, CUDA errors, process termination, non-zero exit code.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_success": """\
Did this shell command complete successfully?

Context: {context}

Answer YES when the output shows the command completed normally.
Answer NO when there are errors, failures, or unclear outcome.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_dangerous": """\
Is this shell command DANGEROUS and should be BLOCKED?

Context: {context}

Answer YES for: rm -rf on system paths (/ or ~), chmod 777 on system dirs,
fork bombs, mkfs, dd without clear target, redirects to /dev/sd*.
Answer NO for: normal file operations, package management, regular shell commands.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_read_only_shell": """\
Is this a read-only diagnostic command (safe to run anytime)?

Context: {context}

Answer YES for: grep, find, cat, ls, head, tail, wc, file, stat, which, type,
echo, pwd, env, printenv, hostname, uname, date, id, whoami, ps, pgrep, nvidia-smi, rocminfo.
Answer NO for: anything that modifies files, installs packages, launches processes, or writes data.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_training_command": """\
Is this ACTUALLY launching a distributed training job?

Context: {context}

Answer YES for: torchrun, deepspeed, flagscale train (not --dryrun), python pretrain_*.py,
mpirun with training script, horovodrun.
Answer NO for: importing modules, config reading, grep/log analysis,
dryrun, --help, --version, training utilities.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_kill_command": """\
Is this a process kill command?

Context: {context}

Answer YES for: kill, pkill, killall, or equivalent.
Answer NO for anything else.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_training_failure": """\
Did this training output indicate a REAL failure that stopped training?

Context: {context}

Answer YES for: crashes, OOM, NCCL errors, unrecoverable exceptions,
process exit with error, missing dependencies.
Answer NO for: warnings that don't stop training, expected init messages,
non-fatal deprecation warnings, wandb/logging issues.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_zombie_gpu": """\
Does this output indicate zombie GPU processes?

Context: {context}

Answer YES if GPU memory is held by dead/stale processes, or memory allocation conflicts.
Answer NO if GPU state is clean or processes are legitimately running.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_stuck_in_loop": """\
Is this agent stuck in a repetitive loop without making progress?

Context: {context}

Answer YES if: the agent is repeating the same operations without gaining new information,
retrying failed operations without changing approach, or cycling between the same tools
with no state change.
Answer NO if: the agent is reading DIFFERENT files/URLs to gather diverse information,
making incremental progress toward a goal, or following a systematic investigation plan
where each step targets different data.
Reply ONLY: {"real": true/false, "need_more": null}""",

    "is_user_porting_confirm": """\
Is the user choosing between Mode B and Mode C?

Context: {context}

Answer "mode_b" if Mode B / Megatron native / 模式B / 原生.
Answer "mode_c" if Mode C / wrapper / 模式C / 包装.
Answer "" (empty) if neither.
Reply ONLY: {"decision": "mode_b"/"mode_c"/""}""",

    "checklist_rule_batch": """\
You are a constraint checker. Below is a tool call and a list of constraints to evaluate.

Tool call context:
{context}

Constraints (each with id, description, prompt):
{items}

For each constraint, follow these two steps:

Step 1 — SCOPE GATE: Each constraint's prompt begins with "SCOPE: <condition>". This condition defines when the constraint is applicable. Read it carefully. If the tool call (tool name + arguments + result) does NOT satisfy the SCOPE condition, this constraint is not applicable — SKIP it entirely.

Step 2 — VIOLATION CHECK: Only if SCOPE matches, check the "CHECK:" part. Does the tool call actually exhibit the violation pattern?

FINAL CROSS-CHECK: Before outputting, review every violation you flagged. For each one, ask: "Does the SCOPE condition genuinely describe this tool call?" If you flagged anything where the answer is no, REMOVE it.

Reply ONLY: {{"violations": [{{"id": "constraint_id", "reason": "one-line explanation"}}]}}
If no constraint is both in-scope AND violated, reply: {{"violations": []}}""",

    "checklist_rule": """\
Evaluate whether a tool action violates a checklist constraint. You are given:

- **Description**: what the constraint requires
- **Prompt**: the specific condition to check for
- **Context**: the tool call details (name, args, result) plus any auto-detected runtime facts

Auto-detected facts (if present) are authoritative — trust them over any inference from the tool call itself. Examples:
- _facts.shared_storage: paths like ["/share/project"] mean shared storage IS available
- _facts.driver_version: the actual NVIDIA driver version from nvidia-smi
- _facts.gpu_count: the actual GPU count detected at startup

Context: {context}

Constraint (id={item_id}): {description}

Check for: {prompt}

Reply ONLY: {{"match": true/false, "reason": "one-line explanation of why this does or does not violate the constraint"}}""",

    "extract_constraints": """\
Read the skill content below and extract ONLY constraints that can be checked by looking at a single tool call result (shell command + output, file write, file read).

Principle: a constraint is valid if, given the tool call context, an LLM can answer "does this tool call violate the rule?" with confidence. If the answer requires knowing what the agent did NOT do, or requires multi-step reasoning about the agent's plan, SKIP it.

For each constraint you extract, output:
- id: snake_case prefixed with the skill name
- description: 1 line
- tool_names: list of tool names that can trigger this constraint, e.g. ["shell"] or ["shell", "write_file"]. MUST be non-empty — every constraint must specify which tools it applies to. Never use [].
- keywords: list of COMPLETE phrases (>= 4 chars each) that identify the scenario. ANY one keyword match triggers the constraint (OR logic). Use multi-word phrases for precision to avoid false triggers from file paths or unrelated text.
- prompt: "SCOPE: <concrete condition>. CHECK: <violation signal>."
- correction: 1-sentence warning to show the agent

CRITICAL keyword rules:
- Keywords use OR logic: if ANY keyword in the list appears in the command, the constraint is triggered for LLM judgment.
- Each keyword MUST be a specific, complete phrase that uniquely identifies the scenario.
- NEVER use single generic words like "flagscale", "torch", "python", "install" alone — these match file paths and unrelated commands, causing false triggers.
- GOOD examples:
  - ["pip install flagscale", "pip3 install flagscale"] — specific action+target
  - ["pip install apex", "pip3 install apex"] — won't match "ls /path/apex/"
  - ["conda create --name", "conda create -n"] — specific flag usage
  - ["rm -rf", "rmdir", "shutil.rmtree"] — multiple variants of same action (OR)
- BAD examples:
  - ["pip install", "flagscale"] — "flagscale" alone matches any path containing it
  - ["torch"] — matches any file path with "torch" in it
  - ["install"] — too generic, matches everything

- tool_names MUST be non-empty. If you cannot determine which tool a constraint applies to, skip it.

Skill content:
{skill_content}

Reply ONLY with a JSON array. If no checkable constraints, return []. Do NOT invent constraints the skill doesn't describe.""",

}


# ── Health judge prompt ──────────────────────────────────────────────────

_HEALTH_JUDGE_PROMPT = """\
You are monitoring a running shell command. Analyze its status and decide
whether it should continue or be terminated.

Command: {command}
Total elapsed: {elapsed}
Output changed since last check: {output_changed}
Consecutive checks with no output change: {stall_count}
Recent output:
{output}

## Phase-aware monitoring

Identify the command's current lifecycle phase and adapt your judgment:

- STARTUP (no output yet, imports loading, initializing): check frequently (10-30s). Early failures are common.
- INSTALLING (pip/conda installing packages, "Installing collected packages", extracting wheels, compiling extensions): moderate (60-120s). Package installation involves disk I/O that produces no stdout — this is NORMAL. Large packages (transformer_engine, torch, apex, flash-attn, onnxscript) can take 3-10 minutes during the "Installing collected packages" phase with zero output.
- COMPILING (gcc/nvcc/ninja building C++/CUDA extensions, "Building wheel", "running build_ext"): very patient (120-300s). Source builds are CPU-intensive and produce minimal output between compilation units.
- DOWNLOADING (wget, curl, git clone, pip downloading): moderate (30-60s). Network operations MUST show progress indicators (percentages, "Receiving objects", transfer rates). If no progress indicator has EVER appeared, the connection may have failed silently.
- LOADING (model weights loading, data loading, progress bars advancing): moderate (30-60s).
- STABLE (training iterations running, loss printing regularly): relaxed (120-300s).
- ANOMALY (errors in output, repeated failures, connection refused, segfault): check soon (10-15s) or kill.

## Key judgment rules

1. NEVER kill a command just because output hasn't changed — first determine the phase.
   Silent phases are EXPECTED for: package installation, source compilation, large file extraction, model weight loading.

2. Kill criteria (ALL must be met):
   - Output has stalled AND
   - The stall duration exceeds what's reasonable for the identified phase AND
   - There is no evidence the operation is still working (no disk/CPU activity indicators in output)

3. Kill immediately if:
   - Repeated error messages or crash signatures in output
   - Network failures with no retry mechanism (ConnectionRefused, DNS failure)
   - Deadlock indicators (process stuck after error, infinite retry loops)
   - The command contains embedded sleep/wait with no way to verify the waited process is alive — after 2-3 minutes of silence, kill so the agent can check external state

4. Phase-specific patience limits (KILL if exceeded with no progress):
   - `pip install` at "Installing collected packages" phase: allow up to 10 minutes of silence
   - Source builds (pip install --no-build-isolation, setup.py, cmake, ninja): allow up to 30 minutes
   - `git clone` / `git fetch`: allow up to 5 minutes ONLY IF progress indicators ("Receiving objects", "Resolving deltas", percentages) have appeared. If the ONLY output is "Cloning into '...'" with NO progress indicators after 2 minutes, the remote is likely unreachable — KILL.
   - `wget` / `curl` downloads: allow up to 5 minutes ONLY IF transfer rate or progress was shown. No progress after 1 minute = KILL.
   - conda create/install: allow up to 10 minutes (solver can be slow)

5. Network operation stall detection:
   - git/wget/curl/pip download phases MUST show progress within the first 2 minutes
   - "Cloning into 'X'..." with no subsequent "Receiving objects" or "remote: Counting objects" = connection hanging
   - "Connecting to..." with no follow-up = connection timeout, kill after 1-2 minutes
   - If total elapsed > 10 minutes and output has NEVER changed from the initial message, KILL unconditionally

6. When uncertain about install/compile phases: increase next_check_seconds rather than killing.
   When uncertain about network operations: KILL — network hangs don't self-resolve.

Reply with ONLY a JSON object:
{{"kill": true/false, "reason": "...", "next_check_seconds": <int 10-300>}}

If everything looks normal and healthy, set reason to empty string.
Only provide a reason when there is something noteworthy - an issue, a phase transition, or a kill decision."""


_RESULT_JUDGE_PROMPT = """\
You are analyzing the output of a shell command run by an AI infrastructure agent.

Command: {command}
Duration: {elapsed:.0f}s
Output (last 3000 chars):
{output}

Identify issues and provide SHORT, actionable annotations. Consider:
- Non-zero exit code or error messages: identify root cause
- CUDA/cuDNN/driver version conflicts: give specific diagnosis commands
- Network errors (connection refused, timeout, DNS): suggest proxy or retry
- Download failures: suggest resume with wget -c or curl -C -
- PyTorch/CUDA incompatibility: suggest version check commands
- Inefficient patterns (sleep+tail for monitoring): suggest find_latest_log or timeout+tail -f
- Log searching with find/ls -R/ls -lt: suggest find_latest_log tool or workspace_experiment list
- Training launch (flagscale/torchrun/deepspeed): remind to verify GPU utilization and logs
- Package install success (pip/conda): remind to verify runtime compatibility
- pip upgraded/downgraded a critical package (torch, numpy, etc.): WARN that this may break CUDA compatibility
- Long duration (>2min) for simple commands: flag as unexpected
- OOM (out of memory): suggest reducing batch size, enabling gradient checkpointing, or adjusting parallelism
- NCCL errors: suggest checking network config, NCCL env vars, and multi-node connectivity
- Training output showing ce_loss or lm_loss near ln(vocab_size) (10.4, 10.8, 11.1, 11.8): WARN: loss indicates random output, check weight loading
- Config file edit containing path values: remind to verify paths exist before launching
- Reading/grepping code from a different workspace than the current one: WARN: source code provenance mismatch
- cp -r from another environment's site-packages: WARN: never copy packages between environments, use pip install
- Checkpoint conversion output showing 'missed' or 'skipped' or 'unexpected' keys: WARN: audit the FULL list
- Checkpoint saved to disk without a reload verification: WARN: verify saved checkpoint
- Training log/output showing crash, error, or exitcode!=0: WARN: update the experiment via workspace_experiment
- Training log showing successful completion: remind to update experiment entry with final metrics

Reply with ONLY a JSON object:
  {{"annotations": ["annotation1", "annotation2"], "severity": "info|warning|error"}}
If no issues: {{"annotations": [], "severity": "info"}}"""


_SKILL_JUDGE_PROMPT = """\
You are deciding which skill (if any) to load for an AI infrastructure agent.

User request: {user_input}
Conversation context: {conversation_context}
Available skills:
{skills_list}
Already loaded: {loaded}

IMPORTANT - dependency chains:
{dependency_chains}

Rules:
- Only suggest a skill if it's clearly relevant to the user's request
- If the user explicitly names a skill or task that maps to one, suggest it
- If the request is ambiguous or general, suggest nothing
- Never suggest a skill that's already loaded
- Match user intent to skill keywords and descriptions listed above
- The system will automatically load all requires/suggests of each skill you select

Reply with ONLY a JSON object:
  {{"skills": ["skill-name"]}} or {{"skills": []}}"""


_COMPLEXITY_JUDGE_PROMPT = """\
You are evaluating whether a user request requires a structured task plan.

User request: {user_input}
Active plan exists: {has_plan}
Session memory context: {memory_context}

A task needs planning when:
- It involves 3+ distinct sequential steps (install -> configure -> run -> verify)
- Steps have dependencies (can't train before data is ready)
- It will take multiple tool calls across different domains (download, config, shell)
- Failure at one step requires knowing what was already done

A task does NOT need planning when:
- Simple question or lookup
- Single command execution
- Continuing an existing plan (plan already exists)
- Quick fix or small edit
- User is asking a follow-up question in an ongoing conversation

Reply with ONLY a JSON object:
  {{"needs_plan": true/false, "reason": "one-line explanation"}}"""


_ROUTE_INTENT_PROMPT = """\
You are routing a user request to the right execution mode.

User request: {user_input}

Determine the execution mode:

- "single": Simple task that one worker can handle directly.
  Examples: "read this file", "run this command", "explain this code",
  "fix this bug", "configure this parameter", "check GPU status",
  "what version is installed", "how do I..."

- "subtask": Multi-stage task that needs a serial pipeline with DAG stages.
  Examples: "set up environment AND reproduce training", "migrate model from HF to Megatron",
  "download source code AND configure AND train", "build env, download data, run training"

- "batch": Comparing multiple independent variants.
  Examples: "compare training with tp=2 vs tp=4", "run experiment A and experiment B",
  "which config is better: X or Y", "try both approaches and compare"

Available profiles: {profiles}

Choose the profile that best matches the task domain. Use "general" for simple shell operations,
file inspection, cleanup, or Q&A that don't need domain-specific skills.

Reference subtask templates (you may reuse or ignore these): {templates}

Reply with ONLY a JSON object:
{{"mode": "single"|"subtask"|"batch", "profile": "<profile_name>", "reason": "<1-sentence explanation of why this mode>"}}

For "subtask" mode — choose one of two approaches:

  A) Reuse an existing template:
     {{
       "mode": "subtask",
       "profile": "<profile_name>",
       "template": "<template_name>"
     }}

  B) Generate custom stages when no template fits:
     {{
       "mode": "subtask",
       "profile": "<default_profile_name>",
       "template": "",
       "dynamic_stages": [
         {{"id": "stage_1", "description": "what to do", "profile": "train-env-setup", "depends_on": []}},
         {{"id": "stage_2", "description": "what to do next", "profile": "training-reproduce", "depends_on": ["stage_1"]}},
         {{"id": "stage_3", "description": "what to do last", "profile": "training-reproduce", "depends_on": ["stage_2"]}}
       ]
     }}
     Each stage needs: id (unique), description (1 sentence), profile (from available profiles),
     depends_on (list of stage ids that must complete first, empty list for first stage).

For "batch" mode, also include:
  "batch_tasks": ["<task1 description>", "<task2 description>"]

For "single" mode, omit template, dynamic_stages, and batch_tasks."""

# Register route_intent in the classify prompts dict (must be after the prompt definition)
_CLASSIFY_PROMPTS["route_intent"] = _ROUTE_INTENT_PROMPT

# ── Continuation detection (skip re-routing for follow-ups) ────────────────

_CLASSIFY_PROMPTS["is_continuation"] = """\
Determine if this user input is a CONTINUATION of the previous conversation turn,
or a NEW independent task.

User input: {user_input}
Previous assistant action summary: {previous_summary}

Answer YES (continuation) for:
- Confirmations: "确认", "好的", "可以", "是的", "yes", "ok", "go ahead", "继续"
- Follow-up questions about the SAME topic just discussed
- Providing additional details requested by the previous turn
- Short replies that only make sense in context of the previous turn
- Corrections or adjustments to the previous action: "不对，应该是...", "改成..."

Answer NO (new task) for:
- A clearly new topic or task unrelated to the previous turn
- Long, self-contained instructions that don't reference the previous turn
- Requests that explicitly start a new task: "帮我做...", "新任务:", "接下来..."

Reply ONLY: {{"real": true/false, "need_more": null}}
(true = this IS a continuation, false = this is a new task)"""

# ── Skill suggestion (semantic, replaces keyword matching) ────────────────────

_CLASSIFY_PROMPTS["skill_suggest"] = """\
Given the user request, decide which skills (if any) should be loaded.

User request: {user_input}

Available skills (not yet loaded):
{available_skills}

Rules:
- ONLY suggest skills that are DIRECTLY and IMMEDIATELY needed for the current task.
- Be CONSERVATIVE: when in doubt, do NOT suggest. Skills can always be loaded later.
- If the task is a sub-step of a larger workflow (e.g., "install dependencies"), only suggest skills for THAT specific step, not the entire workflow.
- Do NOT suggest skills for future steps that haven't started yet.
- For simple operations (delete files, check status, list processes, read files), return empty list.
- Maximum 2 skills per suggestion. If more seem relevant, pick only the most critical ones.
- Return ONLY a JSON array of skill names, e.g. ["train-env-setup"] or [].
"""

_CLASSIFY_PROMPTS["skill_suggest_by_context"] = """\
Based on the agent's recent activity, decide which skills (if any) should be loaded NOW.

Original task: {task}

Recent tool calls (last {window} iterations):
{recent_activity}

Currently loaded skills: {loaded_skills}

Available skills (not yet loaded):
{available_skills}

Rules:
- Suggest a skill ONLY if the recent activity clearly indicates the agent is entering that skill's domain.
- Examples: if the agent just launched training → suggest "train-monitor"; if the agent is debugging OOM → suggest "train-monitor".
- Do NOT suggest skills already loaded.
- Be CONSERVATIVE: max 1 skill per suggestion. Only suggest when the need is obvious.
- Return ONLY a JSON array of skill names, e.g. ["train-monitor"] or [].
"""

# ── Constraint violation judgment (Phase 3) ──────────────────────────────────

_CLASSIFY_PROMPTS["is_constraint_violated"] = """\
Determine if this tool call violates the given constraint.

Constraint: {constraint}
Judgment prompt: {prompt}

Tool call:
- Tool: {tool_name}
- Args: {tool_args}
- Result: {tool_result}

Recent tool history (what the agent already did before this call):
{recent_tool_history}

IMPORTANT RULES for accurate judgment:
1. SELF-REFERENTIAL CHECK: If the constraint says "must do X before Y" and the CURRENT command IS X itself (the prerequisite action), it is NOT a violation. Example: constraint says "run nvidia-smi before installing torch" — if the current command IS nvidia-smi, that's the agent fulfilling the prerequisite, NOT violating it.
2. HISTORY CHECK: Look at the recent tool history above. If the constraint requires a prerequisite action (e.g., "check CUDA version first"), and that action ALREADY APPEARS in the history, the prerequisite is satisfied — NOT a violation.
3. SCOPE CHECK: Only judge what the constraint actually asks about. If the constraint is about "pip install torch" but the current command is something else, it's NOT a violation.
4. When in doubt, answer false (not violated). Only answer true when you are CONFIDENT the constraint is clearly violated.

Reply ONLY: {{"real": true/false, "reason": "one-line explanation of your judgment", "need_more": null}}
(true = constraint IS violated, false = constraint is NOT violated)"""

_CLASSIFY_PROMPTS["is_warning_triggered"] = """\
Determine if this warning applies to the current tool context.

Warning: {warning}
Judgment criteria: {prompt}

Current tool call:
- Tool: {tool_name}
- Args: {tool_args}

Reply ONLY: {{"real": true/false, "need_more": null}}
(true = warning SHOULD fire, false = warning does NOT apply)"""

# ── JudgeBudget ──────────────────────────────────────────────────────────

@dataclass
class JudgeBudget:
    """Per-turn call budget for LLM judges.

    Strategy:
    - Max 64 judge calls per turn for classify operations
    - Health checks have a separate budget (not shared with classify)
    - Each judge type caches independently
    - On exhaustion: health -> heuristic, skill/complexity -> default, classify -> cached/default
    """

    max_calls_per_turn: int = 64
    max_health_per_turn: int = 20
    calls_this_turn: int = 0
    health_calls_this_turn: int = 0
    total_calls: int = 0
    total_saved_by_cache: int = 0
    _skipped_summary: str = ""  # summary of skipped categories for user-visible warning
    _exhausted_warned: bool = False  # only warn once per turn

    @property
    def exhausted(self) -> bool:
        return self.calls_this_turn >= self.max_calls_per_turn

    @property
    def health_exhausted(self) -> bool:
        return self.health_calls_this_turn >= self.max_health_per_turn

    def consume(self) -> bool:
        """Return True if budget allows another call, incrementing if so."""
        if self.calls_this_turn >= self.max_calls_per_turn:
            return False
        self.calls_this_turn += 1
        self.total_calls += 1
        return True

    def consume_health(self) -> bool:
        """Return True if health budget allows another call."""
        if self.health_calls_this_turn >= self.max_health_per_turn:
            return False
        self.health_calls_this_turn += 1
        self.total_calls += 1
        return True

    def note_skipped(self, source: str, category: str):
        """Record a skipped classify call for later reporting."""
        if self._skipped_summary:
            self._skipped_summary += f", {source}/{category}"
        else:
            self._skipped_summary = f"{source}/{category}"

    def reset_turn(self):
        self.calls_this_turn = 0
        self.health_calls_this_turn = 0
        self._skipped_summary = ""

    @property
    def skipped_detail(self) -> str:
        return self._skipped_summary


# ── Judge ────────────────────────────────────────────────────────────────

# ── Classify source tracking ────────────────────────────────────────────

#: Fast-path heuristic returned a confident answer (no LLM call).
SOURCE_FAST = "fast"
#: LLM returned a valid classification.
SOURCE_LLM = "llm"
#: Result was served from local MD5 cache.
SOURCE_CACHE = "cache"
#: Budget exhausted or provider unavailable — default value returned.
SOURCE_DEFAULT = "default"
#: Provider is None (never initialized) — no LLM available at all.
SOURCE_UNAVAILABLE = "unavailable"


class ClassifyTrace:
    """Per-turn trace of classify() calls: category → source.

    Attached to Judge._last_trace after each classify() call.
    Callers (especially safety-critical Guards) can inspect
    trace to decide whether to trust the result or take conservative action.
    """

    def __init__(self):
        self._entries: dict[str, str] = {}  # category → source

    def record(self, category: str, source: str):
        self._entries.setdefault(category, source)

    def source_of(self, category: str) -> str:
        """Return the source for a category, or 'unavailable' if never called."""
        return self._entries.get(category, SOURCE_UNAVAILABLE)

    def any_from(self, *sources: str) -> bool:
        """True if any recorded call has one of the given sources."""
        return any(s in sources for s in self._entries.values())

    def clear(self):
        self._entries.clear()


class Judge:
    """Unified LLM judge with budget control, caching, and multi-round classify.

    classify() returns a (value, source) tuple so safety-critical guards can
    distinguish "LLM said safe" from "Judge unavailable, assuming safe by default."
    """

    _MAX_CLASSIFY_ROUNDS = 3

    def __init__(self, provider, budget: JudgeBudget | None = None):
        self.provider = provider
        self.budget = budget or JudgeBudget()
        self._trace = ClassifyTrace()

        # Caches
        self._health_cache: dict[str, dict] = {}
        self._result_cache: dict[str, list] = {}
        self._skill_cache: dict[str, list] = {}
        self._classify_cache: dict[str, dict] = {}

    def reset_turn(self):
        """Reset per-turn budget and trace. Caches stay warm across turns."""
        self.budget.reset_turn()
        self._trace.clear()

    # ── classify: replaces ALL regex/keyword matching ─────────────────────

    def classify(self, category: str, context: dict, default: Any = None) -> Any:
        """Lightweight LLM classification. Replaces all regex/keyword matching.

        category: one of "is_error", "is_success", "is_dangerous", "is_read_only_shell",
                  "is_training_command", "is_kill_command", "is_training_failure",
                  "is_zombie_gpu", "is_user_porting_confirm", "checklist_rule",
                  "checklist_rule_batch", "route_intent"

        context: dict with relevant fields. LLM can request more in multi-round mode.

        Returns SAME TYPE as before (bool, str, dict, list) — the return value
        contract is unchanged. Callers that need source information should instead
        use classify_traced() or inspect self._trace.source_of(category).
        """
        value, _source = self.classify_traced(category, context, default)
        return value

    def classify_traced(self, category: str, context: dict, default: Any = None) -> tuple[Any, str]:
        """Same as classify() but returns (value, source) tuple.

        Three-tier resolution:
        1. Fast path: heuristic classifiers (instant, no LLM)
        2. Cache path: MD5-keyed cache hit
        3. Deep path: LLM call with multi-round support

        source is one of: SOURCE_FAST, SOURCE_LLM, SOURCE_CACHE, SOURCE_DEFAULT, SOURCE_UNAVAILABLE.

        Safety-critical callers (SafetyGuard) should use this method and
        treat SOURCE_DEFAULT / SOURCE_UNAVAILABLE as "unknown → be conservative."
        """
        # Provider never initialized
        if self.provider is None:
            self._trace.record(category, SOURCE_UNAVAILABLE)
            return (default, SOURCE_UNAVAILABLE)

        # ── Tier 1: Fast path (heuristic) ────────────────────────────────
        fast_result = self._try_fast_path(category, context)
        if fast_result is not None:
            self._trace.record(category, SOURCE_FAST)
            return (fast_result, SOURCE_FAST)

        # ── Tier 2: Cache path ───────────────────────────────────────────
        cache_key = self._classify_cache_key(category, context)
        if cache_key in self._classify_cache:
            self.budget.total_saved_by_cache += 1
            self._trace.record(category, SOURCE_CACHE)
            return (self._classify_cache[cache_key], SOURCE_CACHE)

        # ── Tier 3: Deep path (LLM) ─────────────────────────────────────
        prompt_template = _CLASSIFY_PROMPTS.get(category)
        if not prompt_template:
            self._trace.record(category, SOURCE_DEFAULT)
            return (default, SOURCE_DEFAULT)

        truncated = self._truncate_context(context, max_chars=800)

        for round_num in range(self._MAX_CLASSIFY_ROUNDS):
            if self.budget.exhausted:
                break
            if not self.budget.consume():
                break

            prompt = prompt_template
            if "{context}" in prompt:
                prompt = prompt.replace("{context}", self._format_context(truncated))
            if "{rule}" in prompt:
                prompt = prompt.replace("{rule}", json.dumps(context.get("rule", ""), ensure_ascii=False))
            if "{item_id}" in prompt:
                prompt = prompt.replace("{item_id}", str(context.get("item_id", "")))
            if "{description}" in prompt:
                prompt = prompt.replace("{description}", str(context.get("description", "")))
            if "{prompt}" in prompt:
                # The constraint-specific prompt from ChecklistItem
                prompt = prompt.replace("{prompt}", str(context.get("prompt", "")))
            if "{items}" in prompt:
                # JSON array of {id, description, prompt} for batch checklist evaluation
                prompt = prompt.replace("{items}", json.dumps(context.get("items", []), ensure_ascii=False))
            if "{skill_content}" in prompt:
                prompt = prompt.replace("{skill_content}", str(context.get("skill_content", "")))

            # Generic fallback: substitute any remaining {key} placeholders from context
            for key, val in context.items():
                placeholder = "{" + key + "}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(val))

            data = self._call_and_parse(prompt, default={})
            need_more = data.get("need_more") if isinstance(data, dict) else None
            if need_more and isinstance(need_more, list) and round_num < self._MAX_CLASSIFY_ROUNDS - 1:
                for field in need_more:
                    if field in context and field not in truncated:
                        truncated[field] = self._truncate_one(str(context[field]), max_chars=2000)[:2000]
                continue

            result = self._parse_classify_result(category, data, default)
            self._classify_cache[cache_key] = result
            self._trace.record(category, SOURCE_LLM)
            return (result, SOURCE_LLM)

        self.budget.note_skipped("classify", category)
        self._classify_cache[cache_key] = default
        self._trace.record(category, SOURCE_DEFAULT)
        return (default, SOURCE_DEFAULT)

    # ── Fast path dispatch ──────────────────────────────────────────────────

    _FAST_DISPATCH = {
        "is_read_only_shell": lambda ctx: FastClassifier.is_read_only_shell(
            str(ctx.get("command", ""))
        ),
        "is_dangerous": lambda ctx: FastClassifier.is_dangerous(
            str(ctx.get("command", ""))
        ),
        "is_training_command": lambda ctx: FastClassifier.is_training_command(
            str(ctx.get("command", ""))
        ),
        "is_kill_command": lambda ctx: FastClassifier.is_kill_command(
            str(ctx.get("command", ""))
        ),
    }

    def _try_fast_path(self, category: str, context: dict) -> Any:
        """Try fast-path heuristic for a category.

        Returns the classification result if confident, None to escalate to LLM.
        """
        dispatch = self._FAST_DISPATCH.get(category)
        if dispatch is None:
            return None
        return dispatch(context)

    # ── Batch classify ───────────────────────────────────────────────────

    def classify_batch(
        self, items: list[tuple[str, dict, Any]],
    ) -> list[tuple[Any, str]]:
        """Classify multiple items, batching LLM calls where possible.

        items: list of (category, context, default) tuples.

        Returns list of (value, source) tuples in same order.

        Strategy:
        1. Resolve fast-path and cache hits immediately
        2. Batch remaining items into a single LLM call if they share the same
           category (e.g., multiple is_error checks)
        3. Fall back to individual calls for mixed categories
        """
        results: list[tuple[Any, str] | None] = [None] * len(items)
        pending: list[tuple[int, str, dict, Any]] = []  # (index, category, context, default)

        # Phase 1: resolve fast-path and cache
        for i, (category, context, default) in enumerate(items):
            if self.provider is None:
                self._trace.record(category, SOURCE_UNAVAILABLE)
                results[i] = (default, SOURCE_UNAVAILABLE)
                continue

            # Fast path
            fast_result = self._try_fast_path(category, context)
            if fast_result is not None:
                self._trace.record(category, SOURCE_FAST)
                results[i] = (fast_result, SOURCE_FAST)
                continue

            # Cache path
            cache_key = self._classify_cache_key(category, context)
            if cache_key in self._classify_cache:
                self.budget.total_saved_by_cache += 1
                self._trace.record(category, SOURCE_CACHE)
                results[i] = (self._classify_cache[cache_key], SOURCE_CACHE)
                continue

            pending.append((i, category, context, default))

        # Phase 2: batch LLM calls for same-category items
        if pending and not self.budget.exhausted:
            # Group by category
            by_category: dict[str, list[tuple[int, dict, Any]]] = {}
            for idx, cat, ctx, dflt in pending:
                by_category.setdefault(cat, []).append((idx, ctx, dflt))

            for cat, group in by_category.items():
                if self.budget.exhausted:
                    for idx, ctx, dflt in group:
                        self._trace.record(cat, SOURCE_DEFAULT)
                        results[idx] = (dflt, SOURCE_DEFAULT)
                    continue

                # Single item — just do normal classify
                if len(group) == 1:
                    idx, ctx, dflt = group[0]
                    value, source = self.classify_traced(cat, ctx, dflt)
                    results[idx] = (value, source)
                    continue

                # Multiple items of same category — individual calls
                # (True batching into one prompt is only for checklist_rule_batch)
                for idx, ctx, dflt in group:
                    if self.budget.exhausted:
                        self._trace.record(cat, SOURCE_DEFAULT)
                        results[idx] = (dflt, SOURCE_DEFAULT)
                    else:
                        value, source = self.classify_traced(cat, ctx, dflt)
                        results[idx] = (value, source)

        # Fill any remaining None slots with defaults
        for i, item in enumerate(results):
            if item is None:
                cat, ctx, dflt = items[i]
                self._trace.record(cat, SOURCE_DEFAULT)
                results[i] = (dflt, SOURCE_DEFAULT)

        return results  # type: ignore[return-value]

    # ── Health judge ──────────────────────────────────────────────────────

    def health(
        self, command: str, recent_output: str, elapsed: str,
        output_changed: bool = True, stall_count: int = 0,
    ) -> dict:
        """Evaluate whether a long-running command is healthy."""
        if self.budget.health_exhausted:
            self.budget.note_skipped("health", "health")
            if stall_count >= 3:
                return {"kill": True, "reason": "Output stalled and health check unavailable (judge budget exhausted)"}
            return {"kill": False, "reason": "Judge budget exhausted, health check skipped — command still running"}

        cache_key = hashlib.md5(
            f"{command[:100]}:{elapsed}:{stall_count}".encode()
        ).hexdigest()[:12]
        if cache_key in self._health_cache:
            self.budget.total_saved_by_cache += 1
            return self._health_cache[cache_key]

        if not self.budget.consume_health():
            return {"kill": False}

        prompt = _HEALTH_JUDGE_PROMPT.format(
            command=command, elapsed=elapsed,
            output=recent_output[-2000:],
            output_changed="yes" if output_changed else "no",
            stall_count=stall_count,
        )
        result = self._call_and_parse(prompt, default={"kill": False})
        self._health_cache[cache_key] = result
        return result

    # ── Result judge ──────────────────────────────────────────────────────

    def result(self, command: str, output: str, elapsed: float) -> list[str]:
        """Analyze shell output and return annotations."""
        if self.budget.exhausted:
            return []

        cache_key = hashlib.md5(
            f"{command[:100]}:{output[-500:]}".encode()
        ).hexdigest()[:12]
        if cache_key in self._result_cache:
            self.budget.total_saved_by_cache += 1
            return self._result_cache[cache_key]

        if not self.budget.consume():
            return []

        prompt = _RESULT_JUDGE_PROMPT.format(
            command=command, elapsed=elapsed, output=output[-3000:],
        )
        data = self._call_and_parse(prompt, default={})
        annotations = data.get("annotations", [])
        self._result_cache[cache_key] = annotations
        return annotations

    # ── Skill judge ───────────────────────────────────────────────────────

    def skill(
        self, user_input: str, skills_list: str, loaded: str,
        dependency_chains: str, conversation_context: str, valid_names: set[str],
    ) -> list[str]:
        """Decide which skill to auto-load."""
        if self.budget.exhausted:
            return []

        cache_key = hashlib.md5(user_input[:200].encode()).hexdigest()[:12]
        if cache_key in self._skill_cache:
            self.budget.total_saved_by_cache += 1
            return self._skill_cache[cache_key]

        if not self.budget.consume():
            return []

        prompt = _SKILL_JUDGE_PROMPT.format(
            user_input=user_input[:500],
            conversation_context=conversation_context,
            skills_list=skills_list, loaded=loaded,
            dependency_chains=dependency_chains,
        )
        data = self._call_and_parse(prompt, default={})
        skills = data.get("skills", [])
        result = [s for s in skills if s and s in valid_names]
        self._skill_cache[cache_key] = result
        return result

    # ── Complexity judge ──────────────────────────────────────────────────

    def complexity(
        self, user_input: str, has_plan: bool = False,
        memory_context: str = "",
    ) -> dict:
        """Evaluate whether a user request needs a task plan."""
        if self.budget.exhausted:
            return {"needs_plan": False}

        if not self.budget.consume():
            return {"needs_plan": False}

        prompt = _COMPLEXITY_JUDGE_PROMPT.format(
            user_input=user_input,
            has_plan="yes" if has_plan else "no",
            memory_context=memory_context[:500] if memory_context else "(none)",
        )
        return self._call_and_parse(prompt, default={"needs_plan": False})

    # ── Route intent (replaces Orchestrator regex routing) ─────────────────

    def route(self, user_input: str, profiles: str, templates: str) -> tuple[dict, str]:
        """Route a user request to the right execution mode via LLM.

        Returns ((mode_dict, source)), where source is SOURCE_LLM / SOURCE_UNAVAILABLE.
        mode_dict always contains at least {"mode": "single"}.

        Callers should check source: if SOURCE_UNAVAILABLE, fall back to regex routing.
        This does NOT consume budget — routing happens once per user request,
        before the agent loop starts.
        """
        if self.provider is None:
            return ({"mode": "single"}, SOURCE_UNAVAILABLE)

        context = {
            "user_input": user_input,
            "profiles": profiles,
            "templates": templates,
        }
        value, source = self.classify_traced("route_intent", context,
            default={"mode": "single"})
        if not isinstance(value, dict):
            value = {"mode": "single"}
        return (value, source)

    def suggest_skills(self, user_input: str, available_skills: list[dict]) -> list[str]:
        """Suggest which skills to load based on semantic understanding.

        Args:
            user_input: The user's request text.
            available_skills: List of {"name": ..., "description": ...} for unloaded skills.

        Returns:
            List of skill names to load (may be empty).
        """
        if not available_skills:
            return []

        skills_str = "\n".join(
            f"- {s['name']}: {s.get('description', '')}" for s in available_skills
        )
        context = {
            "user_input": user_input,
            "available_skills": skills_str,
        }
        value, source = self.classify_traced("skill_suggest", context, default=[])
        if not isinstance(value, list):
            return []
        # Validate: only return names that exist in available_skills
        valid_names = {s["name"] for s in available_skills}
        return [n for n in value if isinstance(n, str) and n in valid_names]

    def suggest_skills_by_context(
        self,
        task: str,
        recent_activity: list[dict],
        loaded_skills: list[str],
        available_skills: list[dict],
    ) -> list[str]:
        """Suggest skills based on recent tool activity (mid-turn).

        Unlike suggest_skills() which uses user input, this uses the agent's
        recent tool call history to detect when a new skill domain is entered.

        Args:
            task: The original user task description.
            recent_activity: List of {"tool": ..., "args_summary": ...} dicts.
            loaded_skills: Names of currently loaded skills.
            available_skills: List of {"name": ..., "description": ...} for unloaded skills.

        Returns:
            List of skill names to load (max 1).
        """
        if not available_skills:
            return []

        skills_str = "\n".join(
            f"- {s['name']}: {s.get('description', '')}" for s in available_skills
        )
        activity_str = "\n".join(
            f"  [{a['tool']}] {a.get('args_summary', '')}" for a in recent_activity[-10:]
        )
        context = {
            "task": task or "(unknown)",
            "window": str(len(recent_activity)),
            "recent_activity": activity_str or "(no activity yet)",
            "loaded_skills": ", ".join(loaded_skills) if loaded_skills else "(none)",
            "available_skills": skills_str,
        }
        value, source = self.classify_traced("skill_suggest_by_context", context, default=[])
        if not isinstance(value, list):
            return []
        valid_names = {s["name"] for s in available_skills}
        return [n for n in value[:1] if isinstance(n, str) and n in valid_names]

    def is_continuation(self, user_input: str, previous_summary: str) -> bool:
        """Determine if user_input is a follow-up to the previous turn.

        Uses fast heuristic first, then LLM fallback.
        Returns True if continuation (skip re-routing), False if new task.
        """
        # Fast path: common confirmation/continuation patterns
        stripped = user_input.strip().lower()
        _FAST_CONTINUATIONS = {
            "确认", "好的", "可以", "是的", "对", "行", "嗯", "ok", "yes", "y",
            "go", "go ahead", "sure", "继续", "好", "是", "对的", "没问题",
            "确定", "同意", "proceed", "continue", "right", "yep", "yeah",
        }
        if stripped in _FAST_CONTINUATIONS:
            return True

        # Short input with no verb-like structure → likely continuation
        if len(stripped) <= 5 and not any(c in stripped for c in "帮做请运行执行删除创建"):
            return True

        # LLM path for ambiguous cases
        context = {
            "user_input": user_input,
            "previous_summary": previous_summary,
        }
        result = self.classify("is_continuation", context, default=False)
        return bool(result)

    @property
    def trace(self) -> ClassifyTrace:
        """Expose per-turn classify trace for safety-critical callers."""
        return self._trace

    def extract_constraints(self, skill_content: str) -> list[dict]:
        """Extract checklist constraints from a skill's content via LLM.

        Called once per skill load. Returns a list of constraint dicts
        suitable for ChecklistItem construction.
        """
        # Don't count against the per-turn classify budget — this is
        # initialization, not per-tool-call overhead.
        import hashlib
        cache_key = hashlib.md5(skill_content[:500].encode()).hexdigest()[:12]
        if cache_key in self._classify_cache:
            return self._classify_cache[cache_key]

        result = self.classify("extract_constraints",
            {"skill_content": skill_content}, default=[])
        self._classify_cache[cache_key] = result
        return result

    # ── Classify helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_classify_result(category: str, data: dict, default: Any) -> Any:
        """Extract classification decision from LLM response."""
        if category == "is_constraint_violated":
            # Return dict with violated bool + reason string
            real = data.get("real") if isinstance(data, dict) else None
            reason = data.get("reason", "") if isinstance(data, dict) else ""
            if isinstance(real, bool):
                return {"violated": real, "reason": str(reason)}
            return {"violated": True, "reason": ""}  # conservative fallback
        if category == "is_user_porting_confirm":
            text = str(data.get("decision", "") or data.get("mode", "") or "").lower()
            if "mode_b" in text or "mode b" in text or "b" == text:
                return "mode_b"
            if "mode_c" in text or "mode c" in text or "c" == text:
                return "mode_c"
            return ""
        if category == "checklist_rule_batch":
            # LLM may return a list directly or {"violations": [...]}
            if isinstance(data, list):
                return data
            violations = data.get("violations", []) if isinstance(data, dict) else []
            if isinstance(violations, list):
                return violations
            return []
        if category == "checklist_rule":
            match = data.get("match")
            if isinstance(match, bool):
                return {"match": match, "reason": data.get("reason", "")}
            return {"match": False, "reason": ""}
        if category == "extract_constraints":
            # _parse_json may return a list directly
            if isinstance(data, list):
                return data
            # Sometimes LLM wraps in {"constraints": [...]}
            if isinstance(data, dict):
                constraints = data.get("constraints", [])
                if isinstance(constraints, list):
                    return constraints
            return []
        if category == "route_intent":
            # Return the full dict: {mode, profile, template, batch_tasks, dynamic_stages}
            if isinstance(data, dict):
                return data
            return default if default is not None else {"mode": "single"}
        if category in ("skill_suggest", "skill_suggest_by_context"):
            # LLM should return a JSON array of skill names
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                skills = data.get("skills", [])
                if isinstance(skills, list):
                    return skills
            return []
        # Boolean categories
        real = data.get("real")
        if isinstance(real, bool):
            return real
        decision = data.get("decision")
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, str):
            return decision.lower() in ("yes", "true", "y")
        return default if default is not None else False

    @staticmethod
    def _classify_cache_key(category: str, context: dict) -> str:
        raw = category + json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _truncate_context(context: dict, max_chars: int = 800) -> dict:
        result = {}
        for k, v in context.items():
            result[k] = Judge._truncate_one(str(v), max_chars)
        return result

    @staticmethod
    def _truncate_one(text: str, max_chars: int = 800) -> str:
        """Truncate preserving both head and tail (errors usually at end)."""
        if len(text) <= max_chars:
            return text
        head = text[:max_chars // 4]
        tail = text[-(max_chars - max_chars // 4):]
        return f"{head}\n... [{len(text) - max_chars} chars omitted] ...\n{tail}"

    @staticmethod
    def _format_context(context: dict) -> str:
        lines = []
        for k, v in context.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in v.items():
                    lines.append(f"  {sub_k}: {sub_v}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    # ── LLM helpers ───────────────────────────────────────────────────────

    def _call_and_parse(self, prompt: str, default: dict | list) -> dict | list:
        """Make a single LLM call and parse JSON from response."""
        text = self._call(prompt)
        if not text:
            return default
        result = self._parse_json(text)
        return result if result else default

    def _call(self, prompt: str) -> str:
        """Dispatch LLM call through provider."""
        if self.provider is None:
            return ""
        try:
            response = self.provider.chat(
                [{"role": "user", "content": prompt}], tools=[]
            )
            return (response.get("content") or "").strip()
        except Exception as e:
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """Extract JSON object or array from LLM response text.

        Handles trailing content after the JSON and tries both
        {...} and [...] top-level formats.
        """
        text = text.strip()
        # Try the whole text first
        for candidate in (text,):
            if not candidate:
                continue
            # Trim trailing characters beyond the last ] or }
            if candidate.startswith("["):
                end = candidate.rfind("]")
                if end > 0:
                    candidate = candidate[:end + 1]
            elif candidate.startswith("{"):
                end = candidate.rfind("}")
                if end > 0:
                    candidate = candidate[:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Fallback: find the outermost JSON bounds
        for first_char, last_char in [("[", "]"), ("{", "}")]:
            start = text.find(first_char)
            end = text.rfind(last_char)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
