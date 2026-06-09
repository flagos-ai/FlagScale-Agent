"""Constants and configuration for FlagScale Agent.

Extracted from agent.py to reduce file size and improve maintainability.
"""

# ── Tool Sets ─────────────────────────────────────────────────────────────────

READ_ONLY_TOOLS = {
    "read_file", "grep", "find", "ls", "list_files",
    "memory_read", "memory_list", "plan_status", "web_fetch",
}

CORE_TOOLS = {
    "read_file", "write_file", "edit_file", "shell",
    "load_skill", "web_fetch", "memory_write", "memory_read",
    "memory_list", "monitor", "workspace_experiment",
    "plan_create", "plan_update", "plan_status",
}

PHASE_TOOL_SETS = {
    "idle": {
        "read_file", "shell", "load_skill", "memory_read", "memory_list",
        "web_fetch", "workspace_experiment", "find_latest_log",
        "plan_create", "plan_status", "memory_write", "write_file",
        "edit_file", "monitor", "validate_config",
    },
    "analysis": {
        "read_file", "shell", "memory_read", "memory_list",
        "web_fetch", "load_skill", "workspace_experiment",
        "find_latest_log", "memory_write",
        "plan_create", "plan_update", "plan_status",
        "write_file", "edit_file", "inspect_checkpoint",
        "validate_config",
    },
    "implementation": {
        "read_file", "write_file", "edit_file", "shell",
        "load_skill", "memory_write", "memory_read",
        "plan_update", "plan_status", "workspace_experiment",
        "find_latest_log", "monitor", "validate_config",
        "inspect_checkpoint", "parse_training_metrics",
    },
    "verification": {
        "read_file", "shell", "write_file", "edit_file",
        "monitor", "find_latest_log", "parse_training_metrics",
        "memory_write", "memory_read", "workspace_experiment",
        "plan_update", "plan_status", "load_skill",
        "inspect_checkpoint", "validate_config",
    },
}

# ── Tool Behavior Configuration ───────────────────────────────────────────────

READ_FILE_SUMMARY_THRESHOLD = 8000
READ_FILE_SUMMARY_THRESHOLD_LARGE = 15000
# Backward compat alias
READ_FILE_SUMMARY_THRESHOLD_PORTING = READ_FILE_SUMMARY_THRESHOLD_LARGE
