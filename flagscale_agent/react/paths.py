"""Centralized path management for FlagScale Agent.

All .flagscale directories are now under the project root, not ~/.flagscale.
This provides better isolation and makes cleanup easier.
"""

import os
from pathlib import Path


def _find_project_root() -> str:
    """Find FlagScale project root by walking up from current directory.
    
    Looks for markers: .git, setup.py, pyproject.toml, or 'flagscale' package dir.
    Falls back to current working directory if not found.
    """
    current = Path.cwd().resolve()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for common project markers
        if any((parent / marker).exists() for marker in [".git", "setup.py", "pyproject.toml"]):
            return str(parent)
        # Check if this is the FlagScale repo (has flagscale/ package)
        if (parent / "flagscale" / "__init__.py").exists():
            return str(parent)
    
    # Fallback to cwd
    return str(current)


def get_dot_flagscale_root() -> str:
    """Get the .flagscale root directory (project_root/.flagscale).
    
    Returns:
        Absolute path to .flagscale directory (created if not exists).
    """
    project_root = _find_project_root()
    dot_flagscale = os.path.join(project_root, ".flagscale")
    os.makedirs(dot_flagscale, exist_ok=True)
    return dot_flagscale


def get_sessions_root() -> str:
    """Get sessions directory (.flagscale/sessions)."""
    return os.path.join(get_dot_flagscale_root(), "sessions")


def get_memory_dir() -> str:
    """Get agent memory directory (.flagscale/agent_memory)."""
    return os.path.join(get_dot_flagscale_root(), "agent_memory")


def get_input_history_file() -> str:
    """Get readline input history file (.flagscale/input_history)."""
    return os.path.join(get_dot_flagscale_root(), "input_history")


def get_config_search_paths() -> list[str]:
    """Get agent.yaml config file search paths.
    
    Returns:
        List of paths in priority order:
        1. .flagscale/agent.yaml (project-local)
        2. ~/.flagscale/agent.yaml (user-global, for backward compat)
    """
    return [
        os.path.join(get_dot_flagscale_root(), "agent.yaml"),
        os.path.join(Path.home(), ".flagscale", "agent.yaml"),  # backward compat
    ]


def get_skill_search_paths() -> list[str]:
    """Get skill directory search paths.
    
    Returns:
        List of paths in priority order:
        1. .flagscale/skills (project-local)
        2. ~/.flagscale/skills (user-global, for backward compat)
    """
    return [
        os.path.join(get_dot_flagscale_root(), "skills"),
        os.path.join(Path.home(), ".flagscale", "skills"),  # backward compat
    ]
