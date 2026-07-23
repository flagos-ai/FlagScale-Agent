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

"""Centralized path management for FlagScale Agent.

All agent state lives under ~/.flagscale — one fixed location regardless
of where the agent is launched from. This avoids polluting project directories.
"""

import os
from pathlib import Path


def get_dot_flagscale_root() -> str:
    """Get the .flagscale root directory (~/.flagscale).
    
    Returns:
        Absolute path to ~/.flagscale directory (created if not exists).
    """
    dot_flagscale = os.path.join(Path.home(), ".flagscale")
    os.makedirs(dot_flagscale, exist_ok=True)
    return dot_flagscale


def get_sessions_root() -> str:
    """Get sessions directory (~/.flagscale/sessions)."""
    return os.path.join(get_dot_flagscale_root(), "sessions")


def get_memory_dir() -> str:
    """Get global agent memory directory (~/.flagscale/agent_memory)."""
    return os.path.join(get_dot_flagscale_root(), "agent_memory")


def get_session_memory_dir(session_id: str) -> str:
    """Get per-session memory directory (~/.flagscale/sessions/{session_id}/memory).

    Session memory is isolated per session and never mixed with global memory.
    """
    return os.path.join(get_sessions_root(), session_id, "memory")


def get_input_history_file() -> str:
    """Get readline input history file (~/.flagscale/input_history)."""
    return os.path.join(get_dot_flagscale_root(), "input_history")


def get_config_path() -> str:
    """Get agent.yaml config file path (~/.flagscale/agent.yaml)."""
    return os.path.join(get_dot_flagscale_root(), "agent.yaml")


def get_skill_dir() -> str:
    """Get skill directory (~/.flagscale/skills)."""
    return os.path.join(get_dot_flagscale_root(), "skills")
