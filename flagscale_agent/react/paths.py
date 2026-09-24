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

All agent state lives under a single root directory. Default is ~/.flagscale,
but can be overridden via FLAGSCALE_HOME environment variable for shared storage
or multi-node container setups.
"""

import os
from pathlib import Path


def get_dot_flagscale_root() -> str:
    """Get the .flagscale root directory.
    
    Resolution order:
        1. FLAGSCALE_HOME env var (for shared storage / multi-node setups)
        2. ~/.flagscale (default, backward compatible)
    
    Returns:
        Absolute path to .flagscale directory (created if not exists).
    """
    dot_flagscale = os.environ.get("FLAGSCALE_HOME")
    if not dot_flagscale:
        dot_flagscale = os.path.join(Path.home(), ".flagscale")
    dot_flagscale = os.path.abspath(dot_flagscale)
    os.makedirs(dot_flagscale, exist_ok=True)
    return dot_flagscale


def get_sessions_root() -> str:
    """Get sessions directory (~/.flagscale/sessions)."""
    return os.path.join(get_dot_flagscale_root(), "sessions")


def get_memory_dir() -> str:
    """Get global agent memory directory (~/.flagscale/agent_memory)."""
    return os.path.join(get_dot_flagscale_root(), "agent_memory")


def get_proposals_dir() -> str:
    """Get global improvement-proposal registry directory
    (~/.flagscale/proposals). Global (not per-session) because a proposal is
    reviewed across sessions: an unreviewed proposal raised in one session must
    resurface at the wrap-up of a later one."""
    return os.path.join(get_dot_flagscale_root(), "proposals")


def get_input_history_file() -> str:
    """Get readline input history file (~/.flagscale/input_history)."""
    return os.path.join(get_dot_flagscale_root(), "input_history")


def get_config_path() -> str:
    """Get agent.yaml config file path (~/.flagscale/agent.yaml)."""
    return os.path.join(get_dot_flagscale_root(), "agent.yaml")


def get_skill_dir() -> str:
    """Get skill directory (~/.flagscale/skills)."""
    return os.path.join(get_dot_flagscale_root(), "skills")
