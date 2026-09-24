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

"""PackageSearchGuard — prevents blind searching for package/source locations.

When the agent needs to locate a software package or source code, it should
ask the user directly rather than using find/ls/grep to search the filesystem.
This avoids wasted time and incorrect assumptions about which copy is active.

Exception: locating flagscale_agent's own source via python import is allowed.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Patterns that indicate searching for package/source locations
_SEARCH_PATTERNS = [
    # find commands looking for package directories
    r"find\s+/\S*\s+.*-(?:name|type)\s+.*(?:megatron|flagscale|transformer|vllm|torch|apex|nemo)",
    # ls commands probing multiple directories to find packages
    r"ls\s+/\S*(?:megatron|flagscale|transformer|vllm)",
    # grep/find looking for python package init files
    r"find\s+.*__init__\.py.*(?:megatron|flagscale|transformer)",
    # Broad find commands with maxdepth looking for source trees
    r"find\s+/workspace\s+.*-type\s+d\s*\|.*grep.*(?:megatron|flagscale|transformer|vllm)",
]

# Allowed patterns (exceptions)
_ALLOWED_PATTERNS = [
    # Locating own source via python import
    r"python.*import\s+flagscale_agent",
    # grep inside a known source tree (already located)
    r"grep\s+-[rn]+\s+.*(?:/workspace/\S+/megatron|/workspace/\S+/flagscale)",
]

_SEARCH_RE = [re.compile(p, re.IGNORECASE) for p in _SEARCH_PATTERNS]
_ALLOWED_RE = [re.compile(p, re.IGNORECASE) for p in _ALLOWED_PATTERNS]


class PackageSearchGuard(Guard):
    """Warn when agent blindly searches for package/source locations."""

    name = "package_search"
    priority = 25

    def __init__(self):
        self._warned_this_turn = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None

        command = ctx.tool_args.get("command", "")
        if not command:
            return None

        # Check if this is an allowed pattern
        for pat in _ALLOWED_RE:
            if pat.search(command):
                return None

        # Check if this looks like a blind package search
        for pat in _SEARCH_RE:
            if pat.search(command):
                if not self._warned_this_turn:
                    self._warned_this_turn = True
                    return GuardVerdict.inject(
                        "[PackageSearch] Don't blindly search for package/source locations. "
                        "Ask the user directly: 'Where is the source code for X?' or "
                        "'Which conda environment has X installed?' "
                        "Only proceed after the user provides the path.",
                        reason="blind_package_search",
                        category="package_search",
                    )
                return None

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        self._warned_this_turn = False
