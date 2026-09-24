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

"""PostEditFarEndGuard — inject-only far-end verification reminder after file edits.

Fires on EVERY successful write_file/edit_file, regardless of file type — unlike
UnitTestGuard, which covers only flagscale_agent/ .py sources and only after 2+
accumulated changes. Inject-only by design: it reminds, it never blocks
(user-approved: fire on every edit, no latch, no suppression).
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict

# Per-extension far-end verification commands. The placeholder <path> is replaced
# with the actual edited path. These are the cheapest validity checks per type;
# anything else falls back to a re-read reminder.
_TYPE_HINTS = {
    ".py": "python -m py_compile <path>",
    ".json": "python -c 'import json,sys; json.load(open(sys.argv[1]))' <path>",
    ".yaml": "python -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' <path>",
    ".yml": "python -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' <path>",
    ".toml": "python -c 'import tomllib,sys; tomllib.load(open(sys.argv[1],\"rb\"))' <path>",
    ".sh": "bash -n <path>",
}
_FALLBACK_HINT = "re-read the edited region and confirm it landed as intended"


class PostEditFarEndGuard(Guard):
    """Post-tool guard: verify the FAR end after every successful write/edit."""

    name = "post_edit_far_end"
    priority = 70  # Advisory, same tier as UnitTestGuard — never blocks

    # Only these tools modify files
    WRITE_TOOLS = ("write_file", "edit_file")

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # Inject-only: never blocks or escalates

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only file-writing operations carry a path worth verifying
        if ctx.tool_name not in self.WRITE_TOOLS:
            return None

        result = (ctx.tool_result or "").lstrip()
        # Fire on SUCCESS only. Failures surface as "ERROR: ..." (tool-level) or
        # "Error executing tool: ..." (kernel exception wrapper) — a failed edit
        # has nothing at the far end to verify.
        if not result or result.lower().startswith("error"):
            return None

        path = str(ctx.tool_args.get("path") or "").strip()
        if not path:
            return None

        return GuardVerdict.inject(
            self._message(path),
            reason="post_edit_far_end",
            # Independent category — the registry deduplicates injects by
            # category, so a shared category would silently swallow this
            # reminder whenever another guard injects in the same pass.
            category="post_edit_far_end",
        )

    @classmethod
    def _hint_for(cls, path: str) -> str:
        lower = path.lower()
        for ext, hint in _TYPE_HINTS.items():
            if lower.endswith(ext):
                return hint.replace("<path>", path)
        return _FALLBACK_HINT

    @classmethod
    def _message(cls, path: str) -> str:
        lines = [
            f"[Post-edit] {path} edited. Verify the FAR end now:",
            f"  · valid-for-type: {cls._hint_for(path)}",
            "  · will the consumer actually read it at this exact path?",
        ]
        if cls._is_agent_source(path):
            lines.append(
                "  · flagscale_agent/ source: the LIVE process still runs the OLD "
                "code until /reload."
            )
        return "\n".join(lines)

    @staticmethod
    def _is_agent_source(path: str) -> bool:
        return "flagscale_agent/" in path and path.endswith(".py")
