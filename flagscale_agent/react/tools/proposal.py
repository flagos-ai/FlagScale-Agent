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

"""Proposal tool — register/track harness-improvement proposals across sessions.

Backs the wrap-up routine's HARNESS GAP & CAPTURE item: proposals get a
persistent, stateful home so an unreviewed proposal from an earlier session
resurfaces at a later wrap-up instead of being forgotten.
"""

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.react.proposals import (
    VALID_STATUSES,
    OPEN_STATUSES,
)


class ProposalTool(Tool):
    name = "proposal"
    description = (
        "Register and track improvement proposals for the agent harness itself "
        "(guards/tools/prompts/skills/knowledge). Proposals persist ACROSS "
        "sessions with a status, so an unreviewed one resurfaces at the next "
        "wrap-up instead of being forgotten.\n\n"
        "Actions:\n"
        "- add: register a new proposal (description[, container, topic]). "
        "container is the routing: 'agent-code' | 'skill' | 'knowledge'.\n"
        "- list: show open (unreviewed) proposals. Call this at wrap-up to find "
        "proposals raised earlier that still need the human's decision.\n"
        "- update: change a proposal's status (proposal_id, status[, note]) — OR "
        "resolve several at once by passing 'updates': a list of "
        "{proposal_id, status[, note]} objects.\n\n"
        f"Valid statuses: {', '.join(VALID_STATUSES)}. "
        f"'proposed'/'approved' are OPEN (resurface at wrap-up); "
        "'done'/'rejected'/'superseded' are terminal (closed)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "update"],
                "description": "add a proposal, list open proposals, or update one's status.",
            },
            "description": {
                "type": "string",
                "description": "add: one-line description of the proposed improvement.",
            },
            "container": {
                "type": "string",
                "description": (
                    "add: routing for the proposal — 'agent-code' (guards/tools/"
                    "prompt machinery), 'skill' (multi-step procedure), or "
                    "'knowledge' (missing mechanism/context docs)."
                ),
            },
            "topic": {
                "type": "string",
                "description": "add: short slug naming the subject of the proposal.",
            },
            "proposal_id": {
                "type": "string",
                "description": "update: the id (prop_xxxxxxxx) of the proposal to change.",
            },
            "status": {
                "type": "string",
                "enum": list(VALID_STATUSES),
                "description": (
                    "update: the new status. 'done' = human agreed AND it is "
                    "implemented; 'rejected' = human declined."
                ),
            },
            "note": {
                "type": "string",
                "description": "update: optional short note recorded with the transition.",
            },
            "updates": {
                "type": "array",
                "description": (
                    "update: batch form — a list of "
                    "{proposal_id, status[, note]} objects to resolve several "
                    "proposals in one call. Use instead of the single "
                    "proposal_id/status pair."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "status": {"type": "string", "enum": list(VALID_STATUSES)},
                        "note": {"type": "string"},
                    },
                    "required": ["proposal_id", "status"],
                },
            },
        },
        "required": ["action"],
    }

    def __init__(self, registry, session_id: str = ""):
        self._registry = registry
        self._session_id = session_id

    def _update_batch(self, updates: list) -> str:
        """Apply several status transitions in one call.

        Accepts a list of {proposal_id, status[, note]} dicts. Each item is
        resolved independently (one bad item does not abort the rest), and the
        result line reports ok/error per proposal.
        """
        lines = []
        ok = 0
        for i, item in enumerate(updates):
            if not isinstance(item, dict):
                lines.append(f"  ✗ item {i}: not an object")
                continue
            pid = (item.get("proposal_id") or "").strip()
            st = (item.get("status") or "").strip()
            nt = item.get("note") or ""
            if not pid:
                lines.append(f"  ✗ item {i}: missing 'proposal_id'")
                continue
            if st not in VALID_STATUSES:
                lines.append(
                    f"  ✗ {pid}: 'status' must be one of {VALID_STATUSES} (got {st!r})"
                )
                continue
            try:
                entry = self._registry.set_status(
                    pid, st, session_id=self._session_id, note=nt,
                )
            except ValueError as e:
                lines.append(f"  ✗ {pid}: {e}")
                continue
            ok += 1
            lines.append(f"  ✓ {entry['id']} → {entry.get('status')}")
        header = f"Batch update: {ok}/{len(updates)} applied."
        return "\n".join([header] + lines)

    def execute(self, action: str = "", description: str = "", container: str = "",
                topic: str = "", proposal_id: str = "", status: str = "",
                note: str = "", updates: list = None, **kwargs) -> str:
        action = (action or "").strip()
        if action == "add":
            if not (description or "").strip():
                return "ERROR: action='add' requires a non-empty 'description'."
            if container and container not in ("agent-code", "skill", "knowledge"):
                return (
                    "ERROR: container must be 'agent-code', 'skill', or 'knowledge' "
                    f"(got {container!r})."
                )
            entry = self._registry.add(
                description=description, container=container, topic=topic,
                session_id=self._session_id,
            )
            return (
                f"Registered proposal {entry['id']} (status=proposed): "
                f"{entry['description']}"
            )

        if action == "list":
            open_ = self._registry.list_open()
            if not open_:
                return "No open proposals."
            lines = [f"{len(open_)} open proposal(s):"]
            for e in open_:
                cont = f"[{e.get('container')}] " if e.get("container") else ""
                lines.append(
                    f"  • {e['id']} ({e.get('status')}): {cont}{e.get('description','')}"
                )
            return "\n".join(lines)

        if action == "update":
            if updates:
                return self._update_batch(updates)
            if not proposal_id:
                return "ERROR: action='update' requires 'proposal_id'."
            if status not in VALID_STATUSES:
                return (
                    f"ERROR: 'status' must be one of {VALID_STATUSES} (got {status!r})."
                )
            try:
                entry = self._registry.set_status(
                    proposal_id, status, session_id=self._session_id, note=note,
                )
            except ValueError as e:
                return f"ERROR: {e}"
            return (
                f"Updated {entry['id']} → {entry.get('status')}: "
                f"{entry.get('description','')}"
            )

        return (
            "ERROR: unknown action {!r}. Use one of: add, list, update.".format(action)
        )
