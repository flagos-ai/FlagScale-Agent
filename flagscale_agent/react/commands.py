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

"""Command handlers for FlagScale Agent slash commands."""

import os
import time

from flagscale_agent.react import display
from flagscale_agent.react.session import (
    find_resumable_sessions, load_conversation, get_session_lock_holder,
    SessionLockedError,
)


class CommandHandler:
    """Handles slash commands for WorkerAgent."""

    def __init__(self, agent):
        self.agent = agent

    def handle_slash_command(self, user_input: str) -> bool:
        """Dispatch slash command to appropriate handler.

        Returns True if command was handled, False otherwise.
        """
        # Allow bare "resume" or "resume <arg>" without / prefix
        stripped = user_input.strip()
        if stripped == "resume" or stripped.startswith("resume "):
            self._handle_resume("/" + stripped)
            return True

        cmd = user_input.split()[0] if user_input.startswith("/") else None
        if not cmd:
            return False

        if cmd == "/quit":
            self.agent._exit()
            return True
        elif cmd == "/reload":
            self._handle_reload(user_input)
            return True
        elif cmd == "/resume":
            self._handle_resume(user_input)
            return True
        elif cmd == "/session":
            self._handle_session()
            return True
        return False

    def _handle_session(self):
        """Handle /session command — display current session info."""
        agent = self.agent
        created = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(os.path.getctime(agent._session_dir))
        ) if os.path.exists(agent._session_dir) else "unknown"
        print(f"\n  Session ID:  {agent._session_id}")
        print(f"  Directory:   {agent._session_dir}")
        print(f"  Created:     {created}")
        print(f"  Turns:       {agent.turn_count}")
        print()

    def _handle_resume(self, user_input: str):
        """Handle /resume command - resume previous session.

        Supports:
          /resume         — list resumable sessions
          /resume 1       — resume by numeric index
          /resume f73eb28f — resume by session ID (prefix match)
        """
        sessions = find_resumable_sessions(self.agent._sessions_root)
        if not sessions:
            print("No resumable sessions found.")
            return
        parts = user_input.split()
        if len(parts) >= 2:
            arg = parts[1]
            target = None
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target = sessions[idx]
            else:
                for s in sessions:
                    sid = s.get("session_id", "")
                    if sid.startswith(arg) or sid[:12].startswith(arg):
                        target = s
                        break
            if target:
                # Concurrency guard: refuse to resume a directory another
                # live agent process holds (their saves would overwrite this
                # process's history wholesale — last-writer-wins).
                holder = get_session_lock_holder(target["session_dir"])
                if holder:
                    print(display.red(
                        f"REFUSED: session {target.get('session_id', '?')[:8]} is held by "
                        f"another live agent process (PID {holder.get('pid')}, started "
                        f"{holder.get('start', '?')}: {holder.get('cmd', '?')}). "
                        f"Stop it first or pick a different session."
                    ))
                    return
                data = load_conversation(target["session_dir"])
                if data:
                    try:
                        self.agent._restore_session(data, target["session_dir"])
                    except SessionLockedError as e:
                        # TOCTOU: lock was free at pre-check, taken between
                        # check and restore. _restore_session acquired nothing
                        # in that case (guard raises before binding).
                        print(display.red(str(e)))
                        return
                    sid = target.get("session_id", "?")[:12]
                    print(f"Resumed session {sid} ({target.get('user_turns', 0)} turns)")
                    return
                else:
                    print(f"Failed to load conversation from {target['session_dir']}")
                    return
            print(f"No session matching '{arg}' found.")
            return
        for i, s in enumerate(sessions, 1):
            sid = s.get("session_id", "?")[:8]
            ts = time.strftime("%m-%d %H:%M", time.localtime(s['timestamp']))
            turns = s.get("user_turns", 0)
            summary = s.get("session_summary", "")
            if not summary:
                self.agent._generate_missing_summaries([s])
                summary = s.get("session_summary", "(no summary)")
            print(f"  {i}. {sid}  {ts} ({turns} turns):")
            for line in summary.strip().split("\n"):
                print(f"     {line}")
        print("\nUsage: /resume <number|session_id>")

    def _handle_reload(self, user_input: str):
        """Hot reload: save state, exec new process, auto-resume.

        /reload        — full code reload (restart process)
        /reload config — config-only reload (no restart)
        """
        parts = user_input.split()
        if len(parts) > 1 and parts[1] == "config":
            self.agent.config.reload()
            self.agent.skill_manager.invalidate_cache()
            self.agent._refresh_system_prompt()
            print("Config and skills reloaded (no code reload).")
            return

        print("Saving session state...")
        self.agent._save_conversation(completed=False)

        session_id = self.agent._session_id
        print(f"Restarting process (session: {session_id})...")
        print("All code changes will take effect.\n")

        import sys

        argv = sys.argv[:]
        clean_argv = [a for a in argv if not a.startswith("--auto-resume")]
        clean_argv.append(f"--auto-resume={session_id}")
        os.execv(sys.executable, [sys.executable] + clean_argv)
