"""Command handlers for FlagScale Agent slash commands.

Extracted from agent.py to reduce file size and improve separation of concerns.
"""

import os
import time

from flagscale_agent.react.session import (
    find_resumable_sessions, load_conversation, mark_completed,
)
from flagscale_agent.react.tools.shell import ShellTool


class CommandHandler:
    """Handles slash commands for WorkerAgent.

    This class encapsulates all CLI command handling logic, keeping agent.py
    focused on core agent orchestration.
    """

    def __init__(self, agent):
        """Initialize with reference to parent agent.

        Args:
            agent: WorkerAgent instance that owns this handler
        """
        self.agent = agent

    def handle_slash_command(self, user_input: str) -> bool:
        """Dispatch slash command to appropriate handler.

        Args:
            user_input: Raw user input starting with / (or bare 'resume')

        Returns:
            True if command was handled, False otherwise
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
        elif cmd == "/skill":
            self._handle_skill(user_input)
            return True
        elif cmd == "/file":
            self._handle_file(user_input)
            return True
        elif cmd == "/save":
            self.agent._save_conversation(completed=False)
            print("Conversation saved.")
            return True
        elif cmd == "/load":
            self._handle_load(user_input)
            return True
        elif cmd == "/export":
            self._handle_export(user_input)
            return True
        elif cmd == "/memory":
            self._handle_memory(user_input)
            return True
        elif cmd == "/mode":
            self._handle_mode(user_input)
            return True
        elif cmd == "/plan":
            self._handle_plan(user_input)
            return True
        elif cmd == "/resume":
            self._handle_resume(user_input)
            return True
        elif cmd == "/compact":
            self.agent.history.force_compact(target_ratio=0.50)
            print("History compacted.")
            return True
        return False

    def _handle_skill(self, user_input: str):
        """Handle /skill command - list or load skills."""
        parts = user_input.split()
        if len(parts) < 2:
            skills = self.agent.skill_manager.list_skills()
            print("Available skills:")
            for s in skills:
                print(f"  {s['name']}: {s['description'][:60]}")
            return
        name = parts[1]
        try:
            self.agent.skill_manager.load(name)
            print(f"Skill '{name}' loaded.")
        except FileNotFoundError:
            print(f"Skill '{name}' not found.")

    def _handle_file(self, user_input: str):
        """Handle /file command - read and display file."""
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /file <path>")
            return
        path = parts[1]
        if os.path.isfile(path):
            result = self.agent.tool_registry.execute("read_file", path=path)
            print(result[:2000])
        else:
            print(f"File not found: {path}")

    def _handle_load(self, user_input: str):
        """Handle /load command - load previous session."""
        parts = user_input.split()
        sessions = find_resumable_sessions(self.agent._sessions_root)
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(sessions):
                s = sessions[idx]
                data = load_conversation(s["session_dir"])
                if data:
                    self.agent._restore_session(data)
                    print(f"Loaded session: {s.get('last_user_msg', '')[:60]}")
                    return
        if not sessions:
            print("No resumable sessions found.")
            return
        print("Resumable sessions:")
        for i, s in enumerate(sessions[:10], 1):
            print(f"  {i}. [{time.strftime('%m-%d %H:%M', time.localtime(s['timestamp']))}] {s.get('last_user_msg', '')[:60]}")
        print("Usage: /load <number>")

    def _handle_export(self, user_input: str):
        """Handle /export command - export conversation."""
        path = os.path.join(self.agent._session_dir, "conversation.json")
        print(f"Conversation exported to: {path}")

    def _handle_memory(self, user_input: str):
        """Handle /memory command - manage session memory."""
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /memory list | /memory clear [type] | /memory delete <key>")
            return
        sub = parts[1]
        if sub == "list":
            entries = self.agent.session_memory.list_entries()
            if not entries:
                print("No memory entries.")
                return
            for e in entries:
                key = e.get("key", "?")
                mem_type = e.get("type", "?")
                content = e.get("content", "")
                print(f"  [{mem_type}] {key}: {content[:80]}")
        else:
            print(f"Unknown /memory subcommand: {sub}")

    def _handle_mode(self, user_input: str):
        """Handle /mode command - switch between confirm/auto mode."""
        parts = user_input.split()
        if len(parts) < 2:
            print(f"Current mode: {self.agent.config.mode}")
            print("Available modes: confirm, auto")
            return
        mode = parts[1]
        if mode in ("confirm", "auto"):
            self.agent.config.mode = mode
            if mode == "auto":
                self.agent.config.confirm_commands = False
                self.agent.config.max_iterations = 2**31 - 1
                # Re-register shell tool without confirm
                self.agent.tool_registry._tools.pop("shell", None)
                self.agent.tool_registry.register(
                    ShellTool(
                        remind_interval=self.agent.config.shell_remind_interval,
                        check_dangerous=self.agent.config.dangerous_commands_check,
                        require_confirm=False,
                        env=self.agent.config.shell_env,
                        health_judge_fn=self.agent._health_judge,
                    )
                )
            print(f"Mode set to: {mode}")
        else:
            print(f"Unknown mode: {mode}")

    def _handle_plan(self, user_input: str):
        """Handle /plan command - show active plan."""
        parts = user_input.split()
        if len(parts) < 2:
            active = self.agent.task_plan.get_active()
            if active:
                print(f"Active plan: {active.get('id', '?')}")
                for step in active.get("steps", []):
                    icon = {"pending": " ", "doing": "→", "done": "✓", "skipped": "-", "blocked": "!"}.get(step.get("status", "pending"), " ")
                    title = step.get("title", "") or step.get("description", "")
                    print(f"  [{icon}] {title[:80]}")
            else:
                print("No active plan.")
            return
        print(f"Unknown /plan subcommand: {' '.join(parts[1:])}")

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
                # Match by numeric index
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target = sessions[idx]
            else:
                # Match by session ID prefix
                for s in sessions:
                    sid = s.get("session_id", "")
                    if sid.startswith(arg) or sid[:12].startswith(arg):
                        target = s
                        break
            if target:
                data = load_conversation(target["session_dir"])
                if data:
                    self.agent._restore_session(data, target["session_dir"])
                    sid = target.get("session_id", "?")[:12]
                    print(f"Resumed session {sid} ({target.get('user_turns', 0)} turns)")
                    return
                else:
                    print(f"Failed to load conversation from {target['session_dir']}")
                    return
            print(f"No session matching '{arg}' found.")
        for i, s in enumerate(sessions[:10], 1):
            sid = s.get("session_id", "?")[:12]
            ts = time.strftime("%m-%d %H:%M", time.localtime(s['timestamp']))
            skills = s.get("loaded_skills", [])
            skill_str = f" [{','.join(skills[:2])}]" if skills else ""
            print(f"  {i}. {sid}  {ts}{skill_str}  ({s.get('user_turns', 0)} turns)")
        print("Usage: /resume <number|session_id>")

    def _handle_reload(self, user_input: str):
        """Hot reload: save state, exec new process, auto-resume.

        /reload        — full code reload (restart process)
        /reload config — config-only reload (no restart)
        """
        parts = user_input.split()
        if len(parts) > 1 and parts[1] == "config":
            # Lightweight: just reload config and skills, no process restart
            self.agent.config.reload()
            self.agent.skill_manager.invalidate_cache()
            self.agent._refresh_system_prompt()
            print("Config and skills reloaded (no code reload).")
            return

        # Full code reload via process restart
        print("Saving session state...")
        self.agent._save_conversation(completed=False)

        session_id = self.agent._session_id
        print(f"Restarting process (session: {session_id})...")
        print("All code changes will take effect.\n")

        # Build the command to restart with auto-resume
        import sys
        import os

        # Determine how the agent was launched
        argv = sys.argv[:]

        # Inject --auto-resume flag with session_id
        # Remove any existing --auto-resume to avoid duplication
        clean_argv = [a for a in argv if not a.startswith("--auto-resume")]
        clean_argv.append(f"--auto-resume={session_id}")

        # Use os.execv to replace current process — no orphan processes
        os.execv(sys.executable, [sys.executable] + clean_argv)
