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

"""Tests for FindGuard — block `find` in favor of internal retrieval."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.find_guard import FindGuard


def _shell(cmd):
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


class TestFindBlocked:
    """Every shell command invoking `find` is blocked (overridable)."""

    def test_bare_find_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("find / -name foo.md"))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "find_invocation"
        assert v.overridable is True

    def test_scoped_find_still_blocks(self):
        # User chose "block all" — even a tightly-scoped find is blocked;
        # override releases it.
        g = FindGuard()
        v = g.check_pre(_shell("find ./src -maxdepth 2 -name '*.yaml'"))
        assert v is not None
        assert v.action == "block"

    def test_find_after_cd_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("cd /nfs && find . -name x"))
        assert v is not None
        assert v.action == "block"

    def test_find_after_pipe_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("ls | find . -name x"))
        assert v is not None
        assert v.action == "block"

    def test_message_mentions_retrieval_order(self):
        g = FindGuard()
        v = g.check_pre(_shell("find / -name foo"))
        assert "memory" in v.message
        assert "conversation" in v.message.lower()
        assert "grep" in v.message


class TestFindAllowed:
    """Non-find commands and false-positive substrings pass through."""

    def test_non_find_shell_passes(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls -la")) is None
        assert g.check_pre(_shell("grep -rn foo ./src")) is None

    def test_findutils_substring_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("apt-get install findutils")) is None

    def test_myfind_word_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("myfind . -name x")) is None
        assert g.check_pre(_shell("./myfind_tool /a")) is None

    def test_non_shell_tool_passes(self):
        g = FindGuard()
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "find.txt"})
        assert g.check_pre(ctx) is None

    def test_empty_command_passes(self):
        g = FindGuard()
        assert g.check_pre(_shell("")) is None


class TestEveryIterBlocks:
    """Block fires on EVERY find call, not once-per-turn."""

    def test_repeated_find_all_block(self):
        g = FindGuard()
        # No reset_turn between calls — each independent find is still blocked.
        assert g.check_pre(_shell("find / -name a")).action == "block"
        assert g.check_pre(_shell("find / -name b")).action == "block"
        assert g.check_pre(_shell("find / -name c")).action == "block"

    def test_override_reason_accepted(self):
        # Default accept_override: any reason > 5 chars releases it.
        g = FindGuard()
        assert g.accept_override("root is bounded ./src", _shell("find ./src")) is True
        assert g.accept_override("", _shell("find ./src")) is False


class TestMultilineFindBlocked:
    """A `find` on its own line of a multi-line command is a real invocation.

    Regression: the old regex used `^` without re.MULTILINE, so a find preceded
    only by a newline escaped the guard entirely.
    """

    def test_find_on_own_line_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("cd /tmp\nfind . -name x")).action == "block"

    def test_find_after_assignment_lines_blocks(self):
        g = FindGuard()
        cmd = "SH=/x\nMC=$SH/megatron\nfind / -path '*/m.py' 2>/dev/null | head"
        assert g.check_pre(_shell(cmd)).action == "block"

    def test_find_alone_at_end_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo hi; find")).action == "block"

    def test_bare_find_word_at_line_end_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls\nfind")).action == "block"


class TestQuotedAndHeredocAllowed:
    """`find` inside quotes or a heredoc body is data, not an invocation."""

    def test_find_inside_single_quotes_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo 'a | find b'")) is None

    def test_find_inside_double_quotes_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell('echo "x && find y"')) is None

    def test_find_in_python_heredoc_not_blocked(self):
        g = FindGuard()
        cmd = (
            "python3 - <<'PY'\n"
            "import re\n"
            "print(re.search('find', 'x'))\n"
            "PY\n"
        )
        assert g.check_pre(_shell(cmd)) is None

    def test_find_in_heredoc_with_quoted_body_not_blocked(self):
        g = FindGuard()
        cmd = "cat <<EOF\nfoo | find bar\nEOF\n"
        assert g.check_pre(_shell(cmd)) is None

    def test_real_find_still_blocked_with_quoted_args(self):
        # Quoted arguments are common in real finds; they must still block.
        g = FindGuard()
        assert g.check_pre(_shell("find . -name '*.yaml'")).action == "block"
        assert g.check_pre(_shell('find "$DIR" -name "*.py"')).action == "block"

    def test_find_after_separator_outside_quotes_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls 'a;b' && find . -name x")).action == "block"

    def test_unterminated_quote_does_not_crash(self):
        g = FindGuard()
        # No assertion on outcome beyond "returns without raising".
        g.check_pre(_shell("echo 'unterminated"))
        g.check_pre(_shell('echo "unterminated'))

    def test_bitshift_not_treated_as_heredoc(self):
        g = FindGuard()
        # `2 << 3` is not a heredoc; a following find on its own line still blocks.
        assert g.check_pre(_shell("echo $(( 2 << 3 ))\nfind . -name x")).action == "block"


class TestBroadGrepBlocked:
    """A RECURSIVE grep over a broad root is blocked (overridable)."""

    def test_grep_rn_on_root_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("grep -rn foo /"))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "broad_recursive_grep"

    def test_grep_rn_on_shared_mount_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn 'pattern' /public-nvme")).action == "block"
        assert g.check_pre(_shell("grep -Rl x /mnt")).action == "block"

    def test_grep_rn_on_etc_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn hostname /etc")).action == "block"

    def test_combined_short_flags_recursive_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -RIn foo /usr")).action == "block"

    def test_grep_long_recursive_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep --recursive foo /opt")).action == "block"

    def test_grep_mid_command_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("cd /tmp && grep -rn x /home")).action == "block"

    def test_grep_recursive_cwd_relative_targets_block(self):
        g = FindGuard()
        # `.` / `./` / `..` / `~` resolve to the cwd/parent/home at runtime;
        # the guard cannot prove they are bounded, so a recursive walk from
        # them is broad.
        for target in (".", "./", "..", "~", "~/"):
            v = g.check_pre(_shell(f"grep -rln pat {target}"))
            assert v is not None and v.reason == "broad_recursive_grep", target

    def test_grep_recursive_bare_glob_blocks(self):
        g = FindGuard()
        for target in ("*", ".*", "./*"):
            v = g.check_pre(_shell(f"grep -rln pat {target}"))
            assert v is not None and v.reason == "broad_recursive_grep", target

    def test_grep_recursive_no_target_blocks(self):
        g = FindGuard()
        # No file operand -> grep defaults to `.` (the cwd) -> unbounded.
        v = g.check_pre(_shell("grep -rln pat"))
        assert v is not None and v.reason == "broad_recursive_grep"

    def test_user_real_command_blocks(self):
        g = FindGuard()
        # Regression: a recursive grep piped through head, with a `.` target,
        # was the exact command that escaped the guard before this fix.
        cmd = ('cd /workspace/caozhou/baseline_v2 && ls *.py | head -30; '
               r'echo "==="; grep -rln "initialize_model_parallel\|' 
               'destroy_model_parallel" --include=*.py . 2>/dev/null | '
               'grep -v site-packages | head')
        v = g.check_pre(_shell(cmd))
        assert v is not None and v.reason == "broad_recursive_grep"

    def test_grep_cwd_relative_in_executor_payload_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell('ssh h "grep -rln pat ."'))
        assert v is not None and v.reason == "broad_recursive_grep"


class TestScopedGrepAllowed:
    """Scoped / non-recursive greps pass — they are what the guard recommends."""

    def test_grep_rn_on_subdir_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn foo ./src")) is None
        assert g.check_pre(_shell("grep -rn foo /public-nvme/proj/src")) is None

    def test_non_recursive_grep_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep foo /etc/hosts")) is None

    def test_grep_without_path_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("cat file | grep -n foo")) is None

    def test_grep_recursive_flag_letter_not_recursive(self):
        g = FindGuard()
        # `-i`/`-n` are not recursive even when combined with other letters.
        assert g.check_pre(_shell("grep -in foo ./src")) is None

    def test_grep_inside_heredoc_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("python3 - <<'PY'\ngrep -rn x /\nPY\n")) is None

    def test_grep_word_as_arg_not_command(self):
        g = FindGuard()
        # 'grep' as a non-command token should not trip the broad-grep path.
        assert g.check_pre(_shell("echo grep -rn foo /")) is None

    def test_grep_quoted_pattern_with_scoped_target_allowed(self):
        g = FindGuard()
        # sanitize() blanks the quoted pattern, so the top-level layer must not
        # mistake the scoped target for a pattern (any-positional rule). The
        # lexer path keeps the pattern intact and sees a scoped target.
        assert g.check_pre(_shell('grep -rn "foo" ./src')) is None
        assert g.check_pre(_shell(r"grep -rln 'a\|b' --include=*.py ./src")) is None

    def test_grep_named_component_target_allowed(self):
        g = FindGuard()
        # A named directory component is scoped even without `./`.
        assert g.check_pre(_shell("grep -rln pat src")) is None
        assert g.check_pre(_shell("grep -rln pat /workspace/caozhou/baseline_v2")) is None


class TestHiddenFind:
    """find / broad grep inside executor payloads and command substitution.

    The top-level sanitize pass strips quoted regions (so `echo 'a | find b'`
    stays allowed); but a quoted region handed to ssh/docker/bash is CODE the
    executor will run — often on a remote host or NFS tree. Regression: the
    user's real command escaped detection through ssh -> docker exec ->
    bash -lc -> single-quoted find.
    """

    USER_ESCAPED = (
        'ssh root@10.8.2.152 "docker exec caozhou_v2 bash -lc '
        "'find / -maxdepth 8 -name .session.lock -path \"*sessions*\" "
        '2>/dev/null | head -50\'"'
    )

    # ── block: executor payloads ──

    def test_user_escaped_command_blocks(self):
        v = FindGuard().check_pre(_shell(self.USER_ESCAPED))
        assert v is not None and v.action == "block"

    def test_ssh_docker_bash_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "docker exec c bash -lc \'find /x -name y\'"')) is not None

    def test_ssh_direct_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "find /x -name y"')) is not None

    def test_abs_path_find_in_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "/usr/bin/find / -name y"')) is not None

    def test_xargs_find_in_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "ls | xargs find"')) is not None

    def test_xargs_find_local_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls | xargs find / -name y")) is not None

    def test_broad_grep_in_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "docker exec c bash -c \'grep -r pattern /data\'"')) is not None

    def test_kubectl_exec_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("kubectl exec pod -- find / -name y")) is not None

    # ── block: command substitution executes ──

    def test_cmd_subst_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo $(find / -name y)")) is not None

    def test_backtick_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo `find / -name y`")) is not None

    # ── block: top-level prefix / absolute-path forms (fixed here too) ──

    def test_top_abs_path_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("/usr/bin/find / -name y")) is not None

    def test_top_var_prefix_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("FOO=1 find / -name y")) is not None

    def test_top_nohup_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("nohup find / -name y")) is not None

    def test_top_sudo_find_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("sudo find / -name y")) is not None

    # ── pass: pure data must NOT trip the hidden path ──

    def test_echo_quoted_data_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo 'a | find b'")) is None

    def test_python_c_quoted_data_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("python3 -c \"print('find /x')\"")) is None

    def test_heredoc_body_find_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("python3 - <<'EOF'\nfind / -name y\nEOF")) is None

    def test_plain_ssh_ls_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "ls /tmp"')) is None

    def test_docker_exec_ls_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("docker exec c ls /data")) is None

    def test_ssh_head_pipeline_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "ls /x | head -5"')) is None


class TestHiddenEscapeFamilies:
    """Round-2: systematic shell-escape coverage (obfuscation, wrappers,
    pipe-to-shell, command substitution, variable taint, line continuation).

    Each family pairs a MUST-block positive with controls that must stay
    allowed (data payloads, non-executor words, scoped commands).
    """

    # --- obfuscated command words (quote-in-word / backslash / ANSI-C) ---
    def test_quote_in_word_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("f'in'd /data")) is not None

    def test_backslash_in_word_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("f\\ind /data")) is not None

    def test_quote_glued_pairs_block(self):
        g = FindGuard()
        assert g.check_pre(_shell("'f'ind /data")) is not None
        assert g.check_pre(_shell("fi'nd' /data")) is not None

    def test_ansi_c_quoting_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("$'find' /data")) is not None

    def test_ansi_c_quoting_as_data_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo $'find' /data")) is None

    # --- line continuation splitting the command word ---
    def test_continuation_in_word_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("fi\\\nnd /data")) is not None

    def test_continuation_inside_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "fi\\\nnd /x"')) is not None

    # --- subshell / brace / process substitution ---
    def test_subshell_and_brace_block(self):
        g = FindGuard()
        assert g.check_pre(_shell("(find /data)")) is not None
        assert g.check_pre(_shell("{ find /data; }")) is not None

    def test_process_substitution_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("cat <(find /data)")) is not None

    # --- variable taint ---
    def test_tainted_variable_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("X=find; $X /data")) is not None
        assert g.check_pre(_shell('ssh h "X=find; $X /x"')) is not None

    def test_tainted_non_executor_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("X=findutils; ls $X")) is None

    def test_untainted_var_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls $X")) is None

    # --- executor wrappers (timeout / eval / stdbuf / ...) ---
    def test_timeout_wrapped_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("timeout 5 find /data")) is not None

    def test_eval_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('eval "find /data"')) is not None

    def test_stdbuf_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("stdbuf -o0 find /data")) is not None

    def test_wrapper_without_find_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("timeout 5 ls /data")) is None

    # --- pipe-to-shell (echo payload | sh / bash / <<<) ---
    def test_pipe_to_sh_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo 'find /x' | sh")) is not None

    def test_double_quoted_payload_pipe_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('echo "find /data -name x" | bash')) is not None

    def test_herestring_pipe_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('echo \'find /x\' <<< "" | sh')) is not None

    def test_pipe_without_find_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("cat /etc/hosts | bash")) is None
        assert g.check_pre(_shell("echo hi | wc -l")) is None

    # --- command substitution carrying the executor ---
    def test_cmd_subst_dollar_paren_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls $(find /data)")) is not None

    def test_backtick_subst_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo `find /data`")) is not None

    # --- payload depth: quoted payload executed by an executor ---
    def test_ssh_bare_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "find /data"')) is not None

    def test_ssh_obfuscated_payload_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ssh h \"f'i'n'd /x\"")) is not None

    def test_ssh_echo_data_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell('ssh h "echo find /x"')) is None

    # --- grep symmetry (absolute path / bare token after executor) ---
    def test_grep_absolute_path_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("/usr/bin/grep -rn p /data")) is not None

    def test_grep_bare_token_after_executor_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("kubectl exec pod -- grep -rn p /data")) is not None

    def test_scoped_and_nonrecursive_grep_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep p /etc/hosts")) is None
        assert g.check_pre(_shell("grep -rn p ./src")) is None


class TestCwdRelativeEscapeFamily:
    """Round-3/4 hardening: the FULL cwd-relative / inline-pattern escape set.

    These lock the two adversarial defects found after round-2: (a) deep `.`
    chains (`../..`, `./.`) and always-unbounded env vars (`$HOME`, `${PWD}`)
    escaped; (b) an inline pattern option (`--regexp=.`, `-e.`, `-fF`) was
    mistaken for the pattern, so the real scoped target was dropped and the
    call over-blocked.
    """

    def test_deep_dot_chain_blocks(self):
        g = FindGuard()
        for target in ("../..", "../../", "./.", "././.", "..//.."):
            v = g.check_pre(_shell(f"grep -rln pat {target}"))
            assert v is not None and v.reason == "broad_recursive_grep", target

    def test_cwd_env_vars_block(self):
        g = FindGuard()
        for target in ("$HOME", "${HOME}", "$PWD", "${PWD}", "$OLDPWD"):
            v = g.check_pre(_shell(f"grep -rln pat {target}"))
            assert v is not None and v.reason == "broad_recursive_grep", target

    def test_inline_pattern_with_scoped_target_allowed(self):
        g = FindGuard()
        # The pattern is INSIDE the option; the positional is a real target.
        assert g.check_pre(_shell("grep -rn --regexp=. ./src")) is None
        assert g.check_pre(_shell("grep -rn -e. ./src")) is None
        assert g.check_pre(_shell("grep -rn -f pats.txt ./src")) is None

    def test_inline_pattern_with_broad_target_blocks(self):
        g = FindGuard()
        # Inline pattern + a broad positional target must still block.
        assert g.check_pre(_shell("grep -rn --regexp=. . ./src")).reason == "broad_recursive_grep"
        assert g.check_pre(_shell("grep -rn -e. ./src .")).reason == "broad_recursive_grep"

    def test_bare_pattern_option_keeps_scoped_target_allowed(self):
        g = FindGuard()
        # A BARE -e/--regexp takes the next token as its pattern; the remaining
        # positional is the scoped target -> allowed.
        assert g.check_pre(_shell("grep -rn -e . ./src")) is None
        assert g.check_pre(_shell("grep -rn --regexp . ./src")) is None
