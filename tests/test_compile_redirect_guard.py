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

"""Tests for CompileRedirectGuard — pre-launch block on verbose build commands
that do not persist output to a file."""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.compile_redirect import CompileRedirectGuard


def _ctx(command):
    return GuardContext(tool_name="shell", tool_args={"command": command})


class TestCompileRedirectGuard:

    def setup_method(self):
        self.g = CompileRedirectGuard()

    # --- should BLOCK: compile command whose output is NOT visible to the monitor ---
    @pytest.mark.parametrize("cmd", [
        "make -j$(nproc)",
        "make -j8",
        "cmake --build build",
        "ninja",
        "cargo build --release",
        "opam install coq",
        "gcc -O2 main.c -o main",
        "g++ -std=c++17 a.cpp",
        "go build ./...",
        "make -j | tail",          # piped to pager = the classic hang
        "make 2>&1 | head -100",
        # bare file redirects are now a BLACK BOX to the monitor -> block
        "make -j$(nproc) > build.log 2>&1",
        "make -j8 >build.log",
        "cargo build &> out.txt",
        "opam install coq >> install.log 2>&1",
        "gcc -O2 main.c -o main > compile.log 2>&1",
        # a multi-package install with env setup and bare redirect:
        'unset HTTP_PROXY && eval $(opam env) && opam install pkg1 pkg2 -y > /tmp/opam_install2.log 2>&1; echo "EXIT: $?"',
        # `env -u VAR ...` prefix must NOT let the build slip past the guard: the
        # env branch has to consume its own -u unset pairs before the driver.
        # Seen live on compcert running invisible to the monitor.
        'env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy opam install coq.8.16.1 menhir --switch=compcert -y 2>&1 | tail -30; echo "EXIT: $?"',
        "env -u HTTP_PROXY make -j8 | tail",
        "env -i opam install coq -y | tail -20",
    ])
    def test_blocks_compile_not_visible_to_monitor(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is not None and v.action == "block", f"should block: {cmd}"
        assert v.category == "compile_redirect"

    # --- should PASS: output stays visible to the monitor (tee / tail -f) ---
    @pytest.mark.parametrize("cmd", [
        "cmake --build build 2>&1 | tee build.log",
        "ninja | tee ninja.log",
        "make -j$(nproc) 2>&1 | tee build.log",
        # background redirect + tail -f streams the log back to the terminal
        "opam install coq > install.log 2>&1 & tail -f install.log",
        "make -j8 > build.log 2>&1 & tail -F build.log",
        # env -u prefix with a proper tee stays visible -> must still PASS
        "env -u HTTP_PROXY opam install coq -y 2>&1 | tee install.log",
    ])
    def test_passes_compile_visible_to_monitor(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is None, f"should pass (tee / tail -f keeps output visible): {cmd}"

    # --- should PASS: not a compile command ---
    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "cat build.log",
        "grep error build.log",
        "echo make",              # 'make' as an argument, not the driver
        "tail -f build.log",
        "python train.py",
        "rm -rf build",
    ])
    def test_ignores_non_compile(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is None, f"should ignore: {cmd}"

    def test_non_shell_tool_ignored(self):
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "Makefile"})
        assert self.g.check_pre(ctx) is None

    def test_empty_command_ignored(self):
        assert self.g.check_pre(_ctx("")) is None

    def test_block_is_overridable(self):
        v = self.g.check_pre(_ctx("make -j"))
        assert v.action == "block"
        assert v.overridable is True
        # default accept_override: any reason > 5 chars
        assert self.g.accept_override("already writes its own log file", _ctx("make -j")) is True
        assert self.g.accept_override("", _ctx("make -j")) is False

    def test_stdbuf_wrapped_compile_still_blocks(self):
        # stdbuf -oL make ... still needs a file to be inspectable after
        v = self.g.check_pre(_ctx("stdbuf -oL -eL make -j4"))
        assert v is not None and v.action == "block"

    def test_stateless_across_calls(self):
        # unlike BackupGuard, this guard keys off the command text every time
        assert self.g.check_pre(_ctx("make -j")).action == "block"
        assert self.g.check_pre(_ctx("make -j 2>&1 | tee log")) is None
        assert self.g.check_pre(_ctx("make -j")).action == "block"

    def test_bare_redirect_is_blackbox_to_monitor(self):
        # regression: pure `> file` was previously treated as OK, but it hides
        # all output from the live monitor for the whole run (opam stall).
        assert self.g.check_pre(_ctx("make -j > build.log 2>&1")).action == "block"
        # tee keeps it visible
        assert self.g.check_pre(_ctx("make -j 2>&1 | tee build.log")) is None

    def test_message_guides_probe_then_bounded_parallel_build(self):
        # The block message must steer toward: probe single-threaded (-j1) first
        # under a timeout to surface the first real error cleanly, THEN scale up
        # with MEMORY-BOUNDED parallelism (not full single-threaded, not bare
        # unbounded -j). This balances throughput against OOM on small boxes.
        v = self.g.check_pre(_ctx("make -j > build.log 2>&1"))
        assert v.action == "block"
        msg = v.message
        # probe stage is single-threaded under a timeout
        assert "-j1" in msg
        assert "probe" in msg.lower()
        assert "timeout" in msg
        # real build scales up with a memory-derived -j (bounded parallelism)
        assert "MemAvailable" in msg
        assert "-j$J" in msg or '-j "$J"' in msg
        # must forbid bare UNBOUNDED make -j (the fork-bomb path)
        assert "make -j" in msg
        assert "OOM" in msg or "oom" in msg.lower()
        # verify exit code + binary after build/install
        assert "EXIT" in msg or "exit code" in msg.lower() or "exit:" in msg.lower()
        assert "which" in msg

    def test_message_caps_parallelism_absolutely(self):
        # Regression (caffe -j256): the memory-bounded formula alone does NOT
        # cap J on a big host (256 cores / ~1TB RAM passes the memory cap with a
        # huge J). 256 heavy compilers thrash cache/BW and spawn a process tree
        # that starves the monitor. The message must add an ABSOLUTE cap on top
        # of the memory cap so J settles at a sane value on big hosts.
        v = self.g.check_pre(_ctx("make -j > build.log 2>&1"))
        assert v.action == "block"
        msg = v.message
        # the shell formula must clamp J to an absolute ceiling of 32
        assert "J>32" in msg or "J > 32" in msg
        assert "32" in msg
        # and the prose must explain WHY an unbounded big-host J is harmful
        assert "absolute cap" in msg.lower()
        low = msg.lower()
        assert "thrash" in low or "bandwidth" in low or "starve" in low

    def test_message_makes_parallelism_cgroup_aware(self):
        # Regression (pitfall/flagscale_agent/shell_oom_cgroup_blindspot, caffe
        # -j32 OOM): inside a container `nproc` and /proc/meminfo report the HOST
        # (256 cores / 909GB) even when the cgroup caps the process at 1 CPU /
        # 2GB. The old formula read only the host and picked -j32, forking 32
        # cc1plus that the kernel OOM-killed. The message's shell formula must
        # read the cgroup limit FIRST and take the min against the host.
        v = self.g.check_pre(_ctx("make -j > build.log 2>&1"))
        assert v.action == "block"
        msg = v.message
        # must read the cgroup limit files (v2 primary, v1 fallback)
        assert "/sys/fs/cgroup/cpu.max" in msg
        assert "/sys/fs/cgroup/memory.max" in msg
        assert "cfs_quota_us" in msg or "cfs_period_us" in msg   # v1 cpu path
        assert "memory.limit_in_bytes" in msg                    # v1 mem path
        # prose must explain the host-vs-cgroup trap and name the cc1plus OOM
        low = msg.lower()
        assert "cgroup" in low
        assert "container" in low
        assert "cc1plus" in low

    def test_message_advises_one_by_one_install(self):
        """The block message must advise installing critical packages ONE BY ONE,
        not all at once — a multi-package install can freeze for 10+ minutes."""
        v = self.g.check_pre(_ctx("cargo build --release"))
        assert v.action == "block"
        msg = v.message
        # must mention one-by-one / individual install
        assert "ONE BY ONE" in msg or "one by one" in msg or "individually" in msg
        # must show per-package install examples
        assert "pip install" in msg
        # must advise checking exit code after each package
        assert "EXIT" in msg or "exit code" in msg.lower()
        # must mention that multi-package install can freeze/stall
        assert "freeze" in msg.lower() or "stall" in msg.lower() or "frozen" in msg.lower()
        # must apply to all package managers
        assert "apt" in msg or "pip" in msg or "cargo" in msg or "npm" in msg
        # must NOT contain task-specific package names or tools (no leaking)
        assert "coq" not in msg.lower()
        assert "menhir" not in msg.lower()
        assert "ocamlfind" not in msg.lower()
        assert "opam" not in msg.lower()

    def test_message_advises_overlapping_independent_steps(self):
        """Regression (insight/tbench/caffe_timeout_rootcause_budget_exhaustion):
        the build and a dependency-free download/install were run SERIALLY, so the
        download idled behind the ~8min build and burned the per-task budget. The
        block message must steer toward overlapping the background build with the
        independent expensive steps (download/install) instead of serializing them."""
        v = self.g.check_pre(_ctx("make -j > build.log 2>&1"))
        assert v.action == "block"
        msg = v.message
        low = msg.lower()
        # must name the overlap concept and the background mechanism
        assert "overlap" in low
        assert "background" in low
        # must call out that independent steps do NOT depend on the build output
        assert "independent" in low or "depend" in low
        # must name the concrete independent steps: download and install
        assert "download" in low
        assert "install" in low
        # must express the wall-clock model: longest branch, not the sum
        assert "longest" in low or "not the sum" in low or "concurrent" in low
        # generic, no dataset/task/framework leaking anywhere in the message —
        # the whole guard fires on EVERY build command, so it must not name a
        # concrete benchmark task, dataset, or framework.
        assert "cifar" not in low
        assert "caffe" not in low
        assert "fasttext" not in low
