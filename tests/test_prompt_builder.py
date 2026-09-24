"""Tests for PromptBuilder._build_dashboard and _build_memory_keys_summary."""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────

def make_builder():
    """Create a PromptBuilder with a mock SkillManager."""
    from flagscale_agent.react.prompt_builder import PromptBuilder
    skill_mgr = MagicMock()
    skill_mgr.list_skills.return_value = []
    return PromptBuilder(skill_mgr)


# ── _build_dashboard ─────────────────────────────────────────────────────

class TestBuildDashboard:
    def test_turn_always_present(self):
        b = make_builder()
        b._turn_count = 7
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Turn: 7" in result

    def test_no_plan_no_task_step(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Task:" not in result
        assert "Step:" not in result

    def test_plan_title_extracted(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = '<active-plan title="My Task" status="active">'
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Task: My Task" in result

    def test_plan_step_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[🔄] Step 2: in progress\n'
            '[⬜] Step 3: pending\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_plan_step_pending_when_no_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[⬜] Step 2: next\n'
            '[⬜] Step 3: later\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_session_dir_injected(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="/tmp/sess123")
        assert "Session: /tmp/sess123" in result
        assert "conversation.json: /tmp/sess123/conversation.json" in result
        assert "conversation_full.json: /tmp/sess123/conversation_full.json" in result

    def test_no_session_dir_omits_paths(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "conversation.json" not in result
        assert "Session:" not in result

    def test_memory_domains_present(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/a(2) pitfall/b(1) (3 keys total; memory_list(keyword=...) to search)"):
            result = b._build_dashboard("", session_dir="")
        assert "Memory domains: fact/a(2) pitfall/b(1)" in result

    def test_memory_keys_omitted_when_empty(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Memory domains" not in result

    def test_full_dashboard_all_parts(self):
        """All parts present when plan + session + memory all provided."""
        b = make_builder()
        b._turn_count = 5
        plan_ctx = '<active-plan title="Deploy" >\n[🔄] Step 1:\n[⬜] Step 2:\n'
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/x"):
            result = b._build_dashboard(plan_ctx, session_dir="/home/user/.flagscale/sessions/abc")
        assert "Task: Deploy" in result
        assert "Step: 1/2" in result
        assert "Turn: 5" in result
        assert "Session: /home/user/.flagscale/sessions/abc" in result
        assert "Memory domains: fact/x" in result

# ── Runtime gauges (Ctx / Time / BG) ────────────────────────────────────

class TestDashboardRuntimeGauges:
    """Dashboard gauges: Ctx pressure/evictable/evicted, Time budget, BG jobs."""

    def test_ctx_from_history(self):
        b = make_builder()
        b._turn_count = 1
        hist = MagicMock()
        hist.get_context_pressure.return_value = 0.83
        hist.get_evictable_indexes.return_value = [1, 2, 3, 4, 5]
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="", history=hist)
        assert "Ctx: 83% evictable=5" in result

    def test_ctx_from_runtime_stats_snapshot(self):
        """When no history passed, gauge falls back to runtime_stats snapshot."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"ctx_pressure": 0.5, "evictable": 12, "evict_count": 30}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Ctx: 50% evictable=12 evicted=30" in result

    def test_ctx_absent_when_no_data(self):
        """No history, no snapshot → no Ctx gauge (never a fabricated 0%)."""
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Ctx:" not in result

    def test_time_budget_rendered(self):
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": {"elapsed": 600, "budget": 3600,
                                      "remaining": 3000, "pct": 16.7}}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time: 17% used, 50m left" in result

    def test_time_absent_when_no_budget(self):
        """No external deadline → _task_budget_stats()=None → no Time gauge."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": None}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time:" not in result

    def test_bg_jobs_listed(self):
        b = make_builder()
        b._turn_count = 1
        job = MagicMock()
        job.job_id = "job3"
        job.status_str.return_value = "running"
        job.start = 0.0  # elapsed = time.time() - 0 → large but formatted
        from flagscale_agent.react.tools import shell as shell_mod
        saved = shell_mod._JOB_REGISTRY
        shell_mod._JOB_REGISTRY = MagicMock()
        shell_mod._JOB_REGISTRY.all.return_value = [job]
        try:
            with patch.object(b, "_build_memory_keys_summary", return_value=""):
                result = b._build_dashboard("", session_dir="")
        finally:
            shell_mod._JOB_REGISTRY = saved
        assert "BG: job3:running(" in result

    def test_bg_absent_when_no_jobs(self):
        b = make_builder()
        b._turn_count = 1
        from flagscale_agent.react.tools import shell as shell_mod
        saved = shell_mod._JOB_REGISTRY
        shell_mod._JOB_REGISTRY = MagicMock()
        shell_mod._JOB_REGISTRY.all.return_value = []
        try:
            with patch.object(b, "_build_memory_keys_summary", return_value=""):
                result = b._build_dashboard("", session_dir="")
        finally:
            shell_mod._JOB_REGISTRY = saved
        assert "BG:" not in result

# ── _build_memory_keys_summary ───────────────────────────────────────────

class TestBuildMemoryKeysSummary:
    def test_domain_counts_no_values(self, tmp_path):
        """Second-level groups with counts; values never leak into the summary."""
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/port", "fact", "value: 22")
        mem.put("fact/cluster/ip", "fact", "value: 10.0.0.1")
        mem.put("pitfall/nccl/hang", "pitfall", "symptom: hang")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster(2)" in result
        assert "pitfall/nccl(1)" in result
        # full keys and values must NOT appear
        assert "fact/cluster/port" not in result
        assert "value: 22" not in result
        assert "symptom: hang" not in result
        # total count suffix
        assert "(3 keys total" in result

    def test_malformed_key_falls_back_to_type(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/ok", "fact", "x")
        mem.put("weird", "fact", "y")  # no second segment

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster(1)" in result
        assert "weird(1)" in result

    def test_empty_memory_returns_empty_string(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_exception_returns_empty_string(self):
        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.Memory", side_effect=Exception("boom")):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_multiple_domains_sorted_with_counts(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/agent/a", "fact", "x")
        mem.put("fact/agent/b", "fact", "y")
        mem.put("fact/agent/c", "fact", "z")
        mem.put("pitfall/baseline/p1", "pitfall", "p")
        mem.put("pitfall/baseline/p2", "pitfall", "q")
        mem.put("insight/agent/i1", "insight", "i")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        # sorted alphabetically, counts correct, no truncation
        assert result.startswith("fact/agent(3) insight/agent(1) pitfall/baseline(2)")
        assert "(6 keys total" in result
        # no per-key truncation: every token before " (" is a full "type/domain(n)" pair
        for tok in result.split(" (")[0].split():
            assert tok in ("fact/agent(3)", "insight/agent(1)", "pitfall/baseline(2)")


# ── SYSTEM_PROMPT_STATIC content (regression guards) ─────────────────────

class TestSystemPromptContent:
    """Guard against silent removal of failure-loop / cognitive-mode instructions."""

    def _prompt(self):
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        return SYSTEM_PROMPT_STATIC

    # ── Pitfall Recall section (proactive recall) ────────────────────────

    def test_pitfall_recall_section_present(self):
        assert "## Pitfall Recall — Check Before You Act" in self._prompt()

    def test_pitfall_recall_live_risk_signal(self):
        # Ties the dashboard domain-count line to the recall action.
        assert "LIVE RISK SIGNAL" in self._prompt()

    def test_pitfall_recall_action_chain(self):
        # The three concrete recall commands must all be named.
        p = self._prompt()
        assert "memory_read(key='pitfall/<domain>/')" in p
        assert "memory_read(key='fact/<domain>/')" in p
        assert "memory_list(keyword='...')" in p

    def test_pitfall_recall_timing_rule(self):
        # Recall BEFORE the error, and write-back after >2 debugging rounds.
        p = self._prompt()
        assert "Recall BEFORE the error" in p
        assert ">2 rounds" in p

    def test_stall_is_failure_mode_present(self):
        # Covers regex-chess-style loops: re-analyzing same bug / thinking without acting.
        assert "Stalling is also a failure mode" in self._prompt()

    def test_information_gain_signal_present(self):
        # The stall trigger must be information-gain, not a clock or counter.
        p = self._prompt()
        assert "information gain" in p
        assert "does not depend on a clock or a counter" in p

    def test_protect_working_partial_solution_present(self):
        # Don't let an unsolved detail destroy a working partial solution.
        assert "Do not let an unsolved detail destroy a working partial solution" in self._prompt()

    def test_carry_forward_constraints_present(self):
        # Constraint amnesia: pivoting to clear a blocker must not revert to a proven-bad state.
        p = self._prompt()
        assert "Carry forward the constraints you already established" in p
        assert "Re-breaking a known constraint is negative progress" in p

    def test_switching_requires_fundamental_difference_present(self):
        assert "Switching methods requires fundamental difference, not variants" in self._prompt()

    def test_classification_gate_present(self):
        assert "CLASSIFICATION GATE" in self._prompt()

    def test_same_object_parallel_write_rule_present(self):
        # DON'T rule: parallel read-write on the SAME object = lost-update race
        # (20260908: three parallel edit_file calls to one doc — all reported
        # success, only the last edit survived; re-hit in E29).
        p = self._prompt()
        assert "read-and-write the SAME object in one parallel batch" in p
        assert "Same-file edits MUST be sequential" in p
        assert "only the last edit survives" in p
        assert "(Reads are safe.)" in p

    # ── new three-principle structure guards ─────────────────────────────

    def test_three_principles_structure_present(self):
        # The cognitive block is now organized as three principles, each gating
        # a distinct decision moment. Guard against silent collapse back to a flat list.
        p = self._prompt()
        assert "PRINCIPLE 1 — Understand before you implement" in p
        assert "PRINCIPLE 2 — Serve the real goal, do not self-deceive" in p
        assert "PRINCIPLE 3 — When you fail, escape downward or upward, never sideways" in p

    def test_generalize_not_overfit_present(self):
        # Principle 2 core: a scoring check is only a SAMPLE; the goal is a method
        # that generalizes to the real use, not one that overfits the sampled check.
        p = self._prompt()
        assert "only a SAMPLE of a real-world need" in p
        assert "GENERALIZES to the real use" in p

    def test_near_vs_far_end_verification_present(self):
        # Principle 2 core: the subtler self-deception is running an affirmative test
        # that confirms the NEAR end of the chain (command accepted / channel works)
        # and mistaking it for the FAR end (target actually changed state). Verify
        # where the real consumer observes the effect, not the sender's success report.
        p = self._prompt()
        assert "NEAR end of the chain" in p
        assert "FAR end" in p
        assert "the target's own state" in p
        assert "is not \"the input took effect.\"" in p

    def test_single_sample_overfit_magic_constants_present(self):
        # video-processing lesson: developing against ONE visible sample but graded on
        # hidden inputs — passing the sample is the floor, not generalization. Magic
        # constants tuned to the visible sample are the tell; prefer relative/normalized/
        # structure-derived judgments. Task-agnostic (no jump/video/hurdle specifics).
        p = self._prompt()
        assert "calibrating to the one example you can see" in p
        assert "hardcoded number" in p
        assert "structure-derived" in p
        # leak guard: no task-specific concrete tuned numbers
        import re
        for _lit in ("threshold of 25", "cutoff of 2000", "exceeds 40", "rises more than 40"):
            assert _lit not in p, f"leaked concrete tuned magic number: {_lit!r}"
        for w in ("jump", "hurdle", "takeoff", "jump_analyzer"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)
        for w in ("pixel", "area cutoff", "ffmpeg", "sqlite", "sha256"):
            assert w not in p.lower(), f"leaked task-specific term: {w!r}"
        assert "ffmpeg" not in p.lower()


    def test_observation_vs_argument_litmus_present(self):
        # chess(reward1)/openssl(reward0)/video(reward0) rerun: all three share ONE
        # root — verifying against yourself vs against the consumer's conditions. The
        # decisive tell separating pass from fail is OBSERVATION (chess measured actual
        # pixels) vs ARGUMENT (openssl "runs for me", video "should generalize"). Line85
        # now closes with a single runtime litmus unifying all three near/far forms, plus
        # one general move (reproduce the gap and observe) that needs NO task taxonomy.
        p = self._prompt()
        assert "is my evidence an OBSERVATION" in p
        assert "or an ARGUMENT" in p
        assert "name the GAP between the conditions you developed under" in p
        assert "REPRODUCE that gap and observe the result" in p
        assert "You do not need a taxonomy to know which" in p
        # must stay task-agnostic — no rerun-specific vocabulary leaked in
        import re
        for w in ("chess", "openssl", "cryptography", "hurdle", "jump"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)

    def test_grader_invoke_simulation_present(self):
        # sam-cell-seg / install-windows lesson: agent's deliverable works under
        # its own invocation but fails under the grader's (wrong arg format, wrong
        # path, wrong interface). The prompt must nudge the agent to ask "how will
        # the grader call this?" and test that form before claiming done.
        p = self._prompt()
        assert "how will a hidden grader invoke" in p.lower()
        assert "test the exact form the description implies" in p.lower()
        # When the task doesn't specify positional vs --flag, support BOTH
        assert "support both forms" in p.lower()
        # Constraint listing: re-read task, verify EACH constraint
        assert "re-read the task description" in p.lower()
        assert "list every constraint" in p.lower()

    def test_generalization_actionable_moves_present(self):
        # video-processing rerun lesson: agent matched the example's truth but its
        # algorithm HARD-CRASHED on the hidden test input. The prompt must tell the
        # agent to manufacture stress inputs and degrade to a fallback rather than
        # crashing. Task-agnostic.
        p = self._prompt()
        assert "manufacture a stress input" in p.lower()
        assert "WRONG answer" in p and "HARD CRASH" in p
        assert "degrade to a defensible fallback" in p
        import re
        for w in ("jump", "hurdle", "takeoff", "jump_analyzer"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)

    def test_verification_environment_parity_present(self):
        # Principle 2: a second form of near/far confusion is environmental —
        # verifying in an environment your own actions contaminated (installed a
        # package, set env var, created a helper) and mistaking "runs for me" for
        # "runs where consumed". Deliverable must carry deps or use only guaranteed
        # target tools; a shipped script must not import a locally pip-installed lib.
        p = self._prompt()
        assert "verification environment must be equivalent to the CONSUMER's environment" in p
        assert "must not import a library you pip-installed only locally" in p
        assert "guaranteed in the environment that will actually run this" in p

    def test_deliverable_addressing_invocation_flavor_present(self):
        # Principle 2: a DISTINCT flavor of the environmental near/far trap — not
        # what you ADDED to the environment but HOW the deliverable gets addressed
        # or invoked. The agent's interactive/login shell resolves the artifact via
        # PATH/rc/alias/cwd; the consumer invokes it bare, as a non-login
        # non-interactive subprocess running the plain command by name. "Runs when I
        # TYPE it" != "runs when a PROGRAM calls it". A found-by-name artifact must
        # live in the target's standard install location, verified by invoking it the
        # bare way the consumer will. Must stay generic (no sqlite/.bashrc/symlink).
        p = self._prompt()
        assert "HOW your deliverable gets addressed or invoked" in p
        assert 'is not "it runs when a PROGRAM calls it."' in p
        assert "standard install location the target guarantees" in p
        assert "primed to find it" in p
        # no case-by-case task specifics leaked into the static prompt
        for w in ("sqlite", ".bashrc", "profile.d", "symlink", "/usr/local/bin"):
            assert w not in p

    def test_deliverable_hygiene_present(self):
        # Principle 2 sub-discipline (b): delivery dir holds current best verified version.
        p = self._prompt()
        assert "DELIVERABLE HYGIENE" in p
        # write-through: best-so-far must be on disk, not only in memory
        assert "write-through" in p.lower() or "write it through" in p.lower()
        assert "banked" in p
        assert "NOT banked" in p
        # exact-contents: delivery path holds the named set and nothing more
        assert "EXACT-CONTENTS" in p or "exact-contents" in p.lower()
        assert "nothing more" in p.lower()
        # budget order refinement
        assert "BUDGET ORDER" in p

    def test_deliverable_exact_contents_and_shown_command_byproduct_present(self):
        # EXACT-CONTENTS: delivery path holds named set and NOTHING MORE.
        p = self._prompt()
        low = p.lower()
        assert "exact-contents" in low
        assert "nothing more" in low
        # build artifacts must be cleaned before finishing
        assert "scratch" in low or "build artifacts" in low or "artifacts" in low
        # stays generic: no nouns from the originating task
        for w in ("polyglot", "cmain", "main.py.c", "fibonacci", ".py.c"):
            assert w not in low, f"leaked task-specific term: {w!r}"

    def test_budget_order_present(self):
        # Principle 2 sub-discipline (c): land a crude complete scorable deliverable
        # at the required path FIRST, refine second.
        p = self._prompt()
        assert "BUDGET ORDER" in p
        assert "crude" in p
        assert "that gets scored" in p
        assert "partial answer that exists" in p

    def test_constraint_loyalty_present(self):
        # Principle 2 sub-discipline (a): a stated constraint is non-negotiable;
        # if unmet, report BLOCKED rather than substitute.
        p = self._prompt()
        assert "CONSTRAINT LOYALTY" in p
        assert "non-negotiable" in p
        assert "ZERO tolerance" in p
        assert "BLOCKED" in p
        assert "manufacture the appearance of satisfaction" in p.lower() or "manufacture the APPEARANCE" in p

    def test_qualifier_is_machine_checked_present(self):
        # The grader reads the FAR end — re-parses timestamps, diffs exact version
        # strings, compares byte-for-byte. If the qualifier is not literally
        # satisfied, the task FAILS.
        p = self._prompt()
        assert "byte-for-byte" in p or "byte-for-byte" in p.lower()
        assert "FAILS" in p
        assert "GIVEN" in p and "RANGE" in p
        assert "Never promote a GIVEN to a RANGE" in p

    def test_closed_exemption_list_and_unverified_attribution_present(self):
        # Task-agnostic leak guard — no task-specific vocabulary.
        p = self._prompt()
        for w in ("pyknotid", "planarity", "reconstructed_space_curve"):
            assert w not in p

    def test_minimal_verification_unit_present(self):
        # Principle 1: prove the one load-bearing assumption in a seconds-scale experiment.
        assert "MINIMAL VERIFICATION UNIT" in self._prompt()

    def test_small_sample_first_present(self):
        # Principle 1: validate method on smallest meaningful input before scaling up;
        # use the small-sample run to estimate full-task completion time.
        p = self._prompt()
        assert "SMALL-SAMPLE FIRST" in p
        assert "smallest meaningful input" in p
        assert "estimate" in p or "extrapolate" in p
        assert "scaling" in p or "proportionally" in p

    def test_preserve_irreplaceable_present(self):
        # Principle 1: snapshot/copy an irreplaceable given resource before any action
        # BackupGuard covers the backup reminder; prompt only keeps LOGICAL UNDO concept.
        p = self._prompt()
        assert "LOGICAL UNDO IS NOT BYTE RESTORE" in p

    def test_network_persistence_info_gain_present(self):
        # Environment Resilience: a single failed web_fetch/search is one failed
        # attempt on one path, NOT a verdict that the info is unavailable. Info gain
        # from external research is high-leverage; do not downgrade to a sample-tuned
        # guess at the first network error. "Could not fetch" needs several distinct
        # failed attempts as evidence, not a single error.
        p = self._prompt()
        assert "information gain" in p
        assert "not a verdict" in p.lower() or "is NOT a verdict" in p
        # must frame giving up early as abandoning the highest-leverage move
        assert "highest-leverage" in p or "high-leverage" in p.lower()
        # "could not fetch" is a claim needing multiple failed attempts as evidence
        assert "claim that needs" in p

    def test_logical_undo_not_byte_restore_present(self):
        # Principle 1 / PRESERVE: reverting a change at the logical level does NOT
        # satisfy a byte/hash immutability check (add-then-remove an internal structure
        # leaves the file's bytes changed). Only never-touching / restore-from-backup is safe.
        p = self._prompt()
        assert "LOGICAL UNDO IS NOT BYTE RESTORE" in p
        assert "hash / checksum / exact-bytes" in p
        # Task-agnostic: the add-then-remove round-trip does not reproduce original bytes,
        # stated abstractly (add an internal structure then remove it), NOT via a specific
        # store/engine. Leak guard below asserts no concrete DB engine name is used here.
        assert "adding then removing an internal structure" in p
        # The only safe strategy: never touch the original / restore from backup.
        assert "never touch the original" in p
        # domain leak guard: the PRESERVE passage must teach copy-first as a general
        # reflex, not via one engine's internals. No specific store/format name may
        # appear (sqlite, wal, sqlite_master, sha256 are all case-by-case leakage).
        for w in ("sqlite", "sqlite_master", "sha256"):
            assert w not in p.lower()
        # "wal" must be checked as a STANDALONE word (the DB write-ahead-log term),
        # not a substring — otherwise legitimate words like "wall-clock" false-trip.
        import re as _re
        assert not _re.search(r"\bwal\b", p.lower())

    def test_known_problem_class_first_move_present(self):
        # Principle 1: name the problem class and use its standard technique before enumerating.
        p = self._prompt()
        assert "KNOWN PROBLEM CLASS" in p
        assert "escape is DOWNWARD" in p or "DOWNWARD" in p

    def test_principle1_task_routing_present(self):
        # Principle 1 must route by task class so it does not force web_fetch on
        # infra/ops work (default_to_action + load_knowledge) nor treat a long
        # training run as a failure signal.
        p = self._prompt()
        assert "ROUTE THE TASK" in p
        assert "INFRASTRUCTURE / OPS" in p
        assert "UNFAMILIAR ALGORITHM" in p
        # The mandatory web_fetch checkpoint is scoped to novel-problem tasks, not ANY task.
        assert "UNFAMILIAR ALGORITHM / NOVEL PROBLEM" in p

    def test_cost_weighted_verification_present(self):
        # Principle 1: the price of feedback sets the thinking-vs-running mix.
        # As one iteration gets more expensive, invest more in reasoning and buy
        # fewer, better-targeted observations — WITHOUT weakening the rule that
        # only an observation verifies.
        p = self._prompt()
        assert "COST-WEIGHTED VERIFICATION" in p
        # It must not overturn observation>argument — theory reduces the NUMBER
        # of observations, it does not replace them.
        assert "theory still ENDS in an observation" in p
        # The three self-deception guards must all be stated, since "it's
        # expensive" is a tempting excuse to skip verifying.
        assert "must be OBSERVED" in p          # cost is measured, not asserted
        assert "not license to stop running" in p  # no-cheap-proxy != skip-run
        # Guidance, not case-by-case: no hardcoded time/size threshold leaks in.
        for w in ("minute", "hour", "gb", "second"):
            # allow the word inside larger tokens but not as a standalone threshold
            assert f" {w} " not in p.lower(), f"cost principle must stay threshold-free, found '{w}'"


# ── refresh() session_dir passthrough ───────────────────────────────────

class TestRefreshSessionDir:
    def test_session_dir_appears_in_system_prompt(self, tmp_path):
        """refresh() passes session_dir all the way into the system prompt."""
        from flagscale_agent.react.prompt_builder import PromptBuilder

        skill_mgr = MagicMock()
        skill_mgr.list_skills.return_value = []
        builder = PromptBuilder(skill_mgr)

        history = MagicMock()
        captured = {}
        history.set_system_prompt.side_effect = lambda p: captured.__setitem__("prompt", p)

        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory") as mock_mem_cls:
            mock_mem_cls.return_value.list_entries.return_value = []
            builder.refresh(
                history=history,
                active_skill_content={},
                shared_storage_paths=[],
                session_dir="/fake/session/xyz",
            )

        prompt = captured.get("prompt", "")
        assert "/fake/session/xyz" in prompt
        assert "conversation_full.json" in prompt
        assert "conversation.json" in prompt


# ── Analyze-don't-just-use principle (20260918) ──────────────────────────

class TestAnalyzeDontJustUse:
    """The 'ANALYZE, DON'T JUST USE' principle must not be silently removed.

    Guards the two landing sites: the dedicated paragraph in PRINCIPLE 1
    (after FIRST-ACTION RESEARCH REFLEX) and the depth extension on the
    'Read existing code before writing new code' bullet in Rules/DO.
    """

    def _prompt(self):
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        return SYSTEM_PROMPT_STATIC

    def test_analyze_principle_present(self):
        # The dedicated P1 paragraph and its mechanism-level demand.
        p = self._prompt()
        assert "ANALYZE, DON'T JUST USE" in p
        assert "SUBJECT of study, not a tool to consume" in p
        assert "MECHANISM level" in p

    def test_analyze_resolves_standard_method_tension(self):
        # Dissolves the mis-reading of P1: standard method != current code.
        p = self._prompt()
        assert "standard APPROACH, not whatever the code currently does" in p

    def test_read_existing_code_has_mechanism_depth(self):
        # The Rules/DO bullet must carry the optimization-depth extension.
        p = self._prompt()
        assert "Read existing code before writing new code" in p
        assert "read at the *mechanism* level" in p


    # ── Information Retrieval: check memory for the location before searching ──

    def test_pre_search_memory_location_check_present(self):
        # The retrieval checklist must lead with: consult memory for the named
        # file BEFORE scanning any tree. Regression for the "looked up a symbol
        # that was already recorded" waste.
        p = self._prompt()
        assert "check whether memory already knows the location" in p

    def test_pre_search_memory_uses_memory_list_and_named_file(self):
        p = self._prompt()
        i = p.index("## Information Retrieval")
        j = p.index("## Pitfall Recall")
        blk = p[i:j]
        assert "memory_list(keyword='<term>')" in blk
        assert "read that\nfile directly, instead of scanning a tree" in blk

    def test_pre_search_memory_names_heavy_subtrees(self):
        # Names the concrete heavy-tree markers so the guidance is operational,
        # not abstract ("avoid broad search").
        p = self._prompt()
        i = p.index("## Information Retrieval")
        j = p.index("## Pitfall Recall")
        blk = p[i:j]
        for marker in ("site-packages", "node_modules", "logs", "checkpoints"):
            assert marker in blk, marker
