"""Tests for the background-shell and time-budget guidance added to the
system prompt (prompt.py SYSTEM_PROMPT_STATIC)."""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

SECTION_ORDER = [ln for ln in SYSTEM_PROMPT_STATIC.splitlines() if ln.startswith("## ")]


class TestPromptSectionStructure:
    """Structure regression (user review 20260914): the Dashboard section was
    demoted to a '###' under Tool Guide, splitting the tool list; the ORDER
    bullet was 4-space indented, so CommonMark glued it into the OVERLAP item
    as a continuation line."""

    def test_dashboard_is_top_level_section(self):
        assert "## Dashboard — Your Instrument Panel" in SECTION_ORDER

    def test_dashboard_not_nested_under_tool_guide(self):
        assert "### Dashboard" not in SYSTEM_PROMPT_STATIC

    def test_tool_guide_starts_with_tool_mapping(self):
        after = SYSTEM_PROMPT_STATIC.split("## Tool Guide", 1)[1]
        body = after.split("\n\n", 1)[1].split("## ", 1)[0]
        assert body.lstrip().startswith("- Read/edit files →")

    def test_order_bullet_is_sibling_tactic(self):
        assert "\n- ORDER THE INDEPENDENT STEPS:" in SYSTEM_PROMPT_STATIC

    def test_no_indented_tactic_lines(self):
        for ln in SYSTEM_PROMPT_STATIC.splitlines():
            assert not ln.startswith("    ORDER"), ln[:40]
            assert not ln.startswith("    OVERLAP"), ln[:40]
            assert not ln.startswith("    FAN OUT"), ln[:40]


class TestBackgroundToolGuidance:
    def test_tool_guide_mentions_background_launch(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "background=true" in low
        assert "shell_jobs" in low

    def test_no_stale_timeout_escalation_numbers(self):
        # 20260914 压缩重写: 旧文案的 300s→900s 超时升级示例已过时
        # (guard 实际硬门是 60s), 指针指向 guard 常量而非复述数字。
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "300s" not in low
        assert "900s" not in low
        assert "shelljobswaitguard" in low

    def test_background_bullets_stay_compressed(self):
        # 防回胖: 两条 bullet 合计不得超过 2500 字符
        # (20260914 压缩后实测 1872, 增补元认知枚举后 ~2212)。
        lines = SYSTEM_PROMPT_STATIC.splitlines()
        bullets = [ln for ln in lines if ln.startswith("- Long-running commands")
                   or ln.startswith("- Backgrounding only pays")]
        assert len(bullets) == 2
        assert sum(len(b) for b in bullets) < 2500

    def test_wait_time_covers_metacognitive_work(self):
        # 用户 doctrine (20260914): 等待期的 useful work 不止准备下游步骤,
        # 还包括复盘之前/总结现在(memory)/规划未来(计划)/重审当下(前提)。
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "recap what the last steps taught you" in low
        assert "write memory" in low
        assert "update the plan" in low
        assert "sketch the post-join steps" in low
        assert "re-audit the current approach's premises" in low

    def test_tool_guide_lists_poll_and_wait(self):
        # The agent must know how to check a backgrounded job later.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert 'action="poll"' in low
        assert 'action="wait"' in low

    def test_tool_guide_covers_long_running_classes(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        # Names the kinds of commands that should be backgrounded.
        assert "training" in low
        assert "download" in low or "build" in low

    def test_mentions_auto_detach_by_monitor(self):
        # The prompt should tell the agent the monitor may auto-detach a
        # healthy-but-long command and hand back a job handle.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "detach" in low

    def test_warns_against_wait_spin_loop(self):
        # Task 2 (prompt-only): backgrounding then repeatedly waiting with no
        # other work is synchronous blocking in disguise.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "synchronous blocking" in low
        # Must frame the win as doing OTHER work while the job runs.
        assert "other useful work" in low or "other work" in low

    def test_says_background_jobs_stay_monitored(self):
        # Task 1 surfaced in the prompt: backgrounded/auto-detached jobs stay
        # health-monitored and can be terminated if they later hang.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "health-monitored" in low or "health monitored" in low

    def test_discourages_long_blind_wait_prefers_short(self):
        # Regression (insight/tbench/caffe_timeout_rootcause_budget_exhaustion,
        # fasttext trace jobs/2026-09-04__19-34-14): the agent escalated blind
        # blocking waits (300s->900s->900s) while dependent work was un-prepared,
        # then discovered a fatal output-size cap AFTER the run. The prompt must:
        low = SYSTEM_PROMPT_STATIC.lower()
        # (a) prefer a SHORT bounded wait over one maximal block
        assert "short bounded" in low
        # (b) frame "nothing else to do" as a HIGH bar, not a default
        assert "high bar" in low
        # (c) name preparing dependent/downstream steps as the wait's best use
        assert "prepare" in low or "downstream" in low or "dependent step" in low
        # (d) warn that escalating the timeout run-over-run surrenders budget
        assert "escalate" in low or "walk away" in low or "leaving the room" in low


class TestTimeBudgetGuidance:
    def test_has_time_budget_section(self):
        assert "Time Budget" in SYSTEM_PROMPT_STATIC

    def test_budget_section_says_plan_whole_task(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        # Front-load expensive steps / allocate the shared clock up front.
        assert "front-load" in low or "front load" in low
        assert "budget" in low

    def test_budget_section_warns_against_blocking(self):
        # Counter-example: sit blocked on a long run doing nothing.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "block" in low
        assert "background" in low

    def test_budget_order_preserved(self):
        # BUDGET ORDER guidance must still be present.
        assert "BUDGET ORDER" in SYSTEM_PROMPT_STATIC

    def test_conveys_urgency_from_the_start(self):
        # The section must make the agent tense from the first tool call, not
        # invite a leisurely pace.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "scarce" in low or "urgency" in low or "running out" in low

    def test_no_concrete_default_wall_leaks(self):
        # The 24h/86400 code-uniformity default must NEVER surface in the prompt
        # — the agent must not be told it has a day-scale allowance.
        low = SYSTEM_PROMPT_STATIC.lower()
        for leaked in ("24h", "86400", "24 h", "one day", "a day", "24-hour", "24 hour"):
            assert leaked not in low, f"forbidden time reference leaked: {leaked}"

    def test_mentions_timebudget_advisory(self):
        # The prompt should prime the agent to act on [TimeBudget] advisories.
        assert "[TimeBudget]" in SYSTEM_PROMPT_STATIC

    def test_background_must_be_paired_with_other_work(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        # Backgrounding is only a win if paired with real parallel work.
        assert "in parallel" in low or "other real work" in low

    def test_overlaps_independent_expensive_steps(self):
        # Regression (insight/tbench/caffe_timeout_rootcause_budget_exhaustion):
        # a build and an independent dataset download were run serially, wasting
        # budget. The prompt must teach mapping the dependency graph and running
        # dependency-free expensive steps CONCURRENTLY (build + download + install),
        # not queued one after another.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "overlap independent expensive steps" in low
        # concrete independent steps named
        assert "download" in low
        assert "install" in low
        # the dependency-graph / concurrency framing
        assert "dependency" in low or "depend" in low
        assert "concurrent" in low
        # the wall-clock model: longest branch, not the sum
        assert "longest branch" in low
        assert "not the sum" in low
        # distinguishes from cheap same-turn batching
        assert "batch independent tool calls" in low
        # generic, no task leaking
        assert "caffe" not in low
        assert "cifar" not in low

    def test_fans_out_independent_experiments(self):
        # Regression (insight/tbench/fasttext_blind_wait_discipline): a search
        # that retrained variants ONE AT A TIME wasted budget. The prompt must
        # teach fanning out many same-kind independent runs (sweep/seeds/configs)
        # as concurrent background jobs — a distinct axis from step-overlap.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "fan out independent experiments" in low
        # names the same-kind-many-runs case (sweep / configs / seeds / trials)
        assert "sweep" in low or "trials" in low or "seeds" in low
        # cultivates the concurrency TRADE-OFF rather than naming specific
        # hardware: more concurrency slows each trial, so the goal is the degree
        # that minimizes TOTAL wall-clock, not the maximum you can fit.
        assert "trade-off" in low
        assert "total wall-clock" in low
        assert "minimizes total wall-clock" in low or "best total time" in low
        # optimum is often in the middle, and measure a small batch when unsure
        assert "concurrency degree" in low
        assert "measure one small concurrent batch" in low or "measure" in low
        # the two guards: distinct output paths + write-through-bank the winner
        assert "distinct path" in low or "clobber" in low
        assert "write-through" in low
        # serial vs concurrent cost framing
        assert "serial" in low or "one at a time" in low
        # generic, no task/framework leaking
        assert "fasttext" not in low
        assert "caffe" not in low

    def test_orders_independent_steps_longest_riskiest_first(self):
        # Regression (insight/tbench/caffe_critical_path_and_antifabrication):
        # concurrency alone was not enough — the agent ran apt+build first and
        # only reached the proxy-blocked dataset download 8 minutes in, leaving no
        # budget to find a working mirror. The prompt must teach launching the
        # LONGEST and RISKIEST independent step (a network/proxy download) FIRST,
        # so a block surfaces while budget remains to route around it.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "order the independent steps" in low
        # ranks by duration AND failure risk
        assert "failure risk" in low or "riskiest" in low
        assert "duration" in low
        # launch longest/riskiest first, at t=0, not discovery order
        assert "longest" in low and "riskiest" in low
        assert "discovery order" in low or "t=0" in low
        # the payoff: block surfaces while budget remains to route around
        assert "budget still remains" in low or "budget remains" in low
        # generic, no task leaking
        assert "caffe" not in low
        assert "cifar" not in low

    def test_forbids_fabricating_required_input(self):
        # Regression (insight/tbench/caffe_critical_path_and_antifabrication):
        # when the dataset download failed, the agent SYNTHESIZED fake data
        # (random arrays / sine-pattern) to make the pipeline run, silently
        # swapping an accuracy-gated task for an unmeasurable one. The prompt
        # must classify fabricating a required input as manufacturing the
        # appearance of success — the same violation as empty files.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "fabricating a required input" in low
        # names the synthesize-a-stand-in anti-pattern
        assert "synthesizing a stand-in" in low or "random arrays" in low
        # ties it to the swap: accuracy-gated → unmeasurable, scores zero
        assert "unmeasurable" in low
        # the correct escape: exhaust sources upward, else report BLOCKED
        assert "escape upward" in low or "hf_endpoint" in low
        assert "report blocked" in low or "blocked" in low
        # generic, no task leaking
        assert "caffe" not in low
        assert "cifar" not in low


class TestExpectationViolationAttribution:
    """User doctrine (202609, aligned with GLM RSI blog): an action agent must
    ANALYZE results, not just take them — a result that violates expectation
    obliges the agent to name a mechanism + a controlled comparison, never to
    stop at 'worse' / 'inaccurate'."""

    OBS_LINE = None  # filled in classmethod below

    @classmethod
    def obs_line(cls):
        if cls.OBS_LINE is None:
            lines = [ln for ln in SYSTEM_PROMPT_STATIC.splitlines()
                     if ln.startswith("OBSERVATION SEMANTICS")]
            assert len(lines) == 1, "OBSERVATION SEMANTICS must stay a single line"
            cls.OBS_LINE = lines[0]
        return cls.OBS_LINE

    def test_rule_present_in_observation_semantics_paragraph(self):
        line = self.obs_line()
        assert "treat it as a question, not a verdict" in line
        assert "state the mechanism you propose" in line
        assert "controlled comparison that would test it" in line

    def test_rules_out_verdict_only_language(self):
        line = self.obs_line()
        assert '"worse" or "inaccurate" stand in for a named cause' in line

    def test_no_new_braces_introduced(self):
        # SYSTEM_PROMPT_STATIC is .format()-rendered by the builder with the
        # fixed kwargs cwd/knowledge/skills/tools. Any literal { or } added by
        # an edit raises KeyError/IndexError at prompt-build time (the {N}
        # regression, 20260914).
        import re
        line = self.obs_line()
        assert not re.search(r"[{}]", line), line[:80]
        # and the builder-level invariant: the full static prompt still has
        # exactly the four pre-existing placeholders.
        import re as _re
        assert set(_re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC)) == {
            "cwd", "knowledge", "skills", "tools"}
