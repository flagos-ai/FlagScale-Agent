# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the cross-session improvement-proposal registry."""

import os
import shutil
import tempfile

import pytest

from flagscale_agent.react.proposals import (
    ProposalRegistry,
    VALID_STATUSES,
    OPEN_STATUSES,
    TERMINAL_STATUSES,
)


@pytest.fixture
def reg():
    d = tempfile.mkdtemp()
    yield ProposalRegistry(d)
    shutil.rmtree(d, ignore_errors=True)


class TestAdd:
    def test_add_defaults_to_proposed(self, reg):
        e = reg.add("Add X guard", container="agent-code", session_id="s1",
                    topic="x")
        assert e["status"] == "proposed"
        assert e["id"].startswith("prop_")
        assert e["description"] == "Add X guard"
        assert e["created_session"] == "s1"
        assert e["status_history"][0]["status"] == "proposed"

    def test_add_persists_and_reloads(self, reg):
        e = reg.add("Add Y", session_id="s1")
        # A fresh registry instance over the same dir sees it (cross-session).
        reg2 = ProposalRegistry(reg._dir)
        assert reg2.get(e["id"])["description"] == "Add Y"

    def test_add_invalid_status(self, reg):
        with pytest.raises(ValueError):
            reg.add("bad", status="nonsense")


class TestStatus:
    def test_set_status_appends_history(self, reg):
        e = reg.add("Add Z", session_id="s1")
        reg.set_status(e["id"], "approved", session_id="s2", note="ok")
        got = reg.get(e["id"])
        assert got["status"] == "approved"
        assert [h["status"] for h in got["status_history"]] == ["proposed", "approved"]

    def test_set_status_unknown_id(self, reg):
        with pytest.raises(ValueError):
            reg.set_status("prop_deadbeef", "done")

    def test_set_status_invalid_status(self, reg):
        e = reg.add("A", session_id="s1")
        with pytest.raises(ValueError):
            reg.set_status(e["id"], "bogus")

    def test_invalid_id_rejected_for_get(self, reg):
        # Path-traversal-ish ids must not resolve.
        assert reg.get("../../etc/passwd") is None
        assert reg.get("not_a_prop_id") is None


class TestListing:
    def test_open_vs_terminal(self, reg):
        a = reg.add("A", session_id="s1")
        b = reg.add("B", session_id="s1")
        c = reg.add("C", session_id="s1")
        reg.set_status(b["id"], "done")
        reg.set_status(c["id"], "rejected")
        open_ids = {e["id"] for e in reg.list_open()}
        assert open_ids == {a["id"]}
        assert {e["id"] for e in reg.list_status("done")} == {b["id"]}

    def test_open_statuses_constant(self):
        assert set(OPEN_STATUSES).isdisjoint(TERMINAL_STATUSES)
        assert set(OPEN_STATUSES) | set(TERMINAL_STATUSES) == set(VALID_STATUSES)

    def test_render_open_empty(self, reg):
        assert reg.render_open() == ""

    def test_render_open_lists_open_only(self, reg):
        a = reg.add("A", container="skill", session_id="s1")
        reg.add("B", session_id="s1")  # no container
        text = reg.render_open()
        assert a["id"] in text
        assert "skills" in text or "[skill]" in text
        assert text.startswith("Open improvement proposals")


class TestStaleness:
    def _age(self, reg, pid, days):
        """Rewrite created_at to `days` in the past (write straight to disk)."""
        import yaml
        from datetime import datetime, timezone, timedelta
        p = reg._path(pid)
        with open(p, encoding="utf-8") as f:
            e = yaml.safe_load(f)
        old = datetime.now(timezone.utc) - timedelta(days=days)
        e["created_at"] = old.isoformat(timespec="seconds")
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(e, f)

    def test_fresh_proposed_not_stale(self, reg):
        e = reg.add("A", session_id="s1")
        assert reg.is_stale(reg.get(e["id"])) is False

    def test_old_proposed_is_stale(self, reg):
        e = reg.add("A", session_id="s1")
        self._age(reg, e["id"], 8)  # > 7 days
        assert reg.is_stale(reg.get(e["id"])) is True

    def test_boundary_exactly_seven_days_not_stale(self, reg):
        e = reg.add("A", session_id="s1")
        self._age(reg, e["id"], 6)  # < 7 days
        assert reg.is_stale(reg.get(e["id"])) is False

    def test_approved_never_stale(self, reg):
        e = reg.add("A", session_id="s1")
        reg.set_status(e["id"], "approved")
        self._age(reg, e["id"], 30)
        assert reg.is_stale(reg.get(e["id"])) is False

    def test_render_tags_stale(self, reg):
        e = reg.add("Old", session_id="s1")
        self._age(reg, e["id"], 9)
        text = reg.render_open()
        assert "STALE" in text and e["id"] in text

    def test_render_no_stale_tag_when_fresh(self, reg):
        reg.add("Fresh", session_id="s1")
        assert "STALE" not in reg.render_open()

    def test_stale_after_days_constant_is_seven(self):
        from flagscale_agent.react.proposals import STALE_AFTER_DAYS
        assert STALE_AFTER_DAYS == 7

    def test_open_count(self, reg):
        assert reg.open_count() == 0
        a = reg.add("A", session_id="s1")
        reg.add("B", session_id="s1")
        assert reg.open_count() == 2
        reg.set_status(a["id"], "done")
        assert reg.open_count() == 1

