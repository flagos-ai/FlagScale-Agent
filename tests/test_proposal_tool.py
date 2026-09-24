# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the proposal tool and its wrap-up integration."""

import shutil
import tempfile

import pytest

from flagscale_agent.react.proposals import ProposalRegistry
from flagscale_agent.react.tools.proposal import ProposalTool
from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _TEXT_COMPLETE_HYGIENE,
)


@pytest.fixture
def env():
    d = tempfile.mkdtemp()
    reg = ProposalRegistry(d)
    yield reg, ProposalTool(reg, session_id="s1")
    shutil.rmtree(d, ignore_errors=True)


class TestProposalTool:
    def test_add(self, env):
        reg, tool = env
        out = tool.execute(action="add", description="Add X guard",
                           container="agent-code", topic="x")
        assert "prop_" in out
        assert "proposed" in out
        assert len(reg.list_open()) == 1

    def test_add_requires_description(self, env):
        reg, tool = env
        assert tool.execute(action="add").startswith("ERROR")

    def test_add_rejects_bad_container(self, env):
        reg, tool = env
        out = tool.execute(action="add", description="d", container="bogus")
        assert out.startswith("ERROR")

    def test_list_empty(self, env):
        reg, tool = env
        assert tool.execute(action="list") == "No open proposals."

    def test_list_shows_open(self, env):
        reg, tool = env
        tool.execute(action="add", description="Add Y")
        out = tool.execute(action="list")
        assert "Add Y" in out
        assert "1 open proposal" in out

    def test_update_status(self, env):
        reg, tool = env
        tool.execute(action="add", description="Add Z")
        pid = reg.list_open()[0]["id"]
        out = tool.execute(action="update", proposal_id=pid, status="done",
                           note="shipped")
        assert "done" in out
        assert reg.list_open() == []

    def test_update_requires_valid_status(self, env):
        reg, tool = env
        tool.execute(action="add", description="A")
        pid = reg.list_open()[0]["id"]
        assert tool.execute(action="update", proposal_id=pid,
                            status="nope").startswith("ERROR")

    def test_unknown_action(self, env):
        reg, tool = env
        assert tool.execute(action="frobnicate").startswith("ERROR")


class TestBatchUpdate:
    def test_batch_updates_multiple(self, env):
        reg, tool = env
        tool.execute(action="add", description="A")
        tool.execute(action="add", description="B")
        tool.execute(action="add", description="C")
        ids = [e["id"] for e in reg.list_open()]
        out = tool.execute(action="update", updates=[
            {"proposal_id": ids[0], "status": "done", "note": "shipped"},
            {"proposal_id": ids[1], "status": "rejected"},
            {"proposal_id": ids[2], "status": "approved"},
        ])
        assert "3/3 applied" in out
        assert reg.list_open() == [] or all(
            e["status"] == "approved" for e in reg.list_open()
        )
        assert reg.get(ids[0])["status"] == "done"
        assert reg.get(ids[1])["status"] == "rejected"

    def test_batch_partial_bad_item_does_not_abort(self, env):
        reg, tool = env
        tool.execute(action="add", description="A")
        good = reg.list_open()[0]["id"]
        out = tool.execute(action="update", updates=[
            {"proposal_id": good, "status": "done"},
            {"proposal_id": "prop_nope", "status": "done"},
            {"proposal_id": good, "status": "bogus"},
        ])
        assert "1/3 applied" in out
        assert reg.get(good)["status"] == "done"

    def test_batch_empty_or_none_falls_through_to_single(self, env):
        reg, tool = env
        tool.execute(action="add", description="A")
        pid = reg.list_open()[0]["id"]
        # updates=None/empty => single-id path (which still requires proposal_id)
        assert tool.execute(action="update").startswith("ERROR")
        assert "done" in tool.execute(
            action="update", proposal_id=pid, status="done"
        )


class TestWrapupInjection:
    def test_no_registry_is_unchanged(self):
        g = VerificationGuard(plan=None, proposals=None)
        assert g._text_complete_hygiene_message() == _TEXT_COMPLETE_HYGIENE

    def test_empty_registry_is_unchanged(self):
        d = tempfile.mkdtemp()
        try:
            g = VerificationGuard(plan=None, proposals=ProposalRegistry(d))
            assert g._text_complete_hygiene_message() == _TEXT_COMPLETE_HYGIENE
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_open_proposal_injected(self):
        d = tempfile.mkdtemp()
        try:
            reg = ProposalRegistry(d)
            e = reg.add("Add a timer guard", container="agent-code", session_id="prev")
            g = VerificationGuard(plan=None, proposals=reg)
            msg = g._text_complete_hygiene_message()
            assert "Open proposals already on file" in msg
            assert e["id"] in msg
            assert "Add a timer guard" in msg
            # The original template's re-issue instruction must remain last.
            assert msg.rstrip().endswith("This gate fires once.")
            # The injected block sits inside the message, before the final line.
            assert msg.index("Open proposals already on file") < msg.index(
                "Re-issue [TASK_COMPLETE]"
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_closed_proposal_not_injected(self):
        d = tempfile.mkdtemp()
        try:
            reg = ProposalRegistry(d)
            e = reg.add("Done thing", session_id="prev")
            reg.set_status(e["id"], "rejected")
            g = VerificationGuard(plan=None, proposals=reg)
            assert g._text_complete_hygiene_message() == _TEXT_COMPLETE_HYGIENE
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_template_mentions_registry_and_status(self):
        # The static template must teach registration + reconciliation.
        assert "proposal tool" in _TEXT_COMPLETE_HYGIENE
        assert "RECONCILE THE REGISTRY" in _TEXT_COMPLETE_HYGIENE
