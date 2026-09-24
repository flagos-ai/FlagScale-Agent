# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
"""Unit tests for the knowledge system."""

import os
import tempfile
import shutil

import pytest
import yaml


@pytest.fixture
def knowledge_dir():
    """Create a temporary knowledge directory with test data."""
    tmpdir = tempfile.mkdtemp()
    docs_dir = os.path.join(tmpdir, "docs", "test_repo")
    indexes_dir = os.path.join(tmpdir, "indexes")
    os.makedirs(docs_dir)
    os.makedirs(indexes_dir)

    # Create a test markdown doc
    doc_content = """# Test Chapter 1

## 1.1 Introduction

Some content here about testing.

### 1.1.1 Subtopic

More details.

## 1.2 Core Algorithm

Algorithm description with code.
"""
    with open(os.path.join(docs_dir, "01_test.md"), "w") as f:
        f.write(doc_content)

    # Create config
    config = {
        "know-test-group": {
            "description": "Test knowledge group for unit testing",
            "docs": ["test_repo/01_test.md"],
        }
    }
    config_path = os.path.join(tmpdir, "knowledge_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    yield tmpdir
    shutil.rmtree(tmpdir)


class TestKnowledgeManager:
    def test_init_default(self):
        """Test default initialization uses package directory."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager()
        assert km.root.exists()
        # 17 base groups + know-moe-training (MoE pretraining perf survey)
        # + know-linear-attention (FLA source analysis)
        assert len(km.available_groups) == 19

    def test_init_custom_dir(self, knowledge_dir):
        """Test initialization with custom directory."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager(knowledge_dir)
        assert km.root == __import__("pathlib").Path(knowledge_dir)

    def test_list_groups(self, knowledge_dir):
        """Test listing knowledge groups."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager(knowledge_dir)
        groups = km.list_groups()
        assert len(groups) == 1
        assert groups[0]["name"] == "know-test-group"
        assert groups[0]["doc_count"] == 1
        assert "unit testing" in groups[0]["description"]

    def test_get_doc_content(self, knowledge_dir):
        """Test reading doc content with line ranges."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager(knowledge_dir)

        # Full content
        content = km.get_doc_content("test_repo/01_test.md")
        assert content is not None
        assert "# Test Chapter 1" in content

        # Partial content
        content = km.get_doc_content("test_repo/01_test.md", 3, 5)
        assert content is not None
        lines = content.splitlines()
        assert len(lines) == 3

        # Non-existent file
        content = km.get_doc_content("nonexistent.md")
        assert content is None

    def test_get_group_docs(self, knowledge_dir):
        """Test getting doc list for a group."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager(knowledge_dir)
        docs = km.get_group_docs("know-test-group")
        assert docs == ["test_repo/01_test.md"]

        docs = km.get_group_docs("nonexistent-group")
        assert docs == []

    def test_available_groups(self, knowledge_dir):
        """Test available_groups property."""
        from flagscale_agent.knowledge import KnowledgeManager
        km = KnowledgeManager(knowledge_dir)
        assert "know-test-group" in km.available_groups


class TestGenerateIndex:
    def test_extract_sections(self, knowledge_dir):
        """Test section extraction from markdown."""
        import sys
        sys.path.insert(0, knowledge_dir)
        from flagscale_agent.knowledge.generate_index import extract_sections

        doc_path = os.path.join(knowledge_dir, "docs", "test_repo", "01_test.md")
        sections = extract_sections(doc_path)
        assert len(sections) >= 3
        assert sections[0]["title"] == "Test Chapter 1"
        assert sections[0]["level"] == 1
        assert sections[0]["line"] == 1

    def test_generate_index_for_group(self, knowledge_dir):
        """Test index generation for a group."""
        from flagscale_agent.knowledge.generate_index import generate_index_for_group

        config = {
            "description": "Test group",
            "docs": ["test_repo/01_test.md"],
        }
        docs_root = os.path.join(knowledge_dir, "docs")
        result = generate_index_for_group("know-test", config, docs_root)
        assert "know-test" in result
        assert "01_test.md" in result
        assert "L1:" in result


class TestLoadKnowledgeTool:
    def test_execute_list(self):
        """Test listing available groups."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager()
        tool = LoadKnowledgeTool(km)
        result = tool.execute(name="list")
        assert "know-megatron-parallel" in result
        assert "know-flash-attn" in result

    def test_execute_index(self):
        """Test loading an index."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager()
        tool = LoadKnowledgeTool(km)
        result = tool.execute(name="know-nccl-core", index_only=True)
        assert "nccl/" in result
        assert "L" in result  # Line numbers

    def test_execute_unknown_group(self):
        """Test error on unknown group."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager()
        tool = LoadKnowledgeTool(km)
        result = tool.execute(name="nonexistent-group")
        assert "Unknown group" in result

    def test_tool_attributes(self):
        """Test tool has required attributes."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager()
        tool = LoadKnowledgeTool(km)
        assert tool.name == "load_knowledge"
        assert "name" in tool.parameters["properties"]
        assert "name" in tool.parameters["required"]
        # New precise-read params exposed
        assert "doc" in tool.parameters["properties"]
        assert "start_line" in tool.parameters["properties"]
        assert "end_line" in tool.parameters["properties"]

    def test_execute_read_doc(self, knowledge_dir):
        """Test reading a specific doc via the tool using index-style path."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager(knowledge_dir)
        tool = LoadKnowledgeTool(km)

        # Full doc read
        result = tool.execute(name="know-test-group", doc="test_repo/01_test.md")
        assert "test_repo/01_test.md" in result
        assert "# Test Chapter 1" in result

    def test_execute_read_doc_line_range(self, knowledge_dir):
        """Test reading a line range via the tool."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager(knowledge_dir)
        tool = LoadKnowledgeTool(km)

        result = tool.execute(
            name="know-test-group", doc="test_repo/01_test.md",
            start_line=3, end_line=5,
        )
        assert "lines 3-5" in result
        body = result.split("===\n", 1)[1]
        assert len(body.splitlines()) == 3

    def test_execute_read_doc_not_in_group(self, knowledge_dir):
        """Test clear error when doc path is not part of the group."""
        from flagscale_agent.knowledge import KnowledgeManager
        from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool

        km = KnowledgeManager(knowledge_dir)
        tool = LoadKnowledgeTool(km)

        result = tool.execute(name="know-test-group", doc="wrong/path.md")
        assert "not in group" in result
        assert "test_repo/01_test.md" in result  # lists available docs
