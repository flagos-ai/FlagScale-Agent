"""Tests for SkillManager."""

import os

import pytest

from flagscale_agent.react.skills import SkillManager


@pytest.fixture
def skill_dirs(tmp_path):
    """Create two skill directories with test skills."""
    dir1 = tmp_path / "builtin"
    dir2 = tmp_path / "user"
    dir1.mkdir()
    dir2.mkdir()

    # Skill in builtin dir
    s1 = dir1 / "my_skill"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\nname: my_skill\ndescription: A test skill\n---\nDo something useful."
    )

    # Skill in user dir (overrides builtin)
    s2 = dir2 / "my_skill"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\nname: my_skill\ndescription: User override\n---\nUser version."
    )

    # Another skill only in user dir
    s3 = dir2 / "extra"
    s3.mkdir()
    (s3 / "SKILL.md").write_text(
        "---\nname: extra\ndescription: Extra skill\n---\nExtra content."
    )

    return [str(dir1), str(dir2)]


class TestSkillManager:
    def test_list_skills(self, skill_dirs):
        mgr = SkillManager(skill_dirs)
        skills = mgr.list_skills()
        names = {s["name"] for s in skills}
        assert "my_skill" in names
        assert "extra" in names

    def test_load_priority(self, skill_dirs):
        """Later directories take priority."""
        mgr = SkillManager(skill_dirs)
        content = mgr.load("my_skill")
        assert "User version" in content

    def test_load_missing(self, skill_dirs):
        mgr = SkillManager(skill_dirs)
        with pytest.raises(FileNotFoundError):
            mgr.load("nonexistent")

    def test_empty_dirs(self):
        mgr = SkillManager(["/nonexistent/path"])
        assert mgr.list_skills() == []

    def test_parse_frontmatter_no_yaml(self):
        meta, body = SkillManager._parse_frontmatter("Just plain text")
        assert meta == {}
        assert body == "Just plain text"

    def test_parse_frontmatter_valid(self):
        content = "---\nname: test\ndescription: desc\n---\nBody here."
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta["name"] == "test"
        assert body == "Body here."

    def test_parse_frontmatter_bad_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\nBody."
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {}
        assert body == "Body."


@pytest.fixture
def param_skill_dir(tmp_path):
    """Create a skill with parameters defined in frontmatter."""
    d = tmp_path / "skills"
    d.mkdir()
    s = d / "deploy"
    s.mkdir()
    (s / "SKILL.md").write_text(
        "---\n"
        "name: deploy\n"
        "description: Deploy a model\n"
        "parameters:\n"
        "  - name: model_name\n"
        "    description: Model to deploy\n"
        "    required: false\n"
        "    default: qwen3\n"
        "  - name: replicas\n"
        "    description: Number of replicas\n"
        "    required: false\n"
        "---\n"
        "Deploy {model_name} with {replicas} replicas."
    )
    return [str(d)]


class TestParameterizedSkills:
    def test_load_with_params(self, param_skill_dir):
        mgr = SkillManager(param_skill_dir)
        content = mgr.load("deploy", model_name="llama3", replicas="4")
        assert "llama3" in content
        assert "4 replicas" in content

    def test_load_with_defaults(self, param_skill_dir):
        mgr = SkillManager(param_skill_dir)
        content = mgr.load("deploy", replicas="2")
        assert "qwen3" in content
        assert "2 replicas" in content

    def test_load_no_params_keeps_placeholders(self, param_skill_dir):
        mgr = SkillManager(param_skill_dir)
        content = mgr.load("deploy")
        assert "qwen3" in content
        assert "{replicas}" in content

    def test_list_includes_parameters(self, param_skill_dir):
        mgr = SkillManager(param_skill_dir)
        skills = mgr.list_skills()
        assert len(skills) == 1
        assert len(skills[0]["parameters"]) == 2
        assert skills[0]["parameters"][0]["name"] == "model_name"

    def test_extra_params_ignored(self, param_skill_dir):
        mgr = SkillManager(param_skill_dir)
        content = mgr.load("deploy", model_name="llama3", replicas="4", unknown="x")
        assert "llama3" in content
        assert "{unknown}" not in content
