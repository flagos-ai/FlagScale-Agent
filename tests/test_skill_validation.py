"""Validate all real SKILL.md files parse correctly and are loadable."""

import os

import pytest

from flagscale_agent.react.skills import SkillManager

SKILLS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "flagscale_agent", "skills"
)
SKILLS_DIR = os.path.normpath(SKILLS_DIR)

EXPECTED_SKILLS = {
    "train-env-setup",
    "topo-detect",
    "train-model-porter",
    "train-data-prep",
    "train-config",
    "train-run",
    "train-monitor",
    "train-reproduce",
    "train-precision-alignment",
    "ops-discipline",
    "workspace-layout",
    "train-parallel-strategy",
    "debug-strategy",
}

REQUIRED_FRONTMATTER_FIELDS = {"name", "description", "keywords"}


@pytest.fixture
def skill_manager():
    return SkillManager([SKILLS_DIR])


class TestSkillDirectoryStructure:
    def test_skills_dir_exists(self):
        assert os.path.isdir(SKILLS_DIR), f"Skills directory not found: {SKILLS_DIR}"

    def test_all_expected_skills_present(self):
        found = set()
        for entry in os.listdir(SKILLS_DIR):
            skill_file = os.path.join(SKILLS_DIR, entry, "SKILL.md")
            if os.path.isfile(skill_file):
                found.add(entry)
        missing = EXPECTED_SKILLS - found
        assert not missing, f"Missing skill directories: {missing}"

    def test_no_unexpected_skill_dirs(self):
        found = set()
        for entry in os.listdir(SKILLS_DIR):
            skill_file = os.path.join(SKILLS_DIR, entry, "SKILL.md")
            if os.path.isfile(skill_file):
                found.add(entry)
        unexpected = found - EXPECTED_SKILLS
        assert not unexpected, f"Unexpected skill directories: {unexpected}"


class TestSkillFrontmatter:
    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_frontmatter_has_required_fields(self, skill_name):
        skill_file = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        meta, body = SkillManager._parse_frontmatter(content)
        for field in REQUIRED_FRONTMATTER_FIELDS:
            assert field in meta, f"Skill '{skill_name}' missing frontmatter field: {field}"

    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_name_matches_directory(self, skill_name):
        skill_file = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        meta, _ = SkillManager._parse_frontmatter(content)
        assert meta.get("name") == skill_name, (
            f"Skill directory '{skill_name}' has mismatched name: '{meta.get('name')}'"
        )

    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_keywords_is_nonempty_list(self, skill_name):
        skill_file = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        meta, _ = SkillManager._parse_frontmatter(content)
        keywords = meta.get("keywords", [])
        assert isinstance(keywords, list), f"Skill '{skill_name}' keywords is not a list"
        assert len(keywords) > 0, f"Skill '{skill_name}' has empty keywords"

    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_description_is_nonempty(self, skill_name):
        skill_file = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        meta, _ = SkillManager._parse_frontmatter(content)
        desc = meta.get("description", "")
        assert len(desc.strip()) > 10, f"Skill '{skill_name}' has too short description"


class TestSkillLoading:
    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_skill_loads_successfully(self, skill_manager, skill_name):
        content = skill_manager.load(skill_name)
        assert f'<skill name="{skill_name}">' in content
        assert len(content) > 100, f"Skill '{skill_name}' loaded but content is suspiciously short"

    def test_list_skills_returns_all(self, skill_manager):
        skills = skill_manager.list_skills()
        names = {s["name"] for s in skills}
        assert EXPECTED_SKILLS.issubset(names), (
            f"Missing from list_skills: {EXPECTED_SKILLS - names}"
        )

    @pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
    def test_skill_body_not_empty(self, skill_manager, skill_name):
        skill_file = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        _, body = SkillManager._parse_frontmatter(content)
        assert len(body.strip()) > 50, f"Skill '{skill_name}' has empty or near-empty body"
