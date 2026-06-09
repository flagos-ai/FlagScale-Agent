# Contributing to FlagScale-Agent

Thank you for your interest in contributing to FlagScale-Agent! This document provides guidelines for contributing to the project.

---

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/FlagScale-Agent.git
cd FlagScale-Agent
```

### 2. Set Up Development Environment
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

---

## 📝 Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Run Linter
```bash
ruff check flagscale_agent/
```

### Auto-Fix Issues
```bash
ruff check --fix flagscale_agent/
```

### Configuration
See `pyproject.toml` for Ruff settings:
- Line length: 100
- Selected rules: E (errors), F (pyflakes), UP (pyupgrade), I (isort), B (bugbear)
- Ignored: E402, E501, E722, E731, E741, F403, F405

### Import Order
```python
# Standard library
import os
import sys

# Third-party
import yaml
from anthropic import Anthropic

# Local
from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.tools.base import BaseTool
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_kernel.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=flagscale_agent --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Test Requirements
- New features **must** include tests
- Bug fixes **should** include regression tests
- Maintain or improve coverage (currently >90%)

### Test Structure
- `tests/` mirrors `flagscale_agent/` structure
- Use pytest fixtures for common setup
- Mock external dependencies (LLM calls, file I/O)

---

## 🛠️ Contributing Guidelines

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add support for custom tool plugins
fix: resolve memory leak in history compaction
docs: update README with new examples
test: add tests for CircuitBreakerGuard
refactor: simplify Guard registry lifecycle
```

### Pull Request Process

1. **Before submitting:**
   - Run tests: `pytest tests/ -v`
   - Run linter: `ruff check flagscale_agent/`
   - Update docs if adding new features
   - Add entries to `CHANGELOG.md` (if exists)

2. **PR Title:**
   - Use conventional commit format
   - Example: `feat: add vLLM backend support for inference`

3. **PR Description:**
   - Describe the problem and solution
   - Reference related issues: `Fixes #123`
   - List breaking changes (if any)
   - Include screenshots for UI changes

4. **Review Process:**
   - Address reviewer feedback promptly
   - Keep commits clean (squash if needed)
   - Ensure CI passes

---

## 🎯 What to Contribute

### High-Priority Areas
- 🐛 Bug fixes (check [Issues](https://github.com/flagos-ai/FlagScale-Agent/issues))
- 📚 Documentation improvements
- 🧪 Test coverage for under-tested modules
- 🔧 New domain-specific tools (e.g., profiling, benchmarking)
- 📖 New skills for common workflows
- 🌐 Multi-language support (translations)

### Ideas for Contributions
- **Tools:** Add tools for tensorboard, W&B, or other training infrastructure
- **Skills:** Create skills for specific frameworks (DeepSpeed, FSDP)
- **Guards:** Implement new safety/quality checks
- **Providers:** Add support for more LLM providers (Gemini, Cohere, etc.)
- **Chip Support:** Extend chip capability model for new accelerators
- **Examples:** Add example workflows for common use cases

---

## 📚 Documentation

### Code Documentation
- Public classes/functions **must** have docstrings
- Follow Google-style docstrings:
  ```python
  def my_function(arg1: str, arg2: int) -> bool:
      """Brief description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
      
      Returns:
          Description of return value
      
      Raises:
          ValueError: When arg2 is negative
      """
  ```

### README and Docs
- Keep README.md concise (quick start + overview)
- Detailed guides go in `docs/`
- Update skill READMEs when modifying skills

---

## 🔧 Development Tips

### Running the Agent Locally
```bash
# From source (editable install)
python -m flagscale_agent.cli

# With custom config
FLAGSCALE_AGENT_CONFIG=./my_config.yaml python -m flagscale_agent.cli
```

### Debugging
```python
# Enable verbose logging
export FLAGSCALE_AGENT_LOG_LEVEL=DEBUG

# Or in code
import logging
logging.getLogger("flagscale_agent").setLevel(logging.DEBUG)
```

### Adding a New Tool
1. Create `flagscale_agent/react/tools/my_tool.py`:
   ```python
   from .base import BaseTool, ToolResult
   
   class MyTool(BaseTool):
       name = "my_tool"
       description = "What this tool does"
       parameters = {
           "arg1": {"type": "string", "required": True, "description": "..."}
       }
       
       def execute(self, arg1: str, **kwargs) -> ToolResult:
           # Implementation
           return ToolResult(
               output="Success message",
               success=True
           )
   ```

2. Register in `flagscale_agent/react/tools/__init__.py`:
   ```python
   from .my_tool import MyTool
   
   BUILTIN_TOOLS = [..., MyTool]
   ```

3. Add tests in `tests/test_my_tool.py`

### Adding a New Skill
1. Create `flagscale_agent/skills/my-skill/SKILL.md`
2. Follow the skill template format (see existing skills)
3. Test by loading: `/skill my-skill`

---

## 🤝 Community

### Communication
- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas, general discussion
- **Pull Requests:** Code contributions

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Focus on the technical merit of contributions

---

## 🙏 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md` (if you create one)
- Mentioned in release notes for significant contributions
- Acknowledged in the README for major features

---

## ❓ Questions?

If you're unsure about anything:
1. Check existing [Issues](https://github.com/flagos-ai/FlagScale-Agent/issues) and [Discussions](https://github.com/flagos-ai/FlagScale-Agent/discussions)
2. Open a new Discussion for questions
3. Reach out via email: caozhou.1995@gmail.com

---

**Thank you for contributing to FlagScale-Agent!** 🎉
