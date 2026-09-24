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

"""Agent configuration."""

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}

# Model → context window size mapping (tokens).
# If a model is not listed, falls back to DEFAULT_CONTEXT_TOKENS.
DEFAULT_CONTEXT_TOKENS = 200000

MODEL_CONTEXT_WINDOWS = {
    # Anthropic
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-3-7-sonnet-20250219": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-pro": 200000,
    "o3": 200000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    # DeepSeek
    "deepseek-chat": 200000,
    "deepseek-reasoner": 200000,
}


def _resolve_context_window(model: str) -> int:
    """Resolve context window for a model name, with prefix matching fallback."""
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Prefix match: e.g. "gpt-4o-2024-08-06" matches "gpt-4o"
    for prefix, tokens in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return tokens
    return DEFAULT_CONTEXT_TOKENS


@dataclass
class AgentConfig:
    provider: str = "anthropic"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_iterations: int = 200000
    max_context_tokens: int = 0  # 0 = auto-detect from model
    shell_remind_interval: int = 60
    max_output_tokens: int = 8192
    thinking_budget: int = 0  # 0 = disabled; >0 = enable thinking with budget_tokens cap (must be ≥1024 and < max_output_tokens)
    session_dir: Optional[str] = None
    skill_dirs: List[str] = field(default_factory=list)
    shell_env: Dict[str, str] = field(default_factory=dict)
    max_continuations: int = 200
    time_budget_sec: float = 0.0  # 0 = unset; >0 = per-turn wall-clock budget driving time warnings + wrap-up (NOT a hard kill)
    _config_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.model is None:
            if self.provider == "anthropic":
                env_model = os.environ.get("ANTHROPIC_MODEL")
            elif self.provider == "openai":
                env_model = os.environ.get("OPENAI_DEFAULT_MODEL")
            else:
                env_model = None
            self.model = env_model or DEFAULT_MODELS.get(self.provider, "claude-sonnet-4-20250514")

        # Auto-detect context window from model if not explicitly set
        if self.max_context_tokens <= 0:
            self.max_context_tokens = _resolve_context_window(self.model)

        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = (
                    os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or os.environ.get("ANTHROPIC_API_KEY")
                )
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")

        if self.base_url is None:
            if self.provider == "anthropic":
                self.base_url = os.environ.get("ANTHROPIC_BASE_URL")
            elif self.provider == "openai":
                self.base_url = os.environ.get("OPENAI_BASE_URL")

        if not self.skill_dirs:
            from flagscale_agent.react.paths import get_skill_dir
            builtin_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "skills"
            )
            self.skill_dirs = [builtin_dir, get_skill_dir()]

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
            if var not in self.shell_env:
                val = os.environ.get(var)
                if val:
                    self.shell_env[var] = val

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load config from a YAML file. Unknown keys are ignored."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        config = cls(**filtered)
        config._config_path = path
        return config

    @classmethod
    def auto_load(cls, **overrides) -> "AgentConfig":
        """Try loading from env var or default paths, then apply overrides."""
        config_path = os.environ.get("FLAGSCALE_AGENT_CONFIG")
        if not config_path:
            from flagscale_agent.react.paths import get_config_path
            candidate = get_config_path()
            if os.path.isfile(candidate):
                config_path = candidate

        if config_path and os.path.isfile(config_path):
            config = cls.from_yaml(config_path)
        else:
            config = cls()
            config._config_path = config_path

        # Apply overrides
        for k, v in overrides.items():
            if v is not None and hasattr(config, k):
                setattr(config, k, v)
        
        # Re-run provider-dependent initialization if provider was overridden
        if "provider" in overrides and overrides["provider"] is not None:
            # Re-detect model, api_key, base_url based on the new provider
            if "model" not in overrides or overrides["model"] is None:
                if config.provider == "anthropic":
                    env_model = os.environ.get("ANTHROPIC_MODEL")
                elif config.provider == "openai":
                    env_model = os.environ.get("OPENAI_DEFAULT_MODEL")
                else:
                    env_model = None
                config.model = env_model or DEFAULT_MODELS.get(config.provider, "claude-sonnet-4-20250514")
            
            if "api_key" not in overrides or overrides["api_key"] is None:
                if config.provider == "anthropic":
                    config.api_key = (
                        os.environ.get("ANTHROPIC_AUTH_TOKEN")
                        or os.environ.get("ANTHROPIC_API_KEY")
                    )
                elif config.provider == "openai":
                    config.api_key = os.environ.get("OPENAI_API_KEY")
            
            if "base_url" not in overrides or overrides["base_url"] is None:
                if config.provider == "anthropic":
                    config.base_url = os.environ.get("ANTHROPIC_BASE_URL")
                elif config.provider == "openai":
                    config.base_url = os.environ.get("OPENAI_BASE_URL")
        
        return config

    def reload(self):
        """Reload config from the original YAML file, if available."""
        if not self._config_path or not os.path.isfile(self._config_path):
            return False
        with open(self._config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in self.__dataclass_fields__.values() if not f.name.startswith("_")}
        for k, v in data.items():
            if k in valid_fields:
                setattr(self, k, v)
        # Re-run post-init validation
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
            if var not in self.shell_env:
                val = os.environ.get(var)
                if val:
                    self.shell_env[var] = val
        return True
