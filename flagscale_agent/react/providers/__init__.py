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

"""Provider factory."""

from flagscale_agent.react.providers.base import LLMProvider


def get_provider(provider: str, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192) -> LLMProvider:
    """Create an LLM provider instance."""
    if provider == "openai":
        from flagscale_agent.react.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    elif provider == "anthropic":
        from flagscale_agent.react.providers.anthropic_provider import (
            AnthropicProvider,
        )

        return AnthropicProvider(model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
