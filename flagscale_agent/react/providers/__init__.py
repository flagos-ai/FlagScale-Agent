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
