import os
import litellm

def _apply_api_key(provider: str, kwargs: dict) -> None:
    """
    Looks up the required API key for the provider in the environment
    and safely injects it into either os.environ (for Gemini) or litellm.api_key.
    """
    provider_key_map = {
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    env_key = provider_key_map.get(provider)
    if env_key and os.getenv(env_key):
        if provider == "gemini":
            os.environ["GEMINI_API_KEY"] = os.getenv(env_key)
        else:
            litellm.api_key = os.getenv(env_key)


def _apply_reasoning_params(provider: str, model: str, kwargs: dict) -> None:
    """
    Applies advanced reasoning flags like 'thinking' / 'thinking_level' for providers
    that support it. Removes conflicting settings such as low temperature that 
    cause infinite loops (e.g. Gemini 3+ models).
    """
    if provider == "gemini":
        if "gemini-2.5-pro" in model or "gemini-3" in model:
            # LiteLLM automatically maps reasoning_effort to thinking_level
            kwargs["reasoning_effort"] = "high"
            kwargs.pop("temperature", None)

    elif provider == "openai":
        if model.startswith("o1") or model.startswith("o3"):
            kwargs["reasoning_effort"] = "high"
            kwargs.pop("temperature", None)


def get_llm_kwargs(provider: str, model: str, base_kwargs: dict = None) -> dict:
    """
    Constructs the exact kwargs required for litellm.completion() dynamically.
    """
    if base_kwargs is None:
        base_kwargs = {}
        
    kwargs = base_kwargs.copy()
    kwargs["model"] = f"{provider}/{model}" if provider else model
    
    _apply_api_key(provider, kwargs)
    _apply_reasoning_params(provider, model, kwargs)
    
    # Give complex reasoning models and large context payloads time to process
    # without triggering httpx 60-second read timeouts ("Server disconnected").
    if "timeout" not in kwargs:
        kwargs["timeout"] = 600
        
    # Add retry logic for robustness
    if "num_retries" not in kwargs:
        kwargs["num_retries"] = 3
            
    return kwargs
