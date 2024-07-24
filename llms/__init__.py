from llms.base_llm import BaseLLM


def get_llm(provider: str, *args, **kwargs) -> BaseLLM:
    if provider == "vertexai":
        from .gemini import Gemini
        return Gemini(*args, **kwargs)
    elif provider == "openai":
        from .gpt import GPT
        return GPT(*args, **kwargs)
    elif provider == "mistral-preview":
        from .mistral_preview import MistralPreview
        return MistralPreview(*args, **kwargs)
    elif provider == "llamacpp":
        from .llamacpp import LlamaCPP
        return LlamaCPP(*args, **kwargs)
    elif provider == "llama-gcp":
        from .llama_gcp import LlamaGCP
        return LlamaGCP(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
