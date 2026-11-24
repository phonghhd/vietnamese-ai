"""Extension LLM - Mô hình ngôn ngữ, Wrapper API và Benchmark (extra: `llm` / `local-llm`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "VietnameseLLM": ("vietnamese_ai.llm.vietnamese_llm", "VietnameseLLM"),
    "ModelConfig": ("vietnamese_ai.llm.model_configs", "ModelConfig"),
    "LMEvalHarness": ("vietnamese_ai.llm.lm_eval", "LMEvalHarness"),
    "BenchmarkRunner": ("vietnamese_ai.llm.benchmark_runner", "BenchmarkRunner"),
    "BaseLLMWrapper": ("vietnamese_ai.llm.wrappers", "BaseLLMWrapper"),
    "OpenAIWrapper": ("vietnamese_ai.llm.wrappers", "OpenAIWrapper"),
    "GeminiWrapper": ("vietnamese_ai.llm.wrappers", "GeminiWrapper"),
    "OllamaWrapper": ("vietnamese_ai.llm.wrappers", "OllamaWrapper"),
    "VLLMWrapper": ("vietnamese_ai.llm.wrappers", "VLLMWrapper"),
    "LlamaCppWrapper": ("vietnamese_ai.llm.wrappers", "LlamaCppWrapper"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
