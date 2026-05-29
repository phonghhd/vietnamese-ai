"""Ví dụ: Evaluation & Benchmarks."""

from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner
from vietnamese_ai.llm.lm_eval import LMEvalHarness
from vietnamese_ai.llm.model_configs import ModelConfig
from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM


def vi_du_model_config():
    """Sử dụng ModelConfig presets."""
    print("Danh sách presets:")
    for ten, mo_ta in ModelConfig.danh_sach_presets().items():
        config = ModelConfig.from_preset(ten)
        print(f"  {ten}: {config.so_tham_so_str} - {mo_ta}")


def vi_du_eval():
    """Đánh giá model với LMEvalHarness."""
    corpus = [
        "học máy là một nhánh của trí tuệ nhân tạo",
        "trí tuệ nhân tạo đang phát triển rất nhanh",
        "mạng nơ-ron nhân tạo mô phỏng não bộ con người",
        "học sâu là một kỹ thuật mạnh mẽ trong học máy",
        "xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng Việt",
    ] * 10

    llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
    llm.huan_luyen(corpus)

    harness = LMEvalHarness()
    print("Tasks có sẵn:")
    for ten, mo_ta in harness.danh_sach_tasks().items():
        print(f"  {ten}: {mo_ta}")

    ket_qua = harness.danh_gia(llm, ["vie_perplexity", "vie_sentiment"])
    print("\nKết quả đánh giá:")
    for task, result in ket_qua["ket_qua"].items():
        print(f"  {task}: {result}")

    print(f"\n{harness.bao_cao()}")


def vi_du_benchmark():
    """Chạy benchmarks."""
    corpus = [
        "học máy rất thú vị",
        "trí tuệ nhân tạo tuyệt vời",
        "mạng nơ-ron rất mạnh mẽ",
    ] * 10

    llm = VietnameseLLM(bac=2, toi_thieu_dem=1)
    llm.huan_luyen(corpus)

    runner = BenchmarkRunner()
    runner.chay(llm, benchmarks=["perplexity", "speed"])

    print(f"\n{runner.bao_cao()}")


if __name__ == "__main__":
    print("=== Model Config ===")
    vi_du_model_config()
    print("\n=== Evaluation ===")
    vi_du_eval()
    print("\n=== Benchmark ===")
    vi_du_benchmark()
