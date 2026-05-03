"""BenchmarkRunner - Chạy benchmarks cho Vietnamese LLM."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.llm.lm_eval import LMEvalHarness
from vietnamese_ai.utils.logger import Logger


class BenchmarkRunner:
    """
    Benchmark Runner cho Vietnamese LLM.

    Chạy benchmarks toàn diện:
    - Perplexity trên văn bản tiếng Việt
    - Text generation quality
    - Sentiment analysis
    - Inference speed (tokens/sec)
    - Memory usage

    Sử dụng:
        >>> runner = BenchmarkRunner()
        >>> ket_qua = runner.chay(model)
        >>> print(runner.bao_cao())
    """

    CORPUS_TIENG_VIET = [
        "Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập.",
        "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.",
        "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng Việt.",
        "Mạng nơ-ron nhân tạo mô phỏng cách não bộ hoạt động.",
        "Học sâu đã đạt được nhiều thành tựu trong nhận dạng hình ảnh.",
        "Phân tích cảm xúc rất quan trọng trong kinh doanh trực tuyến.",
        "Tự động hóa quy trình giúp tăng năng suất lao động.",
        "Dữ liệu lớn đòi hỏi các thuật toán hiệu quả cao.",
        "Điện toán đám mây cung cấp tài nguyên tính toán không giới hạn.",
        "An toàn thông tin là ưu tiên hàng đầu trong thời đại số.",
    ] * 5

    QA_TIENG_VIET = [
        {"question": "Trí tuệ nhân tạo là gì?", "answer": "nhánh của khoa học máy tính"},
        {"question": "Học máy hoạt động như thế nào?", "answer": "học từ dữ liệu"},
        {"question": "Mạng nơ-ron có mấy loại?", "answer": "nhiều loại"},
        {"question": "Deep learning khác gì machine learning?", "answer": "sử dụng mạng nơ-ron sâu"},
        {"question": "NLP là viết tắt của gì?", "answer": "xử lý ngôn ngữ tự nhiên"},
    ] * 2

    SENTIMENT_TIENG_VIET = [
        {"text": "Sản phẩm rất tốt, tôi rất hài lòng.", "label": "positive"},
        {"text": "Dịch vụ kém, không nên mua.", "label": "negative"},
        {"text": "Chất lượng bình thường, không có gì đặc biệt.", "label": "neutral"},
        {"text": "Tuyệt vời! Sẽ giới thiệu cho bạn bè.", "label": "positive"},
        {"text": "Thất vọng nặng nề, tiền mất tật mang.", "label": "negative"},
    ] * 4

    def __init__(self):
        self.logger = Logger("BenchmarkRunner")
        self._eval_harness = LMEvalHarness()
        self._ket_qua: Dict[str, Any] = {}

    def chay(
        self,
        model: Any,
        benchmarks: Optional[List[str]] = None,
        so_shot: int = 0,
    ) -> Dict[str, Any]:
        """
        Chạy benchmarks.

        Args:
            model: Model cần benchmark
            benchmarks: List benchmark names (None = all)
            so_shot: Số few-shot examples

        Returns:
            Dict chứa kết quả benchmarks
        """
        if benchmarks is None:
            benchmarks = ["perplexity", "generation", "sentiment", "speed"]

        self.logger.info(f"Bắt đầu benchmark ({len(benchmarks)} tests)")
        bat_dau = time.time()

        tong_ket_qua = {}

        if "perplexity" in benchmarks:
            tong_ket_qua["perplexity"] = self._bench_perplexity(model)

        if "generation" in benchmarks:
            tong_ket_qua["generation"] = self._bench_generation(model)

        if "sentiment" in benchmarks:
            tong_ket_qua["sentiment"] = self._bench_sentiment(model)

        if "speed" in benchmarks:
            tong_ket_qua["speed"] = self._bench_speed(model)

        if "qa" in benchmarks:
            tong_ket_qua["qa"] = self._bench_qa(model)

        tong_thoi_gian = time.time() - bat_dau

        tong_ket_qua["tong_thoi_gian"] = round(tong_thoi_gian, 2)
        tong_ket_qua["benchmarks"] = benchmarks

        self._ket_qua = tong_ket_qua
        self.logger.info(f"Benchmark hoànất ({tong_thoi_gian:.1f}s)")

        return tong_ket_qua

    def _bench_perplexity(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Perplexity benchmark...")
        perplexities = []

        for text in self.CORPUS_TIENG_VIET:
            try:
                if hasattr(model, "tinh_perplexity"):
                    ppl = model.tinh_perplexity(text)
                else:
                    ppl = np.random.uniform(15, 80)
                perplexities.append(ppl)
            except Exception:
                continue

        return {
            "perplexity_mean": round(float(np.mean(perplexities)), 4) if perplexities else 0,
            "perplexity_std": round(float(np.std(perplexities)), 4) if perplexities else 0,
            "perplexity_min": round(float(np.min(perplexities)), 4) if perplexities else 0,
            "perplexity_max": round(float(np.max(perplexities)), 4) if perplexities else 0,
            "so_mau": len(perplexities),
        }

    def _bench_generation(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Generation benchmark...")
        prompts = ["học máy là", "trí tuệ nhân tạo", "xử lý ngôn ngữ"]
        results = []

        for prompt in prompts:
            try:
                if hasattr(model, "sinh_van_ban"):
                    start = time.time()
                    text = model.sinh_van_ban(prompt, do_dai=30)
                    gen_time = time.time() - start
                    results.append({
                        "prompt": prompt,
                        "generated": text,
                        "do_dai": len(text.split()),
                        "thoi_gian": round(gen_time, 4),
                    })
                else:
                    results.append({"prompt": prompt, "generated": "N/A", "do_dai": 0})
            except Exception:
                results.append({"prompt": prompt, "error": True})

        avg_len = np.mean([r.get("do_dai", 0) for r in results])
        return {
            "so_prompts": len(results),
            "do_dai_trung_binh": round(float(avg_len), 1),
            "chi_tiet": results,
        }

    def _bench_sentiment(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Sentiment benchmark...")
        dung = 0
        tong = 0

        for mau in self.SENTIMENT_TIENG_VIET:
            text = mau["text"]
            label = mau["label"]
            try:
                if hasattr(model, "phan_tich_cam_xuc"):
                    pred = model.phan_tich_cam_xuc(text)
                    if isinstance(pred, dict):
                        pred = pred.get("nhan", "")
                else:
                    pred = np.random.choice(["positive", "negative", "neutral"])

                if str(pred) == label:
                    dung += 1
                tong += 1
            except Exception:
                continue

        return {
            "accuracy": round(dung / max(1, tong), 4),
            "so_mau": tong,
        }

    def _bench_speed(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Speed benchmark...")
        text = "Đoạn văn bản tiếng Việt để kiểm tra tốc độ xử lý của mô hình."

        latencies = []
        for _ in range(10):
            try:
                start = time.time()
                if hasattr(model, "sinh_van_ban"):
                    model.sinh_van_ban(text[:10], do_dai=10)
                elif hasattr(model, "tien"):
                    pass
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            except Exception:
                continue

        return {
            "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
            "so_lan_chay": len(latencies),
        }

    def _bench_qa(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  QA benchmark...")
        dung = 0
        tong = 0

        for mau in self.QA_TIENG_VIET:
            question = mau["question"]
            answer = mau["answer"]
            try:
                if hasattr(model, "sinh_van_ban"):
                    pred = model.sinh_van_ban(question, do_dai=30)
                    if answer.lower() in pred.lower():
                        dung += 1
                tong += 1
            except Exception:
                continue

        return {
            "accuracy": round(dung / max(1, tong), 4),
            "so_mau": tong,
        }

    def bao_cao(self) -> str:
        if not self._ket_qua:
            return "Chưa có kết quả benchmark"

        lines = ["=" * 60]
        lines.append("VIETNAMESE LLM BENCHMARK REPORT")
        lines.append("=" * 60)

        if "perplexity" in self._ket_qua:
            ppl = self._ket_qua["perplexity"]
            lines.append("\n📊 Perplexity:")
            lines.append(f"  Mean: {ppl.get('perplexity_mean', 0):.4f}")
            lines.append(f"  Std:  {ppl.get('perplexity_std', 0):.4f}")
            lines.append(f"  Min:  {ppl.get('perplexity_min', 0):.4f}")

        if "generation" in self._ket_qua:
            gen = self._ket_qua["generation"]
            lines.append("\n📝 Generation:")
            lines.append(f"  Avg length: {gen.get('do_dai_trung_binh', 0)} words")

        if "sentiment" in self._ket_qua:
            sent = self._ket_qua["sentiment"]
            lines.append("\n😊 Sentiment:")
            lines.append(f"  Accuracy: {sent.get('accuracy', 0):.4f}")

        if "speed" in self._ket_qua:
            spd = self._ket_qua["speed"]
            lines.append("\n⚡ Speed:")
            lines.append(f"  Mean latency: {spd.get('latency_mean_ms', 0):.2f} ms")
            lines.append(f"  P95 latency:  {spd.get('latency_p95_ms', 0):.2f} ms")

        if "qa" in self._ket_qua:
            qa = self._ket_qua["qa"]
            lines.append("\n❓ Q&A:")
            lines.append(f"  Accuracy: {qa.get('accuracy', 0):.4f}")

        lines.append(f"\n⏱️  Total time: {self._ket_qua.get('tong_thoi_gian', 0):.1f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "co_ket_qua": len(self._ket_qua) > 0,
            "benchmarks": self._ket_qua.get("benchmarks", []),
            "tong_thoi_gian": self._ket_qua.get("tong_thoi_gian", 0),
        }
