"""BenchmarkRunner - Chạy benchmarks cho Vietnamese LLM."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.llm.lm_eval import LMEvalHarness, _tinh_macro_f1, _tinh_token_f1
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
        {
            "question": "Deep learning khác gì machine learning?",
            "answer": "sử dụng mạng nơ-ron sâu",
        },
        {"question": "NLP là viết tắt của gì?", "answer": "xử lý ngôn ngữ tự nhiên"},
    ] * 2

    SENTIMENT_TIENG_VIET = [
        {"text": "Sản phẩm rất tốt, tôi rất hài lòng.", "label": "positive"},
        {"text": "Dịch vụ kém, không nên mua.", "label": "negative"},
        {"text": "Chất lượng bình thường, không có gì đặc biệt.", "label": "neutral"},
        {"text": "Tuyệt vời! Sẽ giới thiệu cho bạn bè.", "label": "positive"},
        {"text": "Thất vọng nặng nề, tiền mất tật mang.", "label": "negative"},
    ] * 4

    def __init__(self) -> None:
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

        tong_ket_qua: Dict[str, Any] = {}

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
        if not (hasattr(model, "tinh_perplexity") or hasattr(model, "tien")):
            return {
                "trang_thai": "khong_ho_tro",
                "loi": "Mô hình không hỗ trợ tính perplexity (thiếu phương thức 'tinh_perplexity' hoặc 'tien').",
                "perplexity_mean": None,
                "perplexity_std": None,
                "perplexity_min": None,
                "perplexity_max": None,
                "so_mau": 0,
            }
        perplexities: List[float] = []

        for text in self.CORPUS_TIENG_VIET:
            try:
                if hasattr(model, "tinh_perplexity"):
                    ppl = float(model.tinh_perplexity(text))
                else:
                    toks = [ord(c) % getattr(model, "so_tu_vung", 256) for c in text[:32]]
                    if len(toks) < 2:
                        toks = [1, 2]
                    x_arr = np.array([toks[:-1]], dtype=int)
                    y_arr = np.array([toks[1:]], dtype=int)
                    logits = model.tien(x_arr)
                    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
                    probs = np.exp(logits_shifted) / np.sum(
                        np.exp(logits_shifted), axis=-1, keepdims=True
                    )
                    nll = -np.mean(
                        np.log(probs[0, np.arange(len(y_arr[0])), y_arr[0]] + 1e-9)
                    )
                    ppl = float(np.exp(min(nll, 10.0)))
                perplexities.append(ppl)
            except Exception:
                continue

        return {
            "trang_thai": "hoan_thanh" if perplexities else "loi_thuc_thi",
            "perplexity_mean": round(float(np.mean(perplexities)), 4) if perplexities else None,
            "perplexity_std": round(float(np.std(perplexities)), 4) if perplexities else None,
            "perplexity_min": round(float(np.min(perplexities)), 4) if perplexities else None,
            "perplexity_max": round(float(np.max(perplexities)), 4) if perplexities else None,
            "so_mau": len(perplexities),
        }

    def _bench_generation(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Generation benchmark...")
        if not hasattr(model, "sinh_van_ban"):
            return {
                "trang_thai": "khong_ho_tro",
                "loi": "Mô hình không hỗ trợ sinh văn bản (thiếu phương thức 'sinh_van_ban').",
                "so_prompts": 0,
                "do_dai_trung_binh": None,
                "chi_tiet": [],
            }
        prompts = ["học máy là", "trí tuệ nhân tạo", "xử lý ngôn ngữ"]
        results: List[Dict[str, Any]] = []

        for prompt in prompts:
            try:
                start = time.perf_counter()
                text = str(model.sinh_van_ban(prompt, do_dai=30))
                gen_time = time.perf_counter() - start
                results.append(
                    {
                        "prompt": prompt,
                        "generated": text,
                        "do_dai": len(text.split()),
                        "thoi_gian": round(gen_time, 4),
                    }
                )
            except Exception as exc:
                results.append({"prompt": prompt, "error": str(exc)})

        do_dai_hop_le = [r["do_dai"] for r in results if "do_dai" in r]
        return {
            "trang_thai": "hoan_thanh" if do_dai_hop_le else "loi_thuc_thi",
            "so_prompts": len(results),
            "do_dai_trung_binh": round(float(np.mean(do_dai_hop_le)), 1) if do_dai_hop_le else None,
            "chi_tiet": results,
        }

    def _bench_sentiment(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Sentiment benchmark...")
        if not (hasattr(model, "phan_tich_cam_xuc") or hasattr(model, "du_doan")):
            return {
                "trang_thai": "khong_ho_tro",
                "loi": "Mô hình không hỗ trợ phân tích cảm xúc (thiếu phương thức 'phan_tich_cam_xuc' hoặc 'du_doan').",
                "accuracy": None,
                "f1": None,
                "so_mau": 0,
            }
        y_true: List[str] = []
        y_pred: List[str] = []

        for mau in self.SENTIMENT_TIENG_VIET:
            text = mau["text"]
            label = mau["label"]
            try:
                if hasattr(model, "phan_tich_cam_xuc"):
                    pred_raw = model.phan_tich_cam_xuc(text)
                    pred = pred_raw.get("nhan", "") if isinstance(pred_raw, dict) else pred_raw
                else:
                    pred_raw = model.du_doan([text])
                    pred = pred_raw[0] if isinstance(pred_raw, (list, tuple, np.ndarray)) and len(pred_raw) > 0 else pred_raw
                y_true.append(label)
                y_pred.append(str(pred))
            except Exception:
                continue

        tong = len(y_true)
        dung = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return {
            "trang_thai": "hoan_thanh" if tong > 0 else "loi_thuc_thi",
            "accuracy": round(dung / tong, 4) if tong > 0 else None,
            "f1": round(_tinh_macro_f1(y_true, y_pred), 4) if tong > 0 else None,
            "so_mau": tong,
        }

    def _bench_speed(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  Speed benchmark...")
        if not (hasattr(model, "sinh_van_ban") or hasattr(model, "tien")):
            return {
                "trang_thai": "khong_ho_tro",
                "loi": "Mô hình không hỗ trợ đo tốc độ suy luận (thiếu phương thức 'sinh_van_ban' hoặc 'tien').",
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_p99_ms": None,
                "so_lan_chay": 0,
            }
        text = "Đoạn văn bản tiếng Việt để kiểm tra tốc độ xử lý của mô hình."
        x_sample = np.array([[ord(c) % getattr(model, "so_tu_vung", 256) for c in text[:16]]], dtype=int)

        latencies: List[float] = []
        for _ in range(10):
            try:
                start = time.perf_counter()
                if hasattr(model, "sinh_van_ban"):
                    model.sinh_van_ban(text[:10], do_dai=10)
                else:
                    model.tien(x_sample)
                latency = (time.perf_counter() - start) * 1000.0
                latencies.append(latency)
            except Exception:
                continue

        return {
            "trang_thai": "hoan_thanh" if latencies else "loi_thuc_thi",
            "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else None,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else None,
            "so_lan_chay": len(latencies),
        }

    def _bench_qa(self, model: Any) -> Dict[str, Any]:
        self.logger.info("  QA benchmark...")
        if not hasattr(model, "sinh_van_ban"):
            return {
                "trang_thai": "khong_ho_tro",
                "loi": "Mô hình không hỗ trợ hỏi đáp (thiếu phương thức 'sinh_van_ban').",
                "accuracy": None,
                "f1": None,
                "so_mau": 0,
            }
        dung = 0
        f1_scores: List[float] = []

        for mau in self.QA_TIENG_VIET:
            question = mau["question"]
            answer = mau["answer"]
            try:
                pred = str(model.sinh_van_ban(question, do_dai=30))
                if answer.lower() in pred.lower():
                    dung += 1
                f1_scores.append(_tinh_token_f1(answer, pred))
            except Exception:
                continue

        tong = len(f1_scores)
        return {
            "trang_thai": "hoan_thanh" if tong > 0 else "loi_thuc_thi",
            "accuracy": round(dung / tong, 4) if tong > 0 else None,
            "f1": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
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
            if ppl.get("perplexity_mean") is None:
                lines.append(f"  Status: {ppl.get('loi') or ppl.get('trang_thai', 'N/A')}")
            else:
                lines.append(f"  Mean: {ppl['perplexity_mean']:.4f}")
                lines.append(f"  Std:  {ppl['perplexity_std']:.4f}")
                lines.append(f"  Min:  {ppl['perplexity_min']:.4f}")

        if "generation" in self._ket_qua:
            gen = self._ket_qua["generation"]
            lines.append("\n📝 Generation:")
            if gen.get("do_dai_trung_binh") is None:
                lines.append(f"  Status: {gen.get('loi') or gen.get('trang_thai', 'N/A')}")
            else:
                lines.append(f"  Avg length: {gen['do_dai_trung_binh']} words")

        if "sentiment" in self._ket_qua:
            sent = self._ket_qua["sentiment"]
            lines.append("\n😊 Sentiment:")
            if sent.get("accuracy") is None:
                lines.append(f"  Status: {sent.get('loi') or sent.get('trang_thai', 'N/A')}")
            else:
                lines.append(f"  Accuracy: {sent['accuracy']:.4f}")
                if sent.get("f1") is not None:
                    lines.append(f"  Macro-F1: {sent['f1']:.4f}")

        if "speed" in self._ket_qua:
            spd = self._ket_qua["speed"]
            lines.append("\n⚡ Speed:")
            if spd.get("latency_mean_ms") is None:
                lines.append(f"  Status: {spd.get('loi') or spd.get('trang_thai', 'N/A')}")
            else:
                lines.append(f"  Mean latency: {spd['latency_mean_ms']:.2f} ms")
                lines.append(f"  P95 latency:  {spd['latency_p95_ms']:.2f} ms")

        if "qa" in self._ket_qua:
            qa = self._ket_qua["qa"]
            lines.append("\n❓ Q&A:")
            if qa.get("accuracy") is None:
                lines.append(f"  Status: {qa.get('loi') or qa.get('trang_thai', 'N/A')}")
            else:
                lines.append(f"  Accuracy: {qa['accuracy']:.4f}")
                if qa.get("f1") is not None:
                    lines.append(f"  Token-F1: {qa['f1']:.4f}")

        lines.append(f"\n⏱️  Total time: {self._ket_qua.get('tong_thoi_gian', 0):.1f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "co_ket_qua": len(self._ket_qua) > 0,
            "benchmarks": self._ket_qua.get("benchmarks", []),
            "tong_thoi_gian": self._ket_qua.get("tong_thoi_gian", 0),
        }
