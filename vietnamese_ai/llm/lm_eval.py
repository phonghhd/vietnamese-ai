"""LMEvalHarness - Evaluation framework tương thích lm-eval-harness."""

import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


def _tinh_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Tính Macro-F1 thực tế giữa danh sách nhãn chuẩn và nhãn dự đoán."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    cac_lop = sorted(set(y_true) | set(y_pred))
    if not cac_lop:
        return 0.0
    f1_list: List[float] = []
    for lop in cac_lop:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lop and p == lop)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lop and p == lop)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lop and p != lop)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))


def _tinh_token_f1(reference: str, prediction: str) -> float:
    """Tính Token-level F1 thực tế cho bài toán hỏi đáp (QA)."""
    ref_toks = reference.lower().split()
    pred_toks = prediction.lower().split()
    if not ref_toks and not pred_toks:
        return 1.0
    if not ref_toks or not pred_toks:
        return 0.0
    common = Counter(ref_toks) & Counter(pred_toks)
    so_khop = sum(common.values())
    if so_khop == 0:
        return 0.0
    prec = so_khop / len(pred_toks)
    rec = so_khop / len(ref_toks)
    return 2.0 * prec * rec / (prec + rec)


class EvalTask:
    """Định nghĩa một evaluation task."""

    def __init__(
        self,
        ten: str,
        loai: str = "text_generation",
        du_lieu: Optional[List[Dict]] = None,
        metrics: Optional[List[str]] = None,
        mo_ta: str = "",
    ) -> None:
        self.ten = ten
        self.loai = loai
        self.du_lieu = du_lieu or []
        self.metrics = metrics or ["accuracy"]
        self.mo_ta = mo_ta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ten": self.ten,
            "loai": self.loai,
            "so_mau": len(self.du_lieu),
            "metrics": self.metrics,
            "mo_ta": self.mo_ta,
        }


class LMEvalHarness:
    """
    LM Evaluation Framework.

    Tương thích với phong cách lm-eval-harness, hỗ trợ đánh giá
    Vietnamese LLM trên nhiều benchmarks.

    Tính năng:
    - Multi-task evaluation
    - Custom task registration
    - Built-in Vietnamese tasks
    - Perplexity, accuracy, generation quality
    - Few-shot evaluation
    - Result aggregation

    Sử dụng:
        >>> harness = LMEvalHarness()
        >>> harness.dang_ky_task(vietnamese_qa_task)
        >>> ket_qua = harness.danh_gia(model, ["task1", "task2"])
    """

    TASKS_MAC_DINH: Dict[str, Dict[str, Any]] = {
        "vie_perplexity": {
            "mo_ta": "Perplexity trên văn bản tiếng Việt",
            "loai": "perplexity",
            "metrics": ["perplexity"],
        },
        "vie_sentiment": {
            "mo_ta": "Phân loại cảm xúc tiếng Việt",
            "loai": "classification",
            "metrics": ["accuracy", "f1"],
        },
        "vie_text_generation": {
            "mo_ta": "Sinh văn bản tiếng Việt",
            "loai": "text_generation",
            "metrics": ["bleu", "rouge"],
        },
        "vie_qa": {
            "mo_ta": "Hỏi đáp tiếng Việt",
            "loai": "question_answering",
            "metrics": ["accuracy", "f1"],
        },
        "vie_cloze": {
            "mo_ta": "Cloze test tiếng Việt",
            "loai": "cloze",
            "metrics": ["accuracy"],
        },
    }

    def __init__(self) -> None:
        self.logger = Logger("LMEvalHarness")
        self._tasks: Dict[str, EvalTask] = {}
        self._ket_qua: Dict[str, Dict] = {}

        for ten, cfg in self.TASKS_MAC_DINH.items():
            self._tasks[ten] = EvalTask(
                ten=ten, loai=cfg["loai"], metrics=cfg["metrics"], mo_ta=cfg["mo_ta"]
            )

    def dang_ky_task(self, task: EvalTask) -> None:
        self._tasks[task.ten] = task
        self.logger.info(f"Đăng ký task: {task.ten}")

    def danh_sach_tasks(self) -> Dict[str, str]:
        return {ten: task.mo_ta for ten, task in self._tasks.items()}

    def danh_gia(
        self,
        model: Any,
        cac_task: Optional[List[str]] = None,
        so_shot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá model trên nhiều tasks.

        Args:
            model: Model cần đánh giá
            cac_task: List task names (None = all)
            so_shot: Số few-shot examples
            limit: Giới hạn số mẫu mỗi task

        Returns:
            Dict chứa kết quả đánh giá
        """
        if cac_task is None:
            cac_task = list(self._tasks.keys())

        self.logger.info(f"Bắt đầu đánh giá {len(cac_task)} tasks (so_shot={so_shot})")
        bat_dau = time.time()

        tong_ket_qua = {}
        for task_name in cac_task:
            if task_name not in self._tasks:
                self.logger.warning(f"Task '{task_name}' không tồn tại, bỏ qua")
                continue

            task = self._tasks[task_name]
            self.logger.info(f"  Đánh giá: {task_name}")

            try:
                task_result = self._danh_gia_task(model, task, so_shot, limit)
                tong_ket_qua[task_name] = task_result
            except Exception as e:
                self.logger.error(f"  Lỗi đánh giá {task_name}: {e}")
                tong_ket_qua[task_name] = {"error": str(e)}

        tong_thoi_gian = time.time() - bat_dau

        self._ket_qua = tong_ket_qua

        return {
            "so_tasks": len(tong_ket_qua),
            "so_shot": so_shot,
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "ket_qua": tong_ket_qua,
            "tong_hop": self._tong_hop(tong_ket_qua),
        }

    def _danh_gia_task(
        self, model: Any, task: EvalTask, so_shot: int, limit: Optional[int]
    ) -> Dict[str, Any]:
        """Đánh giá model trên một task."""
        task_type = task.loai

        if task_type == "perplexity":
            return self._eval_perplexity(model, task, limit)
        elif task_type == "classification":
            return self._eval_classification(model, task, limit)
        elif task_type == "text_generation":
            return self._eval_generation(model, task, limit)
        elif task_type == "question_answering":
            return self._eval_qa(model, task, limit)
        elif task_type == "cloze":
            return self._eval_cloze(model, task, limit)
        else:
            return self._eval_generic(model, task, limit)

    def _eval_perplexity(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Đánh giá perplexity thực tế trên mô hình có phương thức `tinh_perplexity` hoặc `tien`."""
        if not (hasattr(model, "tinh_perplexity") or hasattr(model, "tien")):
            raise NotImplementedError(
                "Mô hình không hỗ trợ tính perplexity (thiếu phương thức 'tinh_perplexity' hoặc 'tien')."
            )
        texts = self._lay_du_lieu(task, limit)
        perplexities: List[float] = []

        for text in texts:
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
            "perplexity": round(float(np.mean(perplexities)), 4) if perplexities else None,
            "perplexity_std": round(float(np.std(perplexities)), 4) if perplexities else None,
            "so_mau": len(perplexities),
        }

    def _eval_classification(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Đánh giá phân loại với Accuracy và Macro-F1 thực tế (không ngẫu nhiên hóa dự đoán)."""
        if not (hasattr(model, "phan_tich_cam_xuc") or hasattr(model, "du_doan")):
            raise NotImplementedError(
                "Mô hình không hỗ trợ phân loại (thiếu phương thức 'phan_tich_cam_xuc' hoặc 'du_doan')."
            )
        du_lieu = task.du_lieu[:limit] if limit else task.du_lieu
        if not du_lieu:
            du_lieu = [{"text": "Sản phẩm rất tốt", "label": "positive"}]

        y_true: List[str] = []
        y_pred: List[str] = []
        for mau in du_lieu:
            text = str(mau.get("text", ""))
            label = str(mau.get("label", ""))

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
        accuracy = dung / tong if tong > 0 else 0.0
        macro_f1 = _tinh_macro_f1(y_true, y_pred) if tong > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "f1": round(macro_f1, 4),
            "so_mau": tong,
        }

    def _eval_generation(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Đánh giá sinh văn bản thực tế bằng BLEU khi có câu tham chiếu chuẩn."""
        if not hasattr(model, "sinh_van_ban"):
            raise NotImplementedError(
                "Mô hình không hỗ trợ sinh văn bản (thiếu phương thức 'sinh_van_ban')."
            )
        du_lieu = task.du_lieu[:limit] if limit else task.du_lieu
        if not du_lieu:
            du_lieu = [{"prompt": "học máy là", "reference": ""}]

        scores: List[float] = []
        so_mau_sinh = 0
        for mau in du_lieu:
            prompt = str(mau.get("prompt", ""))
            reference = str(mau.get("reference", "")).strip()

            try:
                generated = str(model.sinh_van_ban(prompt, do_dai=50))
                so_mau_sinh += 1
                if reference:
                    scores.append(self._tinh_bleu(reference, generated))
            except Exception:
                continue

        return {
            "bleu": round(float(np.mean(scores)), 4) if scores else None,
            "so_mau": so_mau_sinh,
            "so_mau_co_tham_chieu": len(scores),
        }

    def _eval_qa(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Đánh giá hỏi đáp thực tế bằng độ khớp đáp án và Token-level F1."""
        if not hasattr(model, "sinh_van_ban"):
            raise NotImplementedError(
                "Mô hình không hỗ trợ hỏi đáp sinh văn bản (thiếu phương thức 'sinh_van_ban')."
            )
        du_lieu = task.du_lieu[:limit] if limit else task.du_lieu
        if not du_lieu:
            du_lieu = [{"question": "AI là gì?", "answer": "trí tuệ nhân tạo"}]

        dung = 0
        f1_scores: List[float] = []
        for mau in du_lieu:
            question = str(mau.get("question", ""))
            answer = str(mau.get("answer", ""))

            try:
                pred = str(model.sinh_van_ban(question, do_dai=30))
                if answer.lower() in pred.lower():
                    dung += 1
                f1_scores.append(_tinh_token_f1(answer, pred))
            except Exception:
                continue

        tong = len(f1_scores)
        accuracy = dung / tong if tong > 0 else 0.0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return {
            "accuracy": round(accuracy, 4),
            "f1": round(mean_f1, 4),
            "so_mau": tong,
        }

    def _eval_cloze(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Đánh giá Cloze test trên mô hình có phương thức `lay_tu_ke_tiep`."""
        if not hasattr(model, "lay_tu_ke_tiep"):
            raise NotImplementedError(
                "Mô hình không hỗ trợ Cloze test (thiếu phương thức 'lay_tu_ke_tiep')."
            )
        du_lieu = task.du_lieu[:limit] if limit else task.du_lieu
        if not du_lieu:
            du_lieu = [{"text": "học máy là một [MASK] của AI", "answer": "nhánh"}]

        dung = 0
        tong = 0
        for mau in du_lieu:
            text = str(mau.get("text", ""))
            answer = str(mau.get("answer", ""))

            try:
                prefix = text.split("[MASK]")[0]
                goi_y = model.lay_tu_ke_tiep(prefix, top_n=5)
                predictions = [g.get("tu", "") for g in goi_y]
                if answer in predictions:
                    dung += 1
                tong += 1
            except Exception:
                continue

        return {
            "accuracy": round(dung / tong, 4) if tong > 0 else 0.0,
            "so_mau": tong,
        }

    def _eval_generic(self, model: Any, task: EvalTask, limit: Optional[int]) -> Dict[str, Any]:
        """Báo lỗi rõ ràng cho loại task chưa có bộ chấm định lượng hợp lệ."""
        raise NotImplementedError(f"Loại task đánh giá chưa được hỗ trợ: '{task.loai}'.")

    def _lay_du_lieu(self, task: EvalTask, limit: Optional[int]) -> List[str]:
        if task.du_lieu:
            data = task.du_lieu[:limit] if limit else task.du_lieu
            return [d.get("text", d.get("prompt", str(d))) for d in data]
        return ["học máy là một nhánh của trí tuệ nhân tạo"] * 10

    def _tinh_bleu(self, reference: str, hypothesis: str) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens:
            return 0.0

        matches = 0
        for token in hyp_tokens:
            if token in ref_tokens:
                matches += 1

        precision = matches / len(hyp_tokens)
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
        return bp * precision

    def _tong_hop(self, ket_qua: Dict[str, Dict]) -> Dict[str, float]:
        tong_hop: Dict[str, float] = {}
        for task_name, result in ket_qua.items():
            if isinstance(result, dict) and "error" not in result:
                for metric in ["accuracy", "perplexity", "bleu", "f1"]:
                    val = result.get(metric)
                    if isinstance(val, (int, float)):
                        tong_hop[f"{task_name}/{metric}"] = float(val)
        return tong_hop

    def bao_cao(self) -> str:
        if not self._ket_qua:
            return "Chưa có kết quả đánh giá"

        lines = ["=" * 60]
        lines.append("LM EVALUATION REPORT")
        lines.append("=" * 60)

        for task_name, result in self._ket_qua.items():
            lines.append(f"\n[{task_name}]")
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            else:
                for key, value in result.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_tasks": len(self._tasks),
            "tasks": list(self._tasks.keys()),
            "co_ket_qua": len(self._ket_qua) > 0,
        }
