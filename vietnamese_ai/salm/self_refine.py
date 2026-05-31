"""SelfRefine - tự cải thiện output qua nhiều vòng lặp."""

import time
from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class SelfRefine:
    """
    Self-Refinement: model tự đánh giá và cải thiện output qua nhiều vòng.

    Flow: Generate → Evaluate → Feedback → Refine → Repeat

    Dựa trên paper "Self-Refine: Iterative Refinement with Self-Feedback" (2023).
    Model sinh output, tự đánh giá, tự tạo feedback, rồi refine.

    Sử dụng:
        >>> refine = SelfRefine(ham_sinh=generate_fn, ham_danh_gia=eval_fn)
        >>> ket_qua = refine.chay("Viết đoạn văn về AI")
        >>> print(ket_qua["output_cuoi"])
        >>> print(ket_qua["so_vong"])
    """

    def __init__(
        self,
        ham_sinh: Callable[[str], str],
        ham_danh_gia: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        ham_feedback: Optional[Callable[[str, str], str]] = None,
        ham_refine: Optional[Callable[[str, str], str]] = None,
        so_vong_toi_da: int = 3,
        nguong_chat_luong: float = 0.8,
        giam_diem_dung: float = 0.01,
    ):
        self.ham_sinh = ham_sinh
        self.ham_danh_gia = ham_danh_gia
        self.ham_feedback = ham_feedback or ham_sinh
        self.ham_refine = ham_refine or ham_sinh
        self.so_vong_toi_da = so_vong_toi_da
        self.nguong_chat_luong = nguong_chat_luong
        self.giam_diem_dung = giam_diem_dung
        self.logger = Logger("SelfRefine")

        self._lich_su: List[Dict[str, Any]] = []

    def chay(
        self,
        prompt: str,
        nhiet_do: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chạy self-refinement loop.

        Args:
            prompt: Prompt đầu vào
            nhiet_do: Nhiệt độ sinh (cho đa dạng)

        Returns:
            {output_cuoi, lich_su_vong, diem_cuoi, so_vong, tong_thoi_gian}
        """
        bat_dau = time.time()
        lich_su_vong = []

        output_hien_tai = self._sinh_ban_dau(prompt)
        diem_hien_tai = 0.0

        for vong in range(self.so_vong_toi_da):
            # Bước 1: Đánh giá output hiện tại
            danh_gia = self._danh_gia(prompt, output_hien_tai)
            diem_moi = danh_gia.get("diem", 0.0)

            lich_su_vong.append(
                {
                    "vong": vong,
                    "output": output_hien_tai,
                    "diem": diem_moi,
                    "danh_gia": danh_gia,
                }
            )

            # Kiểm tra điều kiện dừng
            if diem_moi >= self.nguong_chat_luong:
                self.logger.info(
                    f"Vòng {vong}: đạt ngưỡng {diem_moi:.3f} >= {self.nguong_chat_luong}"
                )
                break

            if vong > 0 and abs(diem_moi - diem_hien_tai) < self.giam_diem_dung:
                self.logger.info(
                    f"Vòng {vong}: điểm không cải thiện ({diem_moi:.3f} ≈ {diem_hien_tai:.3f})"
                )
                break

            diem_hien_tai = diem_moi

            # Bước 2: Tạo feedback
            feedback = self._tao_feedback(prompt, output_hien_tai, danh_gia)

            # Bước 3: Refine
            output_moi = self._refine(prompt, output_hien_tai, feedback)

            if output_moi == output_hien_tai:
                self.logger.info(f"Vòng {vong}: output không thay đổi, dừng")
                break

            output_hien_tai = output_moi

        thoi_gian = time.time() - bat_dau

        ket_qua = {
            "output_cuoi": output_hien_tai,
            "lich_su_vong": lich_su_vong,
            "diem_cuoi": lich_su_vong[-1]["diem"] if lich_su_vong else 0.0,
            "so_vong": len(lich_su_vong),
            "tong_thoi_gian": round(thoi_gian, 3),
            "dat_nguong": lich_su_vong[-1]["diem"] >= self.nguong_chat_luong
            if lich_su_vong
            else False,
        }

        self._lich_su.append(ket_qua)
        return ket_qua

    def chay_nhieu(
        self,
        prompts: List[str],
        nhiet_do: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Chạy self-refinement cho nhiều prompts."""
        return [self.chay(p, nhiet_do) for p in prompts]

    def _sinh_ban_dau(self, prompt: str) -> str:
        """Sinh output ban đầu."""
        return self.ham_sinh(prompt)

    def _danh_gia(self, prompt: str, output: str) -> Dict[str, Any]:
        """Đánh giá output."""
        if self.ham_danh_gia:
            return self.ham_danh_gia(prompt, output)

        # Default: heuristic scoring
        diem = 0.5

        # Độ dài hợp lý
        so_tu = len(output.split())
        if 20 <= so_tu <= 500:
            diem += 0.1
        elif so_tu < 10:
            diem -= 0.2

        # Có structure không
        if "\n" in output:
            diem += 0.1

        # Không lặp lại
        cac_tu = output.split()
        if len(set(cac_tu)) / max(len(cac_tu), 1) > 0.5:
            diem += 0.1

        # Có nội dung liên quan đến prompt không
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        overlap = len(prompt_words & output_words) / max(len(prompt_words), 1)
        diem += overlap * 0.2

        return {"diem": min(diem, 1.0), "chi_tiet": {"so_tu": so_tu}}

    def _tao_feedback(
        self,
        prompt: str,
        output: str,
        danh_gia: Dict[str, Any],
    ) -> str:
        """Tạo feedback để cải thiện."""
        diem = danh_gia.get("diem", 0.0)
        chi_tiet = danh_gia.get("chi_tiet", {})

        feedback_parts = []
        if diem < 0.3:
            feedback_parts.append("Output chưa đủ chất lượng, cần cải thiện đáng kể.")
        elif diem < 0.6:
            feedback_parts.append("Output ở mức trung bình, cần cải thiện.")
        else:
            feedback_parts.append("Output khá tốt, cần tinh chỉnh thêm.")

        so_tu = chi_tiet.get("so_tu", 0)
        if so_tu < 20:
            feedback_parts.append("Output quá ngắn, cần mở rộng thêm.")
        elif so_tu > 500:
            feedback_parts.append("Output quá dài, cần ngắn gọn hơn.")

        feedback_parts.append(f"Yêu cầu gốc: {prompt}")
        feedback_parts.append(f"Điểm hiện tại: {diem:.2f}/1.0")

        return " ".join(feedback_parts)

    def _refine(
        self,
        prompt: str,
        output_cu: str,
        feedback: str,
    ) -> str:
        """Refine output dựa trên feedback."""
        refine_prompt = (
            f"Yêu cầu: {prompt}\n\n"
            f"Output hiện tại:\n{output_cu}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Hãy cải thiện output dựa trên feedback. "
            f"Trả về output đã cải thiện:"
        )
        return self.ham_refine(refine_prompt)

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử refinement."""
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        if not self._lich_su:
            return {"so_lan_chay": 0}

        so_vong_tb = sum(r["so_vong"] for r in self._lich_su) / len(self._lich_su)
        diem_tb = sum(r["diem_cuoi"] for r in self._lich_su) / len(self._lich_su)
        ty_le_dat = sum(1 for r in self._lich_su if r["dat_nguong"]) / len(self._lich_su)

        return {
            "so_lan_chay": len(self._lich_su),
            "so_vong_tb": round(so_vong_tb, 2),
            "diem_tb": round(diem_tb, 3),
            "ty_le_dat_nguong": round(ty_le_dat, 3),
            "so_vong_toi_da": self.so_vong_toi_da,
            "nguong_chat_luong": self.nguong_chat_luong,
        }

    def __repr__(self) -> str:
        return f"SelfRefine(so_vong_toi_da={self.so_vong_toi_da}, nguong={self.nguong_chat_luong})"
