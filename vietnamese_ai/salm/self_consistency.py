"""SelfConsistency - nhiều reasoning paths với majority voting."""

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class SelfConsistency:
    """
    Self-Consistency: tạo nhiều reasoning paths rồi majority vote.

    Dựa trên paper "Self-Consistency Improves Chain of Thought Reasoning" (2022).
    Thay vì 1 path, tạo N paths khác nhau (temperature cao) rồi chọn answer
    xuất hiện nhiều nhất.

    Sử dụng:
        >>> sc = SelfConsistency(ham_sinh=generate_fn, so_luong=5)
        >>> ket_qua = sc.chay("2 + 2 = ?", che_do="cot")
        >>> print(ket_qua["dap_an"])
    """

    def __init__(
        self,
        ham_sinh: Callable[[str], str],
        so_luong: int = 5,
        ham_trich_xuat: Optional[Callable[[str], str]] = None,
        ham_danh_gia: Optional[Callable[[str, str, str], float]] = None,
    ):
        self.ham_sinh = ham_sinh
        self.so_luong = so_luong
        self.ham_trich_xuat = ham_trich_xuat
        self.ham_danh_gia = ham_danh_gia
        self.logger = Logger("SelfConsistency")

        self._lich_su: List[Dict[str, Any]] = []

    def chay(
        self,
        prompt: str,
        che_do: str = "truc_tiep",
        nhiet_do: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chạy self-consistency.

        Args:
            prompt: Câu hỏi
            che_do: "truc_tiep" (direct) hoặc "cot" (chain-of-thought)
            nhiet_do: Nhiệt độ (cao = đa dạng hơn)

        Returns:
            {dap_an, so_luong_paths, ty_le_dong_nhat, cac_paths, diem}
        """
        bat_dau = time.time()
        cac_paths = []

        for i in range(self.so_luong):
            if che_do == "cot":
                full_prompt = (
                    f"Câu hỏi: {prompt}\n\n"
                    f"Hãy suy nghĩ từng bước rồi trả lời:\n"
                    f"Bước 1:"
                )
            else:
                full_prompt = prompt

            try:
                output = self.ham_sinh(full_prompt)
            except Exception as e:
                self.logger.warning(f"Path {i} lỗi: {e}")
                continue

            # Trích xuất câu trả lời cuối
            dap_an = self._trich_xuat_dap_an(output)

            cac_paths.append({
                "stt": i,
                "output_day_du": output,
                "dap_an": dap_an,
            })

        if not cac_paths:
            return {
                "dap_an": None,
                "so_luong_paths": 0,
                "ty_le_dong_nhat": 0.0,
                "cac_paths": [],
                "diem": 0.0,
            }

        # Majority voting
        dap_an_list = [p["dap_an"] for p in cac_paths if p["dap_an"]]
        dem = Counter(dap_an_list)

        if dem:
            dap_an_tot_nhat, so_lan_chon = dem.most_common(1)[0]
            ty_le_dong_nhat = so_lan_chon / len(dap_an_list)
        else:
            dap_an_tot_nhat = cac_paths[0]["dap_an"]
            ty_le_dong_nhat = 1.0 / len(cac_paths)

        # Nếu có evaluation function, score từng answer
        diem = ty_le_dong_nhat
        if self.ham_danh_gia and dap_an_tot_nhat:
            scores = []
            for p in cac_paths:
                try:
                    s = self.ham_danh_gia(prompt, p["output_day_du"], dap_an_tot_nhat)
                    scores.append(s)
                except Exception:
                    pass
            if scores:
                diem = (ty_le_dong_nhat + np.mean(scores)) / 2

        thoi_gian = time.time() - bat_dau

        ket_qua = {
            "dap_an": dap_an_tot_nhat,
            "so_luong_paths": len(cac_paths),
            "ty_le_dong_nhat": round(ty_le_dong_nhat, 3),
            "cac_paths": cac_paths,
            "phan_phoi": dict(dem.most_common()),
            "diem": round(diem, 3),
            "thoi_gian": round(thoi_gian, 3),
            "che_do": che_do,
        }

        self._lich_su.append(ket_qua)
        return ket_qua

    def chay_nhieu(
        self,
        prompts: List[str],
        che_do: str = "truc_tiep",
    ) -> List[Dict[str, Any]]:
        """Chạy self-consistency cho nhiều câu hỏi."""
        return [self.chay(p, che_do) for p in prompts]

    def _trich_xuat_dap_an(self, output: str) -> str:
        """Trích xuất câu trả lời cuối cùng từ output."""
        if self.ham_trich_xuat:
            return self.ham_trich_xuat(output)

        # Default extraction
        lines = output.strip().split("\n")
        # Tìm dòng cuối có nội dung
        for line in reversed(lines):
            line = line.strip()
            if line:
                # Bỏ prefix "Trả lời:", "Answer:", etc.
                for prefix in ["trả lời:", "answer:", "kết quả:", "đáp án:"]:
                    if line.lower().startswith(prefix):
                        return line[len(prefix):].strip()
                return line

        return output.strip()

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử."""
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        if not self._lich_su:
            return {"so_lan_chay": 0}

        dong_nhat_tb = sum(r["ty_le_dong_nhat"] for r in self._lich_su) / len(self._lich_su)
        diem_tb = sum(r["diem"] for r in self._lich_su) / len(self._lich_su)

        return {
            "so_lan_chay": len(self._lich_su),
            "dong_nhat_tb": round(dong_nhat_tb, 3),
            "diem_tb": round(diem_tb, 3),
            "so_luong_paths": self.so_luong,
        }

    def __repr__(self) -> str:
        return f"SelfConsistency(so_luong={self.so_luong})"
