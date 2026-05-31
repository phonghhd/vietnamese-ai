"""SinhDuLieuTuDong - Self-Generated Training Data."""

import time
from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class SinhDuLieuTuDong:
    """
    Sinh dữ liệu huấn luyện tự động từ model.

    Model tự tạo training data để adapt cho task mới. Dựa trên:
    - Self-Instruct (Wang et al., 2023)
    - Evol-Instruct (Xu et al., 2023)
    - Magpie (Zhang et al., 2024)

    Flow: Seed examples → Generate variations → Filter quality → Use for training

    Sử dụng:
        >>> sinh = SinhDuLieuTuDong(ham_sinh=generate_fn)
        >>> sinh.them_giong_mau("Tóm tắt văn bản về AI", "AI đang thay đổi...")
        >>> du_lieu = sinh.sinh(50, loai="instruction")
    """

    def __init__(
        self,
        ham_sinh: Callable[[str], str],
        ham_danh_gia: Optional[Callable[[Dict[str, str]], float]] = None,
        nguong_chat_luong: float = 0.5,
        toi_da_lap: int = 3,
    ):
        self.ham_sinh = ham_sinh
        self.ham_danh_gia = ham_danh_gia
        self.nguong_chat_luong = nguong_chat_luong
        self.toi_da_lap = toi_da_lap
        self.logger = Logger("SinhDuLieuTuDong")

        self._giong_mau: List[Dict[str, str]] = []
        self._du_lieu_da_sinh: List[Dict[str, str]] = []
        self._lich_su: List[Dict[str, Any]] = []

    def them_giong_mau(
        self,
        instruction: str,
        output: str,
        input_text: str = "",
    ) -> None:
        """Thêm seed example."""
        self._giong_mau.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        )

    def them_nhieu_giong_mau(self, mau_list: List[Dict[str, str]]) -> None:
        """Thêm nhiều seed examples."""
        for mau in mau_list:
            self.them_giong_mau(
                mau.get("instruction", ""),
                mau.get("output", ""),
                mau.get("input", ""),
            )

    def sinh(
        self,
        so_luong: int,
        loai: str = "instruction",
        chu_de: str = "",
    ) -> List[Dict[str, str]]:
        """
        Sinh dữ liệu huấn luyện.

        Args:
            so_luong: Số mẫu cần sinh
            loai: "instruction", "qa", "classification", "completion"
            chu_de: Chủ đề cụ thể (tùy chọn)

        Returns:
            [{instruction, input, output, diem}, ...]
        """
        bat_dau = time.time()
        du_lieu = []

        if not self._giong_mau:
            self.logger.warning("Chưa có seed examples, tạo mẫu mặc định")
            self._tao_giong_mac_dinh(loai)

        for i in range(so_luong):
            for lap in range(self.toi_da_lap):
                mau = self._sinh_mau(loai, chu_de, i)

                # Đánh giá chất lượng
                diem = self._danh_gia_mau(mau)

                if diem >= self.nguong_chat_luong:
                    mau["diem"] = round(diem, 3)
                    du_lieu.append(mau)
                    break
                else:
                    self.logger.debug(
                        f"Mẫu {i} vòng {lap}: điểm {diem:.3f} < {self.nguong_chat_luong}"
                    )

        self._du_lieu_da_sinh.extend(du_lieu)

        thoi_gian = time.time() - bat_dau

        ket_qua = {
            "so_luong_yeu_cau": so_luong,
            "so_luong_sinh": len(du_lieu),
            "ty_le_thanh_cong": len(du_lieu) / max(so_luong, 1),
            "thoi_gian": round(thoi_gian, 3),
        }
        self._lich_su.append(ket_qua)

        self.logger.info(f"Sinh {len(du_lieu)}/{so_luong} mẫu ({loai}), thời gian={thoi_gian:.1f}s")

        return du_lieu

    def _sinh_mau(
        self,
        loai: str,
        chu_de: str,
        chi_so: int,
    ) -> Dict[str, str]:
        """Sinh một mẫu dữ liệu."""
        # Chọn seed ngẫu nhiên
        seed_idx = chi_so % len(self._giong_mau)
        seed = self._giong_mau[seed_idx]

        if loai == "instruction":
            return self._sinh_instruction(seed, chu_de, chi_so)
        elif loai == "qa":
            return self._sinh_qa(seed, chu_de, chi_so)
        elif loai == "classification":
            return self._sinh_classification(seed, chu_de, chi_so)
        else:
            return self._sinh_completion(seed, chu_de, chi_so)

    def _sinh_instruction(
        self,
        seed: Dict[str, str],
        chu_de: str,
        chi_so: int,
    ) -> Dict[str, str]:
        """Sinh instruction-following data."""
        prompt = (
            f"Tạo một ví dụ mới tương tự nhưng khác biệt.\n\n"
            f"Ví dụ mẫu:\n"
            f"Instruction: {seed['instruction']}\n"
            f"Input: {seed.get('input', '')}\n"
            f"Output: {seed['output']}\n\n"
        )
        if chu_de:
            prompt += f"Chủ đề: {chu_de}\n\n"
        prompt += "Tạo instruction mới (chỉ trả về instruction):"

        instruction = self.ham_sinh(prompt).strip()

        # Generate output cho instruction mới
        output_prompt = f"Thực hiện instruction sau:\n{instruction}\n\nOutput:"
        output = self.ham_sinh(output_prompt).strip()

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    def _sinh_qa(
        self,
        seed: Dict[str, str],
        chu_de: str,
        chi_so: int,
    ) -> Dict[str, str]:
        """Sinh QA data."""
        prompt = (
            f"Dựa trên ví dụ, tạo câu hỏi và trả lời mới.\n\n"
            f"Ví dụ:\nQ: {seed['instruction']}\nA: {seed['output']}\n\n"
        )
        if chu_de:
            prompt += f"Chủ đề: {chu_de}\n\n"
        prompt += "Tạo Q&A mới:\nQ:"

        qa_text = self.ham_sinh(prompt).strip()
        parts = qa_text.split("\nA:")

        if len(parts) >= 2:
            return {"instruction": parts[0].strip(), "input": "", "output": parts[1].strip()}
        return {"instruction": qa_text, "input": "", "output": ""}

    def _sinh_classification(
        self,
        seed: Dict[str, str],
        chu_de: str,
        chi_so: int,
    ) -> Dict[str, str]:
        """Sinh classification data."""
        prompt = (
            f"Tạo một câu văn bản mới và nhãn phân loại.\n\n"
            f"Ví dụ:\nText: {seed['instruction']}\nLabel: {seed['output']}\n\n"
            f"Tạo mới:\nText:"
        )
        result = self.ham_sinh(prompt).strip()
        parts = result.split("\nLabel:")

        if len(parts) >= 2:
            return {"instruction": parts[0].strip(), "input": "", "output": parts[1].strip()}
        return {"instruction": result, "input": "", "output": "unknown"}

    def _sinh_completion(
        self,
        seed: Dict[str, str],
        chu_de: str,
        chi_so: int,
    ) -> Dict[str, str]:
        """Sinh completion data."""
        prompt = (
            f"Tạo một đoạn text开头 mới.\n\n"
            f"Ví dụ:\nĐầu vào: {seed['instruction']}\n"
            f"Tiếp theo: {seed['output']}\n\n"
            f"Tạo mới:\nĐầu vào:"
        )
        result = self.ham_sinh(prompt).strip()
        parts = result.split("\nTiếp theo:")

        if len(parts) >= 2:
            return {"instruction": parts[0].strip(), "input": "", "output": parts[1].strip()}
        return {"instruction": result, "input": "", "output": ""}

    def _danh_gia_mau(self, mau: Dict[str, str]) -> float:
        """Đánh giá chất lượng mẫu."""
        if self.ham_danh_gia:
            return self.ham_danh_gia(mau)

        diem = 0.5

        # Không rỗng
        if not mau.get("instruction") or not mau.get("output"):
            return 0.0

        # Độ dài hợp lý
        if 5 <= len(mau["instruction"].split()) <= 100:
            diem += 0.15
        if 5 <= len(mau["output"].split()) <= 500:
            diem += 0.15

        # Đa dạng (không trùng lặp với mẫu đã có)
        for da_sinh in self._du_lieu_da_sinh[-20:]:
            if mau["instruction"].lower() == da_sinh.get("instruction", "").lower():
                diem -= 0.3
                break

        return min(max(diem, 0.0), 1.0)

    def _tao_giong_mac_dinh(self, loai: str) -> None:
        """Tạo seed examples mặc định."""
        if loai == "instruction":
            self._giong_mau = [
                {
                    "instruction": "Tóm tắt văn bản về trí tuệ nhân tạo",
                    "input": "",
                    "output": "Trí tuệ nhân tạo (AI) là ngành khoa học máy tính...",
                },
                {
                    "instruction": "Dịch câu sau sang tiếng Anh: Xin chào thế giới",
                    "input": "",
                    "output": "Hello world",
                },
                {
                    "instruction": "Phân tích ưu nhược điểm của học sâu",
                    "input": "",
                    "output": "Ưu điểm: khả năng học features tự động...",
                },
            ]
        elif loai == "qa":
            self._giong_mau = [
                {
                    "instruction": "AI là gì?",
                    "input": "",
                    "output": "AI là trí tuệ nhân tạo, ngành khoa học...",
                },
                {
                    "instruction": "Machine learning hoạt động như thế nào?",
                    "input": "",
                    "output": "Machine learning hoạt động bằng cách học patterns...",
                },
            ]
        else:
            self._giong_mau = [
                {"instruction": "Ví dụ mẫu", "input": "", "output": "Kết quả mẫu"},
            ]

    def lay_du_lieu_da_sinh(self) -> List[Dict[str, str]]:
        """Lấy tất cả dữ liệu đã sinh."""
        return self._du_lieu_da_sinh.copy()

    def xoa_du_lieu(self) -> None:
        """Xóa dữ liệu đã sinh."""
        self._du_lieu_da_sinh.clear()

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử sinh."""
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_giong_mau": len(self._giong_mau),
            "so_du_lieu_da_sinh": len(self._du_lieu_da_sinh),
            "nguong_chat_luong": self.nguong_chat_luong,
            "so_lan_sinh": len(self._lich_su),
        }

    def __repr__(self) -> str:
        return (
            f"SinhDuLieuTuDong(so_giong={len(self._giong_mau)}, "
            f"da_sinh={len(self._du_lieu_da_sinh)})"
        )
