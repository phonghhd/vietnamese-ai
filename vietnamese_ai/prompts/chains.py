"""ChuoiPrompt - chain-of-thought và few-shot prompting."""

from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.prompts.templates import MauPrompt


class ChuoiPrompt:
    """
    Chain prompt - kết hợp nhiều prompt thành pipeline.

    Hỗ trợ:
    - Chain-of-thought (CoT) prompting
    - Few-shot prompting
    - Sequential prompt chains
    - Conditional branching

    Sử dụng:
        >>> chain = ChuoiPrompt()
        >>> chain.them_buoc("phan_tich", mau_phan_tich)
        >>> chain.them_buoc("tom_tat", mau_tom_tat)
        >>> ket_qua = chain.thuc_hien(ham_sinh, {"chu_de": "AI"})
    """

    def __init__(
        self,
        ham_sinh_mac_dinh: Optional[Callable[[str], str]] = None,
    ):
        self.ham_sinh_mac_dinh = ham_sinh_mac_dinh
        self._buoc: List[Dict[str, Any]] = []
        self._few_shot: List[Dict[str, str]] = []
        self._lich_su: List[Dict[str, Any]] = []

    def them_buoc(
        self,
        ten: str,
        mau: MauPrompt,
        dieu_kien: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> "ChuoiPrompt":
        """Thêm một bước vào chain."""
        self._buoc.append({
            "ten": ten,
            "mau": mau,
            "dieu_kien": dieu_kien,
        })
        return self

    def them_few_shot(
        self,
        dau_vao: str,
        dau_ra: str,
    ) -> "ChuoiPrompt":
        """Thêm một ví dụ few-shot."""
        self._few_shot.append({
            "dau_vao": dau_vao,
            "dau_ra": dau_ra,
        })
        return self

    def thuc_hien(
        self,
        ham_sinh: Optional[Callable[[str], str]] = None,
        bien: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thực hiện chain prompts.

        Args:
            ham_sinh: Hàm sinh text (ghi đè default)
            bien: Giá trị biến cho tất cả steps

        Returns:
            {ket_qua, buoc_thuc_hien, lich_su}
        """
        ham = ham_sinh or self.ham_sinh_mac_dinh
        if ham is None:
            raise RuntimeError("Chưa cung cấp hàm sinh (ham_sinh)")

        bien = bien or {}
        ket_qua_cuoi = ""
        buoc_thuc_hien = []

        # Few-shot prefix
        few_shot_prefix = ""
        if self._few_shot:
            few_shot_prefix = "Dưới đây là một số ví dụ:\n\n"
            for i, vi_du in enumerate(self._few_shot, 1):
                few_shot_prefix += f"Ví dụ {i}:\n"
                few_shot_prefix += f"Đầu vào: {vi_du['dau_vao']}\n"
                few_shot_prefix += f"Đầu ra: {vi_du['dau_ra']}\n\n"

        for buoc in self._buoc:
            # Kiểm tra điều kiện
            if buoc["dieu_kien"] is not None:
                if not buoc["dieu_kien"]({**bien, "ket_qua_truoc": ket_qua_cuoi}):
                    continue

            # Render prompt
            them_bien = {
                **bien,
                "ket_qua_truoc": ket_qua_cuoi,
                "few_shot": few_shot_prefix,
            }

            try:
                prompt = buoc["mau"].render(**them_bien)
            except ValueError:
                prompt = buoc["mau"].render(**bien)

            # Thêm few-shot vào đầu prompt nếu có
            if few_shot_prefix and "few_shot" not in buoc["mau"]._bien:
                prompt = few_shot_prefix + prompt

            # Gọi hàm sinh
            ket_qua = ham(prompt)

            buoc_thuc_hien.append({
                "ten": buoc["ten"],
                "prompt": prompt,
                "ket_qua": ket_qua,
            })

            ket_qua_cuoi = ket_qua
            few_shot_prefix = ""  # Chỉ dùng few-shot ở buoc đầu

        ket_qua_obj = {
            "ket_qua": ket_qua_cuoi,
            "buoc_thuc_hien": buoc_thuc_hien,
            "so_buoc": len(buoc_thuc_hien),
        }

        self._lich_su.append(ket_qua_obj)
        return ket_qua_obj

    def tao_cot_prompt(
        self,
        cau_hoi: str,
        ngu_canh: str = "",
    ) -> str:
        """
        Tạo chain-of-thought prompt.

        Args:
            cau_hoi: Câu hỏi
            ngu_canh: Ngữ cảnh

        Returns:
            CoT prompt
        """
        prompt = f"Câu hỏi: {cau_hoi}\n\n"
        if ngu_canh:
            prompt += f"Ngữ cảnh: {ngu_canh}\n\n"
        prompt += (
            "Hãy suy nghĩ từng bước:\n"
            "Bước 1: Xác định vấn đề chính\n"
            "Bước 2: Phân tích các yếu tố liên quan\n"
            "Bước 3: Đưa ra kết luận\n\n"
            "Suy nghĩ:"
        )
        return prompt

    def tao_few_shot_prompt(
        self,
        cau_hoi: str,
        vi_du: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Tạo few-shot prompt."""
        prompt = ""
        vi_du = vi_du or self._few_shot

        for i, vd in enumerate(vi_du, 1):
            prompt += f"Ví dụ {i}:\n"
            prompt += f"Câu hỏi: {vd['dau_vao']}\n"
            prompt += f"Trả lời: {vd['dau_ra']}\n\n"

        prompt += f"Câu hỏi: {cau_hoi}\n"
        prompt += "Trả lời:"
        return prompt

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử thực hiện."""
        return self._lich_su.copy()

    def __repr__(self) -> str:
        return (
            f"ChuoiPrompt(so_buoc={len(self._buoc)}, "
            f"so_few_shot={len(self._few_shot)})"
        )
