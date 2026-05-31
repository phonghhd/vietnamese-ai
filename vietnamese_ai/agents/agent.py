import json
import re
from typing import Any, Callable, Dict, List, Optional

from .advanced_memory import GraphMemory
from .experience_memory import SoTayKinhNghiem
from .memory import BoNhoTacTu
from .tools import CongCu

# Template hệ thống cơ bản cho Tác tử (ReAct)
REACT_SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh có khả năng sử dụng công cụ để giải quyết vấn đề.
Dưới đây là danh sách các công cụ bạn có thể sử dụng:

{tools_desc}

Bạn PHẢI phản hồi theo định dạng sau:

Suy nghĩ: [Phân tích từng bước những gì bạn cần làm]
Hành động: [Tên của công cụ cần gọi, phải nằm trong danh sách ở trên. Nếu không cần công cụ, bỏ trống phần này]
Tham số: [Tham số truyền vào cho công cụ dưới dạng JSON hợp lệ. Nếu không gọi công cụ, bỏ trống phần này]

Sau khi bạn trả về "Hành động" và "Tham số", hệ thống sẽ thực thi và trả lại cho bạn "Quan sát".
Khi bạn đã có đủ thông tin để trả lời câu hỏi của người dùng, hãy kết thúc bằng:

Suy nghĩ: Tôi đã có đủ thông tin.
Trả lời: [Câu trả lời cuối cùng cho người dùng]

LƯU Ý: Mỗi lượt bạn chỉ được gọi MỘT công cụ. Nếu bạn đã có câu trả lời cuối cùng, KHÔNG trả về "Hành động" và "Tham số".
"""


class TacTu:
    """
    Tác tử (Agent) có khả năng lập kế hoạch và sử dụng công cụ.
    """

    def __init__(
        self,
        llm: Any,
        danh_sach_cong_cu: List[CongCu],
        max_iterations: int = 5,
        ham_xac_nhan: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        so_tay_kinh_nghiem: Optional[SoTayKinhNghiem] = None,
        graph_memory: Optional[GraphMemory] = None,
    ):
        """
        Khởi tạo Tác tử.

        Args:
            llm: Đối tượng LLM (ví dụ: VietnameseLLM) có phương thức sinh_van_ban(prompt) hoặc tương đương.
            danh_sach_cong_cu: Danh sách các đối tượng CongCu.
            max_iterations: Số vòng lặp suy luận tối đa để tránh bị kẹt.
            ham_xac_nhan: Hàm callback cho Human-in-the-Loop, trả về True/False.
            so_tay_kinh_nghiem: Sổ tay ghi nhận bài học trong quá khứ.
        """
        self.llm = llm
        self.cong_cu = {cc.ten: cc for cc in danh_sach_cong_cu}
        self.max_iterations = max_iterations
        self.ham_xac_nhan = ham_xac_nhan
        self.so_tay_kinh_nghiem = so_tay_kinh_nghiem
        self.graph_memory = graph_memory

        tools_desc = ""
        for name, cc in self.cong_cu.items():
            tools_desc += (
                f"- {name}: {cc.mo_ta}. Tham số: {json.dumps(cc.tham_so, ensure_ascii=False)}\n"
            )

        system_prompt = REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc)
        self.bo_nho = BoNhoTacTu(system_prompt=system_prompt)

    def _goi_llm(self, prompt: str) -> str:
        """Hàm bọc gọi LLM. Tùy biến theo LLM cụ thể."""
        if hasattr(self.llm, "sinh_van_ban"):
            # Sử dụng VietnameseLLM
            # Tránh lặp vô hạn bằng stop_words nếu LLM hỗ trợ, ở đây ta giả sử nó trả về str
            res = self.llm.sinh_van_ban(prompt, do_dai=256)
            return res
        elif callable(self.llm):
            return self.llm(prompt)
        else:
            raise ValueError(
                "LLM không được hỗ trợ. Cần có hàm sinh_van_ban(prompt) hoặc là một callable."
            )

    def _phan_tich_phan_hoi(self, response_text: str) -> Dict[str, Any]:
        """Phân tích văn bản phản hồi từ LLM để lấy Action và Action Input."""
        ket_qua = {"suy_nghi": "", "hanh_dong": None, "tham_so": None, "tra_loi": None}

        # Tìm Suy nghĩ
        match_thought = re.search(
            r"Suy nghĩ:(.*?)(?=Hành động:|Trả lời:|$)", response_text, re.DOTALL
        )
        if match_thought:
            ket_qua["suy_nghi"] = match_thought.group(1).strip()

        # Tìm Hành động và Tham số
        match_action = re.search(
            r"Hành động:(.*?)(?=Tham số:|Trả lời:|$)", response_text, re.DOTALL
        )
        match_input = re.search(r"Tham số:(.*?)(?=Trả lời:|$)", response_text, re.DOTALL)

        if match_action and match_action.group(1).strip():
            ket_qua["hanh_dong"] = match_action.group(1).strip()
            if match_input:
                try:
                    tham_so_str = match_input.group(1).strip()
                    # Cố gắng loại bỏ markdown code block nếu có
                    tham_so_str = tham_so_str.removeprefix("```json").removesuffix("```").strip()
                    tham_so_str = tham_so_str.removeprefix("```").strip()
                    ket_qua["tham_so"] = json.loads(tham_so_str)
                except json.JSONDecodeError:
                    ket_qua["tham_so"] = {
                        "error": "Lỗi định dạng JSON",
                        "raw": match_input.group(1).strip(),
                    }

        # Tìm Trả lời (kết thúc)
        match_answer = re.search(r"Trả lời:(.*?)$", response_text, re.DOTALL)
        if match_answer:
            ket_qua["tra_loi"] = match_answer.group(1).strip()

        # Fallback: nếu không thấy "Trả lời:" nhưng cũng không có "Hành động:" -> coi như trả lời
        if not ket_qua["hanh_dong"] and not ket_qua["tra_loi"]:
            ket_qua["tra_loi"] = response_text.strip()

        return ket_qua

    def chay(self, truy_van: str) -> str:
        """
        Thực thi vòng lặp tác tử để trả lời truy vấn.
        """
        self.bo_nho.them_tin_nhan("user", truy_van)

        # Tiêm kiến thức từ GraphMemory
        if getattr(self, "graph_memory", None):
            self.graph_memory.them("user", truy_van)
            ngu_canh_do_thi = self.graph_memory.lay_ngu_canh(truy_van)
            if ngu_canh_do_thi:
                self.bo_nho.them_tin_nhan("system", ngu_canh_do_thi)

        # Tiêm kinh nghiệm từ quá khứ vào trước khi suy luận
        if self.so_tay_kinh_nghiem:
            kinh_nghiem = self.so_tay_kinh_nghiem.truy_xuat_kinh_nghiem(truy_van)
            if kinh_nghiem:
                self.bo_nho.them_tin_nhan(
                    "system", f"BÀI HỌC KINH NGHIỆM TỪ QUÁ KHỨ (Cần lưu ý):\n{kinh_nghiem}"
                )

        for i in range(self.max_iterations):
            prompt_hien_tai = self.bo_nho.lay_noi_dung_chuoi()
            phan_hoi = self._goi_llm(prompt_hien_tai)

            self.bo_nho.them_tin_nhan("assistant", phan_hoi)

            phan_tich = self._phan_tich_phan_hoi(phan_hoi)

            # Đã có câu trả lời cuối cùng
            if phan_tich["tra_loi"]:
                return phan_tich["tra_loi"]

            # Cần gọi công cụ
            ten_cong_cu = phan_tich["hanh_dong"]
            tham_so = phan_tich["tham_so"]

            if ten_cong_cu:
                if ten_cong_cu in self.cong_cu:
                    cc = self.cong_cu[ten_cong_cu]
                    # Nếu lỗi parse JSON tham số
                    if isinstance(tham_so, dict) and "error" in tham_so:
                        quan_sat = f"Lỗi: Không thể phân tích JSON tham số. {tham_so['raw']}"
                    else:
                        # Kiểm tra Human-in-the-Loop (HITL)
                        duoc_phep = True
                        if cc.yeu_cau_xac_nhan and self.ham_xac_nhan:
                            duoc_phep = self.ham_xac_nhan(ten_cong_cu, tham_so or {})

                        if duoc_phep:
                            quan_sat = str(cc.chay(**(tham_so or {})))
                        else:
                            quan_sat = "Lỗi: Con người đã TỪ CHỐI hành động này. Hãy suy nghĩ cách khác an toàn hơn hoặc hỏi lại người dùng."
                else:
                    quan_sat = f"Lỗi: Công cụ '{ten_cong_cu}' không tồn tại."

                # Thêm quan sát vào bộ nhớ để LLM tiếp tục xử lý ở vòng lặp sau
                # Định dạng là User message hoặc Tool message tuỳ prompt. ReAct thường dùng format "Quan sát: "
                self.bo_nho.them_tin_nhan("tool", f"Quan sát: {quan_sat}", ten_cong_cu=ten_cong_cu)

                # Self-Correction Loop
                if str(quan_sat).startswith("Lỗi"):
                    self.bo_nho.them_tin_nhan(
                        "system",
                        "HỆ THỐNG (Tự Sửa Lỗi): Lần gọi công cụ vừa rồi thất bại. Hãy phân tích kỹ lý do lỗi trong phần 'Suy nghĩ' tiếp theo và tìm cách sửa tham số hoặc gọi công cụ khác. KHÔNG lặp lại hành động cũ gây lỗi.",
                    )
            else:
                # LLM không trả về hành động cũng không trả về câu trả lời hợp lệ
                self.bo_nho.them_tin_nhan(
                    "user",
                    "Lỗi: Bạn chưa cung cấp 'Trả lời:' hoặc 'Hành động:'. Vui lòng thử lại theo đúng định dạng.",
                )

        return "Lỗi: Tác tử đã đạt đến số vòng lặp tối đa mà không tìm được câu trả lời."
