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
        danh_sach_cong_cu: Optional[List[CongCu]] = None,
        max_iterations: int = 5,
        ham_xac_nhan: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        so_tay_kinh_nghiem: Optional[SoTayKinhNghiem] = None,
        graph_memory: Optional[GraphMemory] = None,
        bat_tuong_lua: bool = True,
        loc_thong_tin_nhay_cam: bool = False,
        dung_self_refine: bool = False,
        bo_refine: Optional[Any] = None,
        so_vong_refine: int = 2,
        nguong_chat_luong_refine: float = 0.8,
        dung_self_consistency: bool = False,
        bo_consistency: Optional[Any] = None,
        so_paths_consistency: int = 3,
    ):
        """
        Khởi tạo Tác tử (mặc định tích hợp khiên Tường lửa AI Đa tầng).

        Args:
            llm: Đối tượng LLM (ví dụ: VietnameseLLM) có phương thức sinh_van_ban(prompt) hoặc tương đương.
            danh_sach_cong_cu: Danh sách các đối tượng CongCu (tùy chọn).
            max_iterations: Số vòng lặp suy luận tối đa để tránh bị kẹt.
            ham_xac_nhan: Hàm callback cho Human-in-the-Loop, trả về True/False.
            so_tay_kinh_nghiem: Sổ tay ghi nhận bài học trong quá khứ.
            graph_memory: Bộ nhớ đồ thị (tùy chọn).
            bat_tuong_lua: Bật kiểm duyệt Prompt Injection, SSRF/Path Traversal trên tool và DLP đầu ra.
            loc_thong_tin_nhay_cam: Khử và ẩn danh hóa thông tin định danh cá nhân (PII) trong truy vấn đầu vào.
            dung_self_refine: Kích hoạt chu trình tự đánh giá và tinh chỉnh câu trả lời bằng SelfRefine.
            bo_refine: Đối tượng SelfRefine tùy chỉnh (nếu có).
            so_vong_refine: Số vòng tinh chỉnh tối đa khi dùng SelfRefine.
            nguong_chat_luong_refine: Điểm ngưỡng chất lượng dừng sớm trong SelfRefine.
            dung_self_consistency: Kích hoạt cơ chế lấy mẫu đa luồng và biểu quyết đa số (SelfConsistency).
            bo_consistency: Đối tượng SelfConsistency tùy chỉnh (nếu có).
            so_paths_consistency: Số lượng luồng suy luận đồng thời trong SelfConsistency.
        """
        from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

        self.llm = llm
        self.cong_cu = {cc.ten: cc for cc in (danh_sach_cong_cu or [])}
        self.max_iterations = max_iterations
        self.ham_xac_nhan = ham_xac_nhan
        self.so_tay_kinh_nghiem = so_tay_kinh_nghiem
        self.graph_memory = graph_memory
        self.bat_tuong_lua = bat_tuong_lua
        self.tuong_lua = (
            TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000) if bat_tuong_lua else None
        )
        self.loc_thong_tin_nhay_cam = loc_thong_tin_nhay_cam
        if loc_thong_tin_nhay_cam:
            from vietnamese_ai.security.data_sanitizer import DataSanitizer

            self.sanitizer = DataSanitizer
        else:
            self.sanitizer = None

        self.dung_self_refine = dung_self_refine
        self.so_vong_refine = so_vong_refine
        self.nguong_chat_luong_refine = nguong_chat_luong_refine
        self.dung_self_consistency = dung_self_consistency
        self.so_paths_consistency = so_paths_consistency

        self.lich_su_refine: List[Dict[str, Any]] = []
        self.lich_su_consistency: List[Dict[str, Any]] = []

        if bo_consistency is not None:
            self.bo_consistency = bo_consistency
        elif self.dung_self_consistency:
            from vietnamese_ai.salm.self_consistency import SelfConsistency

            self.bo_consistency = SelfConsistency(
                ham_sinh=self._goi_llm_don,
                so_luong=self.so_paths_consistency,
                ham_trich_xuat=self._trich_xuat_dap_an_consistency,
            )
        else:
            self.bo_consistency = None

        if bo_refine is not None:
            self.bo_refine = bo_refine
        elif self.dung_self_refine:
            from vietnamese_ai.salm.self_refine import SelfRefine

            self.bo_refine = SelfRefine(
                ham_sinh=self._goi_llm_don,
                so_vong_toi_da=self.so_vong_refine,
                nguong_chat_luong=self.nguong_chat_luong_refine,
            )
        else:
            self.bo_refine = None

        self._lam_moi_system_prompt()

    def _lam_moi_system_prompt(self) -> None:
        """Cập nhật lại System Prompt với danh sách công cụ mới nhất."""
        tools_desc = ""
        for name, cc in self.cong_cu.items():
            tools_desc += (
                f"- {name}: {cc.mo_ta}. Tham số: {json.dumps(cc.tham_so, ensure_ascii=False)}\n"
            )

        system_prompt = REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc)
        if not hasattr(self, "bo_nho") or self.bo_nho is None:
            self.bo_nho = BoNhoTacTu(system_prompt=system_prompt)
        else:
            self.bo_nho.system_prompt = system_prompt

    def them_cong_cu(self, cong_cu_item: CongCu) -> None:
        """Thêm một công cụ vào danh sách công cụ của Tác tử và làm mới System Prompt."""
        self.cong_cu[cong_cu_item.ten] = cong_cu_item
        self._lam_moi_system_prompt()

    def nap_plugin(self, plugin: Any) -> int:
        """
        Nạp công cụ từ Plugin (PluginCongCu hoặc QuanLyPlugin) vào Tác tử.

        Args:
            plugin: Đối tượng PluginCongCu hoặc QuanLyPlugin.

        Returns:
            Số lượng công cụ mới được nạp.
        """
        from vietnamese_ai.agents.tools import CongCu
        from vietnamese_ai.plugin.plugin_base import PluginCongCu
        from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin

        dem = 0
        if isinstance(plugin, PluginCongCu):
            for cc in plugin.dang_ky_cong_cu():
                ten = cc.get("ten")
                mo_ta = cc.get("mo_ta", "")
                ham = cc.get("ham")
                xac_nhan = cc.get("yeu_cau_xac_nhan", False)
                if ten and callable(ham):
                    self.cong_cu[ten] = CongCu(
                        ten=ten,
                        mo_ta=mo_ta,
                        ham_thuc_thi=ham,
                        yeu_cau_xac_nhan=xac_nhan,
                    )
                    dem += 1
            self._lam_moi_system_prompt()

        elif isinstance(plugin, QuanLyPlugin):
            for p_item in plugin.cac_plugin_cong_cu.values():
                dem += self.nap_plugin(p_item)

        return dem

    def nap_plugin_cong_cu(self, plugin: Any) -> int:
        """Alias cho nap_plugin."""
        return self.nap_plugin(plugin)

    def _goi_llm_don(self, prompt: str) -> str:
        """Hàm bọc gọi LLM đơn lẻ trực tiếp."""
        if hasattr(self.llm, "sinh_van_ban"):
            res = self.llm.sinh_van_ban(prompt, do_dai=256)
            return str(res)
        elif callable(self.llm):
            return str(self.llm(prompt))
        else:
            raise ValueError(
                "LLM không được hỗ trợ. Cần có hàm sinh_van_ban(prompt) hoặc là một callable."
            )

    def _trich_xuat_dap_an_consistency(self, output: str) -> str:
        """Trích xuất câu trả lời hoặc hành động để biểu quyết trong SelfConsistency."""
        pt = self._phan_tich_phan_hoi(output)
        if pt.get("tra_loi"):
            return f"TRA_LOI: {pt['tra_loi']}"
        elif pt.get("hanh_dong"):
            param_key = (
                json.dumps(pt.get("tham_so", {}), sort_keys=True, ensure_ascii=False)
                if pt.get("tham_so")
                else ""
            )
            return f"HANH_DONG: {pt['hanh_dong']} - {param_key}"
        lines = output.strip().split("\n")
        return lines[-1].strip() if lines else output.strip()

    def _goi_llm(self, prompt: str) -> str:
        """Hàm bọc gọi LLM có hỗ trợ SelfConsistency nếu được bật."""
        if self.dung_self_consistency and self.bo_consistency is not None:
            kq_sc = self.bo_consistency.chay(prompt)
            self.lich_su_consistency.append(kq_sc)
            out_best = kq_sc.get("output_tot_nhat")
            if out_best:
                return str(out_best)
            winning_ans = kq_sc.get("dap_an")
            for p in kq_sc.get("cac_paths", []):
                if p.get("dap_an") == winning_ans:
                    return str(p.get("output_day_du", winning_ans))
            return str(winning_ans or self._goi_llm_don(prompt))
        return self._goi_llm_don(prompt)

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
                    if tham_so_str.startswith("```json"):
                        tham_so_str = tham_so_str[7:]
                    if tham_so_str.endswith("```"):
                        tham_so_str = tham_so_str[:-3]
                    tham_so_str = tham_so_str.strip()

                    if tham_so_str.startswith("```"):
                        tham_so_str = tham_so_str[3:]
                    tham_so_str = tham_so_str.strip()
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

    def _thuc_thi_cong_cu(self, ten_cong_cu: str, cc: CongCu, tham_so: dict) -> str:
        """Thực thi công cụ kèm kiểm duyệt Tường lửa (chặn SSRF, Path Traversal & Indirect Prompt Injection)."""
        if self.tuong_lua is not None:
            hop_le, ly_do = self.tuong_lua.kiem_tra_lenh_cong_cu(ten_cong_cu, tham_so or {})
            if not hop_le:
                return f"Lỗi bảo mật (Tường lửa AI): {ly_do}"
        ket_qua_raw = str(cc.chay(**(tham_so or {})))
        if self.tuong_lua is not None:
            kq_sach = self.tuong_lua.lam_sach_tai_lieu_rag(ket_qua_raw)
            return kq_sach["van_ban_sach"]
        return ket_qua_raw

    def chay(self, truy_van: str) -> str:
        """
        Thực thi vòng lặp tác tử để trả lời truy vấn.
        """
        if self.sanitizer is not None:
            truy_van = self.sanitizer.lam_sach(str(truy_van))

        if self.tuong_lua is not None:
            kq_in = self.tuong_lua.kiem_tra_dau_vao(str(truy_van))
            if not kq_in["an_toan"]:
                return f"[Bị chặn bởi Tường lửa AI]: {kq_in['ly_do']}"

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
                tra_loi_cuoi = phan_tich["tra_loi"]
                if self.dung_self_refine and self.bo_refine is not None:
                    kq_refine = self.bo_refine.chay(
                        prompt=f"Câu hỏi: {truy_van}",
                        output_ban_dau=tra_loi_cuoi,
                    )
                    self.lich_su_refine.append(kq_refine)
                    tra_loi_cuoi = str(kq_refine.get("output_cuoi", tra_loi_cuoi))
                if self.tuong_lua is not None:
                    tra_loi_cuoi = self.tuong_lua.loc_dau_ra(str(tra_loi_cuoi))
                return tra_loi_cuoi

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
                            quan_sat = self._thuc_thi_cong_cu(ten_cong_cu, cc, tham_so)
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

    def tinh_chinh_cau_tra_loi(self, cau_tra_loi: str, yeu_cau: Optional[str] = None) -> str:
        """
        Sử dụng SelfRefine để đánh giá và tinh chỉnh câu trả lời.

        Args:
            cau_tra_loi: Câu trả lời dự thảo cần tinh chỉnh.
            yeu_cau: Yêu cầu hoặc truy vấn gốc (tùy chọn).

        Returns:
            Câu trả lời sau khi được tinh chỉnh và nâng cao chất lượng.
        """
        if not self.bo_refine:
            from vietnamese_ai.salm.self_refine import SelfRefine

            self.bo_refine = SelfRefine(
                ham_sinh=self._goi_llm_don,
                so_vong_toi_da=self.so_vong_refine,
                nguong_chat_luong=self.nguong_chat_luong_refine,
            )
        prompt = f"Yêu cầu: {yeu_cau}" if yeu_cau else "Hãy đánh giá và tinh chỉnh câu trả lời:"
        kq = self.bo_refine.chay(prompt=prompt, output_ban_dau=cau_tra_loi)
        self.lich_su_refine.append(kq)
        return str(kq.get("output_cuoi", cau_tra_loi))
