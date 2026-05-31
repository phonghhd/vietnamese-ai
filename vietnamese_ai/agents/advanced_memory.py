"""Advanced Memory - Bộ nhớ nâng cao cho Tác tử."""

from typing import Any, Dict, List


# Giả sử chúng ta có import BaseMemory từ memory.py
# (do kiến trúc hiện tại, ta định nghĩa luôn)
class WindowMemory:
    """Bộ nhớ chỉ giữ lại k tin nhắn gần nhất."""

    def __init__(self, k: int = 10):
        self.k = k
        self.tin_nhan: List[Dict[str, str]] = []

    def them(self, vai_tro: str, noi_dung: str) -> None:
        self.tin_nhan.append({"vai_tro": vai_tro, "noi_dung": noi_dung})
        # Giữ lại k tin nhắn cuối
        if len(self.tin_nhan) > self.k:
            self.tin_nhan = self.tin_nhan[-self.k :]

    def lay_lich_su(self) -> str:
        lich_su = ""
        for tn in self.tin_nhan:
            vai_tro = "Tác tử" if tn["vai_tro"] == "tac_tu" else "Người dùng"
            lich_su += f"{vai_tro}: {tn['noi_dung']}\n"
        return lich_su.strip()

    def lam_sach(self) -> None:
        self.tin_nhan.clear()


class SummaryMemory:
    """Bộ nhớ sử dụng LLM để tóm tắt các cuộc trò chuyện cũ."""

    def __init__(self, llm, so_luong_truoc_khi_tom_tat: int = 5):
        self.llm = llm
        self.so_luong = so_luong_truoc_khi_tom_tat
        self.tom_tat_hien_tai: str = ""
        self.tin_nhan_tam: List[Dict[str, str]] = []

    def them(self, vai_tro: str, noi_dung: str) -> None:
        self.tin_nhan_tam.append({"vai_tro": vai_tro, "noi_dung": noi_dung})

        if len(self.tin_nhan_tam) >= self.so_luong * 2:  # Một vòng QA là 2 tin nhắn
            self._tom_tat_lich_su()

    def _tom_tat_lich_su(self):
        """Gọi LLM để tóm tắt."""
        if not hasattr(self.llm, "sinh_van_ban"):
            return  # Fallback nếu LLM không hợp lệ

        lich_su_moi = ""
        for tn in self.tin_nhan_tam:
            vai_tro = "Tác tử" if tn["vai_tro"] == "tac_tu" else "Người dùng"
            lich_su_moi += f"{vai_tro}: {tn['noi_dung']}\n"

        prompt = (
            f"Tóm tắt hiện tại: {self.tom_tat_hien_tai}\n\n"
            f"Cuộc hội thoại mới:\n{lich_su_moi}\n\n"
            "Hãy cập nhật tóm tắt hiện tại một cách ngắn gọn, không bỏ sót thông tin quan trọng."
        )

        try:
            self.tom_tat_hien_tai = self.llm.sinh_van_ban(prompt)
            # Xóa tạm, chỉ giữ lại tóm tắt
            self.tin_nhan_tam.clear()
        except Exception:
            pass  # Bỏ qua nếu lỗi

    def lay_lich_su(self) -> str:
        ket_qua = f"[TÓM TẮT LỊCH SỬ]: {self.tom_tat_hien_tai}\n" if self.tom_tat_hien_tai else ""
        for tn in self.tin_nhan_tam:
            vai_tro = "Tác tử" if tn["vai_tro"] == "tac_tu" else "Người dùng"
            ket_qua += f"{vai_tro}: {tn['noi_dung']}\n"
        return ket_qua.strip()

    def lam_sach(self) -> None:
        self.tom_tat_hien_tai = ""
        self.tin_nhan_tam.clear()


class GraphMemory:
    """
    Bộ nhớ Đồ thị (Knowledge Graph Memory).
    Sử dụng LLM để trích xuất các bộ ba (Chủ thể, Quan hệ, Đối tượng) từ lịch sử trò chuyện
    và lưu vào NetworkXStore. Quá trình trích xuất chạy ngầm để không ảnh hưởng tốc độ phản hồi.
    """

    def __init__(self, llm: Any, graph_store: Any):
        """
        Khởi tạo GraphMemory.

        Args:
            llm: Tác tử ngôn ngữ có hàm `sinh_van_ban`.
            graph_store: Đối tượng NetworkXStore để lưu trữ đồ thị.
        """
        self.llm = llm
        self.store = graph_store

        # Dùng ThreadPool để chạy các tác vụ trích xuất đồ thị ngầm
        import concurrent.futures

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def them(self, vai_tro: str, noi_dung: str) -> None:
        """Thêm tin nhắn mới vào đồ thị (Chỉ phân tích tin nhắn của user để lấy kiến thức)."""
        if vai_tro != "user":
            return

        self._executor.submit(self._trich_xuat_triplets_ngam, noi_dung)

    def _trich_xuat_triplets_ngam(self, noi_dung: str) -> None:
        """Hàm chạy ngầm để trích xuất và cập nhật đồ thị."""
        if not hasattr(self.llm, "sinh_van_ban"):
            return

        prompt = (
            "Bạn là một hệ thống trích xuất thông tin. Hãy phân tích câu nói sau và trích xuất "
            "các sự kiện dưới dạng danh sách bộ ba [Chủ thể, Quan hệ, Đối tượng].\n"
            "Chỉ trả về danh sách, mỗi bộ ba trên 1 dòng cách nhau bằng dấu phẩy. Không giải thích thêm.\n"
            "Ví dụ: Nếu câu là 'Dự án EvoNet sử dụng ZKP bảo mật', trả về:\n"
            "Dự án EvoNet, sử dụng, ZKP bảo mật\n\n"
            f"Câu nói: {noi_dung}"
        )
        try:
            ket_qua = self.llm.sinh_van_ban(prompt, do_dai=256)
            for dong in ket_qua.strip().split("\n"):
                parts = dong.split(",")
                if len(parts) == 3:
                    chu_the, quan_he, doi_tuong = [p.strip() for p in parts]
                    if chu_the and quan_he and doi_tuong:
                        self.store.them_bo_ba(chu_the, quan_he, doi_tuong)
        except Exception as e:
            print(f"[GraphMemory] Lỗi trích xuất đồ thị: {str(e)}")

    def lay_ngu_canh(self, cau_hoi: str) -> str:
        """Lấy ngữ cảnh từ đồ thị dựa trên câu hỏi hiện tại."""
        if not hasattr(self.llm, "sinh_van_ban"):
            return ""

        prompt = (
            f"Trích xuất các thực thể danh từ quan trọng nhất từ câu hỏi sau. "
            f"Chỉ trả về các danh từ phân tách bằng dấu phẩy, không giải thích.\n"
            f"Câu hỏi: {cau_hoi}"
        )
        try:
            ket_qua = self.llm.sinh_van_ban(prompt, do_dai=128)
            thuc_the_list = [e.strip() for e in ket_qua.split(",")]

            bo_ba_ngu_canh = []
            for tt in thuc_the_list:
                if tt:
                    # Tra cứu các mối quan hệ lân cận với độ sâu 2
                    bo_ba_ngu_canh.extend(self.store.lay_vung_lan_can(tt, do_sau=2))

            # Loại bỏ trùng lặp
            bo_ba_ngu_canh = list(set(bo_ba_ngu_canh))

            if not bo_ba_ngu_canh:
                return ""

            ngu_canh = "THÔNG TIN TỪ TRÍ NHỚ (Đồ Thị Kiến Thức Của Bạn):\n"
            for s, r, o in bo_ba_ngu_canh:
                ngu_canh += f"- {s} {r} {o}\n"

            return ngu_canh
        except Exception:
            return ""

    def lam_sach(self) -> None:
        """Xóa đồ thị hiện tại (thực chất là làm mới đối tượng đồ thị nếu cần)."""
        import networkx as nx

        self.store.graph = nx.DiGraph()
