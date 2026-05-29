import re
from typing import Any, List

from .graph_store import NetworkXStore


class GraphRetriever:
    """
    Truy xuất thông tin từ GraphStore dựa trên câu hỏi của người dùng.
    """
    def __init__(self, graph_store: NetworkXStore, llm: Any = None):
        self.store = graph_store
        self.llm = llm

    def _trich_xuat_thuc_the_cau_hoi(self, cau_hoi: str) -> List[str]:
        """Xác định các thực thể chính trong câu hỏi."""
        if self.llm and hasattr(self.llm, "sinh_van_ban"):
            prompt = f"Xác định các thực thể chính (danh từ riêng, tên gọi, khái niệm) trong câu hỏi sau. Trả về danh sách phân tách bằng dấu phẩy. Không giải thích gì thêm.\nCâu hỏi: {cau_hoi}"
            phan_hoi = self.llm.sinh_van_ban(prompt, do_dai=128)
            thuc_the = [e.strip().lower() for e in phan_hoi.split(',')]
            return [e for e in thuc_the if e]
        else:
            # Rất thô sơ (fallback)
            tu_khoa = re.findall(r'[A-Z][a-z]+', cau_hoi)
            return [tk.lower() for tk in tu_khoa]

    def truy_xuat(self, cau_hoi: str, do_sau: int = 2) -> str:
        """
        Truy xuất ngữ cảnh từ đồ thị dưới dạng văn bản.
        Quy trình: Câu hỏi -> Thực thể -> Subgraph -> Text.
        """
        thuc_the_chinh = self._trich_xuat_thuc_the_cau_hoi(cau_hoi)

        cac_bo_ba = []
        for tt in thuc_the_chinh:
            bo_ba_lan_can = self.store.lay_vung_lan_can(tt, do_sau=do_sau)
            cac_bo_ba.extend(bo_ba_lan_can)

        # Loại bỏ trùng lặp
        cac_bo_ba = list(set(cac_bo_ba))

        if not cac_bo_ba:
            return "Không tìm thấy thông tin liên quan trong tri thức đồ thị."

        # Format lại thành ngữ cảnh dạng văn bản
        ngu_canh = "Các mối quan hệ liên quan:\n"
        for chu_the, quan_he, doi_tuong in cac_bo_ba:
            ngu_canh += f"- {chu_the} {quan_he} {doi_tuong}\n"

        return ngu_canh
