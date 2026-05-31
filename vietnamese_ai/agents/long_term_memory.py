from typing import Any, Optional

from .memory import BoNhoTacTu
from .tools import CongCu


class HethongNhoMemGPT(BoNhoTacTu):
    """
    Quản lý bộ nhớ theo kiến trúc MemGPT.
    Chia làm 2 tầng:
    1. Core Memory (Bộ nhớ lõi): Context hiện tại đang lưu trên VRAM.
    2. Archival Memory (Bộ nhớ lưu trữ): Lưu các hội thoại cũ vào Vector Database.

    Tự động nén (compress) khi Core Memory vượt ngưỡng.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_core_tokens: int = 4000,
        vector_store: Any = None,
    ):
        self.max_core_tokens = max_core_tokens
        self.vector_store = vector_store
        super().__init__(system_prompt)

    def _tinh_so_token_mo_phong(self, text: str) -> int:
        """Hàm mô phỏng tính số token (thực tế dùng tokenizer)."""
        return len(text.split())

    def _kiem_tra_trang_thai_bo_nho(self):
        """Kiểm tra và tự động nén nếu bộ nhớ sắp đầy."""
        tong_tokens = 0
        for msg in self.lich_su:
            tong_tokens += self._tinh_so_token_mo_phong(msg["content"])

        if tong_tokens > self.max_core_tokens:
            self._nen_bo_nho()

    def _nen_bo_nho(self):
        """
        Di chuyển phần nửa đầu của lịch sử (trừ system prompt) vào Archival Memory.
        """
        # Giữ lại system prompt (vị trí 0)
        # Giữ lại khoảng 50% số lượng tin nhắn gần nhất
        if len(self.lich_su) <= 3:
            return  # Quá ít để nén

        system_msg = self.lich_su[0] if self.lich_su[0]["role"] == "system" else None

        idx_chia = len(self.lich_su) // 2
        if system_msg:
            idx_chia = max(1, idx_chia)

        tin_nhan_bi_nen = self.lich_su[1 if system_msg else 0 : idx_chia]
        self.lich_su = ([system_msg] if system_msg else []) + self.lich_su[idx_chia:]

        # Lưu vào Vector Store
        if self.vector_store and tin_nhan_bi_nen:
            texts_to_archive = [f"{m['role']}: {m['content']}" for m in tin_nhan_bi_nen]
            self.vector_store.add_documents(texts_to_archive)
            print(f"[MemGPT] Đã nén {len(tin_nhan_bi_nen)} tin nhắn vào Archival Memory.")

    def them_tin_nhan(self, vai_tro: str, noi_dung: str, ten_cong_cu: Optional[str] = None):
        """Override thêm tin nhắn để tự động theo dõi dung lượng."""
        super().them_tin_nhan(vai_tro, noi_dung, ten_cong_cu)
        self._kiem_tra_trang_thai_bo_nho()

    def tao_cong_cu_truy_van(self) -> CongCu:
        """Tạo công cụ cho phép Agent chủ động lục tìm ký ức cũ."""

        def truy_van_tri_nho_cu(tu_khoa: str) -> str:
            if not self.vector_store:
                return "Lỗi: Hệ thống chưa cấu hình Vector Store cho Archival Memory."

            # Giả định vector_store có hàm similarity_search
            if hasattr(self.vector_store, "similarity_search"):
                ket_qua = self.vector_store.similarity_search(tu_khoa, top_k=3)
                if not ket_qua:
                    return "Không tìm thấy ký ức nào liên quan."
                return "Ký ức cũ: \n" + "\n".join([doc.page_content for doc in ket_qua])
            return "Vector store không hỗ trợ tìm kiếm."

        return CongCu(
            ten="truy_van_tri_nho_cu",
            mo_ta="Lục tìm lại các ký ức hoặc hội thoại trong quá khứ xa. Tham số 'tu_khoa' là từ khóa cần tìm.",
            ham_thuc_thi=truy_van_tri_nho_cu,
        )
