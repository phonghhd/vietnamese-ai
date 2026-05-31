import uuid
from typing import Any, Optional


class SoTayKinhNghiem:
    """
    Sổ tay kinh nghiệm (Experience Memory / Agent Memory 2.0).
    Lưu trữ các lỗi lầm hoặc bài học trong quá khứ của Tác tử vào VectorDB
    để rút kinh nghiệm cho các lần thực thi tiếp theo.
    """

    def __init__(self, vector_store: Any = None):
        """
        Khởi tạo Sổ tay Kinh nghiệm.
        Args:
            vector_store: Một đối tượng lưu trữ Vector (ví dụ: CSDLVector).
                          Nếu không cung cấp, sẽ giả lập lưu trữ tạm thời trong RAM.
        """
        self.vector_store = vector_store
        self._bo_nho_tam_thoi = []  # Fallback nếu không có VectorDB

        # Nếu dùng CSDLVector từ v10, đảm bảo nó đã được import
        if self.vector_store is None:
            try:
                from vietnamese_ai.rag.vector_store import CSDLVector

                self.vector_store = CSDLVector(kich_thuoc=384)  # Mặc định kích thước embeddings
            except ImportError:
                pass

    def ghi_nhan_bai_hoc(self, ngu_canh: str, sai_lam: str, cach_khac_phuc: str):
        """
        Lưu một bài học kinh nghiệm mới.

        Args:
            ngu_canh: Nội dung công việc đang làm (dùng làm vector tìm kiếm)
            sai_lam: Lỗi đã mắc phải
            cach_khac_phuc: Cách làm đúng
        """
        noi_dung = f"[SAI LẦM]: {sai_lam} -> [BÀI HỌC]: {cach_khac_phuc}"

        if self.vector_store and hasattr(self.vector_store, "them_tai_lieu"):
            doc_id = f"exp_{uuid.uuid4().hex[:8]}"
            self.vector_store.them_tai_lieu(
                id_tai_lieu=doc_id, noi_dung=ngu_canh, metadata={"bai_hoc": noi_dung}
            )
        else:
            self._bo_nho_tam_thoi.append({"ngu_canh": ngu_canh, "bai_hoc": noi_dung})

    def truy_xuat_kinh_nghiem(self, truy_van: str, top_k: int = 1) -> Optional[str]:
        """
        Tìm kiếm các bài học liên quan đến truy vấn hiện tại.
        """
        bai_hoc_tim_thay = []

        if self.vector_store and hasattr(self.vector_store, "tim_kiem"):
            try:
                ket_qua = self.vector_store.tim_kiem(truy_van, top_k=top_k)
                for item in ket_qua:
                    # Item thường trả về dict chứa metadata
                    if "metadata" in item and "bai_hoc" in item["metadata"]:
                        bai_hoc_tim_thay.append(item["metadata"]["bai_hoc"])
            except Exception:
                pass
        else:
            # Fake logic cho bộ nhớ tạm (chỉ match keyword cơ bản)
            for item in self._bo_nho_tam_thoi:
                # Nếu từ khóa trong truy vấn khớp với ngữ cảnh
                if any(word.lower() in item["ngu_canh"].lower() for word in truy_van.split()):
                    bai_hoc_tim_thay.append(item["bai_hoc"])

        if bai_hoc_tim_thay:
            return "\n".join(bai_hoc_tim_thay)
        return None
