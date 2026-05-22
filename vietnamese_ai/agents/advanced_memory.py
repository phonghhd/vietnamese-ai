"""Advanced Memory - Bộ nhớ nâng cao cho Tác tử."""

from typing import List, Dict

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
            self.tin_nhan = self.tin_nhan[-self.k:]
            
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
        
        if len(self.tin_nhan_tam) >= self.so_luong * 2: # Một vòng QA là 2 tin nhắn
            self._tom_tat_lich_su()
            
    def _tom_tat_lich_su(self):
        """Gọi LLM để tóm tắt."""
        if not hasattr(self.llm, "sinh_van_ban"):
            return # Fallback nếu LLM không hợp lệ
            
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
            pass # Bỏ qua nếu lỗi
            
    def lay_lich_su(self) -> str:
        ket_qua = f"[TÓM TẮT LỊCH SỬ]: {self.tom_tat_hien_tai}\n" if self.tom_tat_hien_tai else ""
        for tn in self.tin_nhan_tam:
            vai_tro = "Tác tử" if tn["vai_tro"] == "tac_tu" else "Người dùng"
            ket_qua += f"{vai_tro}: {tn['noi_dung']}\n"
        return ket_qua.strip()
        
    def lam_sach(self) -> None:
        self.tom_tat_hien_tai = ""
        self.tin_nhan_tam.clear()
