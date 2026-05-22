import math
from typing import List, Optional, Tuple, Any

from .agent import TacTu

class MCTSNode:
    """Nút trong cây tìm kiếm Monte Carlo Tree Search cho suy luận."""
    def __init__(self, trang_thai: str, cha: Optional['MCTSNode'] = None, hanh_dong: Optional[str] = None):
        self.trang_thai = trang_thai
        self.cha = cha
        self.hanh_dong = hanh_dong # Hành động dẫn đến trạng thái này
        
        self.con: List['MCTSNode'] = []
        self.so_lan_tham = 0
        self.tong_diem = 0.0
        
    def la_la(self) -> bool:
        return len(self.con) == 0

    def tinh_ucb1(self, c_param: float = 1.41) -> float:
        if self.so_lan_tham == 0:
            return float('inf')
        
        diem_khai_thac = self.tong_diem / self.so_lan_tham
        if self.cha is None:
            return diem_khai_thac
            
        diem_khai_pha = c_param * math.sqrt(math.log(self.cha.so_lan_tham) / self.so_lan_tham)
        return diem_khai_thac + diem_khai_pha

class LapKeHoachMCTS:
    """
    Tác tử Lập kế hoạch theo Monte Carlo Tree Search.
    Được sử dụng cho các bài toán phức tạp cần suy nghĩ nhiều bước (o1-like).
    """
    def __init__(self, agent_chinh: TacTu, so_vong_lap: int = 10, c_param: float = 1.41):
        self.agent = agent_chinh
        self.so_vong_lap = so_vong_lap
        self.c_param = c_param

    def _sinh_hanh_dong_kha_thi(self, trang_thai: str) -> List[str]:
        """Sử dụng LLM để sinh ra các bước đi tiềm năng."""
        prompt = f"Dựa trên trạng thái hiện tại: '{trang_thai}'. Hãy đề xuất 3 hành động logic tiếp theo. Trả về dưới dạng danh sách gạch đầu dòng."
        
        # Gọi thẳng hàm _goi_llm thay vì chay() để tránh tool execution ở bước này
        phan_hoi = self.agent._goi_llm(prompt)
        
        hanh_dong = []
        for dong in phan_hoi.split('\n'):
            dong = dong.strip()
            if dong.startswith('-') or dong.startswith('*'):
                hanh_dong.append(dong[1:].strip())
                
        return hanh_dong if hanh_dong else ["Nghiên cứu thêm thông tin"]

    def _chuyen_trang_thai(self, trang_thai: str, hanh_dong: str) -> str:
        """Dự đoán kết quả của hành động."""
        prompt = f"Trạng thái hiện tại: {trang_thai}\nHành động: {hanh_dong}\nHãy mô tả ngắn gọn trạng thái mới sẽ đạt được."
        return self.agent._goi_llm(prompt)

    def _danh_gia_trang_thai(self, trang_thai: str) -> float:
        """Đánh giá chất lượng của trạng thái (Value function). Trả về từ 0.0 đến 1.0"""
        prompt = f"Đánh giá mức độ hoàn thành mục tiêu của trạng thái này: '{trang_thai}'. Trả về một số từ 0.0 (rất tệ) đến 1.0 (hoàn thành). Chỉ trả về số."
        try:
            phan_hoi = self.agent._goi_llm(prompt)
            return min(max(float(phan_hoi.strip()), 0.0), 1.0)
        except:
            return 0.5 # Giá trị mặc định nếu LLM không trả về số

    def _chon_nut(self, nut_goc: MCTSNode) -> MCTSNode:
        """Chọn nút tốt nhất để mở rộng dựa trên UCB1."""
        nut_hien_tai = nut_goc
        while not nut_hien_tai.la_la():
            nut_hien_tai = max(nut_hien_tai.con, key=lambda n: n.tinh_ucb1(self.c_param))
        return nut_hien_tai

    def _mo_rong(self, nut: MCTSNode):
        """Mở rộng nút bằng các hành động tiềm năng."""
        cac_hanh_dong = self._sinh_hanh_dong_kha_thi(nut.trang_thai)
        for hd in cac_hanh_dong:
            trang_thai_moi = self._chuyen_trang_thai(nut.trang_thai, hd)
            nut_con = MCTSNode(trang_thai=trang_thai_moi, cha=nut, hanh_dong=hd)
            nut.con.append(nut_con)

    def _mo_phong(self, nut: MCTSNode) -> float:
        """Mô phỏng (Rollout) từ nút hiện tại để lấy điểm đánh giá."""
        return self._danh_gia_trang_thai(nut.trang_thai)

    def _cap_nhat_nguoc(self, nut: MCTSNode, diem: float):
        """Cập nhật điểm lên gốc."""
        nut_hien_tai = nut
        while nut_hien_tai is not None:
            nut_hien_tai.so_lan_tham += 1
            nut_hien_tai.tong_diem += diem
            nut_hien_tai = nut_hien_tai.cha

    def chay(self, truy_van: str) -> str:
        """Thực thi MCTS Planning và trả về kế hoạch/kết quả tốt nhất."""
        nut_goc = MCTSNode(trang_thai=f"Bắt đầu với yêu cầu: {truy_van}")
        
        for _ in range(self.so_vong_lap):
            # Selection
            nut_chon = self._chon_nut(nut_goc)
            
            # Expansion
            if nut_chon.so_lan_tham > 0 or nut_chon == nut_goc:
                self._mo_rong(nut_chon)
                if nut_chon.con:
                    nut_chon = nut_chon.con[0]
                    
            # Simulation
            diem = self._mo_phong(nut_chon)
            
            # Backpropagation
            self._cap_nhat_nguoc(nut_chon, diem)
            
        # Trích xuất đường đi tốt nhất
        duong_di = []
        nut_hien_tai = nut_goc
        while nut_hien_tai.con:
            nut_hien_tai = max(nut_hien_tai.con, key=lambda n: n.so_lan_tham) # Chọn theo số lần thăm nhiều nhất (robustness)
            duong_di.append(f"Hành động: {nut_hien_tai.hanh_dong} -> Trạng thái: {nut_hien_tai.trang_thai}")
            
        ke_hoach_str = "\n".join(duong_di)
        
        # Yêu cầu Agent thực thi kế hoạch này
        prompt_cuoi = f"Tôi đã lập kế hoạch các bước sau để giải quyết yêu cầu '{truy_van}':\n\n{ke_hoach_str}\n\nHãy thực thi kế hoạch này và cung cấp kết quả cuối cùng."
        
        return self.agent.chay(prompt_cuoi)
