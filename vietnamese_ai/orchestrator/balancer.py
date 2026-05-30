"""
CanBangTai (Load Balancer) - Điều hướng request giữa các Worker Nodes.
"""

import random
from typing import List, Dict, Optional


class CanBangTai:
    """
    Cân bằng tải cho V-Orchestrator.
    Hỗ trợ RoundRobin và Canary (Weighted).
    """

    def __init__(self, chien_luoc: str = "round_robin"):
        """
        Khởi tạo bộ cân bằng tải.

        Args:
            chien_luoc: "round_robin", "random" hoặc "canary"
        """
        self.chien_luoc = chien_luoc
        self._chi_so_hien_tai = 0
        self.trong_so_canary: Dict[str, float] = {}  # {worker_id: weight}

    def cap_nhat_trong_so(self, trong_so: Dict[str, float]):
        """Cập nhật trọng số cho chiến lược Canary (tổng phải <= 1.0)."""
        self.trong_so_canary = trong_so

    def chon_worker(self, danh_sach_worker: List[str]) -> Optional[str]:
        """
        Chọn một worker từ danh sách đang hoạt động (active workers).
        
        Args:
            danh_sach_worker: Danh sách các ID hoặc URL của Worker đang hoạt động.
        
        Returns:
            worker_id được chọn, hoặc None nếu danh sách rỗng.
        """
        if not danh_sach_worker:
            return None

        if self.chien_luoc == "random":
            return random.choice(danh_sach_worker)
            
        elif self.chien_luoc == "canary":
            # Nếu không có trọng số, fallback về random
            if not self.trong_so_canary:
                return random.choice(danh_sach_worker)
                
            # Lọc danh sách worker theo trọng số hợp lệ
            workers_co_trong_so = [w for w in danh_sach_worker if w in self.trong_so_canary]
            if not workers_co_trong_so:
                return random.choice(danh_sach_worker)
                
            trong_so = [self.trong_so_canary[w] for w in workers_co_trong_so]
            # random.choices trả về list, lấy phần tử đầu tiên
            return random.choices(workers_co_trong_so, weights=trong_so, k=1)[0]
            
        else:
            # Mặc định: Round Robin
            self._chi_so_hien_tai = (self._chi_so_hien_tai + 1) % len(danh_sach_worker)
            return danh_sach_worker[self._chi_so_hien_tai]
