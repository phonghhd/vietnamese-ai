from typing import Any, Dict, Optional


class TokenLedger:
    """
    Sổ cái Token (Tokenomics) cho mạng lưới DePIN.
    Lưu trữ số dư EVON Token cho các Edge Node.
    """
    def __init__(self):
        self.so_du: Dict[str, float] = {}

    def thuong_token(self, node_id: str, so_luong: float = 10.0):
        """Thưởng Token khi Node hoàn thành tốt task."""
        if node_id not in self.so_du:
            self.so_du[node_id] = 0.0
        self.so_du[node_id] += so_luong
        print(f"[TokenLedger] Đã thưởng {so_luong} EVON cho Node '{node_id}'. Số dư hiện tại: {self.so_du[node_id]}")

    def lay_so_du(self, node_id: str) -> float:
        return self.so_du.get(node_id, 0.0)

class P2PTracker:
    """
    Sổ đăng ký tập trung mô phỏng mạng ngang hàng (P2P DHT).
    Các Edge Node sẽ đăng ký với Tracker để nhận task từ Router.
    """
    def __init__(self):
        # Lưu thông tin Node: node_id -> dict(engine, toc_do)
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def dang_ky_node(self, node_id: str, engine: Any, toc_do_du_kien: float = 1.0):
        """
        Đăng ký một Node vào mạng.
        Args:
            node_id: ID duy nhất của Node.
            engine: Tham chiếu tới Node Engine (VD: SecureEdgeNode).
            toc_do_du_kien: Điểm tốc độ ước tính (Càng cao càng ưu tiên).
        """
        self.nodes[node_id] = {
            "engine": engine,
            "toc_do": toc_do_du_kien,
            "trang_thai": "san_sang"
        }
        print(f"[P2PTracker] Node '{node_id}' đã gia nhập mạng lưới DePIN.")

    def huy_dang_ky(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]

    def tim_node_tot_nhat(self) -> Optional[Any]:
        """Tìm Edge Node có tốc độ tốt nhất và đang rảnh rỗi."""
        node_tot_nhat = None
        max_toc_do = -1.0

        for node_id, thong_tin in self.nodes.items():
            if thong_tin["trang_thai"] == "san_sang" and thong_tin["toc_do"] > max_toc_do:
                max_toc_do = thong_tin["toc_do"]
                node_tot_nhat = thong_tin["engine"]

        return node_tot_nhat
