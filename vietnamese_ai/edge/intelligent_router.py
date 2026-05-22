from typing import Any, Dict, Optional
import os

from .node_llama import NodeLlamaEngine

class EdgeRouter:
    """
    Gateway định tuyến thông minh (Intelligent Routing Gateway).
    Phân loại yêu cầu từ Agent/User để quyết định chạy inference ở Local (Edge) hay Cloud (Data Center).
    """
    def __init__(
        self, 
        edge_engine: Optional[NodeLlamaEngine] = None,
        cloud_api_key: Optional[str] = None,
        cloud_endpoint: Optional[str] = None
    ):
        self.edge_engine = edge_engine
        self.cloud_api_key = cloud_api_key or os.environ.get("EVONET_CLOUD_KEY")
        self.cloud_endpoint = cloud_endpoint or "https://api.evonet.vn/v1/chat/completions"
        
    def _is_simple_query(self, prompt: str) -> bool:
        """
        Logic phân loại heuristic tạm thời.
        Trong thực tế có thể dùng một mô hình BERT nhỏ để đánh giá.
        """
        # Nếu câu hỏi quá dài, gửi lên cloud
        if len(prompt) > 2000:
            return False
            
        # Các keyword thường cần suy luận sâu hoặc code
        complex_keywords = ["giải thích chi tiết", "viết code", "thuật toán", "kiến trúc", "tổng hợp"]
        if any(kw in prompt.lower() for kw in complex_keywords):
            return False
            
        return True
        
    def sinh_van_ban(self, prompt: str, do_dai: int = 256, force_edge: bool = False, force_cloud: bool = False) -> str:
        """
        Thực thi sinh văn bản dựa trên Routing Logic.
        """
        use_edge = False
        
        if force_edge:
            use_edge = True
        elif force_cloud:
            use_edge = False
        else:
            # Routing logic cơ bản
            use_edge = self._is_simple_query(prompt)
            
        # Kiểm tra tính khả dụng
        if use_edge and not self.edge_engine:
            print("[EdgeRouter] Edge Engine không khả dụng. Fallback sang Cloud.")
            use_edge = False
            
        if not use_edge and not self.cloud_api_key:
            print("[EdgeRouter] Cloud API Key chưa được cấu hình. Fallback sang Edge.")
            use_edge = True
            
        if use_edge and self.edge_engine:
            print("[EdgeRouter] Đang xử lý tại thiết bị (Edge)...")
            return self.edge_engine.sinh_van_ban(prompt, do_dai=do_dai)
        elif not use_edge and self.cloud_api_key:
            print("[EdgeRouter] Đang xử lý trên Cloud (Data Center)...")
            return self._call_cloud(prompt, do_dai)
        else:
            return "Lỗi: Không có engine nào khả dụng (Cả Edge và Cloud đều chưa sẵn sàng)."
            
    def _call_cloud(self, prompt: str, do_dai: int) -> str:
        """Gọi tới Cloud API."""
        try:
            import requests
        except ImportError:
            raise ImportError("Cần cài đặt requests: pip install requests")
            
        payload = {
            "model": "evonet-large",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": do_dai
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cloud_api_key}"
        }
        
        try:
            # Timeout ngắn để tránh treo lâu trong mạng DePIN
            response = requests.post(self.cloud_endpoint, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            # Nếu Cloud sập, cố gắng fallback lại Edge lần cuối
            if self.edge_engine:
                print(f"[EdgeRouter] Lỗi Cloud ({str(e)}). Thử fallback lại Edge...")
                return self.edge_engine.sinh_van_ban(prompt, do_dai=do_dai)
            return f"Lỗi sinh văn bản (Cloud): {str(e)}"
