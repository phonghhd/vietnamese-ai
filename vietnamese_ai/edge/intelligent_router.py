import concurrent.futures
import os
from typing import Optional

from .node_llama import NodeLlamaEngine
from .p2p_network import P2PTracker, TokenLedger


class EdgeRouter:
    """
    Gateway định tuyến thông minh (Intelligent Routing Gateway).
    Phân loại yêu cầu từ Agent/User để quyết định chạy inference ở Local (Edge) hay Cloud (Data Center).
    """

    def __init__(
        self,
        edge_engine: Optional[NodeLlamaEngine] = None,
        cloud_api_key: Optional[str] = None,
        cloud_endpoint: Optional[str] = None,
        p2p_tracker: Optional[P2PTracker] = None,
        token_ledger: Optional[TokenLedger] = None,
    ):
        self.edge_engine = edge_engine
        self.cloud_api_key = cloud_api_key or os.environ.get("EVONET_CLOUD_KEY")
        self.cloud_endpoint = cloud_endpoint or "https://api.evonet.vn/v1/chat/completions"
        self.p2p_tracker = p2p_tracker
        self.token_ledger = token_ledger

    def _is_simple_query(self, prompt: str) -> bool:
        """
        Logic phân loại heuristic tạm thời.
        Trong thực tế có thể dùng một mô hình BERT nhỏ để đánh giá.
        """
        # Nếu câu hỏi quá dài, gửi lên cloud
        if len(prompt) > 2000:
            return False

        # Các keyword thường cần suy luận sâu hoặc code
        complex_keywords = [
            "giải thích chi tiết",
            "viết code",
            "thuật toán",
            "kiến trúc",
            "tổng hợp",
        ]
        if any(kw in prompt.lower() for kw in complex_keywords):
            return False

        return True

    def sinh_van_ban(
        self, prompt: str, do_dai: int = 256, force_edge: bool = False, force_cloud: bool = False
    ) -> str:
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

            # Kiểm tra xem đây có phải là SecureEdgeNode (TEE) không
            if hasattr(self.edge_engine, "chay_suy_luan_an_toan"):
                kq = self.edge_engine.chay_suy_luan_an_toan(prompt)

                # Xác minh tính toàn vẹn (Zero-Knowledge Proof simulation)
                # Trong thực tế public_key sẽ được fetch từ Registry
                is_valid = self.edge_engine.__class__.verify_proof(
                    node_id=kq["node_id"],
                    public_key=self.edge_engine.private_key.decode("utf-8"),  # Chuyển private key dạng bytes sang chuỗi để giả lập public_key
                    cau_hoi=prompt,
                    cau_tra_loi=kq["cau_tra_loi"],
                    proof=kq["proof"],
                    model_hash=kq["model_hash"],
                )

                if not is_valid:
                    print(
                        f"[EdgeRouter] CẢNH BÁO: Node {kq['node_id']} gửi kết quả giả mạo! Chuyển sang Cloud."
                    )
                    return self._call_cloud(prompt, do_dai)

                return kq["cau_tra_loi"]
            else:
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
            "max_tokens": do_dai,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cloud_api_key}",
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

    def sinh_van_ban_song_song(self, prompt: str, do_dai: int = 256) -> str:
        """
        Hybrid Speculative Execution (Chạy Đua).
        Gửi yêu cầu tới cả Edge Network (P2P) và Cloud. Bên nào xong trước lấy bên đó.
        """
        if not self.p2p_tracker:
            print("[EdgeRouter] Không có P2PTracker. Chuyển về luồng thông thường.")
            return self.sinh_van_ban(prompt, do_dai=do_dai)

        edge_node = self.p2p_tracker.tim_node_tot_nhat()
        if not edge_node:
            print("[EdgeRouter] Mạng P2P trống! Chuyển thẳng lên Cloud.")
            return self._call_cloud(prompt, do_dai)

        def _task_edge():
            if hasattr(edge_node, "chay_suy_luan_an_toan"):
                kq = edge_node.chay_suy_luan_an_toan(prompt)

                # ZKP Verification
                public_key = edge_node.private_key.decode("utf-8")
                is_valid = edge_node.__class__.verify_proof(
                    node_id=kq["node_id"],
                    public_key=public_key,
                    cau_hoi=prompt,
                    cau_tra_loi=kq["cau_tra_loi"],
                    proof=kq["proof"],
                    model_hash=kq["model_hash"],
                )

                if not is_valid:
                    raise ValueError(f"Node {kq['node_id']} gửi kết quả giả mạo!")

                # Thưởng Token vì Node đã làm tốt nhiệm vụ
                if self.token_ledger:
                    self.token_ledger.thuong_token(kq["node_id"], 10.0)

                return kq["cau_tra_loi"], "EDGE"
            else:
                return edge_node.sinh_van_ban(prompt, do_dai=do_dai), "EDGE"

        def _task_cloud():
            if not self.cloud_api_key:
                raise ValueError("Chưa cấu hình Cloud API.")
            return self._call_cloud(prompt, do_dai), "CLOUD"

        # Chạy song song
        print("[EdgeRouter] Bắt đầu cuộc đua (Race) giữa P2P Edge và Data Center...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_source = {
                executor.submit(_task_edge): "EDGE",
                executor.submit(_task_cloud): "CLOUD",
            }

            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result, src = future.result()
                    print(f"[EdgeRouter] {src} VỀ ĐÍCH TRƯỚC!")
                    # Trả về kết quả đầu tiên thành công
                    return result
                except Exception as exc:
                    print(f"[EdgeRouter] {source} sinh ra lỗi: {exc}")
                    # Sẽ tiếp tục vòng lặp để lấy kết quả từ luồng kia

        return "Lỗi: Cả Edge và Cloud đều thất bại trong cuộc đua."
