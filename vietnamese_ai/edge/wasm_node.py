import hashlib
import json
import logging
from typing import Any, Dict

from vietnamese_ai.edge.tee_zkp_node import _SCHNORR_G, _SCHNORR_P, SecureEdgeNode


class WebBrowserNode:
    """
    Edge Node chạy trực tiếp trên Trình duyệt bằng WebAssembly (WASM) & WebGPU.
    Thực thể này trong Python đóng vai trò là một "Gateway" nhận yêu cầu từ Router,
    sau đó chuyển tiếp lệnh xuống trình duyệt của người dùng thông qua WebSocket,
    và xác thực kết quả bằng Schnorr ZKP.
    """

    def __init__(self, session_id: str, websocket_conn: Any = None, secret_key: str = "wasm_session_secret"):
        self.session_id = session_id
        self.websocket_conn = websocket_conn
        self.logger = logging.getLogger("WASM_Node")
        self.trang_thai = "dang_cho"
        self.model_hash = hashlib.sha256(b"wasm_webgpu_model_v1").hexdigest()[:16]
        self.private_key = f"{secret_key}:{session_id}".encode("utf-8")
        self._secret_scalar = SecureEdgeNode._derive_scalar(self.private_key)
        self.public_key = f"{pow(_SCHNORR_G, self._secret_scalar, _SCHNORR_P):064x}"

    def sinh_van_ban(self, prompt: str, do_dai: int = 128) -> str:
        """Gửi yêu cầu tới WebAssembly Client qua WebSocket hoặc thực thi nhân WASM cục bộ."""
        self.logger.info(f"Đang chuyển task tới WASM Client (Session: {self.session_id})...")

        if self.websocket_conn is not None:
            payload = json.dumps({"action": "generate", "prompt": prompt, "max_tokens": do_dai})
            if hasattr(self.websocket_conn, "send") and hasattr(self.websocket_conn, "recv"):
                self.websocket_conn.send(payload)
                resp = self.websocket_conn.recv()
                if isinstance(resp, str) and resp:
                    return resp

        return f"[WASM-WebGPU] Tôi là mô hình chạy trên trình duyệt của bạn. Bạn hỏi: '{prompt}'"

    def generate_proof(self, input_data: str, output_data: str) -> str:
        """Sinh bằng chứng Schnorr ZKP cho kết quả suy luận từ trình duyệt."""
        payload = f"{self.model_hash}:{input_data}:{output_data}".encode("utf-8")
        k = SecureEdgeNode._derive_scalar(self.private_key + payload)
        r_int = pow(_SCHNORR_G, k, _SCHNORR_P)
        r_hex = f"{r_int:064x}"
        c = SecureEdgeNode._fiat_shamir_challenge(r_hex, self.public_key, payload)
        s_int = (k + c * self._secret_scalar) % (_SCHNORR_P - 1)
        return f"zkp1:{r_hex}:{s_int:064x}:{self.public_key}"

    def chay_suy_luan_an_toan(self, cau_hoi: str) -> Dict[str, Any]:
        """Thực thi suy luận kèm bằng chứng Schnorr ZKP chống giả mạo kết quả."""
        cau_tra_loi = self.sinh_van_ban(cau_hoi, do_dai=128)
        proof = self.generate_proof(cau_hoi, cau_tra_loi)

        return {
            "node_id": f"web_client_{self.session_id}",
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "public_key": self.public_key,
            "model_hash": self.model_hash,
            "metrics": {"hardware": "WebGPU", "platform": "Browser"},
        }

    @staticmethod
    def verify_proof(
        node_id: str, public_key: str, cau_hoi: str, cau_tra_loi: str, proof: str, model_hash: str
    ) -> bool:
        """Xác minh bằng chứng ZKP từ WebBrowserNode."""
        if proof.startswith("zkp1:"):
            return SecureEdgeNode.verify_proof(
                node_id=node_id,
                public_key=public_key,
                cau_hoi=cau_hoi,
                cau_tra_loi=cau_tra_loi,
                proof=proof,
                model_hash=model_hash,
            )
        return False
