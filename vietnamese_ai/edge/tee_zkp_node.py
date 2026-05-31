import hashlib
import hmac
from typing import Any, Dict

from vietnamese_ai.edge.node_llama import NodeLlamaEngine


class SecureEdgeNode:
    """
    Edge Node tích hợp môi trường thực thi tin cậy (Trusted Execution Environment - TEE)
    dành cho mạng lưới DePIN.

    Trong môi trường mô phỏng này, Node sẽ sinh ra một "Cryptographic Proof"
    bằng cách băm (hashing) đầu vào, đầu ra, và chữ ký (MAC) sử dụng Private Key của Node,
    chứng minh rằng Node đã thực thi chính xác mô hình AI mà không thay đổi dữ liệu.
    """

    def __init__(
        self,
        node_id: str,
        private_key: str,
        model_path: str = "models/llama-3-8b-instruct.Q4_K_M.gguf",
        auto_start: bool = True,
    ):
        self.node_id = node_id
        self.private_key = private_key.encode("utf-8")

        # Bọc Llama Engine vào bên trong "Enclave"
        self.engine = NodeLlamaEngine(
            model_path=model_path, context_size=2048, auto_start=auto_start
        )

        # Giả lập Hash của Model (trong thực tế sẽ băm toàn bộ file weights)
        self.model_hash = hashlib.sha256(model_path.encode("utf-8")).hexdigest()

    def generate_proof(self, input_data: str, output_data: str) -> str:
        """
        Sinh ra Cryptographic Execution Proof.
        Tạo HMAC-SHA256 của chuỗi (Model_Hash + Input + Output) với Private Key.
        """
        payload = f"{self.model_hash}:{input_data}:{output_data}".encode("utf-8")
        signature = hmac.new(self.private_key, payload, hashlib.sha256).hexdigest()
        return signature

    def chay_suy_luan_an_toan(self, cau_hoi: str) -> Dict[str, Any]:
        """
        Thực thi suy luận và đính kèm Proof để Router xác minh.
        """
        # Bước 1: Suy luận trong Enclave
        cau_tra_loi = self.engine.sinh_van_ban(cau_hoi, do_dai=128)

        # Bước 2: Sinh ra Proof
        proof = self.generate_proof(cau_hoi, cau_tra_loi)

        return {
            "node_id": self.node_id,
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "model_hash": self.model_hash,
            "metrics": {},
        }

    @staticmethod
    def verify_proof(
        node_id: str, public_key: str, cau_hoi: str, cau_tra_loi: str, proof: str, model_hash: str
    ) -> bool:
        """
        Hàm dành cho Edge Router (Data Center) để xác minh Proof do Node gửi lên.
        (Trong mô phỏng HMAC này, public_key chính là shared private_key,
        thực tế sẽ dùng thư viện ECDSA signature verification).
        """
        payload = f"{model_hash}:{cau_hoi}:{cau_tra_loi}".encode("utf-8")
        expected_signature = hmac.new(
            public_key.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()

        # So sánh an toàn chống constant-time timing attacks
        return hmac.compare_digest(proof, expected_signature)
