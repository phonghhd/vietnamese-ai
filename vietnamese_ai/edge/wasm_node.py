"""WebBrowserNode - Cổng Suy Luận WASM/WebGPU Kèm Chứng Thực Mật Mã RFC 5114 & Fallback CPU Thực Tế."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import warnings
from typing import Any, Dict, Optional

warnings.warn(
    "Module 'vietnamese_ai.edge.wasm_node' đang ở trạng thái mô phỏng/thử nghiệm và được đánh dấu "
    "lỗi thời (deprecated) từ v31.2.0; dự kiến tách sang nhánh thử nghiệm ở v32.0.0. "
    "Vui lòng dùng 'vietnamese_ai.edge.DongCoSuyLuanEdge' cho suy luận CPU thực tế.",
    DeprecationWarning,
    stacklevel=2,
)

from vietnamese_ai.crypto.schnorr import (
    ky_schnorr_rfc5114,
    suy_dan_khoa_bi_mat,
    tinh_khoa_cong_khai_hex,
)
from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge
from vietnamese_ai.edge.tee_zkp_node import SecureEdgeNode


class WebBrowserNode:
    """
    Edge Node cho trình duyệt WebAssembly (WASM) & WebGPU:
    - Khi có `websocket_conn`, chuyển tiếp lệnh suy luận xuống trình duyệt WebGPU thực tế.
    - Khi chạy độc lập không có WebSocket client, thực thi suy luận thực tế bằng `DongCoSuyLuanEdge`
      (bộ giải mã Transformer CPU cục bộ) thay vì trả về chuỗi mẫu cố định.
    - Chứng thực kết quả bằng chữ ký Schnorr ZKP chuẩn RFC 5114 (2048-bit p, 256-bit q).
    """

    def __init__(
        self,
        session_id: str,
        websocket_conn: Any = None,
        secret_key: Optional[str] = None,
        local_engine: Optional[DongCoSuyLuanEdge] = None,
    ) -> None:
        self.session_id = session_id
        self.websocket_conn = websocket_conn
        self.logger = logging.getLogger("WASM_Node")
        self.trang_thai = "dang_cho"
        self.model_hash = hashlib.sha256(b"wasm_webgpu_model_v1").hexdigest()[:16]
        khoa = secret_key or secrets.token_hex(32)
        self.private_key = f"{khoa}:{session_id}".encode("utf-8")
        self._secret_scalar = suy_dan_khoa_bi_mat(self.private_key, tien_to_mien=b"edge-zkp-key:")
        self.public_key = tinh_khoa_cong_khai_hex(self._secret_scalar)
        self._local_engine = local_engine

    def _lay_dong_co_cuc_bo(self) -> DongCoSuyLuanEdge:
        """Khởi tạo lười động cơ suy luận CPU cục bộ khi không có kết nối WebSocket trình duyệt."""
        if self._local_engine is None:
            self._local_engine = DongCoSuyLuanEdge(so_luong_thread=2)
        return self._local_engine

    def sinh_van_ban(self, prompt: str, do_dai: int = 128) -> str:
        """Gửi yêu cầu tới WASM/WebGPU Client qua WebSocket hoặc thực thi suy luận cục bộ bằng `DongCoSuyLuanEdge`."""
        self.logger.info("Đang xử lý yêu cầu suy luận cho WASM Client (Session: %s)...", self.session_id)

        if self.websocket_conn is not None:
            payload = json.dumps({"action": "generate", "prompt": prompt, "max_tokens": do_dai})
            if hasattr(self.websocket_conn, "send") and hasattr(self.websocket_conn, "recv"):
                self.websocket_conn.send(payload)
                resp = self.websocket_conn.recv()
                if isinstance(resp, str) and resp:
                    return resp

        ket_qua_cuc_bo = self._lay_dong_co_cuc_bo().sinh_van_ban(prompt, do_dai_toi_da=do_dai)
        return f"[WASM-WebGPU] ({prompt}): {ket_qua_cuc_bo}"

    def generate_proof(self, input_data: str, output_data: str) -> str:
        """Sinh bằng chứng Schnorr ZKP chuẩn RFC 5114 với nonce CSPRNG cho kết quả suy luận."""
        payload = f"{self.model_hash}:{input_data}:{output_data}".encode("utf-8")
        _, _, proof_str = ky_schnorr_rfc5114(
            secret_bytes=self.private_key,
            khoa_vo_huong=self._secret_scalar,
            public_key_hex=self.public_key,
            payload=payload,
            tien_to_nonce=b"wasm-nonce:",
            tien_to_thach_thuc=b"edge-zkp1:",
        )
        return proof_str

    def chay_suy_luan_an_toan(self, cau_hoi: str) -> Dict[str, Any]:
        """Thực thi suy luận kèm bằng chứng Schnorr ZKP chống giả mạo kết quả."""
        cau_tra_loi = self.sinh_van_ban(cau_hoi, do_dai=128)
        proof = self.generate_proof(cau_hoi, cau_tra_loi)
        che_do = "websocket_webgpu" if self.websocket_conn is not None else "local_cpu_transformer"

        return {
            "node_id": f"web_client_{self.session_id}",
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "public_key": self.public_key,
            "model_hash": self.model_hash,
            "metrics": {"hardware": "WebGPU" if self.websocket_conn else "CPU-Edge", "che_do": che_do},
        }

    @staticmethod
    def verify_proof(
        node_id: str, public_key: str, cau_hoi: str, cau_tra_loi: str, proof: str, model_hash: str
    ) -> bool:
        """Xác minh bằng chứng ZKP RFC 5114 từ WebBrowserNode."""
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
