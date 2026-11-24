"""SecureEdgeNode - Nút Suy Luận Edge Kèm Chứng Thực Mật Mã Schnorr/Fiat-Shamir Chuẩn RFC 5114."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any, Dict

from vietnamese_ai.crypto.schnorr import (
    P_HEX_WIDTH as _P_HEX_WIDTH,
)
from vietnamese_ai.crypto.schnorr import (
    RFC5114_G as _SCHNORR_G,
)
from vietnamese_ai.crypto.schnorr import (
    THONG_SO_NHOM_SCHNORR,
    kiem_tra_thanh_vien_nhom_con,
    ky_schnorr_rfc5114,
    luy_thua_modulo_rfc5114,
    suy_dan_khoa_bi_mat,
    tinh_khoa_cong_khai_hex,
    tinh_thach_thuc_fiat_shamir,
    xac_thuc_schnorr_rfc5114,
)
from vietnamese_ai.edge.node_llama import NodeLlamaEngine


class SecureEdgeNode:
    """
    Edge Node xác thực suy luận mật mã học (Verifiable Edge Node) cho mạng phân tán:
    - Sử dụng nhóm Schnorr chuẩn RFC 5114 Section 2.3 (trường 2048-bit p, nhóm con nguyên tố 256-bit q).
    - Sinh nonce ngẫu nhiên kết hợp CSPRNG (`secrets`) + HMAC-SHA256 và kiểm tra thành viên nhóm con.
    - Ràng buộc `(model_hash, input_data, output_data)` với bằng chứng Fiat-Shamir không tiết lộ khóa bí mật.
    """

    def __init__(
        self,
        node_id: str,
        private_key: str,
        model_path: str = "models/llama-3-8b-instruct.Q4_K_M.gguf",
        auto_start: bool = True,
    ) -> None:
        self.node_id = node_id
        self.private_key = private_key.encode("utf-8")
        self._secret_scalar = self._derive_scalar(self.private_key)
        self.public_key = tinh_khoa_cong_khai_hex(self._secret_scalar)
        self._public_int = int(self.public_key, 16)

        self.engine = NodeLlamaEngine(
            model_path=model_path, context_size=2048, auto_start=auto_start
        )

        if os.path.isfile(model_path):
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                h.update(f.read(65536))
            self.model_hash = h.hexdigest()
        else:
            self.model_hash = hashlib.sha256(model_path.encode("utf-8")).hexdigest()

    @staticmethod
    def _derive_scalar(secret_bytes: bytes) -> int:
        """Dẫn xuất khóa vô hướng trong nhóm con cấp nguyên tố `[1, q - 1]`."""
        return suy_dan_khoa_bi_mat(secret_bytes, tien_to_mien=b"edge-zkp-key:")

    @staticmethod
    def _la_thanh_vien_nhom_con(phan_tu: int) -> bool:
        """Kiểm tra phần tử thuộc nhóm con cấp nguyên tố q modulo p."""
        return kiem_tra_thanh_vien_nhom_con(phan_tu)

    @staticmethod
    def _fiat_shamir_challenge(r_hex: str, y_hex: str, payload: bytes) -> int:
        """Tính thách thức Fiat-Shamir tách miền trong `[1, q - 1]`."""
        return tinh_thach_thuc_fiat_shamir(r_hex, y_hex, payload, tien_to_mien=b"edge-zkp1:")

    def generate_proof(self, input_data: str, output_data: str) -> str:
        """
        Sinh bằng chứng không tiết lộ thông tin (Non-Interactive Schnorr ZKP RFC 5114)
        theo biến đổi Fiat-Shamir: g^s == R * y^c (mod p) với s = (k + c * x) mod q.
        """
        payload = f"{self.model_hash}:{input_data}:{output_data}".encode("utf-8")
        _, _, proof_str = ky_schnorr_rfc5114(
            secret_bytes=self.private_key,
            khoa_vo_huong=self._secret_scalar,
            public_key_hex=self.public_key,
            payload=payload,
            tien_to_nonce=b"edge-nonce:",
            tien_to_thach_thuc=b"edge-zkp1:",
        )
        return proof_str

    def chay_suy_luan_an_toan(self, cau_hoi: str) -> Dict[str, Any]:
        """Thực thi suy luận cục bộ và đính kèm bằng chứng Schnorr ZKP RFC 5114."""
        cau_tra_loi = self.engine.sinh_van_ban(cau_hoi, do_dai=128)
        proof = self.generate_proof(cau_hoi, cau_tra_loi)

        return {
            "node_id": self.node_id,
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "public_key": self.public_key,
            "model_hash": self.model_hash,
            "chuan_mat_ma": THONG_SO_NHOM_SCHNORR["chuan"],
            "metrics": {},
        }

    @classmethod
    def verify_proof(
        cls,
        node_id: str,
        public_key: str,
        cau_hoi: str,
        cau_tra_loi: str,
        proof: str,
        model_hash: str,
    ) -> bool:
        """
        Xác minh bằng chứng Schnorr ZKP RFC 5114:
        kiểm tra thành viên nhóm con cấp q, miền giá trị s và phương trình g^s == R * y^c (mod p).
        """
        payload = f"{model_hash}:{cau_hoi}:{cau_tra_loi}".encode("utf-8")

        if proof.startswith("zkp1:"):
            parts = proof.split(":")
            if len(parts) != 4:
                return False
            _, r_hex, s_hex, proof_y_hex = parts
            try:
                r_int = int(r_hex, 16)
                s_int = int(s_hex, 16)
                proof_y_int = int(proof_y_hex, 16)
            except ValueError:
                return False

            if len(public_key) == _P_HEX_WIDTH and all(
                ch in "0123456789abcdefABCDEF" for ch in public_key
            ):
                y_int = int(public_key, 16)
                y_hex = public_key.lower()
            else:
                scalar = cls._derive_scalar(public_key.encode("utf-8"))
                y_int = luy_thua_modulo_rfc5114(_SCHNORR_G, scalar)
                y_hex = f"{y_int:0{_P_HEX_WIDTH}x}"

            if y_int != proof_y_int:
                return False

            c = cls._fiat_shamir_challenge(r_hex, y_hex, payload)
            return xac_thuc_schnorr_rfc5114(r_int=r_int, s_int=s_int, y_int=y_int, thach_thuc=c)

        expected_signature = hmac.new(
            public_key.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(proof, expected_signature)
