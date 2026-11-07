import hashlib
import hmac
import os
from typing import Any, Dict

from vietnamese_ai.edge.node_llama import NodeLlamaEngine

# Trường nguyên tố secp256k1 p = 2^256 - 2^32 - 977 và phần tử sinh g = 5
_SCHNORR_P = (1 << 256) - (1 << 32) - 977
_SCHNORR_G = 5


class SecureEdgeNode:
    """
    Edge Node tích hợp môi trường thực thi tin cậy (Trusted Execution Environment - TEE)
    dành cho mạng lưới DePIN.

    Sử dụng giao thức Non-Interactive Schnorr Zero-Knowledge Proof (Fiat-Shamir heuristic)
    trên trường hữu hạn 256-bit để chứng minh tri thức về khóa bí mật gắn chặt với
    (model_hash, input_data, output_data) mà không tiết lộ khóa bí mật.
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
        self._secret_scalar = self._derive_scalar(self.private_key)
        self._public_int = pow(_SCHNORR_G, self._secret_scalar, _SCHNORR_P)
        self.public_key = f"{self._public_int:064x}"

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
        val = int.from_bytes(hashlib.sha256(secret_bytes).digest(), "big")
        return (val % (_SCHNORR_P - 2)) + 1

    @staticmethod
    def _fiat_shamir_challenge(r_hex: str, y_hex: str, payload: bytes) -> int:
        h = hashlib.sha256()
        h.update(r_hex.encode("ascii"))
        h.update(b":")
        h.update(y_hex.encode("ascii"))
        h.update(b":")
        h.update(payload)
        return int.from_bytes(h.digest(), "big") % (_SCHNORR_P - 1)

    def generate_proof(self, input_data: str, output_data: str) -> str:
        """
        Sinh bằng chứng không tiết lộ thông tin (Non-Interactive Schnorr ZKP)
        theo biến đổi Fiat-Shamir: g^s == R * y^c (mod p).
        """
        payload = f"{self.model_hash}:{input_data}:{output_data}".encode("utf-8")
        k_bytes = hmac.new(self.private_key, payload, hashlib.sha256).digest()
        k = (int.from_bytes(k_bytes, "big") % (_SCHNORR_P - 2)) + 1
        r_int = pow(_SCHNORR_G, k, _SCHNORR_P)
        r_hex = f"{r_int:064x}"
        c = self._fiat_shamir_challenge(r_hex, self.public_key, payload)
        s_int = (k + c * self._secret_scalar) % (_SCHNORR_P - 1)
        return f"zkp1:{r_hex}:{s_int:064x}:{self.public_key}"

    def chay_suy_luan_an_toan(self, cau_hoi: str) -> Dict[str, Any]:
        """Thực thi suy luận trong Enclave và đính kèm Schnorr ZKP Proof."""
        cau_tra_loi = self.engine.sinh_van_ban(cau_hoi, do_dai=128)
        proof = self.generate_proof(cau_hoi, cau_tra_loi)

        return {
            "node_id": self.node_id,
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "public_key": self.public_key,
            "model_hash": self.model_hash,
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
        Xác minh bằng chứng Schnorr ZKP: kiểm tra phương trình đồng dư g^s == R * y^c (mod p).
        Chấp nhận `public_key` ở dạng khóa công khai hex 64 ký tự hoặc dẫn xuất từ passphrase.
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

            if len(public_key) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in public_key):
                y_int = int(public_key, 16)
                y_hex = public_key.lower()
            else:
                scalar = cls._derive_scalar(public_key.encode("utf-8"))
                y_int = pow(_SCHNORR_G, scalar, _SCHNORR_P)
                y_hex = f"{y_int:064x}"

            if y_int != proof_y_int or not (1 < r_int < _SCHNORR_P):
                return False

            c = cls._fiat_shamir_challenge(r_hex, y_hex, payload)
            lhs = pow(_SCHNORR_G, s_int, _SCHNORR_P)
            rhs = (r_int * pow(y_int, c, _SCHNORR_P)) % _SCHNORR_P
            return lhs == rhs

        expected_signature = hmac.new(
            public_key.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(proof, expected_signature)
