"""Gói Mật Mã Học & Xác Thực Suy Luận (`vietnamese_ai.crypto`).

Tách riêng các thành phần mật mã học cốt lõi (Merkle Tree tách miền RFC 6962, cam kết Schnorr/Fiat-Shamir
trên nhóm con cấp nguyên tố RFC 5114 và công cụ tự kiểm toán mật mã) theo kiến trúc tải lười PEP 562.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CayMerkleSuyLuan": ("vietnamese_ai.security.verifiable_inference", "CayMerkleSuyLuan"),
    "InferenceMerkleTree": ("vietnamese_ai.security.verifiable_inference", "InferenceMerkleTree"),
    "XacThucSuyLuanMaoHiem": ("vietnamese_ai.security.verifiable_inference", "XacThucSuyLuanMaoHiem"),
    "VerifiableInference": ("vietnamese_ai.security.verifiable_inference", "VerifiableInference"),
    "RFC5114_P": ("vietnamese_ai.crypto.schnorr", "RFC5114_P"),
    "RFC5114_Q": ("vietnamese_ai.crypto.schnorr", "RFC5114_Q"),
    "RFC5114_G": ("vietnamese_ai.crypto.schnorr", "RFC5114_G"),
    "THONG_SO_NHOM_SCHNORR": ("vietnamese_ai.crypto.schnorr", "THONG_SO_NHOM_SCHNORR"),
    "luy_thua_modulo_rfc5114": ("vietnamese_ai.crypto.schnorr", "luy_thua_modulo_rfc5114"),
    "suy_dan_khoa_bi_mat": ("vietnamese_ai.crypto.schnorr", "suy_dan_khoa_bi_mat"),
    "tinh_khoa_cong_khai_hex": ("vietnamese_ai.crypto.schnorr", "tinh_khoa_cong_khai_hex"),
    "kiem_tra_thanh_vien_nhom_con": ("vietnamese_ai.crypto.schnorr", "kiem_tra_thanh_vien_nhom_con"),
    "sinh_nonce_hedged": ("vietnamese_ai.crypto.schnorr", "sinh_nonce_hedged"),
    "tinh_thach_thuc_fiat_shamir": ("vietnamese_ai.crypto.schnorr", "tinh_thach_thuc_fiat_shamir"),
    "ky_schnorr_rfc5114": ("vietnamese_ai.crypto.schnorr", "ky_schnorr_rfc5114"),
    "xac_thuc_schnorr_rfc5114": ("vietnamese_ai.crypto.schnorr", "xac_thuc_schnorr_rfc5114"),
    "KiemToanMatMa": ("vietnamese_ai.crypto.audit", "KiemToanMatMa"),
    "CryptoAudit": ("vietnamese_ai.crypto.audit", "CryptoAudit"),
}

__all__: List[str] = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"Module 'vietnamese_ai.crypto' không có thuộc tính '{name}'")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)
