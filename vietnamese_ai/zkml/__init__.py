"""Gói zkML & Xác Thực Suy Luận Phi Tương Tác (`vietnamese_ai.zkml`).

Namespace chuyên biệt tái xuất lười (PEP 562) các thành phần Proof of Inference (PoI),
Merkle Tree và tự kiểm toán mật mã từ `vietnamese_ai.crypto`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CayMerkleSuyLuan": ("vietnamese_ai.security.verifiable_inference", "CayMerkleSuyLuan"),
    "InferenceMerkleTree": ("vietnamese_ai.security.verifiable_inference", "InferenceMerkleTree"),
    "XacThucSuyLuanMaoHiem": ("vietnamese_ai.security.verifiable_inference", "XacThucSuyLuanMaoHiem"),
    "VerifiableInference": ("vietnamese_ai.security.verifiable_inference", "VerifiableInference"),
    "THONG_SO_NHOM_SCHNORR": ("vietnamese_ai.security.verifiable_inference", "THONG_SO_NHOM_SCHNORR"),
    "KiemToanMatMa": ("vietnamese_ai.crypto.audit", "KiemToanMatMa"),
    "CryptoAudit": ("vietnamese_ai.crypto.audit", "CryptoAudit"),
}

__all__: List[str] = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"Module 'vietnamese_ai.zkml' không có thuộc tính '{name}'")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)
