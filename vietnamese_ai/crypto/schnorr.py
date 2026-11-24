"""Lõi Mật Mã Schnorr / Fiat-Shamir Chuẩn RFC 5114 Section 2.3 (`vietnamese_ai.crypto.schnorr`).

Nguồn chân lý duy nhất cho tham số nhóm MODP-2048-256 (trường 2048-bit p, nhóm con cấp nguyên tố 256-bit q),
kiểm tra thành viên nhóm con, sinh nonce CSPRNG kết hợp HMAC-SHA256 và ký/xác thực Schnorr dùng chung cho
`XacThucSuyLuanMaoHiem`, `SecureEdgeNode` và `WebBrowserNode`.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Dict, Tuple

try:
    import vietnamese_ai_rust as _rust_core  # type: ignore
except ImportError:
    try:
        from vietnamese_ai import vietnamese_ai_rust as _rust_core  # type: ignore
    except ImportError:
        _rust_core = None

RFC5114_P: int = int(
    "87A8E61DB4B6663CFFBBD19C651959998CEEF608660DD0F2"
    "5D2CEED4435E3B00E00DF8F1D61957D4FAF7DF4561B2AA30"
    "16C3D91134096FAA3BF4296D830E9A7C209E0C6497517ABD"
    "5A8A9D306BCF67ED91F9E6725B4758C022E0B1EF4275BF7B"
    "6C5BFC11D45F9088B941F54EB1E59BB8BC39A0BF12307F5C"
    "4FDB70C581B23F76B63ACAE1CAA6B7902D52526735488A0E"
    "F13C6D9A51BFA4AB3AD8347796524D8EF6A167B5A41825D9"
    "67E144E5140564251CCACB83E6B486F6B3CA3F7971506026"
    "C0B857F689962856DED4010ABD0BE621C3A3960A54E710C3"
    "75F26375D7014103A4B54330C198AF126116D2276E11715F"
    "693877FAD7EF09CADB094AE91E1A1597",
    16,
)
RFC5114_Q: int = int(
    "8CF83642A709A097B447997640129DA299B1A47D1EB3750B"
    "A308B0FE64F5FBD3",
    16,
)
RFC5114_G: int = int(
    "3FB32C9B73134D0B2E77506660EDBD484CA7B18F21EF2054"
    "07F4793A1A0BA12510DBC15077BE463FFF4FED4AAC0BB555"
    "BE3A6C1B0C6B47B1BC3773BF7E8C6F62901228F8C28CBB18"
    "A55AE31341000A650196F931C77A57F2DDF463E5E9EC144B"
    "777DE62AAAB8A8628AC376D282D6ED3864E67982428EBC83"
    "1D14348F6F2F9193B5045AF2767164E1DFC967C1FB3F2E55"
    "A4BD1BFFE83B9C80D052B985D182EA0ADB2A3B7313D3FE14"
    "C8484B1E052588B9B7D2BBD2DF016199ECD06E1557CD0915"
    "B3353BBB64E0EC377FD028370DF92B52C7891428CDC67EB6"
    "184B523D1DB246C32F63078490F00EF8D647D148D4795451"
    "5E2327CFEF98C582664B4C0F6CC41659",
    16,
)

P_HEX_WIDTH: int = 512
Q_HEX_WIDTH: int = 64

THONG_SO_NHOM_SCHNORR: Dict[str, Any] = {
    "chuan": "RFC 5114 Section 2.3 (2048-bit MODP Group with 256-bit Prime Order Subgroup)",
    "do_dai_p_bit": RFC5114_P.bit_length(),
    "do_dai_q_bit": RFC5114_Q.bit_length(),
    "ham_bam": "SHA-256",
    "tach_mien_merkle": "RFC-6962 (0x00 leaf / 0x01 node)",
    "nguon_ngau_nhien": "secrets (OS CSPRNG) + HMAC-SHA256 hedged nonce",
    "p": RFC5114_P,
    "q": RFC5114_Q,
    "g": RFC5114_G,
}


def luy_thua_modulo_rfc5114(co_so: int, so_mu: int) -> int:
    """Tính `co_so^so_mu mod RFC5114_P`, tự động dùng nhân Montgomery 2048-bit Rust nếu sẵn sàng."""
    if _rust_core is not None and hasattr(_rust_core, "fast_rfc5114_modpow"):
        try:
            co_so_bytes = (co_so % RFC5114_P).to_bytes(256, "big")
            so_mu_bytes = (so_mu % RFC5114_Q if so_mu < RFC5114_Q else so_mu).to_bytes(32, "big")
            kq_bytes = bytes(_rust_core.fast_rfc5114_modpow(co_so_bytes, so_mu_bytes))
            return int.from_bytes(kq_bytes, "big")
        except Exception:
            pass
    return pow(co_so, so_mu, RFC5114_P)


def suy_dan_khoa_bi_mat(secret_bytes: bytes, tien_to_mien: bytes = b"zkml1-key:") -> int:
    """Dẫn xuất khóa vô hướng trong nhóm con cấp nguyên tố `[1, q - 1]`."""
    digest = hashlib.sha256(tien_to_mien + secret_bytes).digest()
    return (int.from_bytes(digest, "big") % (RFC5114_Q - 1)) + 1


def tinh_khoa_cong_khai_hex(khoa_vo_huong: int) -> str:
    """Tính khóa công khai `y = g^x mod p` định dạng hex cố định 512 ký tự."""
    y_int = luy_thua_modulo_rfc5114(RFC5114_G, khoa_vo_huong)
    return f"{y_int:0{P_HEX_WIDTH}x}"


def kiem_tra_thanh_vien_nhom_con(phan_tu: int) -> bool:
    """Kiểm tra phần tử thuộc nhóm con cấp nguyên tố q modulo p (chống tấn công nhóm con nhỏ)."""
    if not (1 < phan_tu < RFC5114_P):
        return False
    return luy_thua_modulo_rfc5114(phan_tu, RFC5114_Q) == 1


def sinh_nonce_hedged(secret_bytes: bytes, payload: bytes, tien_to_mien: bytes = b"zkml1-nonce:") -> int:
    """Sinh nonce k đều trong `[1, q - 1]` kết hợp OS CSPRNG (`secrets`) và HMAC-SHA256."""
    entropy = secrets.token_bytes(32)
    digest = hmac.new(secret_bytes, tien_to_mien + payload + entropy, hashlib.sha256).digest()
    return (int.from_bytes(digest, "big") % (RFC5114_Q - 1)) + 1


def tinh_thach_thuc_fiat_shamir(
    r_hex: str,
    y_hex: str,
    payload: bytes,
    tien_to_mien: bytes = b"edge-zkp1:",
) -> int:
    """Tính giá trị thách thức Fiat-Shamir tách miền trong `[1, q - 1]`."""
    h = hashlib.sha256(tien_to_mien)
    h.update(r_hex.lower().encode("ascii"))
    h.update(b":")
    h.update(y_hex.lower().encode("ascii"))
    h.update(b":")
    h.update(payload)
    return (int.from_bytes(h.digest(), "big") % (RFC5114_Q - 1)) + 1


def ky_schnorr_rfc5114(
    secret_bytes: bytes,
    khoa_vo_huong: int,
    public_key_hex: str,
    payload: bytes,
    tien_to_nonce: bytes = b"edge-nonce:",
    tien_to_thach_thuc: bytes = b"edge-zkp1:",
) -> Tuple[str, str, str]:
    """Sinh bộ `(r_hex, s_hex, proof_str)` theo lược đồ Schnorr/Fiat-Shamir chuẩn RFC 5114."""
    k = sinh_nonce_hedged(secret_bytes, payload, tien_to_mien=tien_to_nonce)
    r_int = luy_thua_modulo_rfc5114(RFC5114_G, k)
    r_hex = f"{r_int:0{P_HEX_WIDTH}x}"
    c = tinh_thach_thuc_fiat_shamir(r_hex, public_key_hex, payload, tien_to_mien=tien_to_thach_thuc)
    s_int = (k + c * khoa_vo_huong) % RFC5114_Q
    s_hex = f"{s_int:0{Q_HEX_WIDTH}x}"
    return r_hex, s_hex, f"zkp1:{r_hex}:{s_hex}:{public_key_hex}"


def xac_thuc_schnorr_rfc5114(
    r_int: int,
    s_int: int,
    y_int: int,
    thach_thuc: int,
) -> bool:
    """Kiểm tra thành viên nhóm con và phương trình Schnorr `g^s == R * y^c (mod p)`."""
    if not (0 <= s_int < RFC5114_Q):
        return False
    if not (1 <= thach_thuc < RFC5114_Q):
        return False
    if not kiem_tra_thanh_vien_nhom_con(r_int) or not kiem_tra_thanh_vien_nhom_con(y_int):
        return False
    lhs = luy_thua_modulo_rfc5114(RFC5114_G, s_int)
    rhs = (r_int * luy_thua_modulo_rfc5114(y_int, thach_thuc)) % RFC5114_P
    return lhs == rhs


__all__ = [
    "RFC5114_P",
    "RFC5114_Q",
    "RFC5114_G",
    "P_HEX_WIDTH",
    "Q_HEX_WIDTH",
    "THONG_SO_NHOM_SCHNORR",
    "luy_thua_modulo_rfc5114",
    "suy_dan_khoa_bi_mat",
    "tinh_khoa_cong_khai_hex",
    "kiem_tra_thanh_vien_nhom_con",
    "sinh_nonce_hedged",
    "tinh_thach_thuc_fiat_shamir",
    "ky_schnorr_rfc5114",
    "xac_thuc_schnorr_rfc5114",
]
