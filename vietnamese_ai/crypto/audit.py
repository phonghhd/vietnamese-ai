"""Công Cụ Tự Kiểm Toán Mật Mã Học & Sẵn Sàng Kiểm Định Độc Lập (Cryptographic Self-Audit & Audit Readiness).

Cung cấp bộ kiểm tra tự động các bất biến toán học và bảo mật của hệ thống `CayMerkleSuyLuan` và
`XacThucSuyLuanMaoHiem` trước khi gửi mã nguồn kiểm toán độc lập bởi bên thứ ba:
- Kiểm tra tính chất đại số nhóm con RFC 5114 Section 2.3 (độ dài bit p/q, ước số q | (p-1), cấp phần tử sinh g).
- Kiểm tra từ chối phần tử ngoài nhóm con nhỏ (small-subgroup confinement attack).
- Kiểm tra chống giả mạo Fiat-Shamir (forged transcript `(a, e, z)` không có khóa bí mật).
- Kiểm tra tính độc lập của nonce ngẫu nhiên giữa hai lần ký cùng một tensor.
- Kiểm tra tách miền lá/nút nội bộ RFC 6962 của `CayMerkleSuyLuan` chống tấn công tiền ảnh bậc hai.
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.security.verifiable_inference import (
    THONG_SO_NHOM_SCHNORR,
    CayMerkleSuyLuan,
    XacThucSuyLuanMaoHiem,
)


def _kiem_tra_nguyen_to_miller_rabin(n: int) -> bool:
    """Kiểm tra tính nguyên tố xác suất Miller-Rabin với bộ cơ số tất định cố định."""
    if n < 2:
        return False
    so_nho = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in so_nho:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in so_nho:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


class KiemToanMatMa:
    """Bộ kiểm toán nội bộ cho triển khai zkML và Merkle Tree trong `vietnamese_ai.crypto`."""

    @staticmethod
    def kiem_tra_tham_so_nhom() -> Dict[str, Any]:
        """Xác minh toán học các tham số nhóm Schnorr RFC 5114 (p, q, g)."""
        p = int(THONG_SO_NHOM_SCHNORR["p"])
        q = int(THONG_SO_NHOM_SCHNORR["q"])
        g = int(THONG_SO_NHOM_SCHNORR["g"])

        p_bits_ok = p.bit_length() == 2048
        q_bits_ok = q.bit_length() == 256
        chia_het_ok = (p - 1) % q == 0
        cap_g_ok = 1 < g < p and pow(g, q, p) == 1
        q_nguyen_to = _kiem_tra_nguyen_to_miller_rabin(q)
        p_nguyen_to = _kiem_tra_nguyen_to_miller_rabin(p)

        dat = all([p_bits_ok, q_bits_ok, chia_het_ok, cap_g_ok, q_nguyen_to, p_nguyen_to])
        return {
            "hang_muc": "tham_so_nhom_rfc5114",
            "dat": dat,
            "chi_tiet": {
                "do_dai_p_bit": p.bit_length(),
                "do_dai_q_bit": q.bit_length(),
                "q_chia_het_p_tru_1": chia_het_ok,
                "g_co_cap_q": cap_g_ok,
                "q_la_so_nguyen_to": q_nguyen_to,
                "p_la_so_nguyen_to": p_nguyen_to,
            },
        }

    @staticmethod
    def kiem_tra_chong_tan_cong_nhom_con_nho() -> Dict[str, Any]:
        """Kiểm tra cơ chế từ chối phần tử bậc 2 (p - 1) và phần tử ngoài nhóm con cấp q."""
        p = int(THONG_SO_NHOM_SCHNORR["p"])
        phan_tu_bac_2 = p - 1
        bi_tu_choi_bac_2 = not XacThucSuyLuanMaoHiem.xac_thuc_cam_ket(
            y=phan_tu_bac_2,
            a=phan_tu_bac_2,
            e=1,
            z=0,
            cho_phep_khong_rang_buoc=True,
        )
        phan_tu_ngoai_nhom = 2
        if pow(phan_tu_ngoai_nhom, int(THONG_SO_NHOM_SCHNORR["q"]), p) == 1:
            phan_tu_ngoai_nhom = p - 2
        bi_tu_choi_ngoai_nhom = not XacThucSuyLuanMaoHiem.xac_thuc_cam_ket(
            y=phan_tu_ngoai_nhom,
            a=phan_tu_ngoai_nhom,
            e=1,
            z=1,
            cho_phep_khong_rang_buoc=True,
        )
        return {
            "hang_muc": "chong_nhom_con_nho",
            "dat": bi_tu_choi_bac_2 and bi_tu_choi_ngoai_nhom,
            "chi_tiet": {
                "tu_choi_phan_tu_bac_2": bi_tu_choi_bac_2,
                "tu_choi_phan_tu_ngoai_nhom_con": bi_tu_choi_ngoai_nhom,
            },
        }

    @staticmethod
    def kiem_tra_chong_gia_mao_fiat_shamir() -> Dict[str, Any]:
        """Mô phỏng kẻ tấn công tạo bộ (a, e, z) thỏa g^z = a * y^e không biết khóa bí mật và kiểm tra từ chối."""
        p = int(THONG_SO_NHOM_SCHNORR["p"])
        q = int(THONG_SO_NHOM_SCHNORR["q"])
        g = int(THONG_SO_NHOM_SCHNORR["g"])

        verifier = XacThucSuyLuanMaoHiem()
        y = verifier.khoa_cong_khai
        tensor = np.array([1.25, -0.5, 3.75], dtype=np.float32)
        tensor_hash = verifier.bam_tensor(tensor)

        z_gia = secrets.randbelow(q - 1) + 1
        e_gia = secrets.randbelow(q - 1) + 1
        y_inv_e = pow(y, q - e_gia, p)
        a_gia = (pow(g, z_gia, p) * y_inv_e) % p

        bi_tu_choi_khi_rang_buoc = not XacThucSuyLuanMaoHiem.xac_thuc_cam_ket(
            y=y,
            a=a_gia,
            e=e_gia,
            z=z_gia,
            tensor_hash=tensor_hash,
        )
        bi_tu_choi_mac_dinh = not XacThucSuyLuanMaoHiem.xac_thuc_cam_ket(
            y=y,
            a=a_gia,
            e=e_gia,
            z=z_gia,
        )

        cam_ket_that = verifier.tao_cam_ket_zkml(tensor)
        hop_le_that = XacThucSuyLuanMaoHiem.xac_thuc_cam_ket_nghiem_ngat(cam_ket_that, tensor=tensor)

        cam_ket_that_2 = verifier.tao_cam_ket_zkml(tensor)
        nonce_khac_nhau = cam_ket_that["a"] != cam_ket_that_2["a"]

        return {
            "hang_muc": "chong_gia_mao_fiat_shamir_va_nonce",
            "dat": bi_tu_choi_khi_rang_buoc and bi_tu_choi_mac_dinh and hop_le_that and nonce_khac_nhau,
            "chi_tiet": {
                "tu_choi_bang_chung_gia_mao_khong_co_khoa": bi_tu_choi_khi_rang_buoc,
                "tu_choi_mac_dinh_khi_thieu_tensor_hash": bi_tu_choi_mac_dinh,
                "chap_nhan_bang_chung_nghiem_ngat": hop_le_that,
                "nonce_ngau_nhien_khac_nhau_moi_lan_ky": nonce_khac_nhau,
            },
        }

    @staticmethod
    def kiem_tra_tach_mien_merkle() -> Dict[str, Any]:
        """Kiểm tra tách miền 0x00/0x01 ngăn tấn công tiền ảnh bậc hai trên CayMerkleSuyLuan."""
        cay_goc = CayMerkleSuyLuan(["la_a", "la_b"])
        root_goc = cay_goc.lay_goc_merkle()
        la_gia = f"{cay_goc.la_list[0]}:{cay_goc.la_list[1]}".encode("ascii")
        cay_gia_mao = CayMerkleSuyLuan([la_gia])
        root_gia_mao = cay_gia_mao.lay_goc_merkle()
        khong_trung_root = root_goc != root_gia_mao
        return {
            "hang_muc": "tach_mien_merkle_rfc6962",
            "dat": khong_trung_root,
            "chi_tiet": {
                "root_hai_la_khac_root_nut_noi_bo_gia_la": khong_trung_root,
            },
        }

    @classmethod
    def chay_kiem_toan_tu_dong(cls) -> Dict[str, Any]:
        """Chạy toàn bộ bộ tự kiểm toán mật mã và xuất báo cáo chuẩn bị cho bên kiểm toán thứ ba."""
        cac_kiem_tra: List[Dict[str, Any]] = [
            cls.kiem_tra_tham_so_nhom(),
            cls.kiem_tra_chong_tan_cong_nhom_con_nho(),
            cls.kiem_tra_chong_gia_mao_fiat_shamir(),
            cls.kiem_tra_tach_mien_merkle(),
        ]
        tat_ca_dat = all(item["dat"] for item in cac_kiem_tra)
        return {
            "tong_quan_dat": tat_ca_dat,
            "chuan_mat_ma": THONG_SO_NHOM_SCHNORR["chuan"],
            "so_hang_muc_dat": sum(1 for item in cac_kiem_tra if item["dat"]),
            "tong_hang_muc": len(cac_kiem_tra),
            "ket_qua_chi_tiet": cac_kiem_tra,
            "khuyen_nghi_kiem_toan_doc_lap": [
                "Kiểm toán độc lập mô hình mối đe dọa và kênh phụ thời gian (timing side-channel) trên Python big-int.",
                "Đánh giá rõ ranh giới: Schnorr/Fiat-Shamir chỉ chứng thực danh tính nút & toàn vẹn băm, không thay thế zk-SNARK/STARK cho toàn mạch neural.",
                "Quản lý vòng đời khóa bí mật (KMS/HSM) và danh sách khóa công khai tin cậy của từng nút Swarm.",
            ],
        }


CryptoAudit = KiemToanMatMa
