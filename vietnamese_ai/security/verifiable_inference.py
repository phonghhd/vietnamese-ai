"""Hệ Thống Xác Thực Suy Luận Mật Mã Học (Verifiable AI Inference - Merkle Tree & Lightweight zkML).

Cung cấp cơ chế chứng thực tính toàn vẹn và cam kết thực thi cho suy luận AI phân tán (Proof of Inference - PoI):
- Cây băm mật mã Merkle Tree tách miền lá/nút nội bộ (RFC 6962 style 0x00/0x01) chống tấn công tiền ảnh bậc hai.
- Lược đồ cam kết phi tương tác Schnorr / Fiat-Shamir trên nhóm con cấp nguyên tố 256-bit trong trường 2048-bit
  theo chuẩn RFC 5114 Section 2.3 (MODP-2048-256), dùng bộ sinh số ngẫu nhiên mật mã học CSPRNG (`secrets`).
- Cơ chế kiểm tra thành viên nhóm con (subgroup membership), phát hiện và từ chối nút Byzantine gian lận.

Lưu ý phạm vi mật mã:
Lược đồ Schnorr/Fiat-Shamir ở đây chứng minh quyền sở hữu khóa bí mật của nút tính toán và ràng buộc toàn vẹn
băm SHA-256 của tensor/Merkle root; đây KHÔNG phải mạch chứng minh toàn bộ phép nhân ma trận kiểu zk-SNARK/STARK.
Cần kiểm toán độc lập bởi bên thứ ba trước khi triển khai trong hệ thống yêu cầu đảm bảo mật mã nghiêm ngặt.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from vietnamese_ai.crypto.schnorr import (
    P_HEX_WIDTH as _P_HEX_WIDTH,
)
from vietnamese_ai.crypto.schnorr import (
    Q_HEX_WIDTH as _Q_HEX_WIDTH,
)
from vietnamese_ai.crypto.schnorr import (
    RFC5114_G as _RFC5114_G,
)
from vietnamese_ai.crypto.schnorr import (
    RFC5114_P as _RFC5114_P,
)
from vietnamese_ai.crypto.schnorr import (
    RFC5114_Q as _RFC5114_Q,
)
from vietnamese_ai.crypto.schnorr import (
    THONG_SO_NHOM_SCHNORR,
    kiem_tra_thanh_vien_nhom_con,
    luy_thua_modulo_rfc5114,
    sinh_nonce_hedged,
    suy_dan_khoa_bi_mat,
    tinh_khoa_cong_khai_hex,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

_LEAF_DOMAIN_PREFIX = b"\x00"
_NODE_DOMAIN_PREFIX = b"\x01"


class CayMerkleSuyLuan:
    """
    Cây băm mật mã Merkle Tree cho các bước suy luận AI (Inference Execution Merkle Tree):
    - Đóng gói chuỗi băm của hidden states, layer IDs và model weights hash với tách miền 0x00/0x01.
    - Sinh Root Hash và Merkle Audit Path (Proof) với độ phức tạp kiểm tra O(log N).
    """

    def __init__(
        self,
        danh_sach_leaf_data: Optional[Sequence[Union[str, bytes, np.ndarray]]] = None,
    ) -> None:
        self.la_list: List[str] = []
        self.cay: List[List[str]] = []
        if danh_sach_leaf_data:
            for item in danh_sach_leaf_data:
                self.them_la(item)
            self.xay_dung_cay()

    @staticmethod
    def _bam_nut_noi_bo(trai: str, phai: str) -> str:
        """Băm hai nút con với tiền tố tách miền 0x01 chống tấn công tiền ảnh bậc hai."""
        return hashlib.sha256(_NODE_DOMAIN_PREFIX + f"{trai}:{phai}".encode("ascii")).hexdigest()

    @staticmethod
    def bam_du_lieu(du_lieu: Union[str, bytes, np.ndarray]) -> str:
        """Hàm băm SHA-256 tách miền lá (0x00) cho chuỗi, byte hoặc mảng tensor NumPy."""
        h = hashlib.sha256(_LEAF_DOMAIN_PREFIX)
        if isinstance(du_lieu, str):
            h.update(du_lieu.encode("utf-8"))
        elif isinstance(du_lieu, bytes):
            h.update(du_lieu)
        elif isinstance(du_lieu, np.ndarray):
            h.update(du_lieu.tobytes())
        else:
            h.update(str(du_lieu).encode("utf-8"))
        return h.hexdigest()

    def them_la(self, du_lieu: Union[str, bytes, np.ndarray]) -> str:
        """Thêm một nút lá vào cây băm."""
        leaf_hash = self.bam_du_lieu(du_lieu)
        self.la_list.append(leaf_hash)
        return leaf_hash

    def xay_dung_cay(self) -> str:
        """Xây dựng Merkle Tree từ danh sách các nút lá và trả về Merkle Root."""
        if not self.la_list:
            return hashlib.sha256(b"empty_merkle_tree").hexdigest()

        cac_tang: List[List[str]] = [list(self.la_list)]
        tang_hien_tai = cac_tang[0]

        while len(tang_hien_tai) > 1:
            tang_tiep_theo: List[str] = []
            for i in range(0, len(tang_hien_tai), 2):
                trai = tang_hien_tai[i]
                phai = tang_hien_tai[i + 1] if i + 1 < len(tang_hien_tai) else trai
                nut_cha = self._bam_nut_noi_bo(trai, phai)
                tang_tiep_theo.append(nut_cha)
            cac_tang.append(tang_tiep_theo)
            tang_hien_tai = tang_tiep_theo

        self.cay = cac_tang
        return self.lay_goc_merkle()

    def lay_goc_merkle(self) -> str:
        """Lấy giá trị băm của gốc cây Merkle Root."""
        if not self.cay or not self.cay[-1]:
            return ""
        return self.cay[-1][0]

    def lay_root_hash(self) -> str:
        """Alias tương thích cho lay_goc_merkle."""
        return self.lay_goc_merkle()

    def sinh_merkle_proof(self, index_la: int) -> Dict[str, Any]:
        """
        Sinh bằng chứng kiểm toán Merkle Proof cho lá tại vị trí `index_la`.
        Độ phức tạp O(log N).
        """
        if not self.cay or index_la < 0 or index_la >= len(self.la_list):
            raise IndexError("Index nút lá không hợp lệ hoặc cây chưa được xây dựng")

        proof_path: List[Dict[str, str]] = []
        idx = index_la

        for tang in self.cay[:-1]:
            la_ben_trai = idx % 2 == 0
            if la_ben_trai:
                dong_hanh_idx = idx + 1 if idx + 1 < len(tang) else idx
                huong = "phai"
            else:
                dong_hanh_idx = idx - 1
                huong = "trai"

            proof_path.append(
                {
                    "huong": huong,
                    "hash": tang[dong_hanh_idx],
                }
            )
            idx //= 2

        return {
            "root": self.lay_goc_merkle(),
            "leaf_hash": self.la_list[index_la],
            "leaf_index": index_la,
            "proof_path": proof_path,
        }

    def sinh_bang_chung(self, index_la: int) -> List[Dict[str, str]]:
        """Sinh danh sách các bước bằng chứng kiểm toán Merkle Audit Path cho nút lá."""
        if not self.la_list or index_la < 0 or index_la >= len(self.la_list):
            return []
        if not self.cay:
            self.xay_dung_cay()
        proof_dict = self.sinh_merkle_proof(index_la)
        return proof_dict["proof_path"]

    def kiem_tra_bang_chung(
        self,
        du_lieu: Union[str, bytes, np.ndarray],
        index_la: int,
        proof_path: List[Dict[str, str]],
        root: str,
    ) -> bool:
        """Kiểm tra một lá dữ liệu có thuộc về cây với Merkle Root hay không."""
        if index_la < 0:
            return False
        leaf_hash = self.bam_du_lieu(du_lieu)
        current = leaf_hash
        idx = index_la

        for step in proof_path:
            sibling = step.get("hash", "")
            huong = step.get("huong", "")
            la_chan = idx % 2 == 0
            if la_chan and huong != "phai":
                return False
            if not la_chan and huong != "trai":
                return False

            if huong == "phai":
                current = self._bam_nut_noi_bo(current, sibling)
            else:
                current = self._bam_nut_noi_bo(sibling, current)
            idx //= 2

        return hmac.compare_digest(current.lower(), root.lower())

    @classmethod
    def xac_minh_merkle_proof(
        cls,
        root_ky_vong: str,
        leaf_hash: str,
        proof_path: List[Dict[str, str]],
    ) -> bool:
        """Xác minh tính hợp lệ của một Merkle Proof độc lập."""
        current_hash = leaf_hash
        for step in proof_path:
            sibling_hash = step.get("hash", "")
            huong = step.get("huong", "")
            if huong == "phai":
                current_hash = cls._bam_nut_noi_bo(current_hash, sibling_hash)
            elif huong == "trai":
                current_hash = cls._bam_nut_noi_bo(sibling_hash, current_hash)
            else:
                return False

        return hmac.compare_digest(current_hash.lower(), root_ky_vong.lower())


@gan_nhan_tinh_nang(
    TrangThaiTinhNang.EXPERIMENTAL,
    mo_ta=(
        "Cam kết mật mã Schnorr/Fiat-Shamir trên nhóm RFC 5114 (2048-bit p, 256-bit q) "
        "và Merkle Tree kiểm toán suy luận; khuyến nghị kiểm toán mật mã độc lập trước khi dùng cho hệ thống trọng yếu."
    ),
)
class XacThucSuyLuanMaoHiem:
    """
    Bộ Xác Thực Suy Luận AI Mật Mã Học (Verifiable AI Inference Engine):
    - Đóng gói cam kết thực thi mô hình (Model Execution Commitment) trên nhóm con cấp nguyên tố RFC 5114.
    - Sinh bằng chứng Schnorr/Fiat-Shamir NIZK dùng nguồn ngẫu nhiên CSPRNG (`secrets`).
    - Kiểm tra thành viên nhóm con (subgroup membership), phát hiện và gắn cờ các node Byzantine gian lận.
    """

    def __init__(self, private_key: Optional[str] = None) -> None:
        self.secret = private_key or secrets.token_hex(32)
        self.scalar_secret = suy_dan_khoa_bi_mat(self.secret.encode("utf-8"), tien_to_mien=b"zkml1-key:")
        self.public_key = tinh_khoa_cong_khai_hex(self.scalar_secret)

    @property
    def khoa_bi_mat(self) -> int:
        """Khóa bí mật số nguyên trong nhóm con cấp nguyên tố q."""
        return self.scalar_secret

    @property
    def khoa_cong_khai(self) -> int:
        """Khóa công khai số nguyên y = g^secret mod p."""
        return int(self.public_key, 16)

    @staticmethod
    def bam_tensor(tensor: np.ndarray) -> str:
        """Băm tensor NumPy an toàn SHA-256."""
        return hashlib.sha256(tensor.tobytes()).hexdigest()

    @staticmethod
    def _la_thanh_vien_nhom_con(phan_tu: int) -> bool:
        """Kiểm tra phần tử thuộc nhóm con cấp nguyên tố q modulo p (chống tấn công nhóm con nhỏ)."""
        return kiem_tra_thanh_vien_nhom_con(phan_tu)

    def _sinh_nonce_ngau_nhien(self, payload: bytes) -> int:
        """Sinh nonce k đều trong [1, q - 1] kết hợp CSPRNG hệ điều hành và HMAC-SHA256."""
        return sinh_nonce_hedged(self.secret.encode("utf-8"), payload, tien_to_mien=b"zkml1-nonce:")

    @staticmethod
    def _tinh_thach_thuc_tensor(a: int, y: int, tensor_hash: str) -> int:
        """Tính giá trị thách thức Fiat-Shamir e gắn chặt với cam kết a, khóa công khai y và băm tensor."""
        c_hash = hashlib.sha256(
            f"zkml1-tensor:{a:0{_P_HEX_WIDTH}x}:{y:0{_P_HEX_WIDTH}x}:{tensor_hash}".encode("ascii")
        ).digest()
        return (int.from_bytes(c_hash, "big") % (_RFC5114_Q - 1)) + 1

    @staticmethod
    def _tinh_thach_thuc_layer(r_hex: str, pub_key: str, payload: bytes) -> int:
        """Tính giá trị thách thức Fiat-Shamir c gắn chặt với cam kết R, khóa công khai và metadata tầng suy luận."""
        c_hash = hashlib.sha256(
            f"zkml1-layer:{r_hex.lower()}:{pub_key.lower()}:{payload.hex()}".encode("ascii")
        ).digest()
        return (int.from_bytes(c_hash, "big") % (_RFC5114_Q - 1)) + 1

    def tao_cam_ket_zkml(
        self,
        tensor: np.ndarray,
        layer_id: int = 0,
        node_id: str = "node_local",
    ) -> Dict[str, Any]:
        """
        Tạo bằng chứng cam kết mật mã học Fiat-Shamir zkML cho tensor tính toán:
        - Sử dụng lược đồ nhận thực Schnorr NIZK phi tương tác trên nhóm RFC 5114 (2048-bit p, 256-bit q).
        - Thỏa phương trình kiểm chứng g^z == a * (y^e) (mod p) với z = (k + e * x) mod q.
        """
        h_tensor = self.bam_tensor(tensor)
        payload = f"{node_id}:{layer_id}:{h_tensor}".encode("utf-8")
        k = self._sinh_nonce_ngau_nhien(payload)
        a = luy_thua_modulo_rfc5114(_RFC5114_G, k)
        y = self.khoa_cong_khai

        e = self._tinh_thach_thuc_tensor(a=a, y=y, tensor_hash=h_tensor)
        z = (k + e * self.scalar_secret) % _RFC5114_Q

        chuoi_cam_ket = (
            f"zkml1:{a:0{_P_HEX_WIDTH}x}:{z:0{_Q_HEX_WIDTH}x}:"
            f"{y:0{_P_HEX_WIDTH}x}:{e:0{_Q_HEX_WIDTH}x}"
        )
        return {
            "chuoi_cam_ket": chuoi_cam_ket,
            "tensor_hash": h_tensor,
            "y": y,
            "a": a,
            "e": e,
            "z": z,
            "node_id": node_id,
            "layer_id": layer_id,
            "chuan_mat_ma": THONG_SO_NHOM_SCHNORR["chuan"],
        }

    @classmethod
    def xac_thuc_cam_ket(
        cls,
        y: int,
        a: int,
        e: int,
        z: int,
        tensor_hash: Optional[str] = None,
    ) -> bool:
        """
        Xác thực phương trình Schnorr Fiat-Shamir NIZK: g^z == a * y^e (mod p).
        Kiểm tra miền giá trị, thành viên nhóm con cấp q, và đối chiếu lại thách thức e nếu cung cấp `tensor_hash`.
        """
        if not (0 < e < _RFC5114_Q) or not (0 <= z < _RFC5114_Q):
            return False
        if not cls._la_thanh_vien_nhom_con(y) or not cls._la_thanh_vien_nhom_con(a):
            return False
        if tensor_hash is not None:
            e_ky_vong = cls._tinh_thach_thuc_tensor(a=a, y=y, tensor_hash=tensor_hash)
            if e != e_ky_vong:
                return False
        lhs = luy_thua_modulo_rfc5114(_RFC5114_G, z)
        rhs = (a * luy_thua_modulo_rfc5114(y, e)) % _RFC5114_P
        return lhs == rhs

    @classmethod
    def xac_nhan_cam_ket_chuoi(
        cls,
        chuoi_cam_ket: str,
        tensor_hash: Optional[str] = None,
    ) -> bool:
        """Xác thực chuỗi cam kết zkML định dạng `zkml1:a:z:y:e` (ràng buộc Fiat-Shamir nếu có `tensor_hash`)."""
        if not isinstance(chuoi_cam_ket, str) or not chuoi_cam_ket.startswith("zkml1:"):
            return False
        parts = chuoi_cam_ket.split(":")
        if len(parts) != 5:
            return False
        try:
            a = int(parts[1], 16)
            z = int(parts[2], 16)
            y = int(parts[3], 16)
            e = int(parts[4], 16)
        except ValueError:
            return False
        return cls.xac_thuc_cam_ket(y=y, a=a, e=e, z=z, tensor_hash=tensor_hash)

    @classmethod
    def xac_thuc_cam_ket_nghiem_ngat(
        cls,
        cam_ket: Dict[str, Any],
        tensor: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Xác thực nghiêm ngặt gói cam kết zkML từ `tao_cam_ket_zkml`:
        bắt buộc kiểm tra đối chiếu `tensor_hash`, tính lại thách thức Fiat-Shamir `e`,
        kiểm tra nhất quán chuỗi `chuoi_cam_ket`, thành viên nhóm con và phương trình Schnorr.
        """
        tensor_hash = cam_ket.get("tensor_hash")
        chuoi_cam_ket = cam_ket.get("chuoi_cam_ket")
        if not isinstance(tensor_hash, str) or len(tensor_hash) != 64:
            return False
        if tensor is not None and not hmac.compare_digest(cls.bam_tensor(tensor), tensor_hash):
            return False
        if not isinstance(chuoi_cam_ket, str):
            return False
        if not cls.xac_nhan_cam_ket_chuoi(chuoi_cam_ket, tensor_hash=tensor_hash):
            return False
        parts = chuoi_cam_ket.split(":")
        try:
            a_str, z_str, y_str, e_str = (
                int(parts[1], 16),
                int(parts[2], 16),
                int(parts[3], 16),
                int(parts[4], 16),
            )
        except (IndexError, ValueError):
            return False
        return (
            cam_ket.get("a") == a_str
            and cam_ket.get("z") == z_str
            and cam_ket.get("y") == y_str
            and cam_ket.get("e") == e_str
        )

    def ky_ket_qua(self, thong_diep: str) -> str:
        """Ký số kết quả suy luận bằng HMAC-SHA256 với khóa bí mật của node."""
        return hmac.new(self.secret.encode("utf-8"), thong_diep.encode("utf-8"), hashlib.sha256).hexdigest()

    def xac_thuc_chu_ky(self, thong_diep: str, chu_ky: str) -> bool:
        """Xác thực chữ ký số HMAC-SHA256 theo thời gian hằng (constant-time)."""
        ky_dung = self.ky_ket_qua(thong_diep)
        return hmac.compare_digest(ky_dung, chu_ky)

    def sinh_cam_ket_layer(
        self,
        node_id: str,
        layer_id: int,
        tensor_dau_vao: np.ndarray,
        tensor_dau_ra: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Sinh cam kết mật mã học (Layer Activation Commitment) cho một bước tính toán của Node:
        - Tính băm SHA-256 đầu vào, đầu ra và Merkle Root tách miền cho các phân đoạn tensor đầu ra.
        - Sinh chữ ký Schnorr/Fiat-Shamir trên nhóm con RFC 5114.
        """
        h_in = hashlib.sha256(tensor_dau_vao.tobytes()).hexdigest()
        h_out = hashlib.sha256(tensor_dau_ra.tobytes()).hexdigest()

        slices = [
            tensor_dau_ra.flat[i : i + 32].tobytes()
            for i in range(0, min(len(tensor_dau_ra.flat), 256), 32)
        ]
        if not slices:
            slices = [tensor_dau_ra.tobytes()]

        merkle_tree = CayMerkleSuyLuan(slices)
        merkle_root = merkle_tree.lay_goc_merkle()

        payload = f"{node_id}:{layer_id}:{h_in}:{h_out}:{merkle_root}".encode("utf-8")
        k = self._sinh_nonce_ngau_nhien(payload)
        r_int = luy_thua_modulo_rfc5114(_RFC5114_G, k)
        r_hex = f"{r_int:0{_P_HEX_WIDTH}x}"

        c = self._tinh_thach_thuc_layer(r_hex=r_hex, pub_key=self.public_key, payload=payload)
        s_int = (k + c * self.scalar_secret) % _RFC5114_Q

        proof_zk = (
            f"zkml1:{r_hex}:{s_int:0{_Q_HEX_WIDTH}x}:"
            f"{self.public_key}:{c:0{_Q_HEX_WIDTH}x}"
        )

        return {
            "node_id": node_id,
            "layer_id": layer_id,
            "input_hash": h_in,
            "output_hash": h_out,
            "merkle_root": merkle_root,
            "proof_zk": proof_zk,
            "public_key": self.public_key,
            "chuan_mat_ma": THONG_SO_NHOM_SCHNORR["chuan"],
            "thoi_diem": time.time(),
        }

    @classmethod
    def xac_minh_cam_ket(
        cls,
        cam_ket: Dict[str, Any],
        tensor_dau_vao: Optional[np.ndarray] = None,
        tensor_dau_ra: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Kiểm tra tính hợp lệ của cam kết thực thi từ một Node:
        1. Đối chiếu băm SHA-256 với tensor đầu vào/đầu ra nếu được cung cấp (hỗ trợ FP16 LAN).
        2. Xác minh nhất quán khóa công khai, thành viên nhóm con cấp q, thách thức Fiat-Shamir và phương trình Schnorr.
        """
        node_id = cam_ket.get("node_id", "")
        layer_id = cam_ket.get("layer_id", 0)
        h_in = cam_ket.get("input_hash", "")
        h_out = cam_ket.get("output_hash", "")
        merkle_root = cam_ket.get("merkle_root", "")
        proof = cam_ket.get("proof_zk", "")
        pub_key = cam_ket.get("public_key", "")

        if tensor_dau_vao is not None:
            hash_vao = hashlib.sha256(tensor_dau_vao.tobytes()).hexdigest()
            if not hmac.compare_digest(hash_vao, h_in):
                h_in_fp16 = hashlib.sha256(
                    tensor_dau_vao.astype(np.float16).astype(np.float32).tobytes()
                ).hexdigest()
                h_in_raw16 = hashlib.sha256(tensor_dau_vao.astype(np.float16).tobytes()).hexdigest()
                if not (hmac.compare_digest(h_in_fp16, h_in) or hmac.compare_digest(h_in_raw16, h_in)):
                    return False

        if tensor_dau_ra is not None:
            hash_ra = hashlib.sha256(tensor_dau_ra.tobytes()).hexdigest()
            if not hmac.compare_digest(hash_ra, h_out):
                h_out_fp16 = hashlib.sha256(
                    tensor_dau_ra.astype(np.float16).astype(np.float32).tobytes()
                ).hexdigest()
                h_out_raw16 = hashlib.sha256(tensor_dau_ra.astype(np.float16).tobytes()).hexdigest()
                if not (hmac.compare_digest(h_out_fp16, h_out) or hmac.compare_digest(h_out_raw16, h_out)):
                    return False

        if not isinstance(proof, str) or not proof.startswith("zkml1:"):
            return False

        parts = proof.split(":")
        if len(parts) == 5:
            _, r_hex, s_hex, proof_pub, c_hex = parts
        elif len(parts) == 4:
            _, r_hex, s_hex, proof_pub = parts
            c_hex = None
        else:
            return False

        if not isinstance(pub_key, str) or not pub_key:
            return False
        if not hmac.compare_digest(pub_key.lower(), proof_pub.lower()):
            return False

        try:
            r_int = int(r_hex, 16)
            s_int = int(s_hex, 16)
            y_int = int(proof_pub, 16)
        except ValueError:
            return False

        if not (0 <= s_int < _RFC5114_Q):
            return False
        if not cls._la_thanh_vien_nhom_con(r_int) or not cls._la_thanh_vien_nhom_con(y_int):
            return False

        payload = f"{node_id}:{layer_id}:{h_in}:{h_out}:{merkle_root}".encode("utf-8")
        c = cls._tinh_thach_thuc_layer(r_hex=r_hex, pub_key=pub_key, payload=payload)

        if c_hex is not None:
            try:
                if int(c_hex, 16) != c:
                    return False
            except ValueError:
                return False

        lhs = luy_thua_modulo_rfc5114(_RFC5114_G, s_int)
        rhs = (r_int * luy_thua_modulo_rfc5114(y_int, c)) % _RFC5114_P
        return lhs == rhs


class KiemChungFreivalds:
    """
    Bộ Kiểm Chứng Phép Nhân Ma Trận Xác Suất Freivalds (Probabilistic O(n^2) Linear Layer Verifier).

    Kiểm chứng đầu ra tầng tuyến tính `Y = X @ W + b` mà không cần tính lại toàn bộ `X @ W` tốn O(n^3):
    sinh `t` vector Rademacher ngẫu nhiên `r ∈ {-1, +1}^n` và đối chiếu `X @ (W @ r) + b^T r ≈ Y @ r`.
    Xác suất bỏ lọt gian lận bị chặn trên bởi `2^{-t}` (với `t=16`, xác suất bỏ lọt < 1.5e-5).
    """

    @staticmethod
    def _sinh_ma_tran_rademacher(n_out: int, so_vong: int, seed: Optional[int] = None) -> np.ndarray:
        """Sinh ma trận thăm dò Rademacher `{-1, +1}` kích thước `(n_out, so_vong)` bằng CSPRNG hoặc seed xác định."""
        if seed is not None:
            rng = np.random.default_rng(int(seed))
            bits = rng.integers(0, 2, size=(n_out, so_vong), dtype=np.int8)
            return (bits * 2 - 1).astype(np.float64)
        tong_bits = n_out * so_vong
        raw = secrets.token_bytes((tong_bits + 7) // 8)
        arr_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:tong_bits]
        return (arr_bits.reshape(n_out, so_vong).astype(np.float64) * 2.0 - 1.0)

    @classmethod
    def xac_minh_nhan_ma_tran(
        cls,
        x: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        b: Optional[np.ndarray] = None,
        so_vong: int = 16,
        rtol: float = 2e-5,
        atol: float = 1e-3,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Kiểm chứng xác suất `Y ≈ X @ W + b` trong thời gian O(t * n^2) và trả về báo cáo kiểm toán chi tiết."""
        t0 = time.perf_counter()
        use_f32 = (
            np.asarray(x).dtype == np.float32
            and np.asarray(w).dtype == np.float32
            and np.asarray(y).dtype == np.float32
        )
        calc_dtype: Any = np.float32 if use_f32 else np.float64
        x_2d: np.ndarray = np.asarray(x, dtype=calc_dtype)
        w_2d: np.ndarray = np.asarray(w, dtype=calc_dtype)
        y_2d: np.ndarray = np.asarray(y, dtype=calc_dtype)
        if x_2d.ndim == 1:
            x_2d = x_2d.reshape(1, -1)
        if y_2d.ndim == 1:
            y_2d = y_2d.reshape(1, -1)

        if w_2d.ndim != 2 or x_2d.shape[1] != w_2d.shape[0] or y_2d.shape != (x_2d.shape[0], w_2d.shape[1]):
            return {
                "hop_le": False,
                "ly_do": "Kích thước ma trận X, W, Y không tương thích cho kiểm chứng Freivalds",
                "so_vong": int(so_vong),
                "sai_so_toi_da": float("inf"),
                "nguong_cho_phep": float(atol),
                "xac_suat_bo_lot_toi_da": 1.0,
                "thoi_gian_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        t_rounds = max(1, int(so_vong))
        r_mat = cls._sinh_ma_tran_rademacher(w_2d.shape[1], t_rounds, seed=seed).astype(calc_dtype, copy=False)
        wr = w_2d @ r_mat
        lhs: np.ndarray = x_2d @ wr
        if b is not None:
            b_vec = np.asarray(b, dtype=calc_dtype).reshape(1, -1)
            if b_vec.shape[1] == w_2d.shape[1]:
                lhs = lhs + (b_vec @ r_mat)
        rhs = y_2d @ r_mat

        diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        sai_so_toi_da = float(np.max(diff)) if diff.size > 0 else 0.0
        scale_ref = float(
            max(
                np.max(np.abs(lhs)) if lhs.size > 0 else 0.0,
                np.max(np.abs(rhs)) if rhs.size > 0 else 0.0,
            )
        )
        nguong = float(atol + rtol * scale_ref)
        hop_le = bool(np.all(np.isfinite(diff)) and sai_so_toi_da <= nguong)

        return {
            "hop_le": hop_le,
            "so_vong": t_rounds,
            "sai_so_toi_da": round(sai_so_toi_da, 8),
            "nguong_cho_phep": round(nguong, 8),
            "xac_suat_bo_lot_toi_da": float(2.0 ** (-t_rounds)),
            "thoi_gian_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }


VerifiableInference = XacThucSuyLuanMaoHiem
InferenceMerkleTree = CayMerkleSuyLuan
FreivaldsVerifier = KiemChungFreivalds

__all__ = [
    "CayMerkleSuyLuan",
    "InferenceMerkleTree",
    "XacThucSuyLuanMaoHiem",
    "VerifiableInference",
    "KiemChungFreivalds",
    "FreivaldsVerifier",
]
