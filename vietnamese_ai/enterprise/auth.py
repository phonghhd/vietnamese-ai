"""HeThongXacThuc - Hệ thống xác thực và phân quyền (RBAC)."""

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from vietnamese_ai.utils.logger import Logger


class HeThongXacThuc:
    """
    Hệ thống xác thực và phân quyền (RBAC).

    Tính năng:
    - Đăng ký người dùng với vai trò (admin, developer, viewer)
    - Xác thực bằng token
    - Kiểm tra quyền truy cập
    - Quản lý API keys

    Vai trò:
    - admin: Toàn quyền
    - developer: Huấn luyện, dự đoán, quản lý mô hình
    - viewer: Chỉ xem và dự đoán

    Sử dụng:
        >>> auth = HeThongXacThuc()
        >>> auth.dang_ky("admin_user", "password123", vai_tro="admin")
        >>> token = auth.dang_nhap("admin_user", "password123")
        >>> auth.kiem_tra_quyen(token, "train")
        True
    """

    VAI_TRO_QUYEN = {
        "admin": ["train", "predict", "deploy", "manage", "delete", "view", "audit"],
        "developer": ["train", "predict", "manage", "view"],
        "viewer": ["predict", "view"],
    }

    def __init__(self, duong_dan: str = "auth"):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("HeThongXacThuc")
        self._users_file = self.duong_dan / "users.json"
        self._tokens_file = self.duong_dan / "tokens.json"
        self._users = self._tai_json(self._users_file)
        self._tokens = self._tai_json(self._tokens_file)

    @staticmethod
    def _tai_json(duong_dan: Path) -> Dict:
        if duong_dan.exists():
            with open(duong_dan, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _luu_json(duong_dan: Path, data: Dict) -> None:
        with open(duong_dan, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _bam_mat_khau(mat_khau: str) -> str:
        salt = os.urandom(32)
        dk = hashlib.pbkdf2_hmac("sha256", mat_khau.encode(), salt, 310000)
        return salt.hex() + ":" + dk.hex()

    @staticmethod
    def _kiem_tra_mat_khau(mat_khau: str, stored: str) -> bool:
        try:
            salt_hex, dk_hex = stored.split(":")
            salt = bytes.fromhex(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", mat_khau.encode(), salt, 310000)
            return hmac.compare_digest(dk.hex(), dk_hex)
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _tao_token() -> str:
        return hashlib.sha256(os.urandom(32)).hexdigest()[:32]

    def dang_ky(self, ten_dang_nhap: str, mat_khau: str, vai_tro: str = "viewer") -> bool:
        """
        Đăng ký người dùng mới.

        Args:
            ten_dang_nhap: Tên đăng nhập
            mat_khau: Mật khẩu
            vai_tro: Vai trò (admin, developer, viewer)
        """
        if ten_dang_nhap in self._users:
            raise ValueError(f"Tên đăng nhập đã tồn tại: {ten_dang_nhap}")
        if vai_tro not in self.VAI_TRO_QUYEN:
            raise ValueError(f"Vai trò không hợp lệ: {vai_tro}")

        self._users[ten_dang_nhap] = {
            "mat_khau": self._bam_mat_khau(mat_khau),
            "vai_tro": vai_tro,
            "thoi_gian_tao": time.time(),
        }
        self._luu_json(self._users_file, self._users)
        self.logger.info(f"Đã đăng ký: {ten_dang_nhap} ({vai_tro})")
        return True

    def dang_nhap(self, ten_dang_nhap: str, mat_khau: str) -> str:
        """
        Đăng nhập và lấy token.

        Returns:
            Token xác thực
        """
        if ten_dang_nhap not in self._users:
            raise KeyError(f"Không tìm thấy người dùng: {ten_dang_nhap}")

        user = self._users[ten_dang_nhap]
        if not self._kiem_tra_mat_khau(mat_khau, user["mat_khau"]):
            raise ValueError("Mật khẩu không đúng")

        token = self._tao_token()
        self._tokens[token] = {
            "ten_dang_nhap": ten_dang_nhap,
            "vai_tro": user["vai_tro"],
            "thoi_gian": time.time(),
        }
        self._luu_json(self._tokens_file, self._tokens)
        return token

    def xac_thuc(self, token: str) -> Dict:
        """Xác thực token và trả về thông tin người dùng."""
        if token not in self._tokens:
            raise PermissionError("Token không hợp lệ")
        return self._tokens[token]

    def kiem_tra_quyen(self, token: str, quyen: str) -> bool:
        """Kiểm tra người dùng có quyền thực hiện hành động không."""
        try:
            info = self.xac_thuc(token)
            vai_tro = info["vai_tro"]
            return quyen in self.VAI_TRO_QUYEN.get(vai_tro, [])
        except PermissionError:
            return False

    def lay_vai_tro(self, token: str) -> str:
        """Lấy vai trò từ token."""
        info = self.xac_thuc(token)
        return info["vai_tro"]

    def doi_mat_khau(self, ten_dang_nhap: str, mat_khau_cu: str, mat_khau_moi: str) -> bool:
        """Đổi mật khẩu."""
        if ten_dang_nhap not in self._users:
            raise KeyError(f"Không tìm thấy người dùng: {ten_dang_nhap}")
        if not self._kiem_tra_mat_khau(mat_khau_cu, self._users[ten_dang_nhap]["mat_khau"]):
            raise ValueError("Mật khẩu cũ không đúng")

        self._users[ten_dang_nhap]["mat_khau"] = self._bam_mat_khau(mat_khau_moi)
        self._luu_json(self._users_file, self._users)
        return True

    def xoa_nguoi_dung(self, ten_dang_nhap: str) -> None:
        """Xóa người dùng."""
        if ten_dang_nhap in self._users:
            del self._users[ten_dang_nhap]
            self._luu_json(self._users_file, self._users)

    def danh_sach_nguoi_dung(self) -> List[Dict]:
        """Liệt kê người dùng (không hiển thị mật khẩu)."""
        return [
            {"ten_dang_nhap": ten, "vai_tro": info["vai_tro"]}
            for ten, info in self._users.items()
        ]

    def tao_api_key(self, ten_dang_nhap: str) -> str:
        """Tạo API key cho người dùng."""
        if ten_dang_nhap not in self._users:
            raise KeyError(f"Không tìm thấy người dùng: {ten_dang_nhap}")

        api_key = f"vai_{os.urandom(24).hex()}"
        self._users[ten_dang_nhap]["api_key"] = api_key
        self._luu_json(self._users_file, self._users)
        return api_key

    def xac_thuc_api_key(self, api_key: str) -> Dict:
        """Xác thực bằng API key."""
        for ten, info in self._users.items():
            if info.get("api_key") == api_key:
                return {"ten_dang_nhap": ten, "vai_tro": info["vai_tro"]}
        raise PermissionError("API key không hợp lệ")
