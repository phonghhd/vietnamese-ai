"""IO Utils - Lưu và tải dữ liệu, mô hình kèm kiểm tra chữ ký toàn vẹn HMAC-SHA256."""

import hashlib
import hmac
import json
import os
import pickle
import secrets
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler an toàn - chỉ cho phép các loại dữ liệu cơ bản và các module học máy tin cậy."""

    BLOCKED_MODULES = {
        "os",
        "sys",
        "subprocess",
        "posix",
        "nt",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "http",
        "ctypes",
        "importlib",
        "pty",
        "commands",
        "marshal",
    }

    BLOCKED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "input",
    }

    ALLOWED_PREFIXES = (
        "vietnamese_ai.",
        "numpy.",
        "sklearn.",
        "scipy.",
        "torch.",
        "transformers.",
        "xgboost.",
        "lightgbm.",
        "collections.",
        "pathlib.",
    )

    ALLOWED_CLASSES = {
        ("builtins", "dict"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "slice"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "str"),
        ("builtins", "tuple"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("builtins", "range"),
        ("builtins", "NoneType"),
        ("builtins", "object"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "deque"),
        ("pathlib", "PosixPath"),
        ("pathlib", "WindowsPath"),
        ("pathlib", "Path"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy", "float32"),
        ("numpy", "float64"),
        ("numpy", "int32"),
        ("numpy", "int64"),
        ("numpy", "bool_"),
        ("numpy", "array"),
        ("numpy", "matrix"),
        ("_codecs", "encode"),
        ("copyreg", "_reconstruct"),
        ("logging", "getLogger"),
    }

    def __init__(
        self,
        file: Any,
        cho_phep_mo_hinh: bool = False,
        cho_phep_lop_da_ky: bool = False,
    ):
        super().__init__(file)
        self.cho_phep_mo_hinh = cho_phep_mo_hinh
        self.cho_phep_lop_da_ky = cho_phep_lop_da_ky

    def find_class(self, module: str, name: str) -> Any:
        root_mod = module.split(".")[0]
        if root_mod in self.BLOCKED_MODULES or (module == "builtins" and name in self.BLOCKED_BUILTINS):
            raise pickle.UnpicklingError(
                f"Bảo mật: Đã chặn hành vi deserialize nguy hiểm '{module}.{name}'."
            )
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        if self.cho_phep_mo_hinh and (
            module == "vietnamese_ai" or module.startswith(self.ALLOWED_PREFIXES)
        ):
            return super().find_class(module, name)
        if self.cho_phep_lop_da_ky:
            from vietnamese_ai.core.pipeline import Pipeline
            from vietnamese_ai.models.base import BaseModel

            obj = super().find_class(module, name)
            if isinstance(obj, type) and issubclass(obj, (BaseModel, Pipeline)):
                return obj
        raise pickle.UnpicklingError(
            f"Không cho phép deserialize: {module}.{name}. "
            "Sử dụng pickle.load() trực tiếp nếu bạn tin cậy nguồn file."
        )


def tai_an_toan(
    duong_dan: str,
    cho_phep_mo_hinh: bool = False,
    cho_phep_lop_da_ky: bool = False,
) -> Any:
    """Tải mô hình từ pickle với restricted unpickler.

    Chỉ cho phép các loại dữ liệu cơ bản (dict, list, numpy arrays, etc.),
    hoặc các lớp mô hình hợp lệ thuộc `vietnamese_ai`/`sklearn` khi `cho_phep_mo_hinh=True`.
    """
    with open(duong_dan, "rb") as f:
        return RestrictedUnpickler(
            f,
            cho_phep_mo_hinh=cho_phep_mo_hinh,
            cho_phep_lop_da_ky=cho_phep_lop_da_ky,
        ).load()


class LuuTai:
    """
    Công cụ lưu/tải dữ liệu và mô hình.

    Sử dụng:
        >>> LuuTai.luu_mo_hinh(mo_hinh, "model.pkl")
        >>> mo_hinh = LuuTai.tai_mo_hinh("model.pkl")
        >>> LuuTai.luu_numpy(arr, "data.npz", key=arr)
        >>> data = LuuTai.tai_numpy("data.npz")
    """

    _KHOA_CUC_BO: Optional[str] = None

    @classmethod
    def _lay_hoac_tao_khoa_cuc_bo(cls) -> str:
        if cls._KHOA_CUC_BO:
            return cls._KHOA_CUC_BO
        cac_vi_tri = [
            Path.home() / ".vietnamese_ai" / ".signing_key",
            Path(tempfile.gettempdir()) / f".vai_signing_key_{os.getuid() if hasattr(os, 'getuid') else 'local'}",
        ]
        for key_path in cac_vi_tri:
            try:
                if key_path.is_file():
                    raw = key_path.read_text(encoding="utf-8").strip()
                    if len(raw) >= 32:
                        cls._KHOA_CUC_BO = raw
                        return raw
                key_path.parent.mkdir(parents=True, exist_ok=True)
                new_key = secrets.token_hex(32)
                fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_key)
                cls._KHOA_CUC_BO = new_key
                return new_key
            except OSError:
                continue
        cls._KHOA_CUC_BO = secrets.token_hex(32)
        return cls._KHOA_CUC_BO

    @classmethod
    def _khoa_hmac(
        cls,
        khoa_bi_mat: Optional[str] = None,
        yeu_cau_khoa_tuong_minh: bool = False,
    ) -> bytes:
        raw = khoa_bi_mat or os.environ.get("VAI_MODEL_SIGNING_KEY")
        if raw:
            return raw.encode("utf-8")
        if yeu_cau_khoa_tuong_minh or os.environ.get("VAI_STRICT_MODEL_SIGNING") == "1":
            raise ValueError(
                "Thiếu khóa bí mật ký mô hình. Vui lòng thiết lập biến môi trường VAI_MODEL_SIGNING_KEY "
                "hoặc truyền tham số khoa_bi_mat."
            )
        return cls._lay_hoac_tao_khoa_cuc_bo().encode("utf-8")

    @classmethod
    def _tinh_hmac_streaming(cls, duong_dan: Path, khoa_bi_mat: Optional[str] = None) -> str:
        """Tính chữ ký HMAC-SHA256 theo luồng khối 64 KB để hỗ trợ tệp mô hình lớn."""
        mac = hmac.new(cls._khoa_hmac(khoa_bi_mat), digestmod=hashlib.sha256)
        with open(duong_dan, "rb") as f:
            while True:
                khoi = f.read(65536)
                if not khoi:
                    break
                mac.update(khoi)
        return mac.hexdigest()

    @classmethod
    def ky_file(cls, duong_dan: str, khoa_bi_mat: Optional[str] = None) -> str:
        """Tạo chữ ký toàn vẹn HMAC-SHA256 (`.sig`) cho file mô hình."""
        p = Path(duong_dan)
        sig = cls._tinh_hmac_streaming(p, khoa_bi_mat=khoa_bi_mat)
        sig_path = p.with_suffix(p.suffix + ".sig")
        sig_path.write_text(sig, encoding="utf-8")
        return sig

    @classmethod
    def xac_thuc_file(cls, duong_dan: str, khoa_bi_mat: Optional[str] = None) -> bool:
        """Xác thực chữ ký toàn vẹn HMAC-SHA256 (`.sig`) của file mô hình."""
        p = Path(duong_dan)
        sig_path = p.with_suffix(p.suffix + ".sig")
        if not sig_path.exists():
            return False
        expected = sig_path.read_text(encoding="utf-8").strip()
        actual = cls._tinh_hmac_streaming(p, khoa_bi_mat=khoa_bi_mat)
        return hmac.compare_digest(expected, actual)

    @staticmethod
    def luu_mo_hinh(mo_hinh: Any, duong_dan: str, ky_so: bool = False) -> None:
        """Lưu mô hình ra file pickle (tùy chọn tạo chữ ký toàn vẹn `.sig`)."""
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "wb") as f:
            pickle.dump(mo_hinh, f)
        if ky_so:
            LuuTai.ky_file(str(duong_dan_path))

    @staticmethod
    def tai_mo_hinh(duong_dan: str, cho_phep_pickle_tho: bool = False) -> Any:
        """Tải mô hình từ file pickle kèm kiểm tra `.sig` và `RestrictedUnpickler` mặc định."""
        p = Path(duong_dan)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        sig_path = p.with_suffix(p.suffix + ".sig")
        da_ky_hop_le = False
        if sig_path.exists():
            if not LuuTai.xac_thuc_file(str(p)):
                raise ValueError(
                    f"CẢNH BÁO BẢO MẬT: Chữ ký toàn vẹn HMAC-SHA256 của '{duong_dan}' không hợp lệ!"
                )
            da_ky_hop_le = True
        if cho_phep_pickle_tho:
            warnings.warn(
                "Đang tải mô hình từ pickle thô. Chỉ tải file từ nguồn tin cậy.",
                stacklevel=2,
            )
            try:
                with open(duong_dan, "rb") as f:
                    return pickle.load(f)
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")
        try:
            return tai_an_toan(
                duong_dan, cho_phep_mo_hinh=True, cho_phep_lop_da_ky=da_ky_hop_le
            )
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file pickle an toàn: {e}")

    @staticmethod
    def tai_mo_hinh_an_toan(duong_dan: str) -> Any:
        """Tải mô hình với restricted unpickler (chỉ dữ liệu cơ bản)."""
        try:
            return tai_an_toan(duong_dan)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file an toàn: {e}")

    @staticmethod
    def luu_numpy(duong_dan: str, **matran: np.ndarray) -> None:
        """Lưu nhiều mảng numpy vào một file .npz."""
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(duong_dan_path, allow_pickle=True, **matran)

    @staticmethod
    def tai_numpy(duong_dan: str) -> Dict[str, np.ndarray]:
        """Tải mảng numpy từ file .npz."""
        try:
            with np.load(duong_dan) as data:
                return dict(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

    @staticmethod
    def luu_json(data: Dict[str, Any], duong_dan: str) -> None:
        """Lưu dữ liệu ra file JSON."""
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def tai_json(duong_dan: str) -> Dict[str, Any]:
        """Tải dữ liệu từ file JSON."""
        try:
            with open(duong_dan, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except json.JSONDecodeError as e:
            raise ValueError(f"File JSON không hợp lệ: {e}")
