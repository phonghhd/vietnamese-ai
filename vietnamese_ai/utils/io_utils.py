"""IO Utils - Lưu và tải dữ liệu, mô hình."""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler an toàn - chỉ cho phép các loại dữ liệu cơ bản."""

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
        ("collections", "OrderedDict"),
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
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Không cho phép deserialize: {module}.{name}. "
            "Sử dụng pickle.load() trực tiếp nếu bạn tin cậy nguồn file."
        )


def tai_an_toan(duong_dan: str) -> Any:
    """Tải mô hình từ pickle với restricted unpickler.

    Chỉ cho phép các loại dữ liệu cơ bản (dict, list, numpy arrays, etc.).
    Nếu file chứa object tùy chỉnh (sklearn models, custom classes),
    sẽ raise lỗi - dùng pickle.load() trực tiếp trong trường hợp đó.
    """
    with open(duong_dan, "rb") as f:
        return RestrictedUnpickler(f).load()


class LuuTai:
    """
    Công cụ lưu/tải dữ liệu và mô hình.

    Sử dụng:
        >>> LuuTai.luu_mo_hinh(mo_hinh, "model.pkl")
        >>> mo_hinh = LuuTai.tai_mo_hinh("model.pkl")
        >>> LuuTai.luu_numpy(arr, "data.npz", key=arr)
        >>> data = LuuTai.tai_numpy("data.npz")
    """

    @staticmethod
    def luu_mo_hinh(mo_hinh: Any, duong_dan: str) -> None:
        """Lưu mô hình ra file pickle."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "wb") as f:
            pickle.dump(mo_hinh, f)

    @staticmethod
    def tai_mo_hinh(duong_dan: str) -> Any:
        """Tải mô hình từ file pickle.

        Cảnh báo: Chỉ tải file từ nguồn tin cậy.
        Sử dụng tai_an_toan() cho dữ liệu cơ bản.
        """
        warnings.warn(
            "Đang tải mô hình từ pickle. Chỉ tải file từ nguồn tin cậy.",
            stacklevel=2,
        )
        try:
            with open(duong_dan, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")

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
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(duong_dan, **matran)

    @staticmethod
    def tai_numpy(duong_dan: str) -> Dict[str, np.ndarray]:
        """Tải mảng numpy từ file .npz."""
        try:
            data = np.load(duong_dan)
            return dict(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

    @staticmethod
    def luu_json(data: Dict[str, Any], duong_dan: str) -> None:
        """Lưu dữ liệu ra file JSON."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "w", encoding="utf-8") as f:
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
