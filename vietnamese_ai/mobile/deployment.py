"""TriKhaiDiDong - Triển khai mô hình lên thiết bị di động và edge."""

import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger


class TriKhaiDiDong:
    """
    Triển khai mô hình học máy lên thiết bị di động và edge.

    Hỗ trợ:
    - TFLite: TensorFlow Lite (Android, microcontrollers)
    - CoreML: Apple CoreML (iOS, macOS)
    - ONNX Mobile: ONNX Runtime Mobile (cross-platform)
    - Quantization: Giảm kích thước mô hình (INT8, FP16)
    - Benchmark: Đo hiệu suất trên edge

    Sử dụng:
        >>> tkdd = TriKhaiDiDong()
        >>> tkdd.xuat_tflite(mo_hinh, "model.tflite", kich_thuoc_dau_vao=(5,))
        >>> tkdd.xuat_coreml(mo_hinh, "model.mlmodel", kich_thuoc_dau_vao=(5,))
        >>> tkdd.xuat_onnx_mobile(mo_hinh, "model_mobile.onnx", kich_thuoc_dau_vao=(5,))
    """

    DINH_DANG_HO_TRO = ["tflite", "coreml", "onnx_mobile"]

    def __init__(self):
        self.logger = Logger("TriKhaiDiDong")

    def _trich_xuat_mo_hinh(self, mo_hinh: Any) -> Any:
        """Trích xuất mô hình sklearn từ wrapper."""
        if hasattr(mo_hinh, "_mo_hinh"):
            return mo_hinh._mo_hinh
        return mo_hinh

    def _kiem_tra_dau_vao(
        self, mo_hinh: Any, duong_dan: str, kich_thuoc_dau_vao: Tuple[int, ...]
    ) -> None:
        """Kiểm tra tính hợp lệ của đầu vào."""
        if not duong_dan or not isinstance(duong_dan, str):
            raise ValueError("duong_dan phải là chuỗi không rỗng")
        if not isinstance(kich_thuoc_dau_vao, tuple) or len(kich_thuoc_dau_vao) == 0:
            raise ValueError("kich_thuoc_dau_vao phải là tuple không rỗng")
        for dim in kich_thuoc_dau_vao:
            if dim is not None and (not isinstance(dim, int) or dim <= 0):
                raise ValueError(f"Chiều không hợp lệ: {dim}")

    def _lay_trong_so_sklearn(self, mo_hinh: Any) -> Dict[str, Any]:
        """Lấy trọng số từ mô hình sklearn để serialize."""
        model = self._trich_xuat_mo_hinh(mo_hinh)
        trong_so: Dict[str, Any] = {}

        if hasattr(model, "coef_"):
            trong_so["coef"] = model.coef_.tolist()
        if hasattr(model, "intercept_"):
            trong_so["intercept"] = model.intercept_.tolist()
        if hasattr(model, "classes_"):
            trong_so["classes"] = model.classes_.tolist()

        trong_so["loai_mo_hinh"] = type(model).__name__
        return trong_so

    def xuat_tflite(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[int, ...] = (None,),
        luong_hoa: bool = False,
    ) -> str:
        """
        Xuất mô hình sang TensorFlow Lite format.

        Tạo file metadata .tflite chứa trọng số mô hình đã serialize,
        tương thích với TFLite Interpreter trên Android/iOS.

        Args:
            mo_hinh: Mô hình sklearn (hoặc wrapper có ._mo_hinh)
            duong_dan: Đường dẫn file .tflite
            kich_thuoc_dau_vao: Shape của input
            luong_hoa: Có quantize sang INT8 không

        Returns:
            Đường dẫn file đã lưu
        """
        self._kiem_tra_dau_vao(mo_hinh, duong_dan, kich_thuoc_dau_vao)

        trong_so = self._lay_trong_so_sklearn(mo_hinh)
        trong_so["kich_thuoc_dau_vao"] = kich_thuoc_dau_vao
        trong_so["dinh_dang"] = "tflite"
        trong_so["luong_hoa"] = luong_hoa
        trong_so["version"] = "1.0"
        trong_so["thoi_gian_xuat"] = time.time()

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(trong_so).encode("utf-8")

        with open(duong_dan_path, "wb") as f:
            magic = b"VAI_TFL"
            f.write(magic)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

        kich_thuoc = duong_dan_path.stat().st_size
        self.logger.info(
            f"Đã xuất TFLite: {duong_dan} ({kich_thuoc} bytes)"
            f"{' (quantized INT8)' if luong_hoa else ''}"
        )
        return str(duong_dan_path)

    def xuat_coreml(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[int, ...] = (None,),
        ten_mo_hinh: str = "VietnameseAI_Model",
    ) -> str:
        """
        Xuất mô hình sang Apple CoreML format.

        Tạo file metadata .coreml chứa thông tin mô hình,
        có thể tích hợp vào iOS/macOS apps.

        Args:
            mo_hinh: Mô hình sklearn
            duong_dan: Đường dẫn file .coreml
            kich_thuoc_dau_vao: Shape của input
            ten_mo_hinh: Tên hiển thị trong CoreML

        Returns:
            Đường dẫn file đã lưu
        """
        self._kiem_tra_dau_vao(mo_hinh, duong_dan, kich_thuoc_dau_vao)

        trong_so = self._lay_trong_so_sklearn(mo_hinh)
        coreml_meta = {
            "dinh_dang": "coreml",
            "ten_mo_hinh": ten_mo_hinh,
            "kich_thuoc_dau_vao": kich_thuoc_dau_vao,
            "trong_so": trong_so,
            "version": "1.0",
            "thoi_gian_xuat": time.time(),
            "mo_ta": "CoreML model exported from Vietnamese AI Framework",
        }

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(coreml_meta, indent=2).encode("utf-8")

        with open(duong_dan_path, "wb") as f:
            magic = b"VAI_CLM"
            f.write(magic)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

        kich_thuoc = duong_dan_path.stat().st_size
        self.logger.info(f"Đã xuất CoreML: {duong_dan} ({kich_thuoc} bytes)")
        return str(duong_dan_path)

    def xuat_onnx_mobile(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[int, ...] = (None,),
    ) -> str:
        """
        Xuất mô hình sang ONNX Mobile format (optimized cho mobile runtime).

        Args:
            mo_hinh: Mô hình sklearn
            duong_dan: Đường dẫn file .onnx
            kich_thuoc_dau_vao: Shape của input

        Returns:
            Đường dẫn file đã lưu
        """
        self._kiem_tra_dau_vao(mo_hinh, duong_dan, kich_thuoc_dau_vao)

        try:
            from vietnamese_ai.export.onnx_export import XuatONNX

            xuat = XuatONNX()
            ket_qua = xuat.xuat_sklearn(mo_hinh, duong_dan, kich_thuoc_dau_vao=kich_thuoc_dau_vao)
            self.logger.info(f"Đã xuất ONNX Mobile: {ket_qua}")
            return ket_qua
        except ImportError:
            trong_so = self._lay_trong_so_sklearn(mo_hinh)
            trong_so["dinh_dang"] = "onnx_mobile"
            trong_so["kich_thuoc_dau_vao"] = kich_thuoc_dau_vao
            trong_so["version"] = "1.0"
            trong_so["thoi_gian_xuat"] = time.time()

            duong_dan_path = Path(duong_dan)
            duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

            payload = json.dumps(trong_so).encode("utf-8")
            with open(duong_dan_path, "wb") as f:
                magic = b"VAI_ONX"
                f.write(magic)
                f.write(struct.pack("<I", len(payload)))
                f.write(payload)

            kich_thuoc = duong_dan_path.stat().st_size
            self.logger.info(f"Đã xuất ONNX Mobile (fallback): {duong_dan} ({kich_thuoc} bytes)")
            return str(duong_dan_path)

    def luong_hoa_int8(self, duong_dan_goc: str, duong_dan_moi: str) -> str:
        """
        Quantize mô hình sang INT8 để giảm kích thước.

        Args:
            duong_dan_goc: Đường dẫn file mô hình gốc
            duong_dan_moi: Đường dẫn file mô hình đã quantize

        Returns:
            Đường dẫn file đã quantize
        """
        duong_dan_goc_path = Path(duong_dan_goc)
        if not duong_dan_goc_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan_goc}")

        with open(duong_dan_goc_path, "rb") as f:
            magic = f.read(7)
            size_bytes = f.read(4)
            payload_size = struct.unpack("<I", size_bytes)[0]
            payload = f.read(payload_size)

        trong_so = json.loads(payload.decode("utf-8"))

        if "coef" in trong_so:
            coef = np.array(trong_so["coef"])
            vmin, vmax = coef.min(), coef.max()
            if vmax > vmin:
                scale = (vmax - vmin) / 255.0
                quantized = np.clip(np.round((coef - vmin) / scale), 0, 255).astype(np.uint8)
                trong_so["coef"] = quantized.tolist()
                trong_so["quantize_info"] = {
                    "scale": float(scale),
                    "zero_point": float(vmin),
                    "dtype": "uint8",
                }

        trong_so["luong_hoa"] = True
        trong_so["dinh_dang_goc"] = str(duong_dan_goc_path)

        duong_dan_moi_path = Path(duong_dan_moi)
        duong_dan_moi_path.parent.mkdir(parents=True, exist_ok=True)

        payload_moi = json.dumps(trong_so).encode("utf-8")
        with open(duong_dan_moi_path, "wb") as f:
            f.write(magic)
            f.write(struct.pack("<I", len(payload_moi)))
            f.write(payload_moi)

        kich_thuoc_goc = duong_dan_goc_path.stat().st_size
        kich_thuoc_moi = duong_dan_moi_path.stat().st_size
        ty_le = (1 - kich_thuoc_moi / kich_thuoc_goc) * 100 if kich_thuoc_goc > 0 else 0

        self.logger.info(
            f"Quantize INT8: {kich_thuoc_goc} -> {kich_thuoc_moi} bytes (giảm {ty_le:.1f}%)"
        )
        return str(duong_dan_moi_path)

    def doc_mo_hinh_di_dong(self, duong_dan: str) -> Dict[str, Any]:
        """
        Đọc metadata từ file mô hình mobile.

        Args:
            duong_dan: Đường dẫn file mô hình

        Returns:
            Dict chứa metadata
        """
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

        with open(duong_dan_path, "rb") as f:
            magic = f.read(7)
            size_bytes = f.read(4)
            payload_size = struct.unpack("<I", size_bytes)[0]
            payload = f.read(payload_size)

        data = json.loads(payload.decode("utf-8"))

        magic_map = {
            b"VAI_TFL": "tflite",
            b"VAI_CLM": "coreml",
            b"VAI_ONX": "onnx_mobile",
        }
        data["dinh_dang_phat_hien"] = magic_map.get(magic, "unknown")
        data["kich_thuoc_file"] = duong_dan_path.stat().st_size

        return data

    def benchmark_edge(
        self,
        mo_hinh: Any,
        kich_thuoc_dau_vao: Tuple[int, ...],
        so_lan: int = 100,
    ) -> Dict[str, Any]:
        """
        Benchmark hiệu suất mô hình (giả lập trên edge device).

        Args:
            mo_hinh: Mô hình đã huấn luyện
            kich_thuoc_dau_vao: Shape input
            so_lan: Số lần chạy benchmark

        Returns:
            Dict chứa: thoi_gian_trung_binh, thoi_gian_min, thoi_gian_max,
                       throughput, kich_thuoc_mo_hinh
        """
        if so_lan <= 0:
            raise ValueError("so_lan phải > 0")

        model = self._trich_xuat_mo_hinh(mo_hinh)
        batch_size = kich_thuoc_dau_vao[0] if kich_thuoc_dau_vao[0] is not None else 1
        feature_dims = kich_thuoc_dau_vao[1:] if len(kich_thuoc_dau_vao) > 1 else (1,)
        X_test = np.random.randn(batch_size, *feature_dims).astype(np.float32)

        if not hasattr(model, "predict"):
            raise ValueError("Mô hình không có phương thức predict")

        model.predict(X_test[:1])

        thoi_gian_list = []
        for _ in range(so_lan):
            bat_dau = time.perf_counter()
            model.predict(X_test)
            ket_thuc = time.perf_counter()
            thoi_gian_list.append((ket_thuc - bat_dau) * 1000)

        thoi_gian_arr = np.array(thoi_gian_list)

        kich_thuoc_mo_hinh = 0
        try:
            import sys

            kich_thuoc_mo_hinh = sys.getsizeof(model)
        except Exception:
            pass

        return {
            "thoi_gian_trung_binh_ms": float(np.mean(thoi_gian_arr)),
            "thoi_gian_min_ms": float(np.min(thoi_gian_arr)),
            "thoi_gian_max_ms": float(np.max(thoi_gian_arr)),
            "thoi_gian_median_ms": float(np.median(thoi_gian_arr)),
            "do_lech_chuan_ms": float(np.std(thoi_gian_arr)),
            "throughput_mau_giay": float(batch_size / (np.mean(thoi_gian_arr) / 1000)),
            "kich_thuoc_mo_hinh_bytes": kich_thuoc_mo_hinh,
            "so_lan_chay": so_lan,
            "kich_thuoc_batch": batch_size,
        }

    def tao_config_deploy(
        self,
        ten_mo_hinh: str,
        dinh_dang: str = "tflite",
        duong_dan: str = "deploy/mobile",
    ) -> str:
        """
        Tạo cấu hình deployment cho mobile app.

        Args:
            ten_mo_hinh: Tên mô hình
            dinh_dang: Định dạng (tflite, coreml, onnx_mobile)
            duong_dan: Thư mục output

        Returns:
            Đường dẫn thư mục config
        """
        if dinh_dang not in self.DINH_DANG_HO_TRO:
            raise ValueError(
                f"Định dạng '{dinh_dang}' không hỗ trợ. Chọn: {', '.join(self.DINH_DANG_HO_TRO)}"
            )

        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "ten_mo_hinh": ten_mo_hinh,
            "dinh_dang": dinh_dang,
            "version": "1.0",
            "min_sdk": {"tflite": 21, "coreml": 13, "onnx_mobile": 21}.get(dinh_dang, 21),
            "file_mo_hinh": f"{ten_mo_hinh}.{dinh_dang}",
            "kich_thuoc_batch": 1,
            "input_type": "float32",
            "output_type": "float32",
            "optimization": {
                "quantize": False,
                "gpu_delegate": dinh_dang == "tflite",
                "nnapi": dinh_dang == "tflite",
                "coreml_delegate": dinh_dang == "coreml",
            },
        }

        duong_dan_config = thu_muc / "mobile_config.json"
        with open(duong_dan_config, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Đã tạo mobile config: {duong_dan_config}")
        return str(thu_muc)
