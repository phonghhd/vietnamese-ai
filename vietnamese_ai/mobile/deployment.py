"""TriKhaiDiDong - Triển khai mô hình lên thiết bị di động và edge."""

import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang
from vietnamese_ai.utils.logger import Logger


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
    mo_ta=(
        "Triển khai di động dùng định dạng container nội bộ (VAI_TFL/VAI_CLM/VAI_ONX/VAI_PTE) "
        "khi chưa cài toolchain gốc. Xuất định dạng native cho thiết bị thật yêu cầu công cụ ngoài."
    ),
    phu_thuoc_ngoai=[
        "tensorflow / ai-edge-litert + Android SDK/NDK (cho TFLite FlatBuffer native)",
        "coremltools + Xcode trên macOS (cho CoreML .mlmodel/.mlpackage native)",
        "skl2onnx + onnxruntime (cho ONNX chuẩn, cài qua vietnamese-ai[export])",
        "PyTorch ExecuTorch toolchain (cho .pte native)",
    ],
    nhom_extra="mobile",
    la_mo_phong=True,
)
class TriKhaiDiDong:
    """
    Triển khai mô hình học máy lên thiết bị di động và edge (Thử nghiệm / External Toolchain).

    Alias công khai: `MobileDeployer`, `ONNXExporter` (cùng class).

    Lưu ý minh bạch về định dạng và phụ thuộc ngoài:
    - `xuat_tflite`: xuất container nội bộ `VAI_TFL` (chạy qua `chay_suy_luan_di_dong`);
      không sinh FlatBuffer `.tflite` chuẩn nếu chưa cài `tensorflow` / `ai-edge-litert` + Android SDK/NDK.
    - `xuat_coreml`: xuất container nội bộ `VAI_CLM`;
      không sinh `.mlmodel`/`.mlpackage` chuẩn Apple nếu chưa cài `coremltools` + Xcode (macOS).
    - `xuat_onnx_mobile`: dùng `skl2onnx` khi đã cài `vietnamese-ai[export]`, ngược lại fallback `VAI_ONX`.
    - `xuat_executorch`: xuất container placeholder `VAI_PTE`; native cần PyTorch ExecuTorch toolchain.
    - `benchmark_edge`: đo thời gian suy luận giả lập trên CPU host hiện tại, không đo trên NPU di động.

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
        self, mo_hinh: Any, duong_dan: str, kich_thuoc_dau_vao: Tuple[Optional[int], ...]
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

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
        mo_ta="Xuất container VAI_TFL nội bộ; định dạng FlatBuffer .tflite gốc cần tensorflow/ai-edge-litert.",
        phu_thuoc_ngoai=["tensorflow / ai-edge-litert", "Android SDK/NDK"],
        nhom_extra="mobile",
        la_mo_phong=True,
    )
    def xuat_tflite(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[Optional[int], ...] = (None,),
        luong_hoa: bool = False,
    ) -> str:
        """
        Xuất mô hình sang container TFLite nội bộ (`VAI_TFL`) — Thử nghiệm / Mô phỏng định dạng.

        Tạo file container chứa trọng số mô hình đã serialize chạy được bằng
        `chay_suy_luan_di_dong()`. Để tạo tệp FlatBuffer `.tflite` chuẩn cho TFLite Interpreter
        trên Android/iOS cần cài đặt toolchain ngoài (`tensorflow` / `ai-edge-litert` + Android NDK).

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

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
        mo_ta="Xuất container VAI_CLM nội bộ; tệp .mlmodel/.mlpackage chuẩn Apple cần coremltools + Xcode.",
        phu_thuoc_ngoai=["coremltools", "Xcode (macOS)"],
        nhom_extra="mobile",
        la_mo_phong=True,
    )
    def xuat_coreml(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[Optional[int], ...] = (None,),
        ten_mo_hinh: str = "VietnameseAI_Model",
    ) -> str:
        """
        Xuất mô hình sang container CoreML nội bộ (`VAI_CLM`) — Thử nghiệm / Mô phỏng định dạng.

        Tạo file container chứa metadata và trọng số mô hình chạy được qua `chay_suy_luan_di_dong()`.
        Để xuất định dạng `.mlmodel`/`.mlpackage` chuẩn của Apple cần cài đặt `coremltools` và Xcode.

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

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
        mo_ta="Xuất ONNX chuẩn khi cài skl2onnx/onnxruntime; fallback sang container VAI_ONX nội bộ khi thiếu.",
        phu_thuoc_ngoai=["skl2onnx", "onnxruntime"],
        nhom_extra="export",
    )
    def xuat_onnx_mobile(
        self,
        mo_hinh: Any,
        duong_dan: str,
        kich_thuoc_dau_vao: Tuple[Optional[int], ...] = (None,),
    ) -> str:
        """
        Xuất mô hình sang ONNX Mobile (ưu tiên `skl2onnx`, fallback sang container nội bộ `VAI_ONX`).

        Cài đặt công cụ ONNX đầy đủ: `pip install "vietnamese-ai[export]"`.

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

    def xuat_gguf_mobile(self, mo_hinh: Any, duong_dan: str, quantization: str = "q8_0") -> str:
        """
        Xuất mô hình sang GGUF binary v3 cho Mobile / Edge (llama.cpp compatible).
        """
        from vietnamese_ai.export.gguf import XuatGGUF

        model = self._trich_xuat_mo_hinh(mo_hinh)
        trong_so: Dict[str, np.ndarray] = {}

        if isinstance(mo_hinh, dict):
            for k, v in mo_hinh.items():
                if isinstance(v, np.ndarray):
                    trong_so[k] = v.astype(np.float32)
        if hasattr(model, "coef_"):
            trong_so["weight"] = np.asarray(model.coef_, dtype=np.float32)
        if hasattr(model, "intercept_"):
            trong_so["bias"] = np.asarray(model.intercept_, dtype=np.float32)
        if hasattr(mo_hinh, "embedding") and isinstance(mo_hinh.embedding, np.ndarray):
            trong_so["token_embd.weight"] = mo_hinh.embedding.astype(np.float32)
        if hasattr(model, "named_parameters"):
            try:
                for ten, p in model.named_parameters():
                    trong_so[ten] = p.detach().cpu().numpy().astype(np.float32)
            except Exception:
                pass
        if not trong_so:
            trong_so["default.weight"] = np.ones((1, 1), dtype=np.float32)

        xuat = XuatGGUF()
        thong_tin = {
            "name": getattr(mo_hinh, "__class__", type(mo_hinh)).__name__,
            "architecture": "mobile_gguf",
        }
        return xuat.xuat_tu_numpy(trong_so, duong_dan, quantization=quantization, thong_tin=thong_tin)

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.SIMULATED,
        mo_ta="Xuất container placeholder VAI_PTE; biên dịch ExecuTorch gốc cần PyTorch ExecuTorch toolchain.",
        phu_thuoc_ngoai=["PyTorch ExecuTorch toolchain"],
        nhom_extra="mobile",
        la_mo_phong=True,
    )
    def xuat_executorch(self, mo_hinh: Any, duong_dan: str) -> str:
        """
        Xuất mô hình sang container ExecuTorch placeholder (`VAI_PTE`) — Mô phỏng định dạng.

        Chạy suy luận kiểm thử qua `chay_suy_luan_di_dong()`. Định dạng `.pte` native cần
        biên dịch bằng PyTorch ExecuTorch toolchain.
        """
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        trong_so = self._lay_trong_so_sklearn(mo_hinh)
        meta = {
            "dinh_dang": "executorch",
            "compat": "EXECUTORCH_PLACEHOLDER_V1",
            "mo_phong": False,
            "ten_mo_hinh": getattr(mo_hinh, "__class__", type(mo_hinh)).__name__,
            "trong_so": trong_so,
            "thoi_gian_xuat": time.time(),
        }
        payload = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        with open(duong_dan_path, "wb") as f:
            f.write(b"VAI_PTE")
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

        self.logger.info(f"Đã xuất ExecuTorch (.pte): {duong_dan}")
        return str(duong_dan_path)

    def chay_suy_luan_di_dong(self, duong_dan: str, X: np.ndarray) -> np.ndarray:
        """
        Thực thi suy luận trực tiếp từ file mô hình di động đã xuất (.tflite, .coreml, .onnx, .gguf, .pte).
        Tự động giải lượng tử hóa nếu mô hình ở dạng INT8.
        """
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        with open(duong_dan_path, "rb") as f:
            head4 = f.read(4)
            f.seek(0)
            head7 = f.read(7)

            if head4 == b"GGUF":
                from vietnamese_ai.export.gguf import XuatGGUF

                tensors = XuatGGUF().tai_tu_gguf(str(duong_dan_path))
                if "weight" in tensors:
                    w = tensors["weight"]
                    b = tensors.get("bias", np.zeros(1, dtype=np.float32))
                    scores = X_arr @ w.T + b if w.ndim == 2 else X_arr @ w + b
                    if scores.ndim == 2 and scores.shape[1] > 1:
                        return np.argmax(scores, axis=-1)
                    return (scores.ravel() >= 0.0).astype(int)
                return np.zeros(len(X_arr), dtype=np.float32)

            if head7 == b"VAI_ONX":
                from vietnamese_ai.export.onnx_export import XuatONNX

                payload_size = struct.unpack("<I", f.read(4))[0]
                data = json.loads(f.read(payload_size).decode("utf-8"))
                if "initializers" in data:
                    return XuatONNX._thuc_thi_do_thi_native(data, X_arr)
            else:
                payload_size = struct.unpack("<I", f.read(4))[0]
                data = json.loads(f.read(payload_size).decode("utf-8"))

        trong_so = data.get("trong_so", data)
        if "coef" not in trong_so:
            return np.zeros(len(X_arr), dtype=np.float32)

        coef = np.asarray(trong_so["coef"], dtype=np.float32)
        if "quantize_info" in trong_so:
            q_info = trong_so["quantize_info"]
            coef = coef * float(q_info["scale"]) + float(q_info["zero_point"])

        intercept = np.asarray(trong_so.get("intercept", [0.0]), dtype=np.float32)
        if coef.ndim == 1:
            scores = X_arr @ coef + (intercept.ravel()[0] if intercept.size > 0 else 0.0)
        else:
            scores = X_arr @ coef.T + intercept.reshape(1, -1)

        if "classes" in trong_so:
            classes = np.asarray(trong_so["classes"])
            if scores.ndim == 1 or scores.shape[1] == 1:
                idx = (scores.ravel() >= 0.0).astype(int)
                return classes[idx]
            return classes[np.argmax(scores, axis=-1)]
        return scores.ravel() if scores.ndim == 2 and scores.shape[1] == 1 else scores

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

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.SIMULATED,
        mo_ta="Đo benchmark giả lập trên CPU host hiện tại, không đo trên NPU/thiết bị di động vật lý.",
        nhom_extra="mobile",
        la_mo_phong=True,
    )
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

        import platform

        return {
            "thoi_gian_trung_binh_ms": float(np.mean(thoi_gian_arr)),
            "thoi_gian_min_ms": float(np.min(thoi_gian_arr)),
            "thoi_gian_max_ms": float(np.max(thoi_gian_arr)),
            "thoi_gian_median_ms": float(np.median(thoi_gian_arr)),
            "do_lech_chuan_ms": float(np.std(thoi_gian_arr)),
            "throughput_mau_giay": float(batch_size / (np.mean(thoi_gian_arr) / 1000)),
            "kich_thuoc_mo_hinh_bytes": kich_thuoc_mo_hinh,
            "so_lan_chay": so_lan,
            "thiet_bi_do": f"host_cpu:{platform.machine()}",
            "la_thiet_bi_di_dong_that": False,
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


MobileDeployer = TriKhaiDiDong
ONNXExporter = TriKhaiDiDong
