"""ChiMucVectorLuongTu (QuantizedVectorIndex) - Động Cơ Chỉ Mục Vector Lượng Tử Hóa & HNSW Native Rust cho Edge CPU.

Hỗ trợ 6 chế độ tìm kiếm vector hiệu năng cao chạy hoàn toàn trên CPU (có tăng tốc Rust Core
`vietnamese_ai_rust` và dự phòng thuần NumPy tương đương 100%):
- `fp32`: Tìm kiếm cosine chính xác FP32 (Rust Rayon / NumPy).
- `sq8`: Lượng tử hóa vô hướng 8-bit `int8` (nén RAM 4x, nhân tích vô hướng nguyên `i32`).
- `binary`: Lượng tử hóa nhị phân 1-bit + POPCNT phần cứng (`u64::count_ones`, nén RAM 32x).
- `binary_rescore`: Hai tầng bất đối xứng (Binary 1-bit POPCNT lọc thô → SQ8 chấm lại chính xác).
- `hnsw_fp32`: Đồ thị HNSW đa tầng native Rust (FP32, tìm kiếm lô song song Rayon, lưu nhị phân `VHNS`).
- `hnsw_sq8`: Đồ thị HNSW đa tầng native Rust kết hợp lượng tử hóa SQ8 (nén vector 4x + độ phức tạp O(log N)).
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


def _nap_rust_core() -> Optional[Any]:
    """Nạp mô-đun Rust native ưu tiên bản có đủ kernel lượng tử hóa và RustHNSWIndex."""
    candidates: List[Any] = []
    try:
        from vietnamese_ai import vietnamese_ai_rust as _pkg_rust  # type: ignore

        candidates.append(_pkg_rust)
    except ImportError:
        pass
    try:
        import vietnamese_ai_rust as _top_rust  # type: ignore

        candidates.append(_top_rust)
    except ImportError:
        pass
    for mod in candidates:
        if (
            hasattr(mod, "fast_sq8_quantize_bytes")
            and hasattr(mod, "fast_binary_quantize_bytes")
            and hasattr(mod, "RustHNSWIndex")
        ):
            return mod
    return candidates[0] if candidates else None


_rust_core: Any = _nap_rust_core()

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def co_rust_luong_tu() -> bool:
    """Kiểm tra lõi Rust đã có đủ kernel lượng tử hóa vector (SQ8, Binary POPCNT, RustHNSWIndex) hay chưa."""
    return (
        _rust_core is not None
        and hasattr(_rust_core, "fast_sq8_quantize_bytes")
        and hasattr(_rust_core, "fast_binary_quantize_bytes")
        and hasattr(_rust_core, "RustHNSWIndex")
    )


def _chuan_hoa_l2_hang(mat: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 từng hàng của ma trận vector float32."""
    arr = np.ascontiguousarray(mat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 1e-10, norms, 1.0)
    return np.ascontiguousarray(arr / safe, dtype=np.float32)


def _luong_tu_sq8_numpy(unit_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lượng tử hóa vô hướng đối xứng int8 theo từng vector (chuẩn đối chiếu với Rust SQ8)."""
    absmax = np.max(np.abs(unit_mat), axis=1)
    scales = np.where(absmax > 1e-10, absmax / 127.0, 1.0 / 127.0).astype(np.float32)
    inv_scales = np.where(absmax > 1e-10, 127.0 / absmax, 0.0).astype(np.float32)
    codes = np.clip(np.round(unit_mat * inv_scales[:, None]), -127.0, 127.0).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def _luong_tu_nhi_phan_numpy(mat: np.ndarray) -> np.ndarray:
    """Đóng gói dấu bit (>= 0 -> 1) vào mảng uint8 theo khối 64-bit little-endian khớp bố cục Rust."""
    n_docs, dim = mat.shape
    words = (dim + 63) // 64
    padded_bits = words * 64
    bits = np.zeros((n_docs, padded_bits), dtype=np.uint8)
    bits[:, :dim] = (mat >= 0.0).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    return np.ascontiguousarray(packed, dtype=np.uint8)


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Động cơ chỉ mục vector lượng tử hóa SQ8 (4x), Binary 1-bit POPCNT (32x), "
        "Two-Stage Rescore và Native Rust HNSW bền vững chạy 100% trên Edge CPU."
    ),
    nhom_extra="core",
)
class ChiMucVectorLuongTu:
    """Chỉ mục tìm kiếm vector nén RAM 4x–32x và HNSW native Rust cho CPU không GPU."""

    CAC_CHE_DO_HOP_LE = ("fp32", "sq8", "binary", "binary_rescore", "hnsw_fp32", "hnsw_sq8")
    _MAGIC = b"VQI1"

    def __init__(
        self,
        che_do: str = "hnsw_sq8",
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        oversample: int = 8,
        seed: int = 42,
        bat_rust: bool = True,
    ) -> None:
        mode = str(che_do or "hnsw_sq8").strip().lower()
        if mode not in self.CAC_CHE_DO_HOP_LE:
            raise ValueError(
                f"Chế độ lượng tử hóa không hợp lệ: {che_do}. Hỗ trợ: {', '.join(self.CAC_CHE_DO_HOP_LE)}"
            )
        self.che_do = mode
        self.m = max(4, int(m))
        self.ef_construction = max(self.m * 2, int(ef_construction))
        self.ef_search = max(self.m, int(ef_search))
        self.oversample = max(2, int(oversample))
        self.seed = int(seed)
        self.bat_rust = bool(bat_rust)

        self.so_vector: int = 0
        self.so_chieu: int = 0
        self._dung_rust_thuc_te: bool = False

        self._fp32_unit: Optional[np.ndarray] = None
        self._fp32_bytes: bytes = b""
        self._sq8_codes: Optional[np.ndarray] = None
        self._sq8_codes_bytes: bytes = b""
        self._sq8_scales: Optional[np.ndarray] = None
        self._sq8_scales_bytes: bytes = b""
        self._bin_packed: Optional[np.ndarray] = None
        self._bin_bytes: bytes = b""
        self._rust_hnsw: Optional[Any] = None

    def _co_the_dung_rust(self) -> bool:
        return self.bat_rust and co_rust_luong_tu()

    def xay_dung(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Nén và dựng chỉ mục từ ma trận vector float32 `(N, D)`."""
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Ma trận vector đầu vào phải có 2 chiều (N, D), nhận được shape={mat.shape}")
        n_docs, dim = int(mat.shape[0]), int(mat.shape[1])
        self.so_vector = n_docs
        self.so_chieu = dim
        self._fp32_unit = None
        self._fp32_bytes = b""
        self._sq8_codes = None
        self._sq8_codes_bytes = b""
        self._sq8_scales = None
        self._sq8_scales_bytes = b""
        self._bin_packed = None
        self._bin_bytes = b""
        self._rust_hnsw = None
        self._dung_rust_thuc_te = False

        if n_docs == 0 or dim == 0:
            return self.thong_ke_bo_nho()

        mat_c = np.ascontiguousarray(mat, dtype=np.float32)
        use_rust = self._co_the_dung_rust()

        if self.che_do == "fp32":
            self._fp32_unit = _chuan_hoa_l2_hang(mat_c)
            if use_rust:
                self._fp32_bytes = self._fp32_unit.tobytes()
                self._dung_rust_thuc_te = True
        elif self.che_do == "sq8":
            if use_rust:
                codes_b, scales_b = _rust_core.fast_sq8_quantize_bytes(mat_c.tobytes(), n_docs, dim)
                self._sq8_codes_bytes = bytes(codes_b)
                self._sq8_scales_bytes = bytes(scales_b)
                self._dung_rust_thuc_te = True
            else:
                unit = _chuan_hoa_l2_hang(mat_c)
                self._sq8_codes, self._sq8_scales = _luong_tu_sq8_numpy(unit)
        elif self.che_do == "binary":
            if use_rust:
                self._bin_bytes = bytes(_rust_core.fast_binary_quantize_bytes(mat_c.tobytes(), n_docs, dim))
                self._dung_rust_thuc_te = True
            else:
                self._bin_packed = _luong_tu_nhi_phan_numpy(mat_c)
        elif self.che_do == "binary_rescore":
            if use_rust:
                raw_mat = mat_c.tobytes()
                self._bin_bytes = bytes(_rust_core.fast_binary_quantize_bytes(raw_mat, n_docs, dim))
                codes_b, scales_b = _rust_core.fast_sq8_quantize_bytes(raw_mat, n_docs, dim)
                self._sq8_codes_bytes = bytes(codes_b)
                self._sq8_scales_bytes = bytes(scales_b)
                self._dung_rust_thuc_te = True
            else:
                unit = _chuan_hoa_l2_hang(mat_c)
                self._bin_packed = _luong_tu_nhi_phan_numpy(unit)
                self._sq8_codes, self._sq8_scales = _luong_tu_sq8_numpy(unit)
        elif self.che_do in ("hnsw_fp32", "hnsw_sq8"):
            q_mode = "sq8" if self.che_do == "hnsw_sq8" else "fp32"
            if use_rust:
                idx = _rust_core.RustHNSWIndex(
                    dim,
                    m=self.m,
                    ef_construction=self.ef_construction,
                    ef_search=self.ef_search,
                    quantization=q_mode,
                    seed=self.seed,
                )
                idx.build_from_bytes(mat_c.tobytes(), n_docs)
                self._rust_hnsw = idx
                self._dung_rust_thuc_te = True
            else:
                unit = _chuan_hoa_l2_hang(mat_c)
                if q_mode == "sq8":
                    self._sq8_codes, self._sq8_scales = _luong_tu_sq8_numpy(unit)
                else:
                    self._fp32_unit = unit

        return self.thong_ke_bo_nho()

    def tim_kiem(
        self,
        query: np.ndarray,
        top_k: int = 10,
        ef_search: Optional[int] = None,
        oversample: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Tìm `top_k` chỉ số vector gần nhất kèm điểm cosine xấp xỉ/chính xác."""
        if self.so_vector == 0 or top_k <= 0:
            return []
        q_arr = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(-1))
        if q_arr.shape[0] != self.so_chieu:
            raise ValueError(f"Query vector phải có {self.so_chieu} chiều, nhận được {q_arr.shape[0]}")

        k = min(int(top_k), self.so_vector)
        osamp = int(oversample) if oversample is not None else self.oversample

        if self._rust_hnsw is not None:
            res = self._rust_hnsw.search_bytes(
                q_arr.tobytes(),
                top_k=k,
                ef_search=int(ef_search) if ef_search is not None else self.ef_search,
            )
            return [(int(i), float(s)) for i, s in res]

        if self._dung_rust_thuc_te and _rust_core is not None:
            q_bytes = q_arr.tobytes()
            if self.che_do == "fp32":
                res = _rust_core.fast_top_k_cosine_bytes(
                    q_bytes, self._fp32_bytes, self.so_vector, self.so_chieu, k
                )
                return [(int(i), float(s)) for i, s in res]
            if self.che_do == "sq8":
                res = _rust_core.fast_sq8_top_k_bytes(
                    q_bytes,
                    self._sq8_codes_bytes,
                    self._sq8_scales_bytes,
                    self.so_vector,
                    self.so_chieu,
                    k,
                )
                return [(int(i), float(s)) for i, s in res]
            if self.che_do == "binary":
                res = _rust_core.fast_binary_hamming_top_k_bytes(
                    q_bytes, self._bin_bytes, self.so_vector, self.so_chieu, k
                )
                return [(int(i), float(s)) for i, s in res]
            if self.che_do == "binary_rescore":
                res = _rust_core.fast_binary_rescore_sq8_top_k_bytes(
                    q_bytes,
                    self._bin_bytes,
                    self._sq8_codes_bytes,
                    self._sq8_scales_bytes,
                    self.so_vector,
                    self.so_chieu,
                    k,
                    osamp,
                )
                return [(int(i), float(s)) for i, s in res]

        return self._tim_kiem_numpy(q_arr, k, osamp)

    def _tim_kiem_numpy(self, q_arr: np.ndarray, k: int, oversample: int) -> List[Tuple[int, float]]:
        """Đường dự phòng thuần NumPy bảo đảm tương đương toán học khi không biên dịch Rust."""
        q_norm = float(np.linalg.norm(q_arr))
        q_unit = (q_arr / q_norm) if q_norm > 1e-10 else q_arr

        if self.che_do in ("fp32", "hnsw_fp32") and self._fp32_unit is not None:
            scores = self._fp32_unit @ q_unit
        elif self.che_do in ("sq8", "hnsw_sq8") and self._sq8_codes is not None and self._sq8_scales is not None:
            q_codes, q_scales = _luong_tu_sq8_numpy(q_unit.reshape(1, -1))
            dots = self._sq8_codes.astype(np.int32) @ q_codes[0].astype(np.int32)
            scores = dots.astype(np.float32) * (float(q_scales[0]) * self._sq8_scales)
        elif self.che_do == "binary" and self._bin_packed is not None:
            q_bin = _luong_tu_nhi_phan_numpy(q_unit.reshape(1, -1))[0]
            hamming = np.sum(_POPCOUNT_LUT[np.bitwise_xor(self._bin_packed, q_bin)], axis=1)
            scores = 1.0 - (2.0 * hamming.astype(np.float32)) / float(self.so_chieu)
        elif (
            self.che_do == "binary_rescore"
            and self._bin_packed is not None
            and self._sq8_codes is not None
            and self._sq8_scales is not None
        ):
            q_bin = _luong_tu_nhi_phan_numpy(q_unit.reshape(1, -1))[0]
            hamming = np.sum(_POPCOUNT_LUT[np.bitwise_xor(self._bin_packed, q_bin)], axis=1)
            cand_k = min(self.so_vector, max(64, k * oversample))
            cand_idx = np.argpartition(hamming, cand_k - 1)[:cand_k]
            q_codes, q_scales = _luong_tu_sq8_numpy(q_unit.reshape(1, -1))
            dots = self._sq8_codes[cand_idx].astype(np.int32) @ q_codes[0].astype(np.int32)
            sub_scores = dots.astype(np.float32) * (float(q_scales[0]) * self._sq8_scales[cand_idx])
            order = np.argsort(-sub_scores)[:k]
            return [(int(cand_idx[o]), float(sub_scores[o])) for o in order]
        else:
            return []

        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in order]

    def tim_kiem_lo(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[List[Tuple[int, float]]]:
        """Tìm kiếm song song nhiều truy vấn cùng lúc (tận dụng Rayon batch trên Rust HNSW)."""
        q_mat = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        if q_mat.ndim != 2 or q_mat.shape[0] == 0:
            return []
        if self._rust_hnsw is not None:
            res = self._rust_hnsw.search_batch_bytes(
                q_mat.tobytes(),
                int(q_mat.shape[0]),
                top_k=min(int(top_k), self.so_vector),
                ef_search=int(ef_search) if ef_search is not None else self.ef_search,
            )
            return [[(int(i), float(s)) for i, s in row] for row in res]
        return [self.tim_kiem(q_mat[i], top_k=top_k, ef_search=ef_search) for i in range(q_mat.shape[0])]

    def luu_bytes(self) -> bytes:
        """Tuần tự hóa chỉ mục lượng tử hóa / Rust HNSW sang nhị phân an toàn (không dùng pickle)."""
        mode_b = self.che_do.encode("ascii")
        header = struct.pack(
            "<4sBIIIIII",
            self._MAGIC,
            len(mode_b),
            self.so_vector,
            self.so_chieu,
            self.m,
            self.ef_construction,
            self.ef_search,
            self.oversample,
        )
        parts: List[bytes] = [header, mode_b]
        if self._rust_hnsw is not None:
            blob = bytes(self._rust_hnsw.to_bytes())
            parts.append(struct.pack("<BI", 1, len(blob)))
            parts.append(blob)
            return b"".join(parts)

        payload_items: List[bytes] = []
        if self._fp32_unit is not None:
            payload_items.append(self._fp32_unit.astype(np.float32).tobytes())
        elif self._fp32_bytes:
            payload_items.append(self._fp32_bytes)
        else:
            payload_items.append(b"")

        if self._sq8_codes_bytes:
            payload_items.append(self._sq8_codes_bytes)
            payload_items.append(self._sq8_scales_bytes)
        elif self._sq8_codes is not None and self._sq8_scales is not None:
            payload_items.append(self._sq8_codes.astype(np.int8).tobytes())
            payload_items.append(self._sq8_scales.astype(np.float32).tobytes())
        else:
            payload_items.extend([b"", b""])

        if self._bin_bytes:
            payload_items.append(self._bin_bytes)
        elif self._bin_packed is not None:
            payload_items.append(self._bin_packed.astype(np.uint8).tobytes())
        else:
            payload_items.append(b"")

        parts.append(struct.pack("<B", 0))
        for item in payload_items:
            parts.append(struct.pack("<I", len(item)))
            parts.append(item)
        return b"".join(parts)

    @classmethod
    def tai_bytes(cls, raw: bytes, bat_rust: bool = True) -> "ChiMucVectorLuongTu":
        """Khôi phục `ChiMucVectorLuongTu` (kể cả `RustHNSWIndex`) từ luồng nhị phân `VQI1`."""
        hdr_size = struct.calcsize("<4sBIIIIII")
        if len(raw) < hdr_size or raw[:4] != cls._MAGIC:
            raise ValueError("Định dạng nhị phân ChiMucVectorLuongTu không hợp lệ (sai magic VQI1)")
        _, mode_len, n_docs, dim, m, ef_c, ef_s, osamp = struct.unpack("<4sBIIIIII", raw[:hdr_size])
        off = hdr_size
        mode = raw[off : off + mode_len].decode("ascii")
        off += mode_len
        kind = raw[off]
        off += 1

        obj = cls(
            che_do=mode,
            m=m,
            ef_construction=ef_c,
            ef_search=ef_s,
            oversample=osamp,
            bat_rust=bat_rust,
        )
        obj.so_vector = int(n_docs)
        obj.so_chieu = int(dim)

        if kind == 1:
            (blob_len,) = struct.unpack("<I", raw[off : off + 4])
            off += 4
            blob = raw[off : off + blob_len]
            if not (bat_rust and co_rust_luong_tu()):
                raise RuntimeError("Cần lõi Rust `vietnamese_ai_rust` để khôi phục nhị phân RustHNSWIndex")
            obj._rust_hnsw = _rust_core.RustHNSWIndex.from_bytes(blob)
            obj._dung_rust_thuc_te = True
            return obj

        blobs: List[bytes] = []
        for _ in range(4):
            (blen,) = struct.unpack("<I", raw[off : off + 4])
            off += 4
            blobs.append(raw[off : off + blen])
            off += blen

        fp32_b, sq8_c_b, sq8_s_b, bin_b = blobs
        use_rust = bat_rust and co_rust_luong_tu()
        if fp32_b:
            obj._fp32_unit = np.frombuffer(fp32_b, dtype=np.float32).reshape(n_docs, dim).copy()
            if use_rust:
                obj._fp32_bytes = fp32_b
                obj._dung_rust_thuc_te = True
        if sq8_c_b and sq8_s_b:
            if use_rust:
                obj._sq8_codes_bytes = sq8_c_b
                obj._sq8_scales_bytes = sq8_s_b
                obj._dung_rust_thuc_te = True
            else:
                obj._sq8_codes = np.frombuffer(sq8_c_b, dtype=np.int8).reshape(n_docs, dim).copy()
                obj._sq8_scales = np.frombuffer(sq8_s_b, dtype=np.float32).reshape(n_docs).copy()
        if bin_b:
            words = (dim + 63) // 64
            if use_rust:
                obj._bin_bytes = bin_b
                obj._dung_rust_thuc_te = True
            else:
                obj._bin_packed = np.frombuffer(bin_b, dtype=np.uint8).reshape(n_docs, words * 8).copy()
        return obj

    def thong_ke_bo_nho(self) -> Dict[str, Any]:
        """Báo cáo dung lượng bộ nhớ thực tế (bytes/MB) và tỷ lệ nén so với ma trận FP32 chuẩn."""
        fp32_bytes = int(self.so_vector * self.so_chieu * 4)
        vec_bytes = 0
        total_bytes = 0

        if self._rust_hnsw is not None:
            vec_bytes = int(self._rust_hnsw.vector_bytes())
            total_bytes = int(self._rust_hnsw.memory_bytes())
        else:
            if self._fp32_unit is not None:
                vec_bytes += int(self._fp32_unit.nbytes)
            elif self._fp32_bytes:
                vec_bytes += len(self._fp32_bytes)
            if self._sq8_codes_bytes:
                vec_bytes += len(self._sq8_codes_bytes) + len(self._sq8_scales_bytes)
            elif self._sq8_codes is not None and self._sq8_scales is not None:
                vec_bytes += int(self._sq8_codes.nbytes + self._sq8_scales.nbytes)
            if self._bin_bytes:
                vec_bytes += len(self._bin_bytes)
            elif self._bin_packed is not None:
                vec_bytes += int(self._bin_packed.nbytes)
            total_bytes = vec_bytes

        ti_le_nen_vector = round(fp32_bytes / max(1, vec_bytes), 2) if fp32_bytes > 0 else 1.0
        ti_le_nen_tong = round(fp32_bytes / max(1, total_bytes), 2) if fp32_bytes > 0 else 1.0
        return {
            "che_do": self.che_do,
            "dung_rust": self._dung_rust_thuc_te,
            "so_vector": self.so_vector,
            "so_chieu": self.so_chieu,
            "fp32_bytes": fp32_bytes,
            "vector_bytes": vec_bytes,
            "tong_bytes": total_bytes,
            "fp32_mb": round(fp32_bytes / (1024 * 1024), 3),
            "vector_mb": round(vec_bytes / (1024 * 1024), 3),
            "tong_mb": round(total_bytes / (1024 * 1024), 3),
            "ti_le_nen_vector": ti_le_nen_vector,
            "ti_le_nen_tong": ti_le_nen_tong,
            "so_tang_hnsw": int(self._rust_hnsw.num_layers()) if self._rust_hnsw is not None else 0,
        }


QuantizedVectorIndex = ChiMucVectorLuongTu

__all__ = [
    "co_rust_luong_tu",
    "ChiMucVectorLuongTu",
    "QuantizedVectorIndex",
]
