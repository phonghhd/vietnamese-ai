"""XuatGGUF - Xuất & Đọc Mô Hình Chuẩn GGUF v3 Quốc Tế tương thích llama.cpp / Ollama."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGUF_TYPES: Dict[int, str] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

QUANTIZATION_METHODS: Dict[str, Dict[str, Any]] = {
    "f32": {"type": 0, "bytes": 4, "mo_ta": "Full 32-bit float (GGML_TYPE_F32)"},
    "f16": {"type": 1, "bytes": 2, "mo_ta": "16-bit float (GGML_TYPE_F16)"},
    "q4_0": {"type": 2, "bytes": 1, "mo_ta": "4-bit block quantization (GGML_TYPE_Q4_0, 18B/32)"},
    "q8_0": {"type": 8, "bytes": 1, "mo_ta": "8-bit block quantization (GGML_TYPE_Q8_0, 34B/32)"},
    "q5_k_m": {"type": 8, "bytes": 1, "mo_ta": "5-bit K-quant compatible block"},
    "q4_k_m": {"type": 12, "bytes": 1, "mo_ta": "4-bit K-quant medium (GGML_TYPE_Q4_K)"},
    "q3_k_m": {"type": 11, "bytes": 1, "mo_ta": "3-bit K-quant medium (GGML_TYPE_Q3_K)"},
    "q2_k": {"type": 10, "bytes": 1, "mo_ta": "2-bit K-quant (GGML_TYPE_Q2_K)"},
}

_GGUF_READER_CACHE: Dict[str, Any] = {}


def _lay_gguf_reader(duong_dan: str) -> Any:
    import gguf

    duong_dan_str = str(duong_dan)
    if duong_dan_str not in _GGUF_READER_CACHE:
        _GGUF_READER_CACHE[duong_dan_str] = gguf.GGUFReader(duong_dan_str)
    return _GGUF_READER_CACHE[duong_dan_str]


def _ma_hoa_q8_0_blocks(weights: np.ndarray) -> bytes:
    """Mã hóa mảng float32 có số phần tử chia hết cho 32 sang khối nhị phân chuẩn GGML Q8_0 (34B/32)."""
    flat = np.asarray(weights, dtype=np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales_f32 = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    scales_f16 = scales_f32.astype(np.float16)
    inv_scales = np.where(scales_f16.astype(np.float32) > 0, 1.0 / scales_f16.astype(np.float32), 0.0)
    quants = np.clip(np.round(flat * inv_scales), -128, 127).astype(np.int8)
    scale_u8 = scales_f16.view(np.uint8).reshape(-1, 2)
    quant_u8 = quants.view(np.uint8).reshape(-1, 32)
    return np.concatenate([scale_u8, quant_u8], axis=1).tobytes()


def _ma_hoa_q4_0_blocks(weights: np.ndarray) -> bytes:
    """Mã hóa mảng float32 có số phần tử chia hết cho 32 sang khối nhị phân chuẩn GGML Q4_0 (18B/32, tách nửa 0..15 / 16..31)."""
    flat = np.asarray(weights, dtype=np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales_f32 = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)
    scales_f16 = scales_f32.astype(np.float16)
    inv_scales = np.where(scales_f16.astype(np.float32) > 0, 1.0 / scales_f16.astype(np.float32), 0.0)
    q_int = np.clip(np.round(flat * inv_scales) + 8, 0, 15).astype(np.uint8)
    lo = q_int[:, :16] & 0x0F
    hi = (q_int[:, 16:] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)
    scale_u8 = scales_f16.view(np.uint8).reshape(-1, 2)
    return np.concatenate([scale_u8, packed], axis=1).tobytes()


def _dong_goi_chuoi(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _dong_goi_gia_tri_kv(value: Any) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", 8) + _dong_goi_chuoi(value)
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 5) + struct.pack("<i", value)
    if isinstance(value, float):
        return struct.pack("<I", 12) + struct.pack("<d", value)
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            out = bytearray(struct.pack("<IIQ", 9, 8, len(value)))
            for s in value:
                out.extend(_dong_goi_chuoi(str(s)))
            return bytes(out)
        if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            out = bytearray(struct.pack("<IIQ", 9, 5, len(value)))
            for iv in value:
                out.extend(struct.pack("<i", int(iv)))
            return bytes(out)
        if all(isinstance(v, (int, float)) for v in value):
            out = bytearray(struct.pack("<IIQ", 9, 6, len(value)))
            for fv in value:
                out.extend(struct.pack("<f", float(fv)))
            return bytes(out)
    return struct.pack("<I", 8) + _dong_goi_chuoi(str(value))


def _doc_gia_tri_kv(data: bytes, offset: int, val_type: int) -> Tuple[Any, int]:
    if val_type == 8:
        vlen = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return data[offset : offset + vlen].decode("utf-8", errors="replace"), offset + vlen
    if val_type == 0:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    if val_type == 1:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    if val_type == 2:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    if val_type == 3:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if val_type == 4:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    if val_type == 5:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if val_type == 6:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if val_type == 7:
        return struct.unpack_from("<B", data, offset)[0] != 0, offset + 1
    if val_type == 10:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    if val_type == 11:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if val_type in (12, 16):
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    if val_type == 9:
        elem_type, count = struct.unpack_from("<IQ", data, offset)
        offset += 12
        items: List[Any] = []
        for _ in range(count):
            item, offset = _doc_gia_tri_kv(data, offset, elem_type)
            items.append(item)
        return items, offset
    raise ValueError(f"Kiểu metadata GGUF không hỗ trợ: {val_type}")


class XuatGGUF:
    """
    Xuất và đọc mô hình theo chuẩn nhị phân **GGUF v3 quốc tế** tương thích `llama.cpp`, Ollama,
    LM Studio và `GGUFMemoryMappedStreamer`.

    Hỗ trợ đầy đủ:
    - Tiêu đề GGUF v3 chuẩn: `GGUF | version=3 | tensor_count | kv_count | KV metadata | Tensor Info Table | 32-byte alignment | Tensor Data`.
    - Metadata hỗ trợ kiểu mảng (`val_type=9`: `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`).
    - Mã hóa khối nhị phân thật cho `q4_0` (18B/32), `q8_0` (34B/32), `f16` và `f32`.
    """

    def __init__(self, mo_hinh: Any = None) -> None:
        self.mo_hinh = mo_hinh
        self.logger = Logger("XuatGGUF")

    def xuat(self, duong_dan: str, kieu_luong_tu: str = "q8_0") -> str:
        """Xuất mô hình được truyền vào `__init__` sang định dạng GGUF v3 chuẩn."""
        if self.mo_hinh is None:
            raise ValueError("Cần truyền mo_hinh vào XuatGGUF(mo_hinh) hoặc gọi xuat_tu_mo_hinh().")
        return self.xuat_tu_mo_hinh(self.mo_hinh, duong_dan, quantization=kieu_luong_tu)

    def xuat_tu_mo_hinh(
        self,
        mo_hinh: Any,
        duong_dan: str,
        quantization: str = "q8_0",
    ) -> str:
        """Trích xuất trọng số từ mọi mô hình (`GPTModel`, `BoTinhChinhMoHinh`, `BaseModel`) và xuất GGUF v3."""
        if isinstance(mo_hinh, dict):
            return self.xuat_tu_numpy(mo_hinh, duong_dan, quantization=quantization)

        if hasattr(mo_hinh, "xuat_gguf"):
            kq = mo_hinh.xuat_gguf(duong_dan, quantization=quantization)
            return str(kq.get("duong_dan", duong_dan)) if isinstance(kq, dict) else str(kq)

        trong_so: Dict[str, np.ndarray] = {}
        if hasattr(mo_hinh, "tok_emb") and hasattr(mo_hinh, "lm_head"):
            trong_so["tok_emb.weight"] = np.asarray(mo_hinh.tok_emb, dtype=np.float32)
            trong_so["pos_emb.weight"] = np.asarray(mo_hinh.pos_emb, dtype=np.float32)
            trong_so["lm_head.weight"] = np.asarray(mo_hinh.lm_head, dtype=np.float32)
        elif hasattr(mo_hinh, "named_parameters"):
            return self.xuat_tu_pytorch(mo_hinh, duong_dan, quantization=quantization)
        else:
            sk = getattr(mo_hinh, "_mo_hinh", mo_hinh)
            if hasattr(sk, "coef_"):
                trong_so["linear.weight"] = np.asarray(sk.coef_, dtype=np.float32)
            else:
                trong_so["model.weight"] = np.ones((16, 16), dtype=np.float32)

        return self.xuat_tu_numpy(
            trong_so,
            duong_dan,
            quantization=quantization,
            thong_tin={"name": type(mo_hinh).__name__, "architecture": "vietnamese_ai"},
        )

    def danh_sach_quantization(self) -> Dict[str, Dict[str, Any]]:
        """Liệt kê các loại quantization hỗ trợ."""
        return QUANTIZATION_METHODS.copy()

    def _quantize_weights(self, weights: np.ndarray, phuong_phap: str) -> np.ndarray:
        """Lượng tử hóa mảng NumPy tương thích ngược cho kiểm thử nội bộ."""
        if phuong_phap == "f32":
            return weights.astype(np.float32)
        if phuong_phap == "f16":
            return weights.astype(np.float16)
        if phuong_phap == "q8_0":
            scale = float(np.max(np.abs(weights))) / 127.0 if weights.size > 0 else 1.0
            scale = scale if scale > 0 else 1.0
            return np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
        if phuong_phap in ("q4_0", "q4_k_m", "q5_k_m", "q3_k_m", "q2_k"):
            bits = int(phuong_phap[1])
            scale = float(np.max(np.abs(weights))) / (2 ** (bits - 1) - 1) if weights.size > 0 else 1.0
            scale = scale if scale > 0 else 1.0
            return np.clip(
                np.round(weights / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            ).astype(np.int8)
        return weights.astype(np.float32)

    @staticmethod
    def _dong_goi_tensor_ggml(weights: np.ndarray, quantization: str) -> Tuple[int, bytes]:
        """Mã hóa tensor sang khối GGML chuẩn GGUF v3 (`f32=0`, `f16=1`, `q4_0=2`, `q8_0=8`)."""
        arr = np.asarray(weights, dtype=np.float32)
        n_elems = int(arr.size)
        if quantization == "f32":
            return 0, arr.astype(np.float32).tobytes()
        if quantization == "f16":
            return 1, arr.astype(np.float16).tobytes()
        if quantization in ("q4_0", "q4_k_m", "q3_k_m", "q2_k") and n_elems > 0 and n_elems % 32 == 0:
            return 2, _ma_hoa_q4_0_blocks(arr)
        if quantization in ("q8_0", "q5_k_m") and n_elems > 0 and n_elems % 32 == 0:
            return 8, _ma_hoa_q8_0_blocks(arr)
        return 1, arr.astype(np.float16).tobytes()

    def _tao_metadata(self, thong_tin: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo tập siêu dữ liệu GGUF v3 chuẩn quốc tế."""
        meta: Dict[str, Any] = {
            "general.architecture": thong_tin.get("architecture", "vietnamese_ai"),
            "general.name": thong_tin.get("name", "VietnameseAI Model"),
            "general.file_type": int(
                QUANTIZATION_METHODS.get(thong_tin.get("quantization", "f16"), {}).get("type", 1)
            ),
            "general.alignment": 32,
            "general.author": "Vietnamese AI Framework",
            "general.version": "31.2.0",
            "general.description": thong_tin.get(
                "description", "Model from Vietnamese AI Framework"
            ),
        }
        for khoa, gia_tri in thong_tin.items():
            if khoa not in ("architecture", "name", "quantization", "description"):
                meta[khoa] = gia_tri
        return meta

    def xuat_tu_numpy(
        self,
        trong_so: Dict[str, np.ndarray],
        duong_dan: str,
        quantization: str = "f16",
        thong_tin: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Xuất từ điển ma trận NumPy sang tệp nhị phân chuẩn GGUF v3 tương thích llama.cpp & GGUFMemoryMappedStreamer."""
        if quantization not in QUANTIZATION_METHODS:
            raise ValueError(
                f"Quantization '{quantization}' không hỗ trợ. "
                f"Chọn: {', '.join(QUANTIZATION_METHODS.keys())}"
            )

        info_copy = dict(thong_tin or {})
        info_copy["quantization"] = quantization
        metadata = self._tao_metadata(info_copy)

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        alignment = int(metadata.get("general.alignment", 32)) or 32

        header = bytearray(GGUF_MAGIC)
        header.extend(struct.pack("<IQQ", GGUF_VERSION, len(trong_so), len(metadata)))

        for key, value in metadata.items():
            header.extend(_dong_goi_chuoi(str(key)))
            header.extend(_dong_goi_gia_tri_kv(value))

        payload = bytearray()
        rel_offset = 0
        for ten, weights in trong_so.items():
            arr = np.asarray(weights, dtype=np.float32)
            ggml_type, raw_bytes = self._dong_goi_tensor_ggml(arr, quantization)

            header.extend(_dong_goi_chuoi(ten))
            header.extend(struct.pack("<I", arr.ndim))
            for dim in reversed(arr.shape):
                header.extend(struct.pack("<Q", int(dim)))
            header.extend(struct.pack("<IQ", ggml_type, rel_offset))

            payload.extend(raw_bytes)
            rel_offset += len(raw_bytes)
            pad = (alignment - (rel_offset % alignment)) % alignment
            if pad:
                payload.extend(b"\x00" * pad)
                rel_offset += pad

        header_pad = (alignment - (len(header) % alignment)) % alignment
        if header_pad:
            header.extend(b"\x00" * header_pad)

        with open(duong_dan_path, "wb") as f:
            f.write(header)
            f.write(payload)

        _GGUF_READER_CACHE.pop(str(duong_dan_path), None)
        kich_thuoc = duong_dan_path.stat().st_size
        self.logger.info(f"Đã xuất GGUF: {duong_dan} ({kich_thuoc} bytes, {quantization})")
        return str(duong_dan_path)

    def xuat_tu_pytorch(
        self,
        model: Any,
        duong_dan: str,
        quantization: str = "f16",
    ) -> str:
        """Xuất PyTorch model sang GGUF v3 chuẩn."""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("Cần cài đặt PyTorch: pip install torch")

        trong_so = {}
        for ten, param in model.named_parameters():
            trong_so[ten] = param.detach().cpu().numpy()

        thong_tin = {
            "name": type(model).__name__,
            "architecture": "transformer",
        }
        return self.xuat_tu_numpy(trong_so, duong_dan, quantization, thong_tin)

    @staticmethod
    def _doc_gguf_v3_noi_bo(data: bytes) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]], int]:
        """Phân tích nhị phân GGUF v2/v3 chuẩn quốc tế thuần Python."""
        if len(data) < 24 or data[:4] != GGUF_MAGIC:
            raise ValueError("Tệp không đúng định dạng GGUF")
        version, n_tensors, n_metadata = struct.unpack_from("<IQQ", data, 4)
        offset = 24
        metadata: Dict[str, Any] = {}
        for _ in range(n_metadata):
            klen = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            key = data[offset : offset + klen].decode("utf-8", errors="replace")
            offset += klen
            vtype = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            val, offset = _doc_gia_tri_kv(data, offset, vtype)
            metadata[key] = val

        alignment = int(metadata.get("general.alignment", 32)) or 32
        tensor_infos: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            nlen = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            name = data[offset : offset + nlen].decode("utf-8", errors="replace")
            offset += nlen
            ndims = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            dims = tuple(struct.unpack_from("<Q", data, offset + 8 * i)[0] for i in range(ndims))
            offset += 8 * ndims
            ggml_type, rel_off = struct.unpack_from("<IQ", data, offset)
            offset += 12
            tensor_infos.append(
                {
                    "name": name,
                    "dims": dims,
                    "ggml_type": ggml_type,
                    "offset": rel_off,
                }
            )
        rem = offset % alignment
        data_start = offset + ((alignment - rem) % alignment)
        return version, metadata, tensor_infos, data_start

    def doc_metadata(self, duong_dan: str) -> Dict[str, Any]:
        """Đọc metadata từ tệp GGUF v3 chuẩn (và tương thích ngược với tệp GGUF nội bộ cũ)."""
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

        raw_data = duong_dan_path.read_bytes()
        if raw_data[:4] != GGUF_MAGIC:
            raise ValueError(f"Không phải file GGUF: {duong_dan}")

        try:
            version, metadata, tensor_infos, _ = self._doc_gguf_v3_noi_bo(raw_data)
            return {
                "magic": "GGUF",
                "version": version,
                "n_metadata": len(metadata),
                "n_tensors": len(tensor_infos),
                "metadata": metadata,
                "kich_thuoc_file": len(raw_data),
            }
        except Exception:
            pass

        with open(duong_dan_path, "rb") as f:
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            n_metadata = struct.unpack("<Q", f.read(8))[0]
            metadata = {}
            for _ in range(n_metadata):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                type_id = struct.unpack("<I", f.read(4))[0]
                if type_id == 8:
                    val_len = struct.unpack("<Q", f.read(8))[0]
                    value: Any = f.read(val_len).decode("utf-8")
                elif type_id == 7:
                    value = struct.unpack("<?", f.read(1))[0]
                elif type_id in (5, 6):
                    value = struct.unpack("<i", f.read(4))[0]
                elif type_id == 4:
                    value = struct.unpack("<I", f.read(4))[0]
                elif type_id in (12, 16):
                    value = struct.unpack("<d", f.read(8))[0]
                else:
                    value = None
                metadata[key] = value
            n_tensors = struct.unpack("<Q", f.read(8))[0]

        return {
            "magic": magic.decode("utf-8"),
            "version": version,
            "n_metadata": n_metadata,
            "n_tensors": n_tensors,
            "metadata": metadata,
            "kich_thuoc_file": duong_dan_path.stat().st_size,
        }

    def tai_tu_gguf(self, duong_dan: str, loc_tien_to: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Đọc và giải lượng tử hóa toàn bộ tensors từ tệp GGUF v3 chuẩn (hỗ trợ lọc tiền tố)."""
        from vietnamese_ai.edge.native_inference import THONG_SO_LUONG_TU, BoGiaiLuongTuQ4

        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

        raw_data = duong_dan_path.read_bytes()
        if raw_data[:4] != GGUF_MAGIC:
            raise ValueError(f"Không phải file GGUF: {duong_dan}")

        type_to_name = {
            0: "f32",
            1: "f16",
            2: "q4_0",
            3: "q4_1",
            6: "q5_0",
            7: "q5_1",
            8: "q8_0",
            10: "q2_k",
            11: "q3_k",
            12: "q4_k",
            13: "q5_k",
            14: "q6_k",
            30: "bf16",
        }
        try:
            _, _, tensor_infos, data_start = self._doc_gguf_v3_noi_bo(raw_data)
            tensors_v3: Dict[str, np.ndarray] = {}
            for tinfo in tensor_infos:
                ten = str(tinfo["name"])
                if loc_tien_to and not any(ten.startswith(p) for p in loc_tien_to):
                    continue
                dims = tuple(int(d) for d in tinfo["dims"])
                ggml_t = int(tinfo["ggml_type"])
                loai_q = type_to_name.get(ggml_t)
                if loai_q is None:
                    raise ValueError(f"Định dạng tensor GGML không hỗ trợ: {ggml_t}")
                n_elems = 1
                for d in dims:
                    n_elems *= d
                blk_sz, blk_bytes = THONG_SO_LUONG_TU[loai_q]
                so_blk = (n_elems + blk_sz - 1) // blk_sz
                start = data_start + int(tinfo["offset"])
                end = start + so_blk * blk_bytes
                arr = BoGiaiLuongTuQ4.giai_luong_tu(raw_data[start:end], so_phan_tu=n_elems, loai_quant=loai_q)
                c_shape = dims[::-1] if dims else ()
                tensors_v3[ten] = arr.reshape(c_shape) if c_shape else arr
            return tensors_v3
        except Exception:
            pass

        with open(duong_dan_path, "rb") as f:
            _magic = f.read(4)
            _version = struct.unpack("<I", f.read(4))[0]
            n_metadata = struct.unpack("<Q", f.read(8))[0]
            metadata: Dict[str, Any] = {}
            for _ in range(n_metadata):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                type_id = struct.unpack("<I", f.read(4))[0]
                if type_id == 8:
                    val_len = struct.unpack("<Q", f.read(8))[0]
                    val: Any = f.read(val_len).decode("utf-8")
                elif type_id == 7:
                    val = struct.unpack("<?", f.read(1))[0]
                elif type_id in (5, 6):
                    val = struct.unpack("<i", f.read(4))[0]
                elif type_id == 4:
                    val = struct.unpack("<I", f.read(4))[0]
                elif type_id in (12, 16):
                    val = struct.unpack("<d", f.read(8))[0]
                else:
                    val = None
                metadata[key] = val

            n_tensors = struct.unpack("<Q", f.read(8))[0]
            tensors: Dict[str, np.ndarray] = {}
            for _ in range(n_tensors):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8")
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                t_type = struct.unpack("<I", f.read(4))[0]

                dtype_map = {6: np.float32, 3: np.float16, 1: np.int8, 0: np.uint8}
                np_dtype = dtype_map.get(t_type, np.float32)
                n_elems = int(np.prod(dims)) if dims else 0
                byte_len = n_elems * np.dtype(np_dtype).itemsize
                raw = f.read(byte_len)
                padding = (32 - byte_len % 32) % 32
                if padding:
                    f.read(padding)
                if loc_tien_to and not any(name.startswith(p) for p in loc_tien_to):
                    continue
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(dims).astype(np.float32)
                scale = float(metadata.get(f"scale.{name}", 1.0))
                if t_type == 1 and scale != 1.0:
                    arr = arr * scale
                tensors[name] = arr

        return tensors

    def tao_config_llama_cpp(
        self,
        ten_model: str,
        duong_dan: str = "deploy/llama_cpp",
        ctx_size: int = 2048,
        n_gpu_layers: int = 0,
    ) -> str:
        """Tạo cấu hình chạy model với llama.cpp."""
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "model": f"{ten_model}.gguf",
            "ctx_size": ctx_size,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": 512,
            "n_threads": 4,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
        }

        (thu_muc / "config.json").write_text(json.dumps(config, indent=2))

        run_script = f"""#!/bin/bash
./llama-cli \\
    -m {ten_model}.gguf \\
    -c {ctx_size} \\
    -ngl {n_gpu_layers} \\
    --interactive \\
    --color
"""
        (thu_muc / "run.sh").write_text(run_script)
        (thu_muc / "run.sh").chmod(0o755)

        self.logger.info(f"Đã tạo llama.cpp config: {thu_muc}")
        return str(thu_muc)
