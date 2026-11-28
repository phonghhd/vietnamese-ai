"""Hệ Thống Suy Luận Đàn CPU Phân Tán (Distributed Swarm Inference & Pipeline Parallelism trên CPU).

Cho phép kết nối nhiều máy tính CPU cấu hình thấp (Mini PC, máy văn phòng, laptop) qua mạng LAN/WAN
để cùng chia sẻ tải tính toán (Model Sharding / Pipeline Parallelism) một mô hình ngôn ngữ lớn
mà không cần GPU đắt đỏ, tích hợp cơ chế xác thực chứng thực mật mã học Verifiable AI (Proof of Inference).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import socket
import ssl
import struct
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.edge.swarm_auth import (
    XacThucCumSwarm,
    phat_hien_ip_tailscale,
    tao_ma_pin_ngau_nhien,
)
from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc, BoTachTuTiengVietSieuToc
from vietnamese_ai.security.verifiable_inference import KiemChungFreivalds, XacThucSuyLuanMaoHiem
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

logger = logging.getLogger("VietnameseAI.SwarmCPU")


def tao_ssl_context_server(
    certfile: str,
    keyfile: str,
    cafile: Optional[str] = None,
    yeu_cau_mtls: bool = False,
) -> ssl.SSLContext:
    """Khởi tạo `ssl.SSLContext` cho máy chủ TCP Swarm LAN với TLS 1.2+ (tùy chọn mTLS xác thực client)."""
    if yeu_cau_mtls and not cafile:
        raise ValueError("Chế độ mTLS yêu cầu cung cấp chứng chỉ CA (`cafile`) để xác thực client.")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if cafile:
        ctx.load_verify_locations(cafile=cafile)
    if yeu_cau_mtls:
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def tao_ssl_context_client(
    cafile: Optional[str] = None,
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
    kiem_tra_hostname: bool = True,
) -> ssl.SSLContext:
    """Khởi tạo `ssl.SSLContext` cho client TCP Swarm LAN kết nối TLS/mTLS tới nút vật lý."""
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = kiem_tra_hostname
    if certfile and keyfile:
        ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return ctx


class NutSwarmCPU:
    """
    Đại diện cho một nút tính toán CPU văn phòng trong cụm Swarm (Swarm CPU Worker Node):
    - Đảm nhiệm một phân đoạn layer (ví dụ: Layer 0..4 hoặc 4..8).
    - Tính toán forward pass trên CPU với ma trận trọng số cục bộ.
    - Tự động tạo chữ ký và cam kết mật mã học Verifiable AI (Proof of Inference).
    """

    def __init__(
        self,
        node_id: str,
        dai_layer: Tuple[int, int] = (0, 4),
        so_core_cpu: int = 4,
        ram_mb: float = 4096.0,
        private_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        che_do: str = "gia_lap_cuc_bo",
        do_tre_mang_co_dinh_ms: float = 0.0,
        ma_pin: Optional[str] = None,
        yeu_cau_phe_duyet: bool = False,
        nen_fp16: bool = False,
        d_model: int = 128,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self.node_id = node_id
        self.dai_layer = dai_layer  # (start_layer, end_layer)
        self.so_core_cpu = so_core_cpu
        self.ram_mb = ram_mb
        self.so_layer = max(1, dai_layer[1] - dai_layer[0])
        self.verifier = XacThucSuyLuanMaoHiem(private_key=private_key)
        self.endpoint_url = endpoint_url
        self.che_do = che_do
        self.do_tre_mang_co_dinh_ms = do_tre_mang_co_dinh_ms
        self.thoi_gian_tinh_toan_gan_nhat_ms: float = 0.0
        self.thoi_gian_mang_gan_nhat_ms: float = 0.0
        self.duong_dan_gguf_shard: Optional[str] = None
        self.checksum_trong_so_sha256: str = ""
        self.ma_pin = ma_pin
        self.yeu_cau_phe_duyet = yeu_cau_phe_duyet
        self.nen_fp16 = nen_fp16
        self.ssl_context = ssl_context
        self.xac_thuc: Optional[XacThucCumSwarm] = (
            XacThucCumSwarm(ma_pin=ma_pin, yeu_cau_phe_duyet=yeu_cau_phe_duyet)
            if ma_pin is not None
            else None
        )

        # Quản lý Socket Server LAN cục bộ nếu nút này là máy chủ vật lý
        self._server_lan: Optional[asyncio.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pool_lock = threading.Lock()
        self._pool_sock: Optional[Union[socket.socket, ssl.SSLSocket]] = None
        self._pool_token: Optional[str] = None
        self._pool_endpoint: Optional[str] = None
        self._so_lan_tai_su_dung_ket_noi: int = 0
        self._transformer_shard: Optional[Any] = None
        self._kv_phien: Dict[str, Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]] = {}
        self._kv_lock = threading.Lock()

        # Khởi tạo trọng số giả lập xác định (Deterministic Weights) cho các layer phụ trách
        seed = int(hashlib.sha256(f"{node_id}:{dai_layer}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        self.d_model = max(1, int(d_model))
        # Trọng số tuyến tính mô phỏng MLP / Self-Attention block
        self.w_proj = rng.randn(self.so_layer, self.d_model, self.d_model).astype(np.float32) * 0.05
        self.bias = np.zeros((self.so_layer, self.d_model), dtype=np.float32)

    def nap_trong_so_gguf(self, duong_dan_gguf: str) -> Dict[str, Any]:
        """
        Nạp ma trận trọng số thực tế từ tệp GGUF vào bộ nhớ RAM của nút CPU:
        - Tự động trích xuất các layer tương ứng với dải `self.dai_layer`.
        - Giải lượng tử hóa (dequantize) Q8_0, Q4_0 hoặc FP16 thành Float32.
        - Khởi tạo ma trận `w_proj` và `bias` cho forward pass.
        - Tính mã checksum SHA-256 của trọng số phục vụ kiểm toán Proof of Inference.
        """
        from vietnamese_ai.export.gguf import XuatGGUF

        tien_to = []
        for l_abs in range(self.dai_layer[0], self.dai_layer[1]):
            tien_to.extend([f"blk.{l_abs}.", f"layers.{l_abs}.", f"transformer.h.{l_abs}."])

        loader = XuatGGUF()
        tat_ca_tensors = loader.tai_tu_gguf(duong_dan_gguf, loc_tien_to=tien_to)
        self.duong_dan_gguf_shard = str(duong_dan_gguf)

        tensors_thuoc_node: Dict[str, np.ndarray] = {}
        for ten_t, t_arr in tat_ca_tensors.items():
            if any(ten_t.startswith(prefix) for prefix in tien_to):
                tensors_thuoc_node[ten_t] = t_arr

        # Tính checksum SHA256 cho phân mảnh trọng số
        h = hashlib.sha256()
        for ten_t in sorted(tensors_thuoc_node.keys()):
            h.update(ten_t.encode("utf-8"))
            h.update(tensors_thuoc_node[ten_t].tobytes())
        self.checksum_trong_so_sha256 = h.hexdigest()

        # Ánh xạ vào w_proj nếu có tensor phù hợp
        if tensors_thuoc_node:
            for l_rel, l_abs in enumerate(range(self.dai_layer[0], self.dai_layer[1])):
                if l_rel >= self.so_layer:
                    break
                candidates = [v for k, v in tensors_thuoc_node.items() if f".{l_abs}." in k and v.ndim == 2]
                if candidates:
                    cand = candidates[0]
                    h_in, h_out = cand.shape
                    if h_in >= self.d_model and h_out >= self.d_model:
                        w_sub = cand[: self.d_model, : self.d_model].astype(np.float32)
                    elif h_in == self.d_model:
                        w_sub = cand[:, : min(self.d_model, h_out)].astype(np.float32)
                    else:
                        w_sub = cand.astype(np.float32)

                    w_std = float(np.std(w_sub))
                    if w_std > 1e-6:
                        w_norm = (w_sub - float(np.mean(w_sub))) / w_std * (1.0 / np.sqrt(self.d_model))
                    else:
                        w_norm = w_sub * 0.01

                    r_in, r_out = min(self.d_model, w_norm.shape[0]), min(self.d_model, w_norm.shape[1])
                    self.w_proj[l_rel, :r_in, :r_out] = w_norm[:r_in, :r_out]

        return {
            "node_id": self.node_id,
            "dai_layer": self.dai_layer,
            "so_tensors_nap": len(tensors_thuoc_node),
            "checksum_sha256": self.checksum_trong_so_sha256,
            "duong_dan_gguf": self.duong_dan_gguf_shard,
        }

    def nap_phan_manh_transformer_gguf(
        self,
        streamer: Any,
        so_dau: int = 4,
        so_dau_kv: Optional[int] = None,
        d_ff: Optional[int] = None,
        giu_luong_tu: bool = True,
    ) -> Dict[str, Any]:
        """Nạp phân mảnh khối Transformer thực tế (`BoGiaiMaTransformerEdge` với RMSNorm + RoPE + GQA + SwiGLU) cho dải layer của nút."""
        from vietnamese_ai.edge.native_inference import BoGiaiMaTransformerEdge

        chieu_ff = int(d_ff) if d_ff is not None else self.d_model * 2
        bo_giai_ma = BoGiaiMaTransformerEdge(
            d_model=self.d_model,
            so_dau=so_dau,
            d_ff=chieu_ff,
            so_dau_kv=so_dau_kv,
            so_lop=self.so_layer,
        )
        anh_xa = (
            ("attn_q.weight", "w_q"),
            ("attn_k.weight", "w_k"),
            ("attn_v.weight", "w_v"),
            ("attn_output.weight", "w_o"),
            ("ffn_gate.weight", "w_gate"),
            ("ffn_up.weight", "w_up"),
            ("ffn_down.weight", "w_down"),
        )
        da_nap = 0
        for rel_idx, abs_idx in enumerate(range(self.dai_layer[0], self.dai_layer[1])):
            if rel_idx >= bo_giai_ma.so_lop:
                break
            lop = bo_giai_ma._cac_lop[rel_idx]
            for ten_hau_to, ten_attr in anh_xa:
                ten_gguf = f"blk.{abs_idx}.{ten_hau_to}"
                target = lop[ten_attr]
                in_dim, out_dim = int(target.shape[0]), int(target.shape[1])
                if giu_luong_tu and hasattr(streamer, "doc_tensor_tho"):
                    tho = streamer.doc_tensor_tho(ten_gguf)
                    if tho is not None:
                        from vietnamese_ai.edge.native_inference import THONG_SO_LUONG_TU

                        dims = tho["dims"]
                        loai_q = str(tho["loai_quant"])
                        block_size, _ = THONG_SO_LUONG_TU.get(loai_q, (1, 4))
                        khop_ggml = len(dims) == 2 and int(dims[0]) == in_dim and int(dims[1]) == out_dim
                        khop_c = len(dims) == 2 and int(dims[0]) == out_dim and int(dims[1]) == in_dim
                        if (khop_ggml or khop_c) and loai_q not in ("f16", "f32", "bf16") and in_dim % block_size == 0:
                            bo_giai_ma._trong_so_luong_tu[f"{rel_idx}:{ten_attr}"] = {
                                "raw_bytes": tho["raw_bytes"],
                                "shape": (out_dim, in_dim),
                                "loai_quant": loai_q,
                            }
                            da_nap += 1
                            continue
                tensor = streamer.doc_tensor(ten_gguf) if hasattr(streamer, "doc_tensor") else None
                if tensor is not None and tensor.ndim == 2:
                    if tensor.shape == (out_dim, in_dim):
                        tensor = tensor.T
                    r = min(target.shape[0], tensor.shape[0])
                    c = min(target.shape[1], tensor.shape[1])
                    target[:r, :c] = tensor[:r, :c].astype(np.float32)
                    da_nap += 1
            for ten_norm, attr_norm in (("attn_norm.weight", "norm_attn"), ("ffn_norm.weight", "norm_ffn")):
                t_norm = streamer.doc_tensor(f"blk.{abs_idx}.{ten_norm}") if hasattr(streamer, "doc_tensor") else None
                if t_norm is not None:
                    flat_n = t_norm.ravel().astype(np.float32)
                    n_copy = min(lop[attr_norm].shape[0], flat_n.shape[0])
                    lop[attr_norm][:n_copy] = flat_n[:n_copy]
                    da_nap += 1

        self._transformer_shard = bo_giai_ma
        self.xoa_kv_phien()
        return {
            "node_id": self.node_id,
            "dai_layer": self.dai_layer,
            "so_lop": self.so_layer,
            "so_tensor_nap": da_nap,
            "so_tensor_luong_tu": len(bo_giai_ma._trong_so_luong_tu),
            "che_do_transformer_thuc": True,
        }

    def xoa_kv_phien(self, session_id: Optional[str] = None) -> None:
        """Xóa bộ đệm KV-Cache của một phiên hoặc toàn bộ phiên suy luận trên nút."""
        with self._kv_lock:
            if session_id is None:
                self._kv_phien.clear()
            else:
                self._kv_phien.pop(session_id, None)
        if self._transformer_shard is not None:
            self._transformer_shard.xoa_kv_cache()

    def dong_ket_noi_lan(self) -> None:
        """Đóng kết nối TCP/TLS keep-alive đang lưu trong pool của nút."""
        with self._pool_lock:
            if self._pool_sock is not None:
                try:
                    self._pool_sock.close()
                except Exception:
                    pass
                self._pool_sock = None
                self._pool_token = None
                self._pool_endpoint = None

    def xuat_phan_manh_gguf(self, duong_dan_xuat: str, quantization: str = "q8_0") -> str:
        """Xuất riêng phân mảnh trọng số của nút này ra một tệp GGUF độc lập."""
        from vietnamese_ai.export.gguf import XuatGGUF

        dict_tensors: Dict[str, np.ndarray] = {}
        for l_rel, l_abs in enumerate(range(self.dai_layer[0], self.dai_layer[1])):
            if l_rel < self.so_layer:
                dict_tensors[f"blk.{l_abs}.weight"] = self.w_proj[l_rel]

        xuat = XuatGGUF()
        return xuat.xuat_tu_numpy(
            dict_tensors,
            duong_dan_xuat,
            quantization=quantization,
            thong_tin={"name": f"SwarmShard_{self.node_id}", "layers": f"{self.dai_layer}"},
        )

    def khoi_dong_server_lan(
        self,
        host: str = "127.0.0.1",
        port: int = 8890,
        ma_pin: Optional[str] = None,
        yeu_cau_phe_duyet: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> None:
        """
        Khởi động máy chủ TCP Socket (tùy chọn mã hóa TLS/mTLS) lắng nghe suy luận phân tán trên LAN/Tailscale.
        Hoạt động phi đồng bộ trong luồng ngầm (background thread) kèm xác thực Zero-Trust.
        """
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        if ssl_context is not None:
            self.ssl_context = ssl_context
        server_ssl = self.ssl_context

        if ma_pin is not None:
            self.ma_pin = ma_pin
            self.xac_thuc = XacThucCumSwarm(
                ma_pin=ma_pin,
                yeu_cau_phe_duyet=self.yeu_cau_phe_duyet if yeu_cau_phe_duyet is None else yeu_cau_phe_duyet,
            )
        elif yeu_cau_phe_duyet is not None and self.xac_thuc is not None:
            self.xac_thuc.yeu_cau_phe_duyet = yeu_cau_phe_duyet

        def _run_server():
            self._server_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._server_loop)

            async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
                client_peer = writer.get_extra_info("peername")
                client_ip = client_peer[0] if client_peer else "127.0.0.1"

                # Chống tấn công vét cạn brute-force: kiểm tra nếu IP bị tạm khóa
                if self.xac_thuc and self.xac_thuc.kiem_tra_ip_bi_khoa(client_ip):
                    err_res = {"trang_thai": "bi_khoa", "ly_do": "IP bị tạm khóa do thử sai mã PIN quá nhiều lần"}
                    err_bytes = json.dumps(err_res).encode("utf-8")
                    writer.write(struct.pack("!I", len(err_bytes)) + err_bytes)
                    await writer.drain()
                    writer.close()
                    return

                client_authenticated = (self.xac_thuc is None)

                try:
                    max_header_bytes = 2 * 1024 * 1024
                    max_tensor_bytes = 64 * 1024 * 1024
                    while True:
                        raw_header_len = await asyncio.wait_for(reader.readexactly(4), timeout=30.0)
                        header_len = struct.unpack("!I", raw_header_len)[0]
                        if header_len == 0 or header_len > max_header_bytes:
                            err_meta = {"trang_thai": "that_bai", "ly_do": "Kích thước tiêu đề vượt giới hạn an toàn"}
                            err_b = json.dumps(err_meta, ensure_ascii=False).encode("utf-8")
                            writer.write(struct.pack("!I", len(err_b)) + err_b)
                            await writer.drain()
                            break
                        header_bytes = await asyncio.wait_for(reader.readexactly(header_len), timeout=30.0)
                        req_meta = json.loads(header_bytes.decode("utf-8"))

                        loai_req = req_meta.get("loai")
                        res_meta: Dict[str, Any] = {}

                        if loai_req == "auth_challenge_req":
                            if self.xac_thuc is None:
                                res_meta = {"trang_thai": "thanh_cong", "token": "no_auth_required"}
                            else:
                                nonce = self.xac_thuc.tao_thach_thuc()
                                res_meta = {"trang_thai": "challenge", "nonce": nonce}
                            res_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                            writer.write(struct.pack("!I", len(res_bytes)) + res_bytes)
                            await writer.drain()
                            continue

                        elif loai_req == "auth_response":
                            if self.xac_thuc is None:
                                client_authenticated = True
                                res_meta = {"trang_thai": "thanh_cong", "token": "no_auth_required"}
                            else:
                                nonce = req_meta.get("nonce", "")
                                chu_ky = req_meta.get("chu_ky", "")
                                node_id_client = req_meta.get("node_id", "khach")

                                hop_le = self.xac_thuc.xac_minh_thach_thuc(nonce, chu_ky)
                                if not hop_le:
                                    da_khoa = self.xac_thuc.ghi_nhan_that_bai(client_ip)
                                    res_meta = {
                                        "trang_thai": "that_bai",
                                        "ly_do": "Mã PIN xác thực không chính xác" + (" (IP đã bị tạm khóa)" if da_khoa else ""),
                                    }
                                    res_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                                    writer.write(struct.pack("!I", len(res_bytes)) + res_bytes)
                                    await writer.drain()
                                    writer.close()
                                    return

                                self.xac_thuc.xoa_lich_su_that_bai(client_ip)

                                if self.xac_thuc.yeu_cau_phe_duyet:
                                    thong_tin_nut = req_meta.get("thong_tin", {})
                                    fut = self.xac_thuc.dang_ky_cho_duyet(node_id_client, client_ip, thong_tin_nut, loop=self._server_loop)
                                    try:
                                        duoc_duyet = await asyncio.wait_for(fut, timeout=10.0)
                                    except asyncio.TimeoutError:
                                        duoc_duyet = False

                                    if not duoc_duyet:
                                        res_meta = {"trang_thai": "that_bai", "ly_do": "Chủ máy chủ đã từ chối yêu cầu gia nhập cụm"}
                                        res_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                                        writer.write(struct.pack("!I", len(res_bytes)) + res_bytes)
                                        await writer.drain()
                                        writer.close()
                                        return

                                token = self.xac_thuc.tao_session_token(node_id_client, client_ip)
                                client_authenticated = True
                                res_meta = {"trang_thai": "thanh_cong", "token": token}

                            res_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                            writer.write(struct.pack("!I", len(res_bytes)) + res_bytes)
                            await writer.drain()
                            continue

                        # Đối với yêu cầu tính toán tensor ("forward")
                        if self.xac_thuc is not None:
                            token = req_meta.get("token")
                            if not client_authenticated and not (token and self.xac_thuc.xac_minh_session_token(token)):
                                res_meta = {"trang_thai": "that_bai", "ly_do": "Chưa được xác thực (Unauthorized)"}
                                res_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                                writer.write(struct.pack("!I", len(res_bytes)) + res_bytes)
                                await writer.drain()
                                writer.close()
                                return

                        tensor_len = int(req_meta.get("tensor_bytes", 0))
                        if tensor_len < 0 or tensor_len > max_tensor_bytes:
                            err_meta = {"trang_thai": "that_bai", "ly_do": "Kích thước tensor vượt giới hạn an toàn"}
                            err_b = json.dumps(err_meta, ensure_ascii=False).encode("utf-8")
                            writer.write(struct.pack("!I", len(err_b)) + err_b)
                            await writer.drain()
                            break
                        shape = tuple(req_meta.get("shape", [1, self.d_model]))
                        dtype_str = req_meta.get("dtype", "float32")
                        raw_tensor = await asyncio.wait_for(reader.readexactly(tensor_len), timeout=30.0)

                        if dtype_str == "float16":
                            input_tensor = np.frombuffer(raw_tensor, dtype=np.float16).astype(np.float32).reshape(shape)
                        else:
                            input_tensor = np.frombuffer(raw_tensor, dtype=np.float32).reshape(shape)

                        t_comp_start = time.perf_counter()
                        tao_proof = req_meta.get("tao_proof", True)
                        tra_ve_fp16 = req_meta.get("nen_fp16", False) or (dtype_str == "float16") or self.nen_fp16
                        req_session = req_meta.get("session_id")
                        req_vi_tri = int(req_meta.get("vi_tri", 0))
                        if req_session is not None or req_vi_tri != 0:
                            try:
                                output_tensor, proof = self._tinh_toan_cuc_bo(
                                    input_tensor,
                                    tao_proof=tao_proof,
                                    nen_fp16=tra_ve_fp16,
                                    session_id=str(req_session) if req_session is not None else None,
                                    vi_tri=req_vi_tri,
                                )
                            except TypeError:
                                output_tensor, proof = self._tinh_toan_cuc_bo(
                                    input_tensor,
                                    tao_proof=tao_proof,
                                    nen_fp16=tra_ve_fp16,
                                )
                        else:
                            output_tensor, proof = self._tinh_toan_cuc_bo(
                                input_tensor,
                                tao_proof=tao_proof,
                                nen_fp16=tra_ve_fp16,
                            )
                        t_comp_ms = (time.perf_counter() - t_comp_start) * 1000.0

                        if tra_ve_fp16:
                            out_bytes = output_tensor.astype(np.float16).tobytes()
                            out_dtype = "float16"
                        else:
                            out_bytes = output_tensor.astype(np.float32).tobytes()
                            out_dtype = "float32"

                        res_meta = {
                            "trang_thai": "thanh_cong",
                            "node_id": self.node_id,
                            "thoi_gian_cpu_ms": round(t_comp_ms, 3),
                            "shape": list(output_tensor.shape),
                            "tensor_bytes": len(out_bytes),
                            "dtype": out_dtype,
                            "proof": proof,
                        }
                        res_meta_bytes = json.dumps(res_meta, ensure_ascii=False).encode("utf-8")
                        writer.write(struct.pack("!I", len(res_meta_bytes)) + res_meta_bytes + out_bytes)
                        await writer.drain()
                except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError, asyncio.CancelledError, GeneratorExit):
                    pass
                except Exception as e:
                    logger.debug("Lỗi kết nối socket client: %s", e)
                finally:
                    try:
                        if not writer.is_closing():
                            writer.close()
                    except Exception:
                        pass

            async def _start():
                self._server_lan = await asyncio.start_server(_handle_client, host, port, ssl=server_ssl)

            try:
                self._server_loop.run_until_complete(_start())
                self._server_loop.run_forever()
            finally:
                tasks = [t for t in asyncio.all_tasks(self._server_loop) if not t.done()]
                if tasks:
                    self._server_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                self._server_loop.close()

        self._server_thread = threading.Thread(target=_run_server, daemon=True)
        self._server_thread.start()
        time.sleep(0.05)

    def dung_server_lan(self) -> None:
        """Dừng máy chủ TCP Socket LAN."""
        loop = self._server_loop
        if loop is not None and loop.is_running():
            def _stop() -> None:
                if self._server_lan is not None:
                    self._server_lan.close()
                for task in asyncio.all_tasks(loop):
                    if task is not asyncio.current_task(loop):
                        task.cancel()
                loop.stop()

            loop.call_soon_threadsafe(_stop)
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None
        self.dong_ket_noi_lan()

    def phe_duyet_nut(self, nut_id: str, cho_phep: bool = True) -> bool:
        """Phê duyệt hoặc từ chối thiết bị đang chờ gia nhập cụm (One-Click Approval)."""
        if self.xac_thuc is not None:
            return self.xac_thuc.phe_duyet_nut(nut_id, cho_phep)
        return False

    @staticmethod
    def _recv_exact(sock: Union[socket.socket, ssl.SSLSocket], n_bytes: int) -> bytes:
        """Đọc chính xác `n_bytes` từ TCP/TLS socket, xử lý các gói tin bị phân mảnh."""
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = sock.recv(min(65536, n_bytes - len(buf)))
            if not chunk:
                raise ConnectionResetError(f"Socket ngắt kết nối sớm (nhận {len(buf)}/{n_bytes} bytes)")
            buf.extend(chunk)
        return bytes(buf)

    def _mo_va_xac_thuc_socket_lan(
        self,
        host: str,
        port: int,
        timeout: float,
        max_header_bytes: int,
    ) -> Tuple[Union[socket.socket, ssl.SSLSocket], Optional[str]]:
        """Mở kết nối TCP/TLS mới tới nút LAN và hoàn tất bắt tay HMAC PIN nếu được cấu hình."""
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(timeout)
        sock_conn: Union[socket.socket, ssl.SSLSocket]
        if self.ssl_context is not None:
            sock_conn = self.ssl_context.wrap_socket(raw_sock, server_hostname=host)
        else:
            sock_conn = raw_sock

        try:
            sock_conn.connect((host, port))
            session_token: Optional[str] = None
            if self.ma_pin is not None:
                auth_req = {"loai": "auth_challenge_req"}
                auth_req_b = json.dumps(auth_req).encode("utf-8")
                sock_conn.sendall(struct.pack("!I", len(auth_req_b)) + auth_req_b)

                h_len = struct.unpack("!I", self._recv_exact(sock_conn, 4))[0]
                if h_len == 0 or h_len > max_header_bytes:
                    raise ValueError("Tiêu đề challenge từ máy chủ vượt giới hạn an toàn")
                ch_meta = json.loads(self._recv_exact(sock_conn, h_len).decode("utf-8"))

                if ch_meta.get("trang_thai") == "challenge":
                    nonce = ch_meta["nonce"]
                    key = str(self.ma_pin).encode("utf-8")
                    msg = nonce.encode("utf-8")
                    chu_ky = hmac.new(key, msg, hashlib.sha256).hexdigest()

                    resp_req = {
                        "loai": "auth_response",
                        "node_id": self.node_id,
                        "nonce": nonce,
                        "chu_ky": chu_ky,
                        "thong_tin": {"so_core_cpu": self.so_core_cpu, "ram_mb": self.ram_mb},
                    }
                    resp_req_b = json.dumps(resp_req).encode("utf-8")
                    sock_conn.sendall(struct.pack("!I", len(resp_req_b)) + resp_req_b)

                    h_len = struct.unpack("!I", self._recv_exact(sock_conn, 4))[0]
                    if h_len == 0 or h_len > max_header_bytes:
                        raise ValueError("Tiêu đề phản hồi xác thực vượt giới hạn an toàn")
                    res_auth = json.loads(self._recv_exact(sock_conn, h_len).decode("utf-8"))

                    if res_auth.get("trang_thai") != "thanh_cong":
                        raise PermissionError(f"Xác thực cụm Swarm thất bại: {res_auth.get('ly_do')}")
                    session_token = res_auth.get("token")
            return sock_conn, session_token
        except Exception:
            try:
                sock_conn.close()
            except Exception:
                pass
            raise

    def _truyen_tai_qua_socket_lan(
        self,
        hidden_states: np.ndarray,
        tao_proof: bool = True,
        timeout: float = 5.0,
        session_id: Optional[str] = None,
        vi_tri: int = 0,
        giu_ket_noi: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Truyền tensor qua socket TCP/TLS tới máy tính vật lý LAN/Tailscale (hỗ trợ keep-alive connection pool)."""
        t_net_start = time.perf_counter()
        max_header_bytes = 2 * 1024 * 1024
        max_tensor_bytes = 64 * 1024 * 1024
        endpoint = str(self.endpoint_url or "")
        url = endpoint.replace("tls://", "").replace("tcp://", "").replace("http://", "")
        if ":" in url:
            host, port_str = url.split(":", 1)
            port = int(port_str)
        else:
            host, port = url, 8890

        if self.nen_fp16:
            x_arr = np.asarray(hidden_states, dtype=np.float16)
            raw_x = x_arr.tobytes()
            dtype_str = "float16"
        else:
            x_arr = np.asarray(hidden_states, dtype=np.float32)
            raw_x = x_arr.tobytes()
            dtype_str = "float32"

        with self._pool_lock:
            for lan_thu in range(2):
                reused = False
                if giu_ket_noi and self._pool_sock is not None and self._pool_endpoint == endpoint:
                    sock = self._pool_sock
                    session_token = self._pool_token
                    sock.settimeout(timeout)
                    reused = True
                else:
                    if self._pool_sock is not None:
                        try:
                            self._pool_sock.close()
                        except Exception:
                            pass
                        self._pool_sock = None
                        self._pool_token = None
                        self._pool_endpoint = None
                    sock, session_token = self._mo_va_xac_thuc_socket_lan(
                        host=host,
                        port=port,
                        timeout=timeout,
                        max_header_bytes=max_header_bytes,
                    )

                try:
                    req_meta: Dict[str, Any] = {
                        "loai": "forward",
                        "node_id": self.node_id,
                        "shape": list(x_arr.shape),
                        "tensor_bytes": len(raw_x),
                        "dtype": dtype_str,
                        "nen_fp16": self.nen_fp16,
                        "tao_proof": tao_proof,
                        "token": session_token,
                        "vi_tri": int(vi_tri),
                    }
                    if session_id is not None:
                        req_meta["session_id"] = session_id
                    meta_bytes = json.dumps(req_meta).encode("utf-8")
                    sock.sendall(struct.pack("!I", len(meta_bytes)) + meta_bytes + raw_x)

                    resp_header_len = struct.unpack("!I", self._recv_exact(sock, 4))[0]
                    if resp_header_len == 0 or resp_header_len > max_header_bytes:
                        raise ValueError("Tiêu đề phản hồi từ máy chủ vượt giới hạn an toàn")
                    resp_meta_bytes = self._recv_exact(sock, resp_header_len)
                    resp_meta = json.loads(resp_meta_bytes.decode("utf-8"))

                    if resp_meta.get("trang_thai") != "thanh_cong":
                        raise RuntimeError(
                            f"Lỗi suy luận từ node vật lý {self.node_id}: {resp_meta.get('ly_do', 'Không xác định')}"
                        )

                    tensor_len = int(resp_meta.get("tensor_bytes", 0))
                    if tensor_len < 0 or tensor_len > max_tensor_bytes:
                        raise ValueError("Kích thước tensor phản hồi vượt giới hạn an toàn")
                    tensor_shape = tuple(resp_meta.get("shape", list(x_arr.shape)))
                    raw_tensor = self._recv_exact(sock, tensor_len)

                    resp_dtype = resp_meta.get("dtype", "float32")
                    if resp_dtype == "float16":
                        out_tensor = np.frombuffer(raw_tensor, dtype=np.float16).astype(np.float32).reshape(tensor_shape)
                    else:
                        out_tensor = np.frombuffer(raw_tensor, dtype=np.float32).reshape(tensor_shape)

                    if reused:
                        self._so_lan_tai_su_dung_ket_noi += 1
                    if giu_ket_noi:
                        self._pool_sock = sock
                        self._pool_token = session_token
                        self._pool_endpoint = endpoint
                    else:
                        try:
                            sock.close()
                        except Exception:
                            pass

                    self.thoi_gian_tinh_toan_gan_nhat_ms = float(resp_meta.get("thoi_gian_cpu_ms", 0.0))
                    self.thoi_gian_mang_gan_nhat_ms = (
                        (time.perf_counter() - t_net_start) * 1000.0 - self.thoi_gian_tinh_toan_gan_nhat_ms
                    )
                    return out_tensor, resp_meta.get("proof")
                except (OSError, ConnectionResetError, BrokenPipeError, EOFError):
                    try:
                        sock.close()
                    except Exception:
                        pass
                    if self._pool_sock is sock:
                        self._pool_sock = None
                        self._pool_token = None
                        self._pool_endpoint = None
                    if reused and lan_thu == 0:
                        continue
                    raise
                except Exception:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    if self._pool_sock is sock:
                        self._pool_sock = None
                        self._pool_token = None
                        self._pool_endpoint = None
                    raise
            raise RuntimeError(f"Không thể gửi tensor qua kết nối LAN tới nút {self.node_id}")

    def _tinh_toan_cuc_bo(
        self,
        hidden_states: np.ndarray,
        tao_proof: bool = True,
        nen_fp16: Optional[bool] = None,
        session_id: Optional[str] = None,
        vi_tri: int = 0,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Thực thi forward pass cục bộ trên CPU (ưu tiên phân mảnh Transformer GGUF + KV-Cache theo phiên nếu đã nạp)."""
        t_start = time.perf_counter()
        ap_dung_fp16 = self.nen_fp16 if nen_fp16 is None else nen_fp16
        x = hidden_states.copy()
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self._transformer_shard is not None:
            with self._kv_lock:
                shard = self._transformer_shard
                if session_id is not None:
                    saved = self._kv_phien.get(session_id)
                    if saved is None:
                        shard.xoa_kv_cache()
                    else:
                        shard._kv_local.k_caches = [list(c) for c in saved[0]]
                        shard._kv_local.v_caches = [list(c) for c in saved[1]]
                elif vi_tri == 0:
                    shard.xoa_kv_cache()

                ket_qua_rows: List[np.ndarray] = []
                for row_idx in range(x.shape[0]):
                    ket_qua_rows.append(shard.buoc_suy_luan(x[row_idx], vi_tri=int(vi_tri) + row_idx))
                x = np.vstack(ket_qua_rows).astype(np.float32)

                if session_id is not None:
                    k_cur, v_cur = shard._lay_kv_caches_luong()
                    self._kv_phien[session_id] = ([list(c) for c in k_cur], [list(c) for c in v_cur])
        else:
            for l_idx in range(self.so_layer):
                w = self.w_proj[l_idx]
                b = self.bias[l_idx]
                x_next = np.dot(x, w) + b
                x_next = np.clip(x_next, -50.0, 50.0)
                x = np.clip(x + 0.1 * x_next, -100.0, 100.0)

        if ap_dung_fp16:
            x = x.astype(np.float16).astype(np.float32)

        self.thoi_gian_tinh_toan_gan_nhat_ms = (time.perf_counter() - t_start) * 1000.0

        if self.do_tre_mang_co_dinh_ms > 0:
            time.sleep(self.do_tre_mang_co_dinh_ms / 1000.0)
            self.thoi_gian_mang_gan_nhat_ms = self.do_tre_mang_co_dinh_ms
        else:
            self.thoi_gian_mang_gan_nhat_ms = 0.0

        proof = None
        if tao_proof:
            proof = self.verifier.sinh_cam_ket_layer(
                node_id=self.node_id,
                layer_id=self.dai_layer[0],
                tensor_dau_vao=hidden_states,
                tensor_dau_ra=x,
            )

        return x, proof

    def thuc_thi_layer(
        self,
        hidden_states: np.ndarray,
        tao_proof: bool = True,
        session_id: Optional[str] = None,
        vi_tri: int = 0,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Thực hiện tính toán chuyển tiếp (Forward Pass) qua các layer phụ trách trên CPU:
        - Nếu cấu hình là nút mạng LAN vật lý (`endpoint_url`), gửi tensor qua TCP socket thực tế (keep-alive pool).
        - Nếu chạy cục bộ, tính toán trực tiếp trên CPU.
        """
        if self.endpoint_url is not None and self.che_do == "mang_vat_ly_lan":
            try:
                return self._truyen_tai_qua_socket_lan(
                    hidden_states,
                    tao_proof=tao_proof,
                    session_id=session_id,
                    vi_tri=vi_tri,
                )
            except PermissionError:
                raise
            except Exception as e:
                logger.warning(
                    "Không thể kết nối nút LAN %s: %s. Chuyển sang tính toán dự phòng cục bộ.",
                    self.endpoint_url,
                    e,
                )
                return self._tinh_toan_cuc_bo(
                    hidden_states,
                    tao_proof=tao_proof,
                    session_id=session_id,
                    vi_tri=vi_tri,
                )

        return self._tinh_toan_cuc_bo(
            hidden_states,
            tao_proof=tao_proof,
            session_id=session_id,
            vi_tri=vi_tri,
        )

    def lay_thong_tin_cau_hinh(self) -> Dict[str, Any]:
        """Trả về thông tin cấu hình phần cứng và dải layer của Node."""
        return {
            "node_id": self.node_id,
            "dai_layer": self.dai_layer,
            "so_core_cpu": self.so_core_cpu,
            "ram_mb": self.ram_mb,
            "so_layer": self.so_layer,
            "nen_fp16": self.nen_fp16,
            "public_key": self.verifier.public_key,
            "endpoint_url": self.endpoint_url,
            "che_do": self.che_do,
            "la_node_mang_vat_ly": self.endpoint_url is not None or self.che_do == "mang_vat_ly_lan",
            "ma_hoa_tls": self.ssl_context is not None,
            "che_do_transformer_thuc": self._transformer_shard is not None,
            "so_phien_kv_cache": len(self._kv_phien),
            "so_lan_tai_su_dung_ket_noi": self._so_lan_tai_su_dung_ket_noi,
        }

    def lan_truyen_tien(
        self,
        hidden_states: np.ndarray,
        buoc: int = 0,
        tao_proof: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Alias phương thức lan truyền tiến (Forward Pass) cho Node."""
        x, proof = self.thuc_thi_layer(
            hidden_states,
            tao_proof=tao_proof,
            session_id=session_id,
            vi_tri=buoc,
        )
        p_dict = proof.copy() if proof else {}
        if "chuoi_cam_ket" not in p_dict and "proof_zk" in p_dict:
            p_dict["chuoi_cam_ket"] = p_dict["proof_zk"]
        return x, p_dict


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
    mo_ta=(
        "Điều phối Pipeline Parallelism trên CPU (hỗ trợ cả chế độ giả lập nội bộ và TCP LAN/Tailscale). "
        "Tầng giải mã token mặc định hiện ở mức thử nghiệm dựa trên từ điển ghép."
    ),
    phu_thuoc_ngoai=["gguf (tùy chọn khi phân mảnh tệp GGUF lớn)", "tailscale CLI (tùy chọn cho mesh WAN)"],
    nhom_extra="export",
)
class DieuPhoiSwarmCPU:
    """
    Bộ Điều Phối Suy Luận Đàn CPU (Distributed Swarm Inference Orchestrator):
    - Quản lý danh sách các node CPU văn phòng trong mạng LAN.
    - Tự động phân chia mô hình thành các block layer (Pipeline Parallelism).
    - Vận hành quy trình suy luận phân tán, truyền activation và kiểm tra Byzantine proof.
    """

    def __init__(
        self,
        d_model: int = 128,
        tong_so_layer: int = 12,
        bat_xac_thuc_mat_ma: bool = True,
        bat_tuong_lua: bool = True,
        ma_pin: Optional[str] = None,
        yeu_cau_phe_duyet: bool = False,
        nen_fp16: bool = False,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self.d_model = d_model
        self.tong_so_layer = tong_so_layer
        self.bat_xac_thuc_mat_ma = bat_xac_thuc_mat_ma
        self.bat_tuong_lua = bat_tuong_lua
        self.ma_pin = ma_pin
        self.yeu_cau_phe_duyet = yeu_cau_phe_duyet
        self.nen_fp16 = nen_fp16
        self.ssl_context = ssl_context
        self.xac_thuc: Optional[XacThucCumSwarm] = (
            XacThucCumSwarm(ma_pin=ma_pin, yeu_cau_phe_duyet=yeu_cau_phe_duyet)
            if ma_pin is not None
            else None
        )
        self.ip_tailscale: Optional[str] = None
        self.port_tailscale: int = 9999
        self.danh_sach_node: List[NutSwarmCPU] = []
        self.tokenizer = BoTachTuTiengVietSieuToc()
        self.embedder = BoNhungVanBanSieuToc(so_chieu=d_model, tokenizer=self.tokenizer)
        self._vocab_tokens: List[str] = sorted(self.tokenizer.TU_DIEN_TU_GHEP_MAC_DINH)
        self._lm_head_matrix: np.ndarray = np.vstack(
            [self.embedder.nhung(tu) for tu in self._vocab_tokens]
        ).astype(np.float32)
        self.nhat_ky_kiem_toan: List[Dict[str, Any]] = []

        if bat_tuong_lua:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            self.tuong_lua: Optional[TuongLuaBaoVeFramework] = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        else:
            self.tuong_lua = None

    @classmethod
    def khoi_tao_qua_tailscale(
        cls,
        port: int = 9999,
        ma_pin: Optional[str] = None,
        yeu_cau_phe_duyet: bool = False,
        d_model: int = 128,
        tong_so_layer: int = 12,
        bat_xac_thuc_mat_ma: bool = True,
        bat_tuong_lua: bool = True,
    ) -> DieuPhoiSwarmCPU:
        """
        Khởi tạo Bộ điều phối Swarm tự động liên kết mạng Tailscale:
        - Tự động phát hiện IP Tailscale của máy chủ (dải 100.64.0.0/10).
        - Tự động sinh mã PIN bảo mật 6 số nếu chưa được chỉ định.
        """
        ip_ts = phat_hien_ip_tailscale()
        pin = ma_pin or tao_ma_pin_ngau_nhien(6)
        instance = cls(
            d_model=d_model,
            tong_so_layer=tong_so_layer,
            bat_xac_thuc_mat_ma=bat_xac_thuc_mat_ma,
            bat_tuong_lua=bat_tuong_lua,
            ma_pin=pin,
            yeu_cau_phe_duyet=yeu_cau_phe_duyet,
        )
        instance.ip_tailscale = ip_ts
        instance.port_tailscale = port
        return instance

    def phe_duyet_nut(self, nut_id: str, cho_phep: bool = True) -> bool:
        """Phê duyệt hoặc từ chối thiết bị xin gia nhập cụm (One-Click Approval)."""
        if self.xac_thuc is not None:
            return self.xac_thuc.phe_duyet_nut(nut_id, cho_phep)
        return False

    def lay_danh_sach_cho_duyet(self) -> List[Dict[str, Any]]:
        """Lấy danh sách các thiết bị đang đợi phê duyệt tham gia cụm Swarm."""
        if self.xac_thuc is not None:
            return self.xac_thuc.lay_danh_sach_cho_duyet()
        return []

    @property
    def cac_node(self) -> List[NutSwarmCPU]:
        """Danh sách các node tham gia cụm Swarm."""
        return self.danh_sach_node

    def them_node(
        self,
        node_id: str,
        dai_layer: Optional[Tuple[int, int]] = None,
        so_core_cpu: int = 4,
        ram_mb: float = 4096.0,
        private_key: Optional[str] = None,
        nen_fp16: Optional[bool] = None,
    ) -> NutSwarmCPU:
        """Thêm một máy tính CPU văn phòng vào cụm Swarm (chế độ cục bộ / đa tiến trình)."""
        if dai_layer is None:
            # Tự động gán dải layer nếu chưa xác định
            hien_tai = sum(n.so_layer for n in self.danh_sach_node)
            phan_doan = max(1, (self.tong_so_layer - hien_tai))
            dai_layer = (hien_tai, min(self.tong_so_layer, hien_tai + phan_doan))

        node = NutSwarmCPU(
            node_id=node_id,
            dai_layer=dai_layer,
            so_core_cpu=so_core_cpu,
            ram_mb=ram_mb,
            private_key=private_key,
            che_do="gia_lap_cuc_bo",
            nen_fp16=self.nen_fp16 if nen_fp16 is None else nen_fp16,
            d_model=self.d_model,
        )
        self.danh_sach_node.append(node)
        return node

    def them_node_mang_vat_ly(
        self,
        node_id: str,
        endpoint_url: str,
        dai_layer: Optional[Tuple[int, int]] = None,
        so_core_cpu: int = 4,
        ram_mb: float = 4096.0,
        private_key: Optional[str] = None,
        do_tre_mang_co_dinh_ms: float = 1.5,
        ma_pin: Optional[str] = None,
        nen_fp16: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> NutSwarmCPU:
        """
        Đăng ký một nút mạng vật lý thực tế qua mạng LAN/Tailscale (Physical LAN/Tailscale Node):
        - Hỗ trợ endpoint TCP Socket nhị phân tốc độ cao kèm xác thực HMAC Zero-Trust.
        - Tách biệt rõ độ trễ truyền tải mạng vật lý (network transfer latency)
          với thời gian tính toán nội tại trên CPU (compute latency) để đảm bảo tính
          minh bạch và tái lập thực nghiệm khoa học chuẩn IEEE / Elsevier.
        """
        if dai_layer is None:
            hien_tai = sum(n.so_layer for n in self.danh_sach_node)
            phan_doan = max(1, (self.tong_so_layer - hien_tai))
            dai_layer = (hien_tai, min(self.tong_so_layer, hien_tai + phan_doan))

        pin_dung = ma_pin if ma_pin is not None else self.ma_pin
        node = NutSwarmCPU(
            node_id=node_id,
            dai_layer=dai_layer,
            so_core_cpu=so_core_cpu,
            ram_mb=ram_mb,
            private_key=private_key,
            endpoint_url=endpoint_url,
            che_do="mang_vat_ly_lan",
            do_tre_mang_co_dinh_ms=do_tre_mang_co_dinh_ms,
            ma_pin=pin_dung,
            nen_fp16=self.nen_fp16 if nen_fp16 is None else nen_fp16,
            d_model=self.d_model,
            ssl_context=ssl_context if ssl_context is not None else self.ssl_context,
        )
        self.danh_sach_node.append(node)
        return node

    def tu_dong_phan_manh(
        self,
        danh_sach_node_id: List[str],
        so_core_cpu: int = 4,
        ram_mb: float = 4096.0,
    ) -> List[NutSwarmCPU]:
        """Tự động phân chia đều các layer mô hình cho danh sách các node CPU tham gia."""
        self.danh_sach_node.clear()
        so_node = len(danh_sach_node_id)
        if so_node == 0:
            return []

        layers_per_node = max(1, self.tong_so_layer // so_node)
        curr = 0

        for idx, nid in enumerate(danh_sach_node_id):
            end_layer = curr + layers_per_node if idx < so_node - 1 else self.tong_so_layer
            node = NutSwarmCPU(
                node_id=nid,
                dai_layer=(curr, end_layer),
                so_core_cpu=so_core_cpu,
                ram_mb=ram_mb,
                nen_fp16=self.nen_fp16,
                d_model=self.d_model,
            )
            self.danh_sach_node.append(node)
            curr = end_layer

        return self.danh_sach_node

    @gan_nhan_tinh_nang(
        trang_thai=TrangThaiTinhNang.SIMULATED,
        mo_ta="Chế độ giả lập đa worker nội bộ (In-Process Simulation) chạy trong cùng 1 tiến trình Python.",
        la_mo_phong=True,
    )
    def khoi_tao_cum_gia_lap(
        self,
        so_node: int = 3,
        so_core_moi_node: int = 4,
        ram_mb: float = 4096.0,
    ) -> List[NutSwarmCPU]:
        """Khởi tạo cụm CPU văn phòng giả lập để thử nghiệm Pipeline Parallelism."""
        node_ids = [f"cpu_node_{i+1}" for i in range(so_node)]
        return self.tu_dong_phan_manh(node_ids, so_core_cpu=so_core_moi_node, ram_mb=ram_mb)

    def lay_trang_thai_cum(self) -> Dict[str, Any]:
        """Thống kê tài nguyên và trạng thái phân bổ layer của cụm Swarm."""
        so_node = len(self.danh_sach_node)
        tong_core = sum(n.so_core_cpu for n in self.danh_sach_node)
        tong_ram = sum(n.ram_mb for n in self.danh_sach_node)
        return {
            "so_node": so_node,
            "tong_so_core_cpu": tong_core,
            "tong_ram_mb": tong_ram,
            "tong_so_layer": self.tong_so_layer,
            "nen_fp16": self.nen_fp16,
            "ma_hoa_tls": self.ssl_context is not None,
            "xac_thuc_mat_ma": self.bat_xac_thuc_mat_ma,
            "danh_sach_node": [n.lay_thong_tin_cau_hinh() for n in self.danh_sach_node],
        }

    @staticmethod
    def _xac_minh_danh_tinh_proof_node(node: NutSwarmCPU, proof: Optional[Dict[str, Any]]) -> bool:
        """Đảm bảo proof tồn tại và khóa công khai khớp với danh tính đã ghim của node."""
        if not isinstance(proof, dict) or not proof:
            return False
        pub_trong_proof = str(proof.get("public_key", "")).lower()
        if not pub_trong_proof:
            return False
        if node.che_do != "mang_vat_ly_lan":
            return hmac.compare_digest(pub_trong_proof, node.verifier.public_key.lower())
        pinned = getattr(node, "_pinned_public_key", None)
        if pinned is None:
            setattr(node, "_pinned_public_key", pub_trong_proof)
            return True
        return hmac.compare_digest(pub_trong_proof, str(pinned).lower())

    @staticmethod
    def xac_minh_freivalds_nut(
        node: NutSwarmCPU,
        h_in: np.ndarray,
        h_out: np.ndarray,
        so_vong: int = 12,
    ) -> Dict[str, Any]:
        """Kiểm chứng xác suất Freivalds cho phép chiếu tuyến tính của một nút đơn tầng không bão hòa."""
        if node._transformer_shard is not None or node.so_layer != 1:
            return {
                "hop_le": None,
                "bo_qua": True,
                "trang_thai": "khong_ap_dung",
                "ly_do": "Nút nhiều tầng hoặc Transformer phi tuyến không hỗ trợ kiểm chứng Freivalds đơn tầng",
            }
        x_2d = np.asarray(h_in, dtype=np.float64)
        if x_2d.ndim == 1:
            x_2d = x_2d.reshape(1, -1)
        y_2d = np.asarray(h_out, dtype=np.float64)
        if y_2d.ndim == 1:
            y_2d = y_2d.reshape(1, -1)
        y_delta = (y_2d - x_2d) * 10.0
        rtol = 2e-2 if node.nen_fp16 else 1e-3
        atol = 5e-2 if node.nen_fp16 else 1e-3
        return KiemChungFreivalds.xac_minh_nhan_ma_tran(
            x=x_2d,
            w=node.w_proj[0],
            y=y_delta,
            b=node.bias[0],
            so_vong=so_vong,
            rtol=rtol,
            atol=atol,
        )

    def lan_truyen_pipeline(
        self,
        hidden_states: np.ndarray,
        tao_proof: Optional[bool] = None,
        session_id: Optional[str] = None,
        vi_tri: int = 0,
        kiem_chung_freivalds: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Lan truyền tensor trạng thái ẩn qua toàn bộ chuỗi Pipeline Parallelism:
        - Kiểm tra cam kết chứng thực mật mã học ở mỗi node (bắt buộc có proof & khớp khóa công khai).
        - Hỗ trợ kiểm chứng tính toán tuyến tính Freivalds O(n^2) khi `kiem_chung_freivalds=True`.
        - Ném ngoại lệ ValueError nếu phát hiện nút Byzantine gian lận.
        """
        if not self.danh_sach_node:
            self.them_node("local_worker_cpu", dai_layer=(0, self.tong_so_layer))

        ap_dung_proof = self.bat_xac_thuc_mat_ma if tao_proof is None else tao_proof
        h_curr = hidden_states.copy()
        if h_curr.ndim == 1:
            h_curr = h_curr.reshape(1, -1)

        cam_kets: List[Dict[str, Any]] = []

        for node in self.danh_sach_node:
            if session_id is not None or vi_tri != 0:
                try:
                    h_next, proof = node.lan_truyen_tien(
                        h_curr,
                        buoc=vi_tri,
                        tao_proof=ap_dung_proof,
                        session_id=session_id,
                    )
                except TypeError:
                    h_next, proof = node.lan_truyen_tien(h_curr, tao_proof=ap_dung_proof)
            else:
                h_next, proof = node.lan_truyen_tien(h_curr, tao_proof=ap_dung_proof)
            if ap_dung_proof:
                if not proof:
                    raise ValueError(f"Nút {node.node_id} không gửi bằng chứng chứng thực mật mã (Byzantine)")
                if "chuoi_cam_ket" in proof:
                    proof["proof_zk"] = proof["chuoi_cam_ket"]

                hop_le = self._xac_minh_danh_tinh_proof_node(node, proof) and XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
                    cam_ket=proof,
                    tensor_dau_vao=h_curr,
                    tensor_dau_ra=h_next,
                )
                if not hop_le:
                    raise ValueError(f"Phát hiện nút gian lận Byzantine trong cụm Swarm: {node.node_id}")
                if kiem_chung_freivalds:
                    fv = self.xac_minh_freivalds_nut(node, h_curr, h_next)
                    proof["freivalds"] = fv
                    if fv.get("hop_le") is False:
                        raise ValueError(
                            f"Phát hiện sai lệch tính toán ma trận qua kiểm chứng Freivalds tại nút {node.node_id}"
                        )
                cam_kets.append(proof)
            elif kiem_chung_freivalds:
                fv = self.xac_minh_freivalds_nut(node, h_curr, h_next)
                if fv.get("hop_le") is False:
                    raise ValueError(
                        f"Phát hiện sai lệch tính toán ma trận qua kiểm chứng Freivalds tại nút {node.node_id}"
                    )
            h_curr = h_next

        return h_curr, cam_kets

    def suy_luan_swarm(
        self,
        prompt: str,
        so_token_sinh: int = 16,
        xac_minh_proof: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực hiện suy luận phân tán Pipeline Parallelism trên cụm CPU:
        1. Tokenize & Nhúng văn bản ban đầu thành hidden states.
        2. Truyền hidden states tuần tự qua từng Node trong cụm Swarm.
        3. Thu thập và xác minh bằng chứng Proof of Inference (PoI) từ mỗi node.
        4. Giải mã hidden states ở Node cuối cùng thành chuỗi phản hồi tiếng Việt.
        """
        if not self.danh_sach_node:
            # Tạo node mặc định nếu chưa có node nào
            self.them_node("local_worker_cpu", dai_layer=(0, self.tong_so_layer))

        # Kiểm tra qua Tường lửa AI bảo vệ đầu vào
        if self.tuong_lua is not None:
            prompt_sach = str(prompt or "").strip()
            kt = self.tuong_lua.kiem_tra_dau_vao(prompt_sach)
            if not kt.get("an_toan", True):
                ly_do = kt.get("ly_do", "Vi phạm chính sách an toàn")
                return {
                    "prompt": prompt,
                    "cau_tra_loi": f"[Tường lửa AI chặn]: {ly_do}",
                    "an_toan": False,
                    "danh_sach_node": [],
                    "tong_so_layer": self.tong_so_layer,
                    "thoi_gian_xu_ly_ms": 0.0,
                    "tokens_moi_giay": 0.0,
                    "so_proof_da_xac_minh": 0,
                    "cam_ket_chi_tiet": [],
                }

        t0 = time.perf_counter()
        _ = self.tokenizer.tach_tu(prompt, noi_tu_ghep=True)

        # 1. Trạng thái ẩn khởi tạo
        h_state = self.embedder.nhung(prompt).reshape(1, self.d_model)

        cam_ket_cac_node: List[Dict[str, Any]] = []
        node_hop_le: List[str] = []
        tong_xac_minh_ms = 0.0

        # 2. Pipeline Pass qua các Node CPU
        for node in self.danh_sach_node:
            h_next, proof = node.thuc_thi_layer(h_state, tao_proof=xac_minh_proof)

            if xac_minh_proof:
                if not proof:
                    raise PermissionError(f"Node {node.node_id} không gửi bằng chứng chứng thực (Byzantine)")
                t_ver_start = time.perf_counter()
                hop_le = self._xac_minh_danh_tinh_proof_node(node, proof) and XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
                    cam_ket=proof,
                    tensor_dau_vao=h_state,
                    tensor_dau_ra=h_next,
                )
                tong_xac_minh_ms += (time.perf_counter() - t_ver_start) * 1000.0
                if not hop_le:
                    raise PermissionError(f"Phát hiện Node gian lận Byzantine trong cụm Swarm: {node.node_id}")
                cam_ket_cac_node.append(proof)

            node_hop_le.append(node.node_id)
            h_state = h_next

        dt = time.perf_counter() - t0

        ket_qua_sinh: List[str] = []
        da_chon: List[int] = []
        h_curr_dec = np.nan_to_num(h_state.reshape(-1).astype(np.float32), nan=0.0)
        for _ in range(so_token_sinh):
            norm_h = float(np.linalg.norm(h_curr_dec))
            h_unit = h_curr_dec / (norm_h + 1e-9)
            logits = (self._lm_head_matrix @ h_unit).astype(np.float64)
            for idx_da in set(da_chon):
                if 0 <= idx_da < len(logits):
                    logits[idx_da] = logits[idx_da] / 1.5 if logits[idx_da] > 0 else logits[idx_da] * 1.5
            idx_tu = int(np.argmax(logits))
            da_chon.append(idx_tu)
            tu = self._vocab_tokens[idx_tu]
            ket_qua_sinh.append(tu)
            h_curr_dec = 0.7 * h_curr_dec + 0.3 * self._lm_head_matrix[idx_tu]

        cau_tra_loi = f"{prompt.strip()} -> " + " ".join(ket_qua_sinh[:so_token_sinh])
        tps = so_token_sinh / dt if dt > 0 else 0.0

        tong_cpu_ms = sum(getattr(n, "thoi_gian_tinh_toan_gan_nhat_ms", 0.0) for n in self.danh_sach_node)
        tong_mang_ms = sum(getattr(n, "thoi_gian_mang_gan_nhat_ms", 0.0) for n in self.danh_sach_node)
        co_node_mang = any(getattr(n, "endpoint_url", None) is not None or getattr(n, "che_do", "") == "mang_vat_ly_lan" for n in self.danh_sach_node)
        che_do_thuc_thi = "mang_vat_ly_lan" if co_node_mang else "gia_lap_da_tien_trinh"

        nhat_ky = {
            "prompt": prompt,
            "cau_tra_loi": cau_tra_loi,
            "danh_sach_node": node_hop_le,
            "tong_so_layer": self.tong_so_layer,
            "nen_fp16": self.nen_fp16,
            "thoi_gian_xu_ly_ms": round(dt * 1000.0, 2),
            "tokens_moi_giay": round(tps, 2),
            "so_proof_da_xac_minh": len(cam_ket_cac_node),
            "cam_ket_chi_tiet": cam_ket_cac_node,
            "che_do_thuc_thi": che_do_thuc_thi,
            "phan_ra_do_tre": {
                "tong_thoi_gian_ms": round(dt * 1000.0, 2),
                "tinh_toan_cpu_ms": round(tong_cpu_ms, 2),
                "truyen_tai_mang_lan_ms": round(tong_mang_ms, 2),
                "xac_minh_mat_ma_ms": round(tong_xac_minh_ms, 2),
            },
        }
        self.nhat_ky_kiem_toan.append(nhat_ky)
        return nhat_ky

    def phan_manh_mo_hinh_gguf(self, duong_dan_gguf: str) -> Dict[str, Any]:
        """
        Tự động phân bổ các phân mảnh trọng số từ tệp GGUF tới toàn bộ các node CPU trong cụm.
        """
        from vietnamese_ai.export.gguf import XuatGGUF

        loader = XuatGGUF()
        meta = loader.doc_metadata(duong_dan_gguf)

        ket_qua_phan_manh: List[Dict[str, Any]] = []
        for node in self.danh_sach_node:
            kq_node = node.nap_trong_so_gguf(duong_dan_gguf)
            ket_qua_phan_manh.append(kq_node)

        return {
            "duong_dan_gguf": duong_dan_gguf,
            "tong_so_tensors": meta["n_tensors"],
            "danh_sach_node": ket_qua_phan_manh,
            "trang_thai": "da_phan_manh_thanh_cong",
        }

    def suy_luan_streaming(self, prompt: str, so_token: int = 12) -> Generator[str, None, None]:
        """Sinh tokens dạng streaming thời gian thực từ cụm Swarm."""
        if self.tuong_lua is not None:
            prompt_sach = str(prompt or "").strip()
            kt = self.tuong_lua.kiem_tra_dau_vao(prompt_sach)
            if not kt.get("an_toan", True):
                yield f"[Tường lửa AI chặn]: {kt.get('ly_do')}"
                return

        res = self.suy_luan_swarm(prompt, so_token_sinh=so_token, xac_minh_proof=False)
        parts = res["cau_tra_loi"].split("->")
        raw_tokens = parts[1].strip().split() if len(parts) > 1 else parts[0].split()
        tokens = raw_tokens[:so_token]
        for t in tokens:
            yield t + " "

    def sinh_stream(self, prompt: str, do_dai_toi_da: int = 16) -> Generator[str, None, None]:
        """Sinh tokens dạng streaming (alias tương thích)."""
        yield from self.suy_luan_streaming(prompt, so_token=do_dai_toi_da)

    def tao_cau_noi_orchestrator(
        self,
        port_http: int = 8085,
        host: str = "127.0.0.1",
        ma_bi_mat_cum: Optional[str] = None,
    ) -> Any:
        """
        Tạo Cầu nối HTTP Worker để liên kết cụm Swarm vào NutChinh (Cluster Orchestrator).
        """
        from vietnamese_ai.orchestrator.swarm_bridge import NutCauNoiSwarm

        return NutCauNoiSwarm(
            swarm=self,
            host=host,
            port=port_http,
            ma_bi_mat_cum=ma_bi_mat_cum,
        )

    def dang_ky_vao_nut_chinh(
        self,
        nut_chinh: Any,
        port_http: int = 8085,
        host: str = "127.0.0.1",
        trong_so_canary: Optional[float] = None,
    ) -> str:
        """
        Đăng ký trực tiếp cụm Swarm này vào Master Node (NutChinh / DieuPhoiCum).
        """
        return nut_chinh.them_nut_swarm(
            port=port_http,
            swarm=self,
            host=host,
            trong_so_canary=trong_so_canary,
            chay_ngam=True,
        )


# Aliases
SwarmInferenceOrchestrator = DieuPhoiSwarmCPU
SwarmCPUNode = NutSwarmCPU
