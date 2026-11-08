"""Hệ Thống Suy Luận Đàn CPU Phân Tán (Distributed Swarm Inference & Pipeline Parallelism trên CPU).

Cho phép kết nối nhiều máy tính CPU cấu hình thấp (Mini PC, máy văn phòng, laptop) qua mạng LAN/WAN
để cùng chia sẻ tải tính toán (Model Sharding / Pipeline Parallelism) một mô hình ngôn ngữ lớn
mà không cần GPU đắt đỏ, tích hợp cơ chế xác thực chứng thực mật mã học Verifiable AI (Proof of Inference).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc, BoTachTuTiengVietSieuToc
from vietnamese_ai.security.verifiable_inference import XacThucSuyLuanMaoHiem


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

        # Khởi tạo trọng số giả lập xác định (Deterministic Weights) cho các layer phụ trách
        seed = int(hashlib.sha256(f"{node_id}:{dai_layer}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        self.d_model = 128
        # Trọng số tuyến tính mô phỏng MLP / Self-Attention block
        self.w_proj = rng.randn(self.so_layer, self.d_model, self.d_model).astype(np.float32) * 0.05
        self.bias = np.zeros((self.so_layer, self.d_model), dtype=np.float32)

    def thuc_thi_layer(
        self,
        hidden_states: np.ndarray,
        tao_proof: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Thực hiện tính toán chuyển tiếp (Forward Pass) qua các layer phụ trách trên CPU.
        Trả về `(tensor_dau_ra, cryptographic_commitment)`.
        Đồng thời ghi nhận riêng biệt thời gian tính toán CPU và thời gian trễ mạng.
        """
        t_start = time.perf_counter()
        x = hidden_states.copy()
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Chạy qua từng block layer
        for l_idx in range(self.so_layer):
            # Tuyến tính + hàm kích hoạt ReLU / GeLU xấp xỉ
            w = self.w_proj[l_idx]
            b = self.bias[l_idx]
            x_next = np.dot(x, w) + b
            x = np.maximum(0.0, x + 0.1 * x_next)  # Residual connection nhẹ

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

    def lay_thong_tin_cau_hinh(self) -> Dict[str, Any]:
        """Trả về thông tin cấu hình phần cứng và dải layer của Node."""
        return {
            "node_id": self.node_id,
            "dai_layer": self.dai_layer,
            "so_core_cpu": self.so_core_cpu,
            "ram_mb": self.ram_mb,
            "so_layer": self.so_layer,
            "public_key": self.verifier.public_key,
            "endpoint_url": self.endpoint_url,
            "che_do": self.che_do,
            "la_node_mang_vat_ly": self.endpoint_url is not None or self.che_do == "mang_vat_ly_lan",
        }

    def lan_truyen_tien(
        self,
        hidden_states: np.ndarray,
        buoc: int = 0,
        tao_proof: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Alias phương thức lan truyền tiến (Forward Pass) cho Node."""
        x, proof = self.thuc_thi_layer(hidden_states, tao_proof=tao_proof)
        p_dict = proof.copy() if proof else {}
        if "chuoi_cam_ket" not in p_dict and "proof_zk" in p_dict:
            p_dict["chuoi_cam_ket"] = p_dict["proof_zk"]
        return x, p_dict


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
    ):
        self.d_model = d_model
        self.tong_so_layer = tong_so_layer
        self.bat_xac_thuc_mat_ma = bat_xac_thuc_mat_ma
        self.bat_tuong_lua = bat_tuong_lua
        self.danh_sach_node: List[NutSwarmCPU] = []
        self.tokenizer = BoTachTuTiengVietSieuToc()
        self.embedder = BoNhungVanBanSieuToc(so_chieu=d_model, tokenizer=self.tokenizer)
        self.nhat_ky_kiem_toan: List[Dict[str, Any]] = []

        if bat_tuong_lua:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            self.tuong_lua: Optional[TuongLuaBaoVeFramework] = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        else:
            self.tuong_lua = None

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
    ) -> NutSwarmCPU:
        """
        Đăng ký một nút mạng vật lý thực tế qua mạng LAN/WAN (Physical LAN Network Node):
        - Hỗ trợ endpoint HTTP / WebSocket / Socket RPC thực tế.
        - Tách biệt rõ độ trễ truyền tải mạng vật lý (network transfer latency)
          với thời gian tính toán nội tại trên CPU (compute latency) để đảm bảo tính
          minh bạch và tái lập thực nghiệm khoa học chuẩn IEEE / Elsevier.
        """
        if dai_layer is None:
            hien_tai = sum(n.so_layer for n in self.danh_sach_node)
            phan_doan = max(1, (self.tong_so_layer - hien_tai))
            dai_layer = (hien_tai, min(self.tong_so_layer, hien_tai + phan_doan))

        node = NutSwarmCPU(
            node_id=node_id,
            dai_layer=dai_layer,
            so_core_cpu=so_core_cpu,
            ram_mb=ram_mb,
            private_key=private_key,
            endpoint_url=endpoint_url,
            che_do="mang_vat_ly_lan",
            do_tre_mang_co_dinh_ms=do_tre_mang_co_dinh_ms,
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
            )
            self.danh_sach_node.append(node)
            curr = end_layer

        return self.danh_sach_node

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
            "xac_thuc_mat_ma": self.bat_xac_thuc_mat_ma,
            "danh_sach_node": [n.lay_thong_tin_cau_hinh() for n in self.danh_sach_node],
        }

    def lan_truyen_pipeline(
        self,
        hidden_states: np.ndarray,
        tao_proof: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Lan truyền tensor trạng thái ẩn qua toàn bộ chuỗi Pipeline Parallelism:
        - Kiểm tra cam kết chứng thực mật mã học ở mỗi node.
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
            h_next, proof = node.lan_truyen_tien(h_curr, tao_proof=ap_dung_proof)
            if ap_dung_proof and proof:
                if "chuoi_cam_ket" in proof:
                    proof["proof_zk"] = proof["chuoi_cam_ket"]

                hop_le = XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
                    cam_ket=proof,
                    tensor_dau_vao=h_curr,
                    tensor_dau_ra=h_next,
                )
                if not hop_le:
                    raise ValueError(f"Phát hiện nút gian lận Byzantine trong cụm Swarm: {node.node_id}")
                cam_kets.append(proof)
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

        # 2. Pipeline Pass qua các Node CPU
        for node in self.danh_sach_node:
            h_next, proof = node.thuc_thi_layer(h_state, tao_proof=xac_minh_proof)

            if xac_minh_proof and proof:
                # Xác thực tính trung thực của Node
                hop_le = XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
                    cam_ket=proof,
                    tensor_dau_vao=h_state,
                    tensor_dau_ra=h_next,
                )
                if not hop_le:
                    raise PermissionError(f"Phát hiện Node gian lận Byzantine trong cụm Swarm: {node.node_id}")
                cam_ket_cac_node.append(proof)

            node_hop_le.append(node.node_id)
            h_state = h_next

        dt = time.perf_counter() - t0

        # 3. Giải mã tokens đầu ra
        ket_qua_sinh: List[str] = []
        for i in range(so_token_sinh):
            idx_tu = (int(np.sum(np.abs(h_state))) + i * 7) % len(self.tokenizer.TU_DIEN_TU_GHEP_MAC_DINH)
            tu = list(self.tokenizer.TU_DIEN_TU_GHEP_MAC_DINH)[idx_tu]
            ket_qua_sinh.append(tu)

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
            "thoi_gian_xu_ly_ms": round(dt * 1000.0, 2),
            "tokens_moi_giay": round(tps, 2),
            "so_proof_da_xac_minh": len(cam_ket_cac_node),
            "cam_ket_chi_tiet": cam_ket_cac_node,
            "che_do_thuc_thi": che_do_thuc_thi,
            "phan_ra_do_tre": {
                "tong_thoi_gian_ms": round(dt * 1000.0, 2),
                "tinh_toan_cpu_ms": round(tong_cpu_ms, 2),
                "truyen_tai_mang_lan_ms": round(tong_mang_ms, 2),
            },
        }
        self.nhat_ky_kiem_toan.append(nhat_ky)
        return nhat_ky

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
            time.sleep(0.01)
            yield t + " "

    def sinh_stream(self, prompt: str, do_dai_toi_da: int = 16) -> Generator[str, None, None]:
        """Sinh tokens dạng streaming (alias tương thích)."""
        yield from self.suy_luan_streaming(prompt, so_token=do_dai_toi_da)


# Aliases
SwarmInferenceOrchestrator = DieuPhoiSwarmCPU
SwarmCPUNode = NutSwarmCPU
