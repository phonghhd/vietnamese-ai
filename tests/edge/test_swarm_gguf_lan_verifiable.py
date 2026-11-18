"""
Kiểm thử toàn diện Trụ cột 2: Swarm CPU GGUF Shards & Physical LAN Pipeline Parallelism với Verifiable AI.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU, NutSwarmCPU
from vietnamese_ai.export.gguf import XuatGGUF
from vietnamese_ai.security.verifiable_inference import XacThucSuyLuanMaoHiem


def _tim_port_trong() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_gguf_shard_loading_va_export():
    """Kiểm tra xuất mô hình GGUF và nạp phân mảnh (Shards) vào NutSwarmCPU."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        duong_dan_gguf = str(Path(tmp_dir) / "model_test.gguf")

        # 1. Tạo tệp GGUF đa tầng (4 layers: 0..3)
        d_model = 128
        trong_so = {
            "tok_emb.weight": np.random.randn(50, d_model).astype(np.float32),
            "blk.0.weight": np.random.randn(d_model, d_model).astype(np.float32),
            "blk.1.weight": np.random.randn(d_model, d_model).astype(np.float32),
            "blk.2.weight": np.random.randn(d_model, d_model).astype(np.float32),
            "blk.3.weight": np.random.randn(d_model, d_model).astype(np.float32),
            "lm_head.weight": np.random.randn(50, d_model).astype(np.float32),
        }

        xuat = XuatGGUF()
        xuat.xuat_tu_numpy(trong_so, duong_dan_gguf, quantization="q8_0")
        assert Path(duong_dan_gguf).exists()

        # 2. Khởi tạo 2 node Swarm đảm nhiệm 2 phân đoạn layer: (0, 2) và (2, 4)
        node_1 = NutSwarmCPU("cpu_node_shard_1", dai_layer=(0, 2))
        node_2 = NutSwarmCPU("cpu_node_shard_2", dai_layer=(2, 4))

        kq_nap_1 = node_1.nap_trong_so_gguf(duong_dan_gguf)
        kq_nap_2 = node_2.nap_trong_so_gguf(duong_dan_gguf)

        assert kq_nap_1["so_tensors_nap"] >= 2
        assert kq_nap_2["so_tensors_nap"] >= 2
        assert len(node_1.checksum_trong_so_sha256) == 64
        assert len(node_2.checksum_trong_so_sha256) == 64
        # Checksum của 2 shard khác nhau phải khác nhau
        assert node_1.checksum_trong_so_sha256 != node_2.checksum_trong_so_sha256

        # 3. Xuất ngược shard độc lập ra tệp mini-GGUF
        duong_dan_shard_1 = str(Path(tmp_dir) / "shard_1.gguf")
        xuat_shard = node_1.xuat_phan_manh_gguf(duong_dan_shard_1)
        assert Path(xuat_shard).exists()


def test_nut_swarm_server_lan_socket():
    """Kiểm tra khởi động máy chủ TCP Socket trên nút vật lý và truyền nhận tensor qua LAN."""
    port = _tim_port_trong()
    node_server = NutSwarmCPU(
        node_id="physical_cpu_node_1",
        dai_layer=(0, 2),
        so_core_cpu=4,
        ram_mb=4096.0,
    )

    # Khởi động Socket Server trên port
    node_server.khoi_dong_server_lan(host="127.0.0.1", port=port)

    try:
        # Nút client kết nối qua mạng LAN
        node_client = NutSwarmCPU(
            node_id="physical_cpu_node_1",
            dai_layer=(0, 2),
            endpoint_url=f"tcp://127.0.0.1:{port}",
            che_do="mang_vat_ly_lan",
        )

        x_in = np.ones((1, 128), dtype=np.float32) * 0.5
        x_out, proof = node_client.thuc_thi_layer(x_in, tao_proof=True)

        assert x_out.shape == (1, 128)
        assert proof is not None
        assert "proof_zk" in proof
        assert proof["node_id"] == "physical_cpu_node_1"

        # Đo lường độ trễ mạng thực tế qua socket
        assert node_client.thoi_gian_mang_gan_nhat_ms > 0.0

        # Xác minh chữ ký mật mã ZKP của kết quả
        hop_le = XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
            cam_ket=proof,
            tensor_dau_vao=x_in,
            tensor_dau_ra=x_out,
        )
        assert hop_le is True

    finally:
        node_server.dung_server_lan()


def test_dieu_phoi_cum_lan_da_nut_kem_phan_manh_gguf():
    """Kiểm tra cụm Swarm đa nút kết hợp phân mảnh GGUF và truyền phát LAN Pipeline Parallelism."""
    port_1 = _tim_port_trong()
    port_2 = _tim_port_trong()

    # Nút vật lý 1
    server_1 = NutSwarmCPU("lan_worker_1", dai_layer=(0, 2))
    server_1.khoi_dong_server_lan(host="127.0.0.1", port=port_1)

    # Nút vật lý 2
    server_2 = NutSwarmCPU("lan_worker_2", dai_layer=(2, 4))
    server_2.khoi_dong_server_lan(host="127.0.0.1", port=port_2)

    try:
        orchestrator = DieuPhoiSwarmCPU(d_model=128, tong_so_layer=4, bat_tuong_lua=False)

        # Đăng ký 2 nút mạng vật lý LAN
        orchestrator.them_node_mang_vat_ly(
            node_id="lan_worker_1",
            endpoint_url=f"tcp://127.0.0.1:{port_1}",
            dai_layer=(0, 2),
        )
        orchestrator.them_node_mang_vat_ly(
            node_id="lan_worker_2",
            endpoint_url=f"tcp://127.0.0.1:{port_2}",
            dai_layer=(2, 4),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            f_gguf = str(Path(tmp_dir) / "cum_model.gguf")
            trong_so = {
                "blk.0.weight": np.ones((128, 128), dtype=np.float32) * 0.01,
                "blk.1.weight": np.ones((128, 128), dtype=np.float32) * 0.01,
                "blk.2.weight": np.ones((128, 128), dtype=np.float32) * 0.01,
                "blk.3.weight": np.ones((128, 128), dtype=np.float32) * 0.01,
            }
            XuatGGUF().xuat_tu_numpy(trong_so, f_gguf, quantization="f32")

            # Phân bổ mô hình tới cụm
            res_pm = orchestrator.phan_manh_mo_hinh_gguf(f_gguf)
            assert res_pm["trang_thai"] == "da_phan_manh_thanh_cong"

        # Thực thi suy luận Swarm phân tán qua mạng LAN
        kq = orchestrator.suy_luan_swarm("Quy trình cấp phép thành lập doanh nghiệp", so_token_sinh=4)

        assert kq["che_do_thuc_thi"] == "mang_vat_ly_lan"
        assert kq["so_proof_da_xac_minh"] == 2
        assert len(kq["danh_sach_node"]) == 2
        assert "xac_minh_mat_ma_ms" in kq["phan_ra_do_tre"]

    finally:
        server_1.dung_server_lan()
        server_2.dung_server_lan()


def test_byzantine_fault_detection_verifiable_ai():
    """Kiểm tra phát hiện và ngăn chặn tức thì nút Byzantine gian lận kết quả trên mạng."""
    port = _tim_port_trong()

    # Nút giả mạo cố ý gửi output sai khác với cam kết mật mã học
    node_rogue = NutSwarmCPU("rogue_node", dai_layer=(0, 2))
    node_rogue.khoi_dong_server_lan(host="127.0.0.1", port=port)

    try:
        orchestrator = DieuPhoiSwarmCPU(tong_so_layer=2, bat_tuong_lua=False)
        orchestrator.them_node_mang_vat_ly(
            node_id="rogue_node",
            endpoint_url=f"tcp://127.0.0.1:{port}",
            dai_layer=(0, 2),
        )

        # Trạng thái ban đầu: Node chạy bình thường
        h_in = np.ones((1, 128), dtype=np.float32)
        h_out, proof = orchestrator.danh_sach_node[0].thuc_thi_layer(h_in, tao_proof=True)
        assert proof is not None

        # Giả lập tấn công Byzantine: Sửa đổi giá trị 1 điểm trong tensor
        h_out_tampered = h_out.copy()
        h_out_tampered[0, 0] += 50.0

        # Kiểm tra Verifiable AI phát hiện gian lận
        with pytest.raises(PermissionError, match="Byzantine"):
            # Khi xác minh cam kết với tensor bị giả mạo
            hop_le = XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
                cam_ket=proof,
                tensor_dau_vao=h_in,
                tensor_dau_ra=h_out_tampered,
            )
            if not hop_le:
                raise PermissionError("Phát hiện Node gian lận Byzantine trong cụm Swarm: rogue_node")

    finally:
        node_rogue.dung_server_lan()


def test_nut_swarm_nen_fp16_lan_socket():
    """Kiểm tra nén trạng thái ẩn FP16 qua TCP Socket giúp giảm 50% băng thông mạng LAN."""
    port = _tim_port_trong()
    node_server = NutSwarmCPU(
        node_id="fp16_cpu_server",
        dai_layer=(0, 2),
        so_core_cpu=4,
        ram_mb=4096.0,
    )
    node_server.khoi_dong_server_lan(host="127.0.0.1", port=port)

    try:
        # Nút client cấu hình nén FP16
        node_client = NutSwarmCPU(
            node_id="fp16_cpu_client",
            dai_layer=(0, 2),
            endpoint_url=f"tcp://127.0.0.1:{port}",
            che_do="mang_vat_ly_lan",
            nen_fp16=True,
        )

        assert node_client.nen_fp16 is True
        info = node_client.lay_thong_tin_cau_hinh()
        assert info["nen_fp16"] is True

        x_in = np.ones((1, 128), dtype=np.float32) * 0.25
        x_out, proof = node_client.thuc_thi_layer(x_in, tao_proof=True)

        assert x_out.shape == (1, 128)
        assert proof is not None
        assert "proof_zk" in proof

        # Kiểm tra tính toàn vẹn xác thực ZKP với tensor FP16
        hop_le = XacThucSuyLuanMaoHiem.xac_minh_cam_ket(
            cam_ket=proof,
            tensor_dau_vao=x_in,
            tensor_dau_ra=x_out,
        )
        assert hop_le is True

        # Kiểm tra DieuPhoiSwarmCPU tích hợp nén FP16
        orchestrator = DieuPhoiSwarmCPU(tong_so_layer=4, bat_tuong_lua=False, nen_fp16=True)
        assert orchestrator.nen_fp16 is True
        trang_thai = orchestrator.lay_trang_thai_cum()
        assert trang_thai["nen_fp16"] is True

        kq = orchestrator.suy_luan_swarm("Kiểm tra nén dữ liệu truyền tải Swarm", so_token_sinh=2)
        assert kq["nen_fp16"] is True
    finally:
        node_server.dung_server_lan()


def test_swarm_tampering_phat_hien_byzantine_qua_tcp():
    """Kiểm tra phát hiện nút giả mạo tensor gian lận kết quả truyền tải qua mạng TCP Socket."""
    port = _tim_port_trong()

    class RogueNutSwarmCPU(NutSwarmCPU):
        def _tinh_toan_cuc_bo(self, hidden_states, tao_proof=True, nen_fp16=False):
            out_tensor, proof = super()._tinh_toan_cuc_bo(hidden_states, tao_proof=tao_proof, nen_fp16=nen_fp16)
            # Gian lận: làm giả dữ liệu tensor đầu ra sau khi cam kết mật mã học đã sinh
            tampered_tensor = out_tensor.copy()
            tampered_tensor[0, 0] += 999.0
            return tampered_tensor, proof

    server_rogue = RogueNutSwarmCPU(
        node_id="nut_gian_lan_tcp",
        dai_layer=(0, 2),
        so_core_cpu=4,
        ram_mb=4096.0,
    )
    server_rogue.khoi_dong_server_lan(host="127.0.0.1", port=port)

    try:
        orchestrator = DieuPhoiSwarmCPU(
            tong_so_layer=2,
            bat_tuong_lua=False,
            bat_xac_thuc_mat_ma=True,
        )
        orchestrator.them_node_mang_vat_ly(
            node_id="nut_gian_lan_tcp",
            endpoint_url=f"tcp://127.0.0.1:{port}",
            dai_layer=(0, 2),
        )

        h_in = np.ones((1, 128), dtype=np.float32)

        # Bộ điều phối Swarm phải ném lỗi Byzantine khi nhận tensor bị giả mạo qua mạng TCP
        with pytest.raises(ValueError, match="Byzantine"):
            orchestrator.lan_truyen_pipeline(h_in, tao_proof=True)

    finally:
        server_rogue.dung_server_lan()
