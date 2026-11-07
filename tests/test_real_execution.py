"""Kiểm thử thực thi thật (Real Execution) toàn diện cho Vietnamese AI Framework v30.1.0."""


import numpy as np
import pytest
from PIL import Image

from vietnamese_ai.agents.experience_memory import SoTayKinhNghiem
from vietnamese_ai.agents.long_term_memory import HethongNhoMemGPT
from vietnamese_ai.agents.spatial_tools import (
    _SPATIAL_STORE,
    cong_cu_di_chuyen_robot,
    cong_cu_quet_radar_3d,
    giai_dong_hoc_nguoc_3dof,
)
from vietnamese_ai.agents.tools import cong_cu_phan_tich_anh
from vietnamese_ai.distributed.ring_attention import RingAttentionCluster
from vietnamese_ai.edge.tee_zkp_node import SecureEdgeNode
from vietnamese_ai.edge.wasm_node import WebBrowserNode
from vietnamese_ai.export.gguf import XuatGGUF
from vietnamese_ai.export.onnx_export import XuatONNX
from vietnamese_ai.federated.learning import HocLienKet
from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer
from vietnamese_ai.fine_tuning.reward_model import RewardModel
from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer
from vietnamese_ai.mobile.bridge import MobileAppBridge
from vietnamese_ai.mobile.browser_copilot import BrowserCopilot
from vietnamese_ai.mobile.deployment import TriKhaiDiDong
from vietnamese_ai.mobile.mobile_tools import (
    MobileDeviceState,
    cong_cu_chup_anh_camera,
    cong_cu_lay_toa_do_gps,
)
from vietnamese_ai.models import PhanLoai
from vietnamese_ai.nlp.pretrained.phobert import PhoBERTWrapper
from vietnamese_ai.rag.embedder import BoNhungVanBan
from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder
from vietnamese_ai.rag.realtime_sync import DocumentWatcher, RAGEventBus
from vietnamese_ai.rag.vector_store import CSDLVector
from vietnamese_ai.serving.paged_attention import BlockManager, PagedAttentionSimulation
from vietnamese_ai.transformer.gpt_model import GPTModel
from vietnamese_ai.transformer.pretrainer import PreTrainer, TextDataset


def test_gpt_real_backprop_and_pretrainer():
    model = GPTModel(
        d_model=16,
        so_dau=2,
        d_ff=32,
        so_block=2,
        so_tu_vung=64,
        do_dai_toi_da=32,
        dropout=0.0,
    )
    x = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    y = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)

    loss_first = model.buoc_huan_luyen(x, y, toc_do_hoc=1e-2)
    loss_last = loss_first
    for _ in range(15):
        loss_last = model.buoc_huan_luyen(x, y, toc_do_hoc=1e-2)

    assert loss_last < loss_first

    ds = TextDataset(do_dai_window=8, buoc_nhay=4)
    ds.tai_corpus(["xin chao viet nam tri tue nhan tao " * 8], vocab_size=64)
    pt = PreTrainer(so_vong=3, kich_thuoc_batch=2, toc_do_hoc=5e-3)
    res = pt.huan_luyen(model, ds)
    assert res["history"]["train_loss"][-1] <= res["history"]["train_loss"][0]


def test_sft_dpo_and_reward_model_numpy():
    gpt = GPTModel(
        d_model=16,
        so_dau=2,
        d_ff=32,
        so_block=1,
        so_tu_vung=64,
        do_dai_toi_da=32,
        dropout=0.0,
    )

    sft = SFTTrainer(so_vong=3, kich_thuoc_batch=2, toc_do_hoc=5e-3)
    sft_res = sft.huan_luyen(
        gpt,
        [
            {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [2, 3, 4, 5, 6, 7]},
            {"input_ids": [2, 3, 4, 5, 6, 7], "labels": [3, 4, 5, 6, 7, 8]},
        ],
    )
    assert sft_res["history"]["train_loss"][-1] <= sft_res["history"]["train_loss"][0]

    dpo = DPOTrainer(so_vong=5, toc_do_hoc=0.05)
    pref_data = [
        {
            "prompt": "Thủ đô của Việt Nam là gì?",
            "chosen": "Thủ đô của Việt Nam là Hà Nội.",
            "rejected": "Không biết.",
        },
        {
            "prompt": "AI là gì?",
            "chosen": "AI là trí tuệ nhân tạo.",
            "rejected": "xyz 123",
        },
    ]
    dpo_res = dpo.huan_luyen(model=None, ref_model=None, preference_data=pref_data)
    assert dpo_res["history"]["train_loss"][-1] < dpo_res["history"]["train_loss"][0]
    assert dpo_res["final_reward_margin"] > 0.0

    rm = RewardModel(toc_do_hoc=0.05)
    rm.huan_luyen(model=None, preference_data=pref_data, so_vong=6)
    scores = rm.diem_danh_gia(
        model=None,
        cac_van_ban=[
            "Thủ đô của Việt Nam là gì? Thủ đô của Việt Nam là Hà Nội.",
            "Thủ đô của Việt Nam là gì? Không biết.",
        ],
    )
    assert scores[0]["score"] > scores[1]["score"]


def test_gguf_onnx_and_mobile_real_inference(tmp_path):
    rng = np.random.RandomState(42)
    X = rng.randn(40, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = PhanLoai(thuat_toan="logistic")
    clf.huan_luyen(X, y)

    # 1. GGUF roundtrip
    gguf = XuatGGUF()
    weights = {"w1": rng.randn(8, 4).astype(np.float32), "b1": rng.randn(8).astype(np.float32)}
    for qtype in ("f32", "f16", "q8_0", "q4_k_m"):
        p = str(tmp_path / f"model_{qtype}.gguf")
        gguf.xuat_tu_numpy(weights, p, quantization=qtype)
        loaded = gguf.tai_tu_gguf(p)
        assert "w1" in loaded
        assert loaded["w1"].shape == (8, 4)
        if qtype in ("f32", "f16", "q8_0"):
            assert np.allclose(loaded["w1"], weights["w1"], atol=0.15)

    # 2. ONNX native export & execution
    onnx_exp = XuatONNX()
    onnx_path = str(tmp_path / "clf.onnx")
    onnx_exp.xuat_sklearn(clf, onnx_path, kich_thuoc_dau_vao=(None, 4))
    info = onnx_exp.kiem_tra_onnx(onnx_path)
    assert len(info["inputs"]) >= 1
    preds_onnx = onnx_exp.chay_onnx(onnx_path, X[:5])
    assert np.array_equal(preds_onnx.ravel(), clf.du_doan(X[:5]).ravel())

    # 3. Mobile deployment real inference across formats
    mob = TriKhaiDiDong()
    p_tflite = str(tmp_path / "mob.tflite")
    mob.xuat_tflite(clf, p_tflite, kich_thuoc_dau_vao=(4,), luong_hoa="int8")
    p_coreml = str(tmp_path / "mob.mlmodel")
    mob.xuat_coreml(clf, p_coreml, kich_thuoc_dau_vao=(4,))
    p_onnx = str(tmp_path / "mob.onnx")
    mob.xuat_onnx_mobile(clf, p_onnx, kich_thuoc_dau_vao=(4,))
    p_gguf = str(tmp_path / "mob.gguf")
    mob.xuat_gguf_mobile(clf, p_gguf)
    p_pte = str(tmp_path / "mob.pte")
    mob.xuat_executorch(clf, p_pte)

    for fpath in (p_tflite, p_coreml, p_onnx, p_gguf, p_pte):
        preds_mob = mob.chay_suy_luan_di_dong(fpath, X[:5])
        assert preds_mob.shape == (5,)


def test_federated_learning_real_weights():
    rng = np.random.RandomState(7)
    X = rng.randn(60, 3)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    fl = HocLienKet(so_client=2, so_vong=2)
    fl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")

    preds = fl.du_doan(PhanLoai, X[:15], thuat_toan="logistic")
    acc = float(np.mean(preds == y[:15]))
    assert acc >= 0.7


def test_semantic_embeddings_multimodal_and_memory(tmp_path):
    emb = BoNhungVanBan(kich_thuoc=128, phuong_phap="hash")
    v1 = emb.nhung_mot("trí tuệ nhân tạo và học máy sâu")
    v2 = emb.nhung_mot("mô hình trí tuệ nhân tạo học máy")
    v3 = emb.nhung_mot("công thức nấu món phở bò truyền thống")
    assert float(v1 @ v2) > float(v1 @ v3)

    phobert = PhoBERTWrapper(che_do_mo_phong=True, kich_thuoc_an=128)
    assert phobert.do_tuong_dong("xử lý ngôn ngữ tự nhiên tiếng Việt", "ngôn ngữ tự nhiên tiếng Việt") > phobert.do_tuong_dong(
        "xử lý ngôn ngữ tự nhiên tiếng Việt", "bóng đá ngoại hạng anh"
    )

    # Real image feature extraction
    img_red_1 = str(tmp_path / "red1.png")
    img_red_2 = str(tmp_path / "red2.png")
    img_blue = str(tmp_path / "blue.png")
    Image.new("RGB", (64, 64), color=(240, 20, 20)).save(img_red_1)
    Image.new("RGB", (64, 64), color=(230, 30, 25)).save(img_red_2)
    Image.new("RGB", (64, 64), color=(10, 20, 245)).save(img_blue)

    img_emb = ImageEmbedder()
    vecs = np.asarray(img_emb.nhung_hinh_anh([img_red_1, img_red_2, img_blue]))
    sim_red_red = float(vecs[0] @ vecs[1])
    sim_red_blue = float(vecs[0] @ vecs[2])
    assert sim_red_red > sim_red_blue

    analysis = cong_cu_phan_tich_anh.chay(duong_dan_anh=img_red_1)
    assert "64x64px" in analysis
    assert "đỏ (Red)" in analysis

    # CSDLVector + SoTayKinhNghiem + HethongNhoMemGPT
    store = CSDLVector(kich_thuoc=128)
    exp = SoTayKinhNghiem(vector_store=store)
    exp.ghi_nhan_bai_hoc("lỗi kết nối cơ sở dữ liệu postgres", "quên mở port 5432", "kiểm tra pg_hba.conf")
    recalled = exp.truy_xuat_kinh_nghiem("kết nối cơ sở dữ liệu postgres")
    assert recalled is not None and "pg_hba.conf" in recalled

    mem = HethongNhoMemGPT(max_core_tokens=10, vector_store=CSDLVector(kich_thuoc=128))
    for i in range(6):
        mem.them_tin_nhan("user", f"Thông tin mật số {i}: mật khẩu máy chủ là alpha_beta_{i}")
    tool = mem.tao_cong_cu_truy_van()
    kq_mem = tool.chay(tu_khoa="mật khẩu máy chủ alpha_beta")
    assert "alpha_beta" in kq_mem

    # DocumentWatcher real directory scan
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "tai_lieu.txt").write_text("Nội dung tài liệu thực tế cho RAG", encoding="utf-8")
    events = []
    RAGEventBus().dang_ky_lang_nghe("NEW_DOCUMENT", lambda payload: events.append(payload))
    watcher = DocumentWatcher(str(doc_dir))
    found = watcher.quet_thu_muc(dong_bo=True)
    assert "tai_lieu.txt" in found
    assert any(e["ma"] == "tai_lieu.txt" for e in events)

    # BrowserCopilot real HTML extraction
    html_file = tmp_path / "page.html"
    html_file.write_text(
        "<html><head><script>var a=1;</script></head><body><h1>Báo cáo Quý 4</h1><p>Doanh thu đạt 500 tỷ đồng.</p></body></html>",
        encoding="utf-8",
    )

    class _DummyMobileAgent:
        def chay(self, prompt: str) -> str:
            return prompt

    copilot = BrowserCopilot(_DummyMobileAgent())
    extracted = copilot.cao_du_lieu_web(str(html_file))
    assert "Doanh thu đạt 500 tỷ đồng" in extracted
    assert "var a=1" not in extracted


def test_paged_attention_cow_and_ring_attention():
    bm = BlockManager(num_blocks=6, block_size=2, d_model=8)
    k1 = np.ones(8, dtype=np.float32)
    v1 = np.full(8, 2.0, dtype=np.float32)
    bm.append_token("s1", 10, k_vec=k1, v_vec=v1)

    # Copy-on-Write fork
    bm.fork_sequence("s1", "s2")
    bm.append_token("s2", 20, k_vec=k1 * 0.5, v_vec=v1 * 3.0)

    pa = PagedAttentionSimulation(bm)
    out_s1 = pa.forward(np.ones((1, 8), dtype=np.float32), "s1", return_tensor=True)
    out_s2 = pa.forward(np.ones((1, 8), dtype=np.float32), "s2", return_tensor=True)
    assert np.allclose(out_s1, 2.0)
    assert not np.allclose(out_s1, out_s2)

    # Ring Attention exact match with full standard attention
    rng = np.random.RandomState(99)
    Q = rng.randn(2, 8, 16)
    K = rng.randn(2, 8, 16)
    V = rng.randn(2, 8, 16)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(16.0)
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    expected = weights @ V

    cluster = RingAttentionCluster(d_model=16, total_nodes=4)
    actual = cluster.forward(Q, K, V)
    assert np.allclose(actual, expected, atol=1e-6)


@pytest.mark.asyncio
async def test_aead_bridge_schnorr_zkp_and_spatial_mobile_tools(tmp_path):
    class _Agent:
        def chay(self, msg: str) -> str:
            return f"Echo: {msg}"

    bridge = MobileAppBridge(_Agent(), e2e_key="khoa_bi_mat_256", strict_aead=True)
    ct = bridge.ma_hoa_aead({"message": "Xin chào AEAD"})
    assert ct.startswith("AEAD1.")

    resp_ct = await bridge.handle_request("chat", ct)
    resp = bridge.giai_ma_aead(resp_ct)
    assert resp["text"] == "Echo: Xin chào AEAD"

    # Tamper detection
    tampered = ct[:-4] + ("AAAA" if not ct.endswith("AAAA") else "BBBB")
    with pytest.raises(Exception):
        bridge.giai_ma_aead(tampered)

    # Schnorr ZKP verification using true public_key (not private_key)
    node = SecureEdgeNode("edge_01", private_key="secret_passphrase", auto_start=False)
    node.engine.sinh_van_ban = lambda q, do_dai=128: f"Kết quả: {q}"
    kq = node.chay_suy_luan_an_toan("Kiểm tra ZKP")
    assert node.public_key != "secret_passphrase"
    assert SecureEdgeNode.verify_proof(
        node_id=kq["node_id"],
        public_key=node.public_key,
        cau_hoi="Kiểm tra ZKP",
        cau_tra_loi=kq["cau_tra_loi"],
        proof=kq["proof"],
        model_hash=kq["model_hash"],
    )

    wasm = WebBrowserNode("sess_1", secret_key="browser_secret")
    wasm_kq = wasm.chay_suy_luan_an_toan("Hello WASM")
    assert WebBrowserNode.verify_proof(
        node_id=wasm_kq["node_id"],
        public_key=wasm.public_key,
        cau_hoi="Hello WASM",
        cau_tra_loi=wasm_kq["cau_tra_loi"],
        proof=wasm_kq["proof"],
        model_hash=wasm_kq["model_hash"],
    )

    # 3-DOF Inverse Kinematics & Spatial Point Cloud
    ik = giai_dong_hoc_nguoc_3dof(3.0, 4.0, 2.0)
    assert abs(ik["khoang_cach"] - np.sqrt(29.0)) < 1e-3
    assert "IK 3-DOF" in cong_cu_di_chuyen_robot.chay(x=3.0, y=4.0, z=2.0)
    cong_cu_quet_radar_3d.chay()
    assert _SPATIAL_STORE.so_luong() >= 3

    # Mobile tools state & real camera file creation
    MobileDeviceState.thu_muc_anh = str(tmp_path)
    MobileDeviceState.cap_nhat_gps(21.028511, 105.804817, "Hà Nội")
    assert "Hà Nội" in cong_cu_lay_toa_do_gps.chay()
    cam_msg = cong_cu_chup_anh_camera.chay(mat_truoc=False)
    img_saved = tmp_path / "evonet_camera" / "IMG_2026.jpg"
    assert img_saved.is_file()
    assert str(img_saved) in cam_msg
