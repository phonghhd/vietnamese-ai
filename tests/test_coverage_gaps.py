"""Bổ sung coverage cho các module trước đây thiếu test."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestImageEmbedderDeterministic:
    def test_cung_input_cung_vector(self):
        from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder

        emb = ImageEmbedder()
        a = emb.nhung_van_ban("xin chào")
        b = emb.nhung_van_ban("xin chào")
        assert a == b
        assert len(a[0]) == 512

    def test_khac_input_khac_vector(self):
        from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder

        emb = ImageEmbedder()
        a = emb.nhung_hinh_anh("a.jpg")
        b = emb.nhung_hinh_anh("b.jpg")
        assert a != b


class TestPhoBERTMoPhong:
    def test_ma_hoa_mo_phong(self):
        from vietnamese_ai.nlp import PhoBERTWrapper

        pb = PhoBERTWrapper(che_do_mo_phong=True)
        vecs = pb.ma_hoa(["học máy", "học máy"])
        assert vecs.shape == (2, 768)
        assert np.allclose(vecs[0], vecs[1])

    def test_tim_tu_giong(self):
        from vietnamese_ai.nlp import PhoBERTWrapper

        pb = PhoBERTWrapper(che_do_mo_phong=True)
        kq = pb.tim_tu_giong("AI", ["AI Việt Nam", "ẩm thực"], top_n=1)
        assert len(kq) == 1
        assert kq[0][0] == "AI Việt Nam"


class TestDocumentLoadersSplitters:
    def test_text_loader(self, tmp_path):
        from vietnamese_ai.rag.document_loaders import TextLoader

        f = tmp_path / "doc.txt"
        f.write_text("Xin chào Việt Nam.\nDòng hai.", encoding="utf-8")
        docs = TextLoader(str(f)).load()
        assert len(docs) == 1
        assert "Việt Nam" in docs[0].page_content

    def test_recursive_splitter(self):
        from vietnamese_ai.rag.text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5)
        text = "Câu một. Câu hai. Câu ba. " * 10
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)


class TestCollaborationAgents:
    def test_manager_worker_khoi_tao(self):
        from vietnamese_ai.agents.collaboration import ManagerAgent, WorkerAgent

        llm = MagicMock()
        llm.sinh_van_ban = MagicMock(return_value="ok")
        worker = WorkerAgent(
            ten="nghien_cuu",
            mo_ta_chuyen_mon="tìm kiếm",
            llm=llm,
            danh_sach_cong_cu=[],
        )
        manager = ManagerAgent(llm=llm, workers=[worker])
        assert worker.ten == "nghien_cuu"
        assert "nghien_cuu" in manager.workers


class TestDatalakePhienBan:
    def test_commit_va_lich_su(self, tmp_path):
        from vietnamese_ai.datalake.dvc_core import QuanLyPhienBan

        dvc = QuanLyPhienBan(thu_muc_goc=str(tmp_path))
        data = tmp_path / "data.txt"
        data.write_text("hello dvc", encoding="utf-8")
        ma = dvc.commit(str(data), tin_nhan="lần 1")
        assert len(ma) == 64
        lich_su = dvc.lay_lich_su()
        assert len(lich_su) == 1
        assert lich_su[0]["tin_nhan"] == "lần 1"


class TestMobilePlaceholder:
    def test_gguf_placeholder(self, tmp_path):
        from vietnamese_ai.mobile.deployment import TriKhaiDiDong

        tk = TriKhaiDiDong()
        out = tmp_path / "model.gguf"
        path = tk.xuat_gguf_mobile(object(), str(out))
        content = Path(path).read_bytes()
        assert b"PLACEHOLDER" in content or b"GGUF" in content

    def test_executorch_placeholder(self, tmp_path):
        from vietnamese_ai.mobile.deployment import TriKhaiDiDong

        tk = TriKhaiDiDong()
        out = tmp_path / "model.pte"
        path = tk.xuat_executorch(object(), str(out))
        assert b"PLACEHOLDER" in Path(path).read_bytes()


class TestAPIGateway:
    def test_fastapi_server_khoi_tao(self):
        pytest.importorskip("fastapi")
        from vietnamese_ai.api.fastapi_server import FastAPIServer

        def bo_xu_ly(truy_van: str) -> str:
            return f"Echo: {truy_van}"

        server = FastAPIServer(bo_xu_ly=bo_xu_ly)
        assert server.app is not None
        assert server.ten == "VietnameseAI-API"

    def test_gateway_version_route(self):
        pytest.importorskip("fastapi")
        from vietnamese_ai.api.gateway import EvoNetGateway

        server = EvoNetGateway(bo_xu_ly=lambda q: q)
        routes = {getattr(r, "path", None) for r in server.app.routes}
        assert "/sdk/version" in routes
        assert "/sdk/openapi.json" in routes


class TestEvoJIT:
    def test_khoi_tao(self):
        from vietnamese_ai.extreme import EvoJITCompiler

        jit = EvoJITCompiler()
        assert jit.use_openmp is False


class TestWebXRNode:
    def test_telemetry_va_sinh(self):
        from vietnamese_ai.edge.webxr_node import WebXRNode

        node = WebXRNode(session_id="s1")
        node.cap_nhat_telemetry({"toa_do": {"x": 1.0, "y": 2.0, "z": 3.0}})
        assert node.toa_do_hien_tai["x"] == 1.0
        out = node.sinh_van_ban("xin chào")
        assert "WebXR" in out


class TestSecurityMiddleware:
    def test_khoi_tao_khong_fastapi(self):
        pytest.importorskip("fastapi")
        from starlette.applications import Starlette

        from vietnamese_ai.api.security_middleware import AISecurityMiddleware

        app = Starlette()
        mw = AISecurityMiddleware(app, ghi_log=False)
        assert mw.tuong_lua is not None
