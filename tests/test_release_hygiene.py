"""Kiểm tra vệ sinh release v30: version, export, packaging."""

from pathlib import Path

import vietnamese_ai


class TestVersionDongBo:
    def test_package_version(self):
        assert vietnamese_ai.__version__ == "30.1.0"

    def test_pyproject_version(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "30.1.0"' in text

    def test_setup_version(self):
        text = Path("setup.py").read_text(encoding="utf-8")
        assert 'version="30.1.0"' in text


class TestDieuPhoiCum:
    def test_import_dieu_phoi_cum(self):
        from vietnamese_ai import DieuPhoiCum
        from vietnamese_ai.orchestrator import NutChinh

        assert DieuPhoiCum is NutChinh

    def test_khoi_tao(self):
        from vietnamese_ai import DieuPhoiCum

        cum = DieuPhoiCum(port=18080)
        assert cum.port == 18080


class TestModuleExports:
    def test_core_exports(self):
        from vietnamese_ai.core import Engine, KiemDinhCheo, Pipeline, TimKiemThamSo

        assert Engine is not None
        assert Pipeline is not None
        assert KiemDinhCheo is not None
        assert TimKiemThamSo is not None

    def test_nlp_exports(self):
        from vietnamese_ai.nlp import PhanTichCamXuc, PhoBERTWrapper

        assert PhanTichCamXuc is not None
        assert PhoBERTWrapper is not None

    def test_serving_exports(self):
        from vietnamese_ai.serving import BlockManager, SpeculativeEngine

        assert SpeculativeEngine is not None
        assert BlockManager is not None

    def test_enterprise_exports(self):
        from vietnamese_ai.enterprise import NenTangDichVu

        assert NenTangDichVu is not None

    def test_extreme_package(self):
        from vietnamese_ai.extreme import EvoJITCompiler

        assert EvoJITCompiler is not None

    def test_agents_memgpt(self):
        from vietnamese_ai.agents import HethongNhoMemGPT

        assert HethongNhoMemGPT is not None


class TestPyprojectExtras:
    def test_extras_llm_api_rag(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        assert "llm =" in text or "llm=" in text
        assert "api =" in text or "api=" in text
        assert "rag =" in text or "rag=" in text
        assert "chromadb" in text
        assert "fastapi" in text
