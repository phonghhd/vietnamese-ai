"""Bộ kiểm thử hồi quy cho Giai đoạn P3 — Phương án A (Deprecation mềm v31.2.0, giữ tương thích ngược)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    [
        "vietnamese_ai.edge.wasm_node",
        "vietnamese_ai.edge.webxr_node",
        "vietnamese_ai.edge.tee_zkp_node",
        "vietnamese_ai.mobile.mobile_tools",
        "vietnamese_ai.salm",
        "vietnamese_ai.federated",
    ],
)
def test_p3_soft_deprecation_warnings_on_simulated_modules(module_name: str) -> None:
    """Xác minh 6 module mô phỏng/thử nghiệm phát DeprecationWarning khi nạp nhưng vẫn tương thích ngược."""
    code = (
        "import importlib, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always', DeprecationWarning)\n"
        f"    importlib.import_module({module_name!r})\n"
        "assert any(issubclass(w.category, DeprecationWarning) for w in caught), caught\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_p3_evonet_os_app_js_syntax_and_deprecation_docs() -> None:
    """Xác minh evonet_os/app.js không còn lỗi cú pháp template literal và có tài liệu deprecation rõ ràng."""
    app_js = ROOT_DIR / "evonet_os" / "app.js"
    noi_dung = app_js.read_text(encoding="utf-8")
    assert r"\`" not in noi_dung
    assert r"\${" not in noi_dung

    if shutil.which("node"):
        proc = subprocess.run(
            ["node", "--check", str(app_js)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0, proc.stderr

    assert (ROOT_DIR / "sdks" / "README.md").is_file()
    assert (ROOT_DIR / "evonet_os" / "README.md").is_file()


def test_p3_finance_moved_to_plugins_finance_with_compat_shims() -> None:
    """Xác minh nhóm tài chính đã chuyển sang vietnamese_ai.plugins.finance và giữ shim tương thích ngược."""
    with warnings.catch_warnings(record=True) as clean_warnings:
        warnings.simplefilter("always", DeprecationWarning)
        import vietnamese_ai.plugin as plugin_pkg
        import vietnamese_ai.plugins.finance as fin_pkg
        from vietnamese_ai.plugins.plugin_chungkhoan_crypto import PluginChungKhoanCrypto

        assert plugin_pkg.QuanLyPlugin is not None
        assert PluginChungKhoanCrypto is not None
        assert fin_pkg.BoDieuPhoiThiTruong is not None
        assert fin_pkg.BoDuDoanXuHuongNganHan is not None
    assert not [w for w in clean_warnings if issubclass(w.category, DeprecationWarning)]

    for sub in ("market_feed", "market_predictor", "micro_features", "quant_indicators"):
        legacy_mod = importlib.import_module(f"vietnamese_ai.plugin.{sub}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.reload(legacy_mod)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    import vietnamese_ai.plugin as plugin_pkg
    import vietnamese_ai.plugins.finance as fin_pkg

    with warnings.catch_warnings(record=True) as lazy_caught:
        warnings.simplefilter("always", DeprecationWarning)
        cls_legacy = getattr(plugin_pkg, "BoDieuPhoiThiTruong")
    assert cls_legacy is fin_pkg.BoDieuPhoiThiTruong
    assert any(issubclass(w.category, DeprecationWarning) for w in lazy_caught)


def test_p3_legal_namespaces_full_parity_with_rag_legal() -> None:
    """Xác minh vietnamese_ai.legal và vietnamese_ai.legal.rag đồng bộ 100% exports với vietnamese_ai.rag.legal."""
    import vietnamese_ai.legal as legal_ns
    import vietnamese_ai.legal.rag as legal_rag_ns
    import vietnamese_ai.rag.legal as rag_legal_ns

    assert set(legal_ns.__all__) == set(rag_legal_ns.__all__)
    assert set(legal_rag_ns.__all__) == set(rag_legal_ns.__all__)

    for sym in (
        "BoTruyHoiPhapLyThoiGian",
        "BoPhanTichThoiGianPhapLy",
        "BoKiemChungTrichDanPhapLy",
        "BoNapVanBanPhapQuyTuDong",
        "PhienBanDieuKhoan",
    ):
        assert getattr(legal_ns, sym) is getattr(rag_legal_ns, sym)
        assert getattr(legal_rag_ns, sym) is getattr(rag_legal_ns, sym)


def test_p3_subpackages_depend_on_core_and_release_check_v31_2_0() -> None:
    """Xác minh 5 gói con phụ thuộc vietnamese-ai>=31.3.0,<32.0.0 và giữ nguyên 349 public APIs."""
    import vietnamese_ai
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    assert vietnamese_ai.__version__ == "31.3.0"
    assert len(vietnamese_ai.__all__) == 349

    for ten_goi in (
        "vietnamese-ai-nlp",
        "vietnamese-ai-edge",
        "vietnamese-ai-legal",
        "vietnamese-ai-zkml",
        "vietnamese-ai-classic-ml",
    ):
        toml_txt = (ROOT_DIR / "packages" / ten_goi / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "31.3.0"' in toml_txt
        assert '"vietnamese-ai>=31.3.0,<32.0.0"' in toml_txt

    bc = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh(ROOT_DIR)
    assert bc["san_sang_phat_hanh"] is True
    assert bc["phien_ban"] == "31.3.0"
    assert bc["so_bieu_tuong_cong_khai"] == 349
    assert bc["dong_bo_du_lieu_phap_ly"] is True
