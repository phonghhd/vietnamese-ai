"""Kiểm thử Giai đoạn 1: Tái cấu trúc Core/Extensions, vá eval()/Sandbox, nhãn tính năng và vệ sinh dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from vietnamese_ai.agents import danh_gia_bieu_thuc_toan
from vietnamese_ai.agents.code_healing import TacTuLapTrinhTuSua, _KiemTraASTAnToan
from vietnamese_ai.edge.c_bridge import CauNoiCtypesNative
from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge
from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU
from vietnamese_ai.extreme.cpu_kernel import EvoKernelCPU
from vietnamese_ai.extreme.jit_engine import EvoJITCompiler
from vietnamese_ai.extreme.kernel_fusion import FusedDenseNormReLU
from vietnamese_ai.mobile.deployment import TriKhaiDiDong
from vietnamese_ai.mobile.sdk_generator import SDKGenerator
from vietnamese_ai.sandbox.executor import ThucThiDocLap
from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy
from vietnamese_ai.security.blue_team import BlueTeamAgent
from vietnamese_ai.security.red_team import RedTeamAgent, RedTeamSimulator
from vietnamese_ai.utils.feature_status import (
    BAN_DO_MODULAR,
    TrangThaiTinhNang,
    lay_thong_tin_tinh_nang,
    liet_ke_tinh_nang,
)
from vietnamese_ai.utils.io_utils import RestrictedUnpickler
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc


def test_phan_tach_core_va_extensions_lazy_khong_nap_som():
    code = (
        "import sys; "
        "import vietnamese_ai.core, vietnamese_ai.core.edge, vietnamese_ai.core.preprocessing, "
        "vietnamese_ai.extensions, vietnamese_ai.extensions.vision, "
        "vietnamese_ai.extensions.timeseries, vietnamese_ai.extensions.federated, "
        "vietnamese_ai.extensions.fine_tuning, vietnamese_ai.extensions.transformer, "
        "vietnamese_ai.extensions.llm, vietnamese_ai.extensions.nlp, "
        "vietnamese_ai.extensions.mobile, vietnamese_ai.extensions.web3, "
        "vietnamese_ai.extensions.robotics, vietnamese_ai.extensions.extreme; "
        "assert 'numpy' not in sys.modules, 'numpy bị nạp sớm!'; "
        "assert 'sklearn' not in sys.modules, 'sklearn bị nạp sớm!'; "
        "assert 'torch' not in sys.modules, 'torch bị nạp sớm!'; "
        "print('LAZY_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "LAZY_OK"


def test_core_va_extensions_reexport_dong_nhat():
    from vietnamese_ai.core import (
        BoGiaiLuongTuQ4,
        Engine,
        GGUFMemoryMappedStreamer,
        Pipeline,
        XuLySo,
    )
    from vietnamese_ai.core import (
        CauNoiCtypesNative as CoreCBridge,
    )
    from vietnamese_ai.core import (
        DongCoSuyLuanEdge as CoreEdgeEngine,
    )
    from vietnamese_ai.core.edge import DongCoSuyLuanEdge as CoreEdgeSub
    from vietnamese_ai.core.preprocessing import XuLySo as CoreXuLySo
    from vietnamese_ai.extensions import DANH_SACH_EXTENSIONS
    from vietnamese_ai.extensions.extreme import EvoJITCompiler as ExtJIT
    from vietnamese_ai.extensions.federated import HocLienKet
    from vietnamese_ai.extensions.fine_tuning import BoTinhChinhMoHinh
    from vietnamese_ai.extensions.llm import VietnameseLLM
    from vietnamese_ai.extensions.mobile import TriKhaiDiDong as ExtMobile
    from vietnamese_ai.extensions.nlp import BoTachTuTiengVietSieuToc
    from vietnamese_ai.extensions.robotics import giai_dong_hoc_nguoc_3dof
    from vietnamese_ai.extensions.timeseries import DuDoanChuoiThoiGian
    from vietnamese_ai.extensions.transformer import GPTModel
    from vietnamese_ai.extensions.vision import PhanLoaiHinhAnh
    from vietnamese_ai.extensions.web3 import P2PTracker, TokenLedger

    assert Engine is not None
    assert Pipeline is not None
    assert XuLySo is CoreXuLySo
    assert CoreEdgeEngine is CoreEdgeSub is DongCoSuyLuanEdge
    assert CoreCBridge is CauNoiCtypesNative
    assert BoGiaiLuongTuQ4 is not None
    assert GGUFMemoryMappedStreamer is not None
    assert ExtJIT is EvoJITCompiler
    assert ExtMobile is TriKhaiDiDong
    assert all(
        obj is not None
        for obj in (
            PhanLoaiHinhAnh,
            DuDoanChuoiThoiGian,
            HocLienKet,
            BoTinhChinhMoHinh,
            GPTModel,
            VietnameseLLM,
            BoTachTuTiengVietSieuToc,
            P2PTracker,
            TokenLedger,
            giai_dong_hoc_nguoc_3dof,
        )
    )
    assert "core" in BAN_DO_MODULAR
    assert set(DANH_SACH_EXTENSIONS.keys()) >= {
        "vision",
        "timeseries",
        "federated",
        "fine_tuning",
        "transformer",
        "llm",
        "nlp",
        "mobile",
        "web3",
        "robotics",
        "extreme",
    }


def test_va_lo_hong_eval_readme_va_parser_an_toan():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "return eval(bieu_thuc)" not in readme

    assert danh_gia_bieu_thuc_toan("15 * 8") == 120
    assert danh_gia_bieu_thuc_toan("sqrt(16) + 2 ** 3") == 12.0

    for payload_doc in (
        "__import__('os').system('id')",
        "eval('1+1')",
        "exec('x=1')",
        "open('/etc/passwd')",
        "().__class__.__bases__",
    ):
        with pytest.raises(ValueError):
            danh_gia_bieu_thuc_toan(payload_doc)


def test_code_healing_sandbox_cach_ly_chan_dunder_escape():
    checker = _KiemTraASTAnToan()
    for payload in (
        "x = ().__class__.__bases__[0].__subclasses__()",
        "x = getattr(__builtins__, 'open')",
        "import importlib\nimportlib.import_module('os')",
    ):
        an_toan, _ = checker.kiem_tra(payload)
        assert an_toan is False

    agent = TacTuLapTrinhTuSua(so_lan_thu_lai_toi_da=1)
    agent.sinh_ma_fn = lambda _: "def hack():\n    return ().__class__.__mro__\n"
    kq = agent.lap_trinh_tu_sua("Viết hàm hack")
    assert kq["thanh_cong"] is False
    assert "thuộc tính hệ thống" in kq["loi_traceback"]

    kq_ok = TacTuLapTrinhTuSua(so_lan_thu_lai_toi_da=2).lap_trinh_tu_sua(
        "Viết hàm tinh_khoang_cach_euclid(x, y)"
    )
    assert kq_ok["thanh_cong"] is True


def test_restricted_unpickler_chan_marshal():
    assert "marshal" in RestrictedUnpickler.BLOCKED_MODULES
    assert " marshal" not in RestrictedUnpickler.BLOCKED_MODULES


def test_gan_nhan_tinh_nang_thu_nghiem_va_phu_thuoc_ngoai():
    meta_mobile = lay_thong_tin_tinh_nang(TriKhaiDiDong)
    assert meta_mobile is not None
    assert meta_mobile["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL
    assert meta_mobile["la_mo_phong"] is True
    assert len(meta_mobile["phu_thuoc_ngoai"]) >= 3

    for method in (
        TriKhaiDiDong.xuat_tflite,
        TriKhaiDiDong.xuat_coreml,
        TriKhaiDiDong.xuat_onnx_mobile,
        TriKhaiDiDong.xuat_executorch,
        TriKhaiDiDong.benchmark_edge,
    ):
        meta_m = lay_thong_tin_tinh_nang(method)
        assert meta_m is not None
        assert meta_m["trang_thai"] in (
            TrangThaiTinhNang.EXPERIMENTAL,
            TrangThaiTinhNang.SIMULATED,
            TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
        )

    assert lay_thong_tin_tinh_nang(SDKGenerator)["la_mo_phong"] is True
    assert lay_thong_tin_tinh_nang(RedTeamSimulator)["la_mo_phong"] is True
    assert lay_thong_tin_tinh_nang(RedTeamAgent)["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL
    assert lay_thong_tin_tinh_nang(BlueTeamAgent)["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL
    assert lay_thong_tin_tinh_nang(ThucThiDocLap)["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL
    assert lay_thong_tin_tinh_nang(MoiTruongCachLy)["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL

    meta_jit = lay_thong_tin_tinh_nang(EvoJITCompiler)
    assert meta_jit is not None
    assert meta_jit["trang_thai"] == TrangThaiTinhNang.EXTERNAL_DEPENDENCY
    assert any("g++" in dep for dep in meta_jit["phu_thuoc_ngoai"])
    assert lay_thong_tin_tinh_nang(EvoKernelCPU)["trang_thai"] == TrangThaiTinhNang.EXTERNAL_DEPENDENCY
    assert (
        lay_thong_tin_tinh_nang(FusedDenseNormReLU)["trang_thai"]
        == TrangThaiTinhNang.EXTERNAL_DEPENDENCY
    )

    assert lay_thong_tin_tinh_nang(DongCoSuyLuanEdge)["trang_thai"] == TrangThaiTinhNang.CORE
    assert (
        lay_thong_tin_tinh_nang(CauNoiCtypesNative)["trang_thai"]
        == TrangThaiTinhNang.EXTERNAL_DEPENDENCY
    )
    assert lay_thong_tin_tinh_nang(DieuPhoiSwarmCPU)["trang_thai"] == TrangThaiTinhNang.EXPERIMENTAL
    assert (
        lay_thong_tin_tinh_nang(DieuPhoiSwarmCPU.khoi_tao_cum_gia_lap)["trang_thai"]
        == TrangThaiTinhNang.SIMULATED
    )
    assert len(liet_ke_tinh_nang()) >= 10


def test_khong_xung_dot_phu_thuoc_va_tach_requirements():
    req_prod = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert req_prod == ["numpy>=1.20.0,<3.0.0"]

    req_dev_path = Path("requirements-dev.txt")
    assert req_dev_path.is_file()
    req_dev_text = req_dev_path.read_text(encoding="utf-8")
    assert "-r requirements.txt" in req_dev_text
    assert "pytest>=" in req_dev_text
    assert "ruff>=" in req_dev_text
    assert "mypy>=" in req_dev_text

    bao_cao = QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc()
    assert bao_cao["hop_le"] is True, f"Phát hiện xung đột phụ thuộc: {bao_cao['xung_dot']}"

    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    for extra in (
        "vietnamese-nlp =",
        "vision =",
        "export =",
        "mobile =",
        "web3 =",
        "robotics =",
        "tracking =",
        "local-llm =",
        "grpc =",
    ):
        assert extra in pyproject_text

    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(pyproject_text)
    opt: Dict[str, List[str]] = data["project"]["optional-dependencies"]
    specs_by_pkg: Dict[str, set] = {}
    for group_specs in [data["project"]["dependencies"], *opt.values()]:
        for raw in group_specs:
            base = raw.split(";")[0].strip()
            pkg = (
                base.split(">=")[0]
                .split("<=")[0]
                .split("==")[0]
                .split("<")[0]
                .split(">")[0]
                .split("[")[0]
                .strip()
                .lower()
            )
            specs_by_pkg.setdefault(pkg, set()).add(base)

    conflicts = {pkg: sorted(specs) for pkg, specs in specs_by_pkg.items() if len(specs) > 1}
    assert conflicts == {}, f"Xung đột phiên bản trong pyproject.toml: {conflicts}"


def test_security_md_cap_nhat_minh_bach():
    sec = Path("SECURITY.md").read_text(encoding="utf-8")
    assert "31.0.x" in sec
    assert "External Toolchain Requirements" in sec
    assert "RedTeamSimulator" in sec
    assert "ThucThiDocLap" in sec
    assert "g++" in sec
