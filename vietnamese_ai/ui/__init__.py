from typing import Any

from .components import (
    BangThongKe,
    BieuDoTriThuc,
    BoNhoTrinhDuyet,
    CameraVaMicro,
    ChiaSeManHinh,
    DieuKhienIoT,
    DinhViGPS,
    GiaoDienChat,
    KhuVucTaiTaiLieu,
    KinhThucTeAo,
    TinhToanGPU,
    TrangThaiPin,
    TrinhChieu3D,
    TroLyGiongNoi,
)
from .core import Component, KieuDang, UIApp

_LAZY_UI_EXPORTS = {
    "StudioDashboard": ("vietnamese_ai.ui.studio_dashboard", "StudioDashboard"),
    "MayChuStudioDashboard": ("vietnamese_ai.ui.studio_dashboard", "MayChuStudioDashboard"),
    "QuanLyPhienChatWorkspace": ("vietnamese_ai.ui.workspace_store", "QuanLyPhienChatWorkspace"),
    "StudioWorkspaceManager": ("vietnamese_ai.ui.workspace_store", "StudioWorkspaceManager"),
    "UngDungWeb": ("vietnamese_ai.ui.web_app", "UngDungWeb"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_UI_EXPORTS:
        import importlib

        mod_path, attr = _LAZY_UI_EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'vietnamese_ai.ui' has no attribute {name!r}")


__all__ = [
    "Component",
    "UIApp",
    "KieuDang",
    "GiaoDienChat",
    "BangThongKe",
    "KhuVucTaiTaiLieu",
    "CameraVaMicro",
    "BieuDoTriThuc",
    "TroLyGiongNoi",
    "ChiaSeManHinh",
    "DinhViGPS",
    "BoNhoTrinhDuyet",
    "TrinhChieu3D",
    "TinhToanGPU",
    "DieuKhienIoT",
    "TrangThaiPin",
    "KinhThucTeAo",
    "StudioDashboard",
    "MayChuStudioDashboard",
    "QuanLyPhienChatWorkspace",
    "StudioWorkspaceManager",
    "UngDungWeb",
]
