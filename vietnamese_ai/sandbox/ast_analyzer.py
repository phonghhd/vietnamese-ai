"""
PhanTichAST (AST Analyzer) - Lớp khiên phòng thủ tĩnh đầu tiên.
Sử dụng Abstract Syntax Tree (ast) để quét và chặn các câu lệnh nguy hiểm
trước khi mã được đưa vào Executor.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Iterable, Optional, Set

logger = logging.getLogger("V-Sandbox")

_RE_FORMAT_ESCAPE = re.compile(
    r"\{[^}]*(?:__|\b(?:gi_frame|gi_code|cr_frame|ag_frame|f_back|f_globals|f_locals|f_builtins|f_code|tb_frame|tb_next|func_globals|func_code)\b)"
)


class LoiAnNinh(Exception):
    """Lỗi sinh ra khi phát hiện mã độc hại."""

    pass


class ASTKhachViengTham(ast.NodeVisitor):
    """Đi dạo qua cây AST để phát hiện mã độc."""

    MODULE_CAM = {
        "os",
        "posix",
        "nt",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "pty",
        "builtins",
        "importlib",
        "ctypes",
        "tempfile",
        "threading",
        "multiprocessing",
        "io",
        "pathlib",
        "operator",
        "inspect",
        "gc",
        "signal",
        "resource",
        "codecs",
        "asyncio",
        "pickle",
        "marshal",
        "shelve",
        "linecache",
        "pdb",
        "runpy",
        "webbrowser",
        "sqlite3",
        "code",
        "codeop",
        "pkgutil",
        "zipimport",
        "venv",
    }

    HAM_CAM = {
        "eval",
        "exec",
        "open",
        "compile",
        "globals",
        "locals",
        "__import__",
        "setattr",
        "delattr",
        "getattr",
        "type",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "memoryview",
    }

    THUOC_TINH_CAM = {
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "tb_frame",
        "tb_next",
        "func_globals",
        "func_code",
    }

    def __init__(
        self,
        module_cam_bo_sung: Optional[Iterable[str]] = None,
        ham_cam_bo_sung: Optional[Iterable[str]] = None,
    ) -> None:
        self._module_cam: Set[str] = set(self.MODULE_CAM)
        if module_cam_bo_sung:
            self._module_cam.update(module_cam_bo_sung)
        self._ham_cam: Set[str] = set(self.HAM_CAM)
        if ham_cam_bo_sung:
            self._ham_cam.update(ham_cam_bo_sung)

    def visit_Import(self, node: ast.Import) -> None:
        """Bắt lỗi khi dùng lệnh import trực tiếp."""
        for alias in node.names:
            tu_khoa = alias.name.split(".")[0]
            if tu_khoa in self._module_cam:
                raise LoiAnNinh(f"Không được phép import module '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Bắt lỗi khi dùng lệnh from ... import."""
        if node.module:
            tu_khoa = node.module.split(".")[0]
            if tu_khoa in self._module_cam:
                raise LoiAnNinh(f"Không được phép import từ module '{node.module}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Chặn mọi tham chiếu gián tiếp tới hàm dựng sẵn nguy hiểm (ví dụ `f = open`)."""
        if isinstance(node.ctx, ast.Load) and node.id in self._ham_cam:
            raise LoiAnNinh(
                f"Không được phép sử dụng hàm '{node.id}' (Cấm sử dụng hàm nguy hiểm: '{node.id}()')"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Bắt lỗi khi gọi các hàm dựng sẵn (built-in) hoặc phương thức nguy hiểm."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self._ham_cam:
                raise LoiAnNinh(
                    f"Không được phép sử dụng hàm '{node.func.id}' (Cấm sử dụng hàm nguy hiểm: '{node.func.id}()')"
                )

        if isinstance(node.func, ast.Attribute):
            if (
                node.func.attr.startswith("__")
                and node.func.attr.endswith("__")
                and node.func.attr != "__init__"
            ):
                raise LoiAnNinh(f"Cấm truy cập thuộc tính dunder: '{node.func.attr}'")
            if node.func.attr in self.THUOC_TINH_CAM:
                raise LoiAnNinh(f"Cấm truy cập thuộc tính nội bộ nguy hiểm: '{node.func.attr}'")
            if node.func.attr in ("format", "format_map"):
                if isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
                    if _RE_FORMAT_ESCAPE.search(node.func.value.value):
                        raise LoiAnNinh("Cấm dùng chuỗi định dạng truy cập thuộc tính nội bộ Python.")
                else:
                    raise LoiAnNinh(
                        "Cấm gọi phương thức format/format_map trên biểu thức động không phải hằng chuỗi."
                    )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Chặn truy cập vào các dunder và frame/generator attributes để leo thang đặc quyền."""
        if node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__":
            raise LoiAnNinh(f"Cấm truy cập thuộc tính hệ thống: '{node.attr}'")
        if node.attr in self.THUOC_TINH_CAM:
            raise LoiAnNinh(f"Cấm truy cập thuộc tính nội bộ nguy hiểm: '{node.attr}'")
        self.generic_visit(node)


class PhanTichAST:
    """Trình quét và phân tích mã độc tĩnh."""

    @staticmethod
    def kiem_tra(
        ma_nguon: str,
        module_cam_bo_sung: Optional[Iterable[str]] = None,
        ham_cam_bo_sung: Optional[Iterable[str]] = None,
    ) -> bool:
        """
        Quét mã nguồn xem có an toàn để chạy không.

        Args:
            ma_nguon: Đoạn code Python cần chạy.
            module_cam_bo_sung: Tập module bổ sung cần chặn ngoài danh sách mặc định.
            ham_cam_bo_sung: Tập hàm dựng sẵn bổ sung cần chặn ngoài danh sách mặc định.

        Returns:
            True nếu an toàn.

        Raises:
            LoiAnNinh nếu phát hiện mã độc.
            SyntaxError nếu code bị lỗi cú pháp.
        """
        try:
            cay_ast = ast.parse(ma_nguon)
            khach = ASTKhachViengTham(
                module_cam_bo_sung=module_cam_bo_sung,
                ham_cam_bo_sung=ham_cam_bo_sung,
            )
            khach.visit(cay_ast)
            return True
        except SyntaxError as e:
            logger.error(f"[PhanTichAST] Lỗi cú pháp: {e}")
            raise
