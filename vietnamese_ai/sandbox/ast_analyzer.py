"""
PhanTichAST (AST Analyzer) - Lớp khiên phòng thủ tĩnh đầu tiên.
Sử dụng Abstract Syntax Tree (ast) để quét và chặn các câu lệnh nguy hiểm
trước khi mã được đưa vào Executor.
"""

import ast
import logging

logger = logging.getLogger("V-Sandbox")


class LoiAnNinh(Exception):
    """Lỗi sinh ra khi phát hiện mã độc hại."""
    pass


class ASTKhachViengTham(ast.NodeVisitor):
    """Đi dạo qua cây AST để phát hiện mã độc."""

    MODULE_CAM = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 
        'urllib', 'requests', 'pty', 'builtins', 'importlib',
        'ctypes', 'tempfile', 'threading', 'multiprocessing'
    }

    HAM_CAM = {
        'eval', 'exec', 'open', 'compile', 'globals', 'locals',
        '__import__', 'setattr', 'delattr', 'getattr'
    }

    def visit_Import(self, node):
        """Bắt lỗi khi dùng lệnh import trực tiếp."""
        for alias in node.names:
            tu_khoa = alias.name.split('.')[0]
            if tu_khoa in self.MODULE_CAM:
                raise LoiAnNinh(f"Cấm sử dụng module nguy hiểm: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Bắt lỗi khi dùng lệnh from ... import."""
        if node.module:
            tu_khoa = node.module.split('.')[0]
            if tu_khoa in self.MODULE_CAM:
                raise LoiAnNinh(f"Cấm sử dụng module nguy hiểm: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        """Bắt lỗi khi gọi các hàm dựng sẵn (built-in) nguy hiểm."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.HAM_CAM:
                raise LoiAnNinh(f"Cấm sử dụng hàm nguy hiểm: '{node.func.id}()'")
                
        # Ngăn chặn sử dụng thuộc tính dunder như __class__, __subclasses__
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                raise LoiAnNinh(f"Cấm truy cập thuộc tính dunder: '{node.func.attr}'")
                
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Chặn truy cập vào các dunder attributes để leo thang đặc quyền."""
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise LoiAnNinh(f"Cấm truy cập thuộc tính hệ thống: '{node.attr}'")
        self.generic_visit(node)


class PhanTichAST:
    """Trình quét và phân tích mã độc tĩnh."""

    @staticmethod
    def kiem_tra(ma_nguon: str) -> bool:
        """
        Quét mã nguồn xem có an toàn để chạy không.
        
        Args:
            ma_nguon: Đoạn code Python cần chạy.
            
        Returns:
            True nếu an toàn.
            
        Raises:
            LoiAnNinh nếu phát hiện mã độc.
            SyntaxError nếu code bị lỗi cú pháp.
        """
        try:
            cay_ast = ast.parse(ma_nguon)
            khach = ASTKhachViengTham()
            khach.visit(cay_ast)
            return True
        except SyntaxError as e:
            logger.error(f"[PhanTichAST] Lỗi cú pháp: {e}")
            raise
