import ast
import os
import subprocess
import tempfile
from typing import Tuple


class MoiTruongCachLy:
    """
    Môi trường cách ly (Sandbox) sử dụng subprocess để chạy mã Python do LLM sinh ra.
    Ngăn chặn việc gọi các lệnh hệ thống nguy hiểm bằng cách:
    1. Kiểm tra Cây cú pháp (AST) trước khi chạy để chặn các module cấm.
    2. Chạy trên tiến trình riêng biệt với timeout.
    """

    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "sandbox_rules.json")

    @classmethod
    def _tai_cau_hinh(cls) -> dict:
        import json

        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "module_cam": ["os", "sys", "subprocess", "shutil", "socket", "requests", "urllib"],
            "ham_cam": ["eval", "exec", "open", "__import__"],
        }

    @classmethod
    def _kiem_tra_ast(cls, ma_nguon: str) -> Tuple[bool, str]:
        """Phân tích AST để tìm các module hoặc hàm cấm."""
        cau_hinh = cls._tai_cau_hinh()
        try:
            cay = ast.parse(ma_nguon)
        except SyntaxError as e:
            return False, f"Lỗi cú pháp: {e}"

        for node in ast.walk(cay):
            # Chặn import statement
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in cau_hinh.get("module_cam", []):
                        return False, f"Bảo mật: Không được phép import module '{alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in cau_hinh.get("module_cam", []):
                    return False, f"Bảo mật: Không được phép import từ module '{node.module}'"

            # Chặn các built-in nguy hiểm (vd: eval, exec, open)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cau_hinh.get("ham_cam", []):
                        return False, f"Bảo mật: Không được phép sử dụng hàm '{node.func.id}'"

        return True, "An toàn"

    @classmethod
    def thuc_thi(cls, ma_nguon: str, timeout_giay: int = 5) -> str:
        """
        Thực thi mã nguồn trong môi trường cách ly (subprocess).

        Args:
            ma_nguon (str): Mã Python cần chạy.
            timeout_giay (int): Thời gian chờ tối đa.

        Returns:
            str: Kết quả in ra (stdout) hoặc thông báo lỗi.
        """
        # Bước 1: Phân tích cú pháp để phát hiện mã độc
        an_toan, ly_do = cls._kiem_tra_ast(ma_nguon)
        if not an_toan:
            return f"Lỗi bảo mật (AST): {ly_do}"

        # Bước 2: Tạo file tạm và chạy qua subprocess
        fd, path = tempfile.mkstemp(suffix=".py", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(ma_nguon)

            import sys

            # Chạy tiến trình con, tắt khả năng gọi mạng nếu có cấu hình OS (chưa áp dụng ở mức OS, chỉ giới hạn timeout)
            ket_qua = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=timeout_giay
            )

            if ket_qua.returncode == 0:
                out = ket_qua.stdout
                return (
                    out
                    if out.strip()
                    else "Mã đã chạy thành công nhưng không có kết quả in ra (không dùng print)."
                )
            else:
                return f"Lỗi khi chạy mã:\n{ket_qua.stderr}"

        except subprocess.TimeoutExpired:
            return f"Lỗi bảo mật: Mã thực thi vượt quá thời gian cho phép ({timeout_giay} giây) (Có thể do lặp vô hạn)."
        except Exception as e:
            return f"Lỗi hệ thống khi chạy sandbox: {str(e)}"
        finally:
            if os.path.exists(path):
                os.remove(path)
