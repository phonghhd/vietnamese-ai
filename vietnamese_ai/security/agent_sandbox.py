import os
from typing import Tuple

from vietnamese_ai.sandbox.ast_analyzer import ASTKhachViengTham, LoiAnNinh, PhanTichAST
from vietnamese_ai.sandbox.executor import ThucThiDocLap
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
    mo_ta=(
        "Sandbox mức ứng dụng (PhanTichAST + ThucThiDocLap spawn + safe_builtins + POSIX setrlimit); "
        "không thay thế cách ly cấp hệ điều hành/container (Docker, gVisor, seccomp)."
    ),
)
class MoiTruongCachLy:
    """
    Môi trường cách ly mức ứng dụng (Agent Sandbox) dùng `PhanTichAST` và `ThucThiDocLap`.

    Giới hạn bảo mật: Đây là lớp bảo vệ phòng thủ chiều sâu ở tầng Python, không áp đặt
    cgroup/seccomp/network namespace ở cấp nhân OS. Khi thực thi mã không tin cậy trên
    môi trường production, bắt buộc kết hợp container cách ly (Docker, gVisor, Firecracker).
    """

    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "sandbox_rules.json")

    @classmethod
    def _tai_cau_hinh(cls) -> dict:
        import json

        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "module_cam": sorted(ASTKhachViengTham.MODULE_CAM),
            "ham_cam": sorted(ASTKhachViengTham.HAM_CAM),
        }

    @classmethod
    def _kiem_tra_ast(cls, ma_nguon: str) -> Tuple[bool, str]:
        """Phân tích AST thống nhất qua `PhanTichAST` kèm quy tắc bổ sung từ cấu hình Sandbox."""
        cau_hinh = cls._tai_cau_hinh()
        module_cam = cau_hinh.get("module_cam", [])
        ham_cam = cau_hinh.get("ham_cam", [])
        try:
            PhanTichAST.kiem_tra(
                ma_nguon,
                module_cam_bo_sung=module_cam,
                ham_cam_bo_sung=ham_cam,
            )
            return True, "An toàn"
        except SyntaxError as e:
            return False, f"Lỗi cú pháp: {e}"
        except LoiAnNinh as e:
            return False, f"Bảo mật: {e}"

    @classmethod
    def thuc_thi(cls, ma_nguon: str, timeout_giay: int = 5, gioi_han_ram_mb: int = 256) -> str:
        """
        Thực thi mã nguồn trong tiến trình cách ly `ThucThiDocLap` với builtins giới hạn và `setrlimit`.

        Args:
            ma_nguon (str): Mã Python cần chạy.
            timeout_giay (int): Thời gian chờ tối đa.
            gioi_han_ram_mb (int): Giới hạn RAM bổ sung tối đa (MB).

        Returns:
            str: Kết quả in ra (stdout) hoặc thông báo lỗi.
        """
        an_toan, ly_do = cls._kiem_tra_ast(ma_nguon)
        if not an_toan:
            return f"Lỗi bảo mật (AST): {ly_do}"

        cau_hinh = cls._tai_cau_hinh()
        executor = ThucThiDocLap(timeout_giay=timeout_giay, gioi_han_ram_mb=gioi_han_ram_mb)
        ok, du_lieu, out = executor.chay_kem_stdout(
            ma_nguon,
            module_cam_bo_sung=cau_hinh.get("module_cam", []),
            ham_cam_bo_sung=cau_hinh.get("ham_cam", []),
        )
        if ok:
            return (
                out
                if out.strip()
                else "Mã đã chạy thành công nhưng không có kết quả in ra (không dùng print)."
            )
        msg = str(du_lieu)
        if "Timeout" in msg or "vượt quá thời gian" in msg.lower() or "quá thời gian" in msg.lower():
            return (
                f"Lỗi bảo mật: Mã thực thi vượt quá thời gian cho phép ({timeout_giay} giây) "
                "(Có thể do lặp vô hạn)."
            )
        if msg.startswith("Lỗi An Ninh:"):
            return f"Lỗi bảo mật (AST): {msg}"
        return f"Lỗi khi chạy mã:\n{msg}"
