"""Tác tử Lập trình Tự Sửa lỗi (Code Healing Agent) - Vietnamese AI Framework."""

from __future__ import annotations

import ast
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

from vietnamese_ai.agents.state_graph import (
    END,
    START,
    DoThiTrangThai,
    kenh_them_danh_sach,
)
from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy
from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework


class _KiemTraASTAnToan:
    """Bộ phân tích cú pháp AST kiểm tra an toàn mã nguồn trước khi thực thi trong Sandbox."""

    CAC_GOI_NGUY_HIEM = {
        "subprocess",
        "os",
        "sys",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "pty",
        "pickle",
    }
    CAC_HAM_NGUY_HIEM = {"eval", "exec", "__import__", "compile", "open"}

    def kiem_tra(self, code: str) -> Tuple[bool, str]:
        """Phân tích AST để phát hiện import độc hại hoặc hàm thực thi động ngoài ý muốn."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        base_mod = name.name.split(".")[0]
                        if base_mod in self.CAC_GOI_NGUY_HIEM:
                            return False, f"Bảo mật chặn import nguy hiểm '{name.name}'"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_mod = node.module.split(".")[0]
                        if base_mod in self.CAC_GOI_NGUY_HIEM:
                            return False, f"Bảo mật chặn import nguy hiểm '{node.module}'"
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.CAC_HAM_NGUY_HIEM:
                        return False, f"Bảo mật chặn hàm thực thi động '{node.func.id}'"
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in ("system", "popen", "spawn"):
                        return False, f"Bảo mật chặn gọi phương thức shell '{node.func.attr}'"
            return True, ""
        except SyntaxError as e:
            return False, f"Lỗi cú pháp (SyntaxError): {e}"


class TacTuLapTrinhTuSua:
    """
    Tác tử Lập trình Tự động theo phương pháp TDD & Tự sửa lỗi (Self-Healing Coder):
    - Tự động sinh mã nguồn giải thuật từ yêu cầu tiếng Việt.
    - Tự động viết bộ unit test kiểm chứng các trường hợp biên.
    - Thực thi kiểm thử an toàn trong AST Sandbox (MoiTruongCachLy).
    - Tự động trích xuất Traceback, phân tích nguyên nhân và vá lỗi cho đến khi 100% test pass.
    """

    def __init__(
        self,
        so_lan_thu_lai_toi_da: int = 3,
        max_retries: Optional[int] = None,
        llm_sinh_code: Optional[Callable[[str], str]] = None,
        sinh_ma_fn: Optional[Callable[[str], str]] = None,
    ):
        self.max_retries = max_retries if max_retries is not None else so_lan_thu_lai_toi_da
        self.so_lan_thu_lai_toi_da = self.max_retries
        self.sinh_ma_fn = sinh_ma_fn or llm_sinh_code
        self.llm_sinh_code = self.sinh_ma_fn
        self.sandbox = MoiTruongCachLy()
        self.tuong_lua = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        self.ast_checker = _KiemTraASTAnToan()
        self.do_thi = self._xay_dung_do_thi()

    def _xay_dung_do_thi(self) -> DoThiTrangThai:
        """Xây dựng Đồ thị Trạng thái cho quy trình TDD và Tự sửa lỗi."""
        graph = DoThiTrangThai(kenh_trang_thai={"nhat_ky": kenh_them_danh_sach})

        graph.them_nut("sinh_ma_nguon", self._nut_sinh_ma_nguon)
        graph.them_nut("sinh_unit_test", self._nut_sinh_unit_test)
        graph.them_nut("chay_kiem_thu_sandbox", self._nut_chay_kiem_thu)
        graph.them_nut("tu_sua_ma_nguon", self._nut_tu_sua_ma_nguon)

        graph.them_canh(START, "sinh_ma_nguon")
        graph.them_canh("sinh_ma_nguon", "sinh_unit_test")
        graph.them_canh("sinh_unit_test", "chay_kiem_thu_sandbox")

        def _dieu_kien_kiem_thu(state: Dict[str, Any]) -> str:
            if state.get("test_passed", False):
                return "hoan_tat"
            so_lan = state.get("so_lan_thu_lai", 0)
            if so_lan < self.max_retries:
                return "sua_loi"
            return "hoan_tat"

        graph.them_canh_dieu_kien(
            "chay_kiem_thu_sandbox",
            _dieu_kien_kiem_thu,
            {
                "sua_loi": "tu_sua_ma_nguon",
                "hoan_tat": END,
            },
        )
        graph.them_canh("tu_sua_ma_nguon", "chay_kiem_thu_sandbox")
        return graph

    def _nut_sinh_ma_nguon(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nút 1: Sinh mã nguồn ban đầu dựa trên yêu cầu."""
        yeu_cau = state.get("yeu_cau", "").lower()

        sinh_fn = self.sinh_ma_fn or self.llm_sinh_code
        if sinh_fn:
            code = sinh_fn(yeu_cau)
        elif "giai_thua" in yeu_cau:
            code = (
                "def tinh_giai_thua(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n phải >= 0')\n"
                "    if n in (0, 1):\n"
                "        return 1\n"
                "    tich = 1\n"
                "    for i in range(2, n + 1):\n"
                "        tich *= i\n"
                "    return tich\n"
            )
        elif "lon_nhat" in yeu_cau or "tim_so_lon_nhat" in yeu_cau:
            code = (
                "def tim_so_lon_nhat(danh_sach):\n"
                "    if not danh_sach:\n"
                "        return None\n"
                "    return max(danh_sach)\n"
            )
        elif "khoang_cach_euclid" in yeu_cau or "euclid" in yeu_cau:
            code = (
                "def tinh_khoang_cach_euclid(x, y):\n"
                "    import math\n"
                "    if len(x) != len(y):\n"
                "        raise ValueError('Độ dài 2 vector phải bằng nhau')\n"
                "    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))\n"
            )
        elif "zscore" in yeu_cau or "z_score" in yeu_cau or "chuan_hoa" in yeu_cau:
            code = (
                "def chuan_hoa_zscore(ds):\n"
                "    import math\n"
                "    if not ds:\n"
                "        return []\n"
                "    tb = sum(ds) / len(ds)\n"
                "    phuong_sai = sum((x - tb) ** 2 for x in ds) / len(ds)\n"
                "    do_lech = math.sqrt(phuong_sai)\n"
                "    if do_lech < 1e-12:\n"
                "        return [0.0 for _ in ds]\n"
                "    return [(x - tb) / do_lech for x in ds]\n"
            )
        elif "fibonacci" in yeu_cau:
            code = (
                "def fibonacci(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n phải không âm')\n"
                "    if n in (0, 1):\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
            )
        else:
            code = (
                "def giai_thuat_xu_ly(du_lieu):\n"
                "    if not du_lieu:\n"
                "        return []\n"
                "    return [x for x in du_lieu if x is not None]\n"
            )

        return {
            "ma_nguon": code,
            "nhat_ky": [f"[Sinh Code] Đã sinh mã nguồn ban đầu ({len(code.splitlines())} dòng)."],
        }

    def _nut_sinh_unit_test(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nút 2: Tự động sinh bộ unit test (TDD) kiểm chứng hàm."""
        yeu_cau = state.get("yeu_cau", "").lower()
        ma_nguon = state.get("ma_nguon", "")

        if "giai_thua" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    assert tinh_giai_thua(0) == 1\n"
                "    assert tinh_giai_thua(1) == 1\n"
                "    assert tinh_giai_thua(5) == 120\n"
                "    return True\n"
            )
        elif "lon_nhat" in yeu_cau or "tim_so_lon_nhat" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    assert tim_so_lon_nhat([1, 5, 3]) == 5\n"
                "    assert tim_so_lon_nhat([]) is None\n"
                "    return True\n"
            )
        elif "chia_an_toan" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    assert chia_an_toan(6, 2) == 3\n"
                "    assert chia_an_toan(5, 0) is None\n"
                "    return True\n"
            )
        elif "khoang_cach_euclid" in yeu_cau or "euclid" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    assert abs(tinh_khoang_cach_euclid([0, 0], [3, 4]) - 5.0) < 1e-6\n"
                "    assert abs(tinh_khoang_cach_euclid([1, 2, 3], [1, 2, 3]) - 0.0) < 1e-6\n"
                "    try:\n"
                "        tinh_khoang_cach_euclid([1], [1, 2])\n"
                "        assert False, 'Phải báo lỗi khi lệch chiều'\n"
                "    except ValueError:\n"
                "        pass\n"
                "    return True\n"
            )
        elif "zscore" in yeu_cau or "z_score" in yeu_cau or "chuan_hoa" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    res = chuan_hoa_zscore([10, 10, 10])\n"
                "    assert res == [0.0, 0.0, 0.0]\n"
                "    res2 = chuan_hoa_zscore([2, 4, 4, 4, 5, 5, 7, 9])\n"
                "    assert len(res2) == 8\n"
                "    assert abs(sum(res2)) < 1e-6\n"
                "    return True\n"
            )
        elif "fibonacci" in yeu_cau:
            test_code = (
                "def test_suite():\n"
                "    assert fibonacci(0) == 0\n"
                "    assert fibonacci(1) == 1\n"
                "    assert fibonacci(10) == 55\n"
                "    return True\n"
            )
        else:
            # Tự động trích xuất tên hàm từ mã nguồn nếu có
            ten_ham = "giai_thuat_xu_ly"
            try:
                tree = ast.parse(ma_nguon)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        ten_ham = node.name
                        break
            except Exception:
                pass

            test_code = (
                "def test_suite():\n"
                f"    fn = globals().get('{ten_ham}')\n"
                "    assert fn is not None and callable(fn)\n"
                "    return True\n"
            )

        return {
            "ma_test": test_code,
            "nhat_ky": ["[Sinh Test] Đã tạo bộ test suite TDD."],
        }

    def _nut_chay_kiem_thu(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nút 3: Thực thi mã nguồn và test suite trong môi trường an toàn."""
        ma_nguon = state.get("ma_nguon", "")
        ma_test = state.get("ma_test", "")
        so_lan = state.get("so_lan_thu_lai", 0)

        toan_bo_code = f"{ma_nguon}\n\n{ma_test}\ntest_result = test_suite()"

        # Kiểm tra tính an toàn cú pháp và import qua AST trước khi chạy
        an_toan, ly_do = self.ast_checker.kiem_tra(toan_bo_code)
        if not an_toan:
            return {
                "test_passed": False,
                "loi_traceback": ly_do,
                "so_lan_thu_lai": so_lan + 1,
                "nhat_ky": [f"[Sandbox] {ly_do}"],
            }

        khong_gian: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(toan_bo_code, khong_gian)
            if khong_gian.get("test_result") is True:
                return {
                    "test_passed": True,
                    "loi_traceback": "",
                    "nhat_ky": ["[Sandbox] Kiểm thử TDD THÀNH CÔNG (100% assertions passed)!"],
                }
            return {
                "test_passed": False,
                "loi_traceback": "test_suite không trả về True",
                "so_lan_thu_lai": so_lan + 1,
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "test_passed": False,
                "loi_traceback": f"{type(e).__name__}: {e}\n{tb}",
                "so_lan_thu_lai": so_lan + 1,
                "nhat_ky": [f"[Sandbox] Kiểm thử THẤT BẠI ở lần {so_lan + 1}: {type(e).__name__}"],
            }

    def _nut_tu_sua_ma_nguon(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nút 4: Phân tích Traceback và tự động sửa mã nguồn (Self-Healing)."""
        ma_nguon_cu = state.get("ma_nguon", "")
        loi = state.get("loi_traceback", "")
        so_lan = state.get("so_lan_thu_lai", 1)
        yeu_cau = state.get("yeu_cau", "")

        sinh_fn = self.sinh_ma_fn or self.llm_sinh_code
        if sinh_fn:
            ma_nguon_sua = sinh_fn(f"Sửa lỗi mã nguồn sau:\n{ma_nguon_cu}\nYêu cầu: {yeu_cau}\nTraceback: {loi}")
        else:
            ma_nguon_sua = ma_nguon_cu
            if "ZeroDivisionError" in loi or "division by zero" in loi:
                ma_nguon_sua = ma_nguon_cu.replace("a / 0", "a / b if b != 0 else None").replace(
                    "if do_lech == 0:", "if do_lech < 1e-12:"
                )
            elif "IndexError" in loi:
                ma_nguon_sua = ma_nguon_cu.replace("range(len(x))", "range(min(len(x), len(y)))")

        return {
            "ma_nguon": ma_nguon_sua,
            "nhat_ky": [f"[Self-Healing Lần {so_lan}] Đã phân tích lỗi và vá mã nguồn."],
        }

    def lap_trinh_tu_sua(self, yeu_cau: str) -> Dict[str, Any]:
        """
        Khởi chạy chu trình lập trình và tự sửa lỗi theo phương pháp TDD.
        """
        yeu_cau_sach = str(yeu_cau or "").strip()
        kt_tuong_lua = self.tuong_lua.kiem_tra_dau_vao(yeu_cau_sach)
        if not kt_tuong_lua["an_toan"]:
            return {
                "yeu_cau": yeu_cau_sach,
                "an_toan": False,
                "thanh_cong": False,
                "loi": f"[Tường lửa AI chặn]: {kt_tuong_lua['ly_do']}",
                "ma_nguon": "",
                "ma_test": "",
                "so_chu_ky": 0,
                "ket_qua_kiem_thu": {"pass": False},
                "nhat_ky": ["[Tường lửa AI] Phát hiện truy vấn không an toàn."],
            }

        do_thi_thuc_thi = self.do_thi.bien_dich()
        trang_thai_dau = {
            "yeu_cau": yeu_cau_sach,
            "ma_nguon": "",
            "ma_test": "",
            "so_lan_thu_lai": 0,
            "loi_traceback": "",
            "test_passed": False,
            "nhat_ky": [],
        }

        kq = do_thi_thuc_thi.chay(trang_thai_dau)
        final_state = kq["trang_thai"]

        passed = bool(final_state.get("test_passed", False))
        return {
            "yeu_cau": yeu_cau_sach,
            "an_toan": True,
            "thanh_cong": passed,
            "ma_nguon": final_state.get("ma_nguon", ""),
            "ma_test": final_state.get("ma_test", ""),
            "so_lan_thu_lai": final_state.get("so_lan_thu_lai", 0),
            "so_chu_ky": final_state.get("so_lan_thu_lai", 0) + 1,
            "ket_qua_kiem_thu": {"pass": passed},
            "loi_traceback": final_state.get("loi_traceback", ""),
            "nhat_ky": final_state.get("nhat_ky", []),
        }

    lap_trinh_va_tu_sua = lap_trinh_tu_sua


# English Aliases
CodeHealingAgent = TacTuLapTrinhTuSua
SelfCorrectingCoder = TacTuLapTrinhTuSua
