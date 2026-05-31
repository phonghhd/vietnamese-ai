import os

from vietnamese_ai.agents.devops import viet_code_an_toan


def test_devops_tool_ast():
    # Test trực tiếp tool với lỗi Syntax
    ma_nguon = "def tong(a, b)\n    return a+b"
    res = viet_code_an_toan("loi_cu_phap.py", ma_nguon=ma_nguon)
    assert "SyntaxError" in res


def test_devops_tool_dry_run():
    # Test trực tiếp tool với lỗi Logic (Test Fail)
    ma_nguon = "def tong(a, b):\n    return a * b"
    ma_test = "import sys\nsys.path.append('.')\nfrom tinh_tong_loi import tong\ndef test_tong():\n    assert tong(1,2) == 3"
    res = viet_code_an_toan("tinh_tong_loi.py", ma_nguon=ma_nguon, ma_test=ma_test)
    assert "Test Failed" in res


def test_devops_tool_success():
    # Test trực tiếp tool thành công
    ma_nguon = "def tong(a, b):\n    return a + b"
    ma_test = "import sys\nsys.path.append('.')\nfrom tinh_tong_dung import tong\ndef test_tong():\n    assert tong(1,2) == 3"
    res = viet_code_an_toan("tinh_tong_dung.py", ma_nguon=ma_nguon, ma_test=ma_test)
    assert "Thành công" in res

    # Dọn dẹp
    try:
        os.remove("/home/phong/V-Neural/scratch/tinh_tong_dung.py")
        os.remove("/home/phong/V-Neural/scratch/test_tinh_tong_dung.py")
    except Exception:
        pass
