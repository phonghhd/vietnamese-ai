import pytest
import time
from vietnamese_ai.sandbox import PhanTichAST, ThucThiDocLap, LoiAnNinh


def test_ast_ngan_chan_module_nguy_hiem():
    code_doc_1 = "import os\nos.system('echo hacked')"
    code_doc_2 = "from subprocess import Popen"
    
    with pytest.raises(LoiAnNinh):
        PhanTichAST.kiem_tra(code_doc_1)
        
    with pytest.raises(LoiAnNinh):
        PhanTichAST.kiem_tra(code_doc_2)

def test_ast_ngan_chan_ham_nguy_hiem():
    code_doc_1 = "eval('1 + 1')"
    code_doc_2 = "open('/etc/passwd', 'r')"
    
    with pytest.raises(LoiAnNinh):
        PhanTichAST.kiem_tra(code_doc_1)
        
    with pytest.raises(LoiAnNinh):
        PhanTichAST.kiem_tra(code_doc_2)

def test_ast_cho_phep_code_an_toan():
    code_an_toan = "kq = sum([1, 2, 3])"
    assert PhanTichAST.kiem_tra(code_an_toan) is True

def test_thuc_thi_thanh_cong():
    sandbox = ThucThiDocLap()
    code = """
def tinh_tong(a, b):
    return a + b
    
x = 10
y = 20
ket_qua = tinh_tong(x, y)
"""
    thanh_cong, du_lieu = sandbox.chay(code)
    assert thanh_cong is True
    assert du_lieu["x"] == 10
    assert du_lieu["ket_qua"] == 30

def test_thuc_thi_timeout():
    sandbox = ThucThiDocLap(timeout_giay=1) # Set timeout cực ngắn
    code_treo = """
while True:
    pass
"""
    start = time.time()
    thanh_cong, thong_diep = sandbox.chay(code_treo)
    end = time.time()
    
    assert thanh_cong is False
    assert "Timeout" in thong_diep
    # Đảm bảo tiến trình thực sự bị ngắt sớm chứ không phải treo mãi
    assert end - start < 3.0

def test_thuc_thi_truy_cap_nguy_hiem_luc_chay():
    sandbox = ThucThiDocLap()
    # Code này vượt qua AST (do giấu import hoặc gọi hàm __builtins__ lạ) 
    # nhưng sẽ bị Executor chặn vì globals rỗng.
    code_an_gian = """
try:
    __import__('os').system('ls')
    hack_thanh_cong = True
except Exception as e:
    loi = str(e)
"""
    thanh_cong, thong_diep = sandbox.chay(code_an_gian)
    assert thanh_cong is False
    assert "Cấm sử dụng hàm nguy hiểm: '__import__()'" in thong_diep
