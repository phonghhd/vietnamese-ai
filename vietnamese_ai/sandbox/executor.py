"""
ThucThiDocLap (Isolated Executor) - Lớp khiên phòng thủ động.
Đưa mã nguồn vào một Process riêng biệt với không gian biến toàn cục (globals)
trống rỗng và đặt giới hạn thời gian chạy (Timeout).
"""

import multiprocessing
import queue
import logging
from typing import Any, Dict, Tuple

from .ast_analyzer import PhanTichAST, LoiAnNinh

logger = logging.getLogger("V-Sandbox")


def _tien_trinh_thuc_thi(ma_nguon: str, ket_qua_queue: multiprocessing.Queue):
    """
    Hàm được chạy trong một Process riêng.
    Không gian tên ở đây hoàn toàn trống rỗng để tránh mã độc leo thang.
    """
    # Xây dựng môi trường an toàn (Safe builtins)
    safe_builtins = {
        'print': print,
        'range': range,
        'len': len,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'map': map,
        'filter': filter,
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        # CHỦ Ý LOẠI BỎ: eval, exec, open, __import__, globals, locals, compile
    }
    
    # Context (Globals) dùng để chạy code
    safe_globals = {
        '__builtins__': safe_builtins
    }
    
    try:
        # Chạy code
        # Code có thể định nghĩa biến và hàm vào safe_globals
        exec(ma_nguon, safe_globals)
        
        # Lọc kết quả trả về (chỉ lấy các biến do code định nghĩa, bỏ qua builtins)
        ket_qua = {}
        for k, v in safe_globals.items():
            if k != '__builtins__' and not callable(v):
                # Chỉ trả về dữ liệu cơ bản (str, int, list, dict) để an toàn khi qua queue
                if isinstance(v, (int, float, str, bool, list, dict, set, tuple)):
                    ket_qua[k] = v
                    
        ket_qua_queue.put(("THANH_CONG", ket_qua))
        
    except Exception as e:
        # Bắt mọi lỗi xảy ra trong lúc chạy
        ket_qua_queue.put(("LOI", str(e)))


class ThucThiDocLap:
    """Môi trường hộp cát (Sandbox) thực thi mã."""

    def __init__(self, timeout_giay: int = 5):
        """
        Khởi tạo Sandbox.
        
        Args:
            timeout_giay: Thời gian sống tối đa của tiến trình thực thi (mặc định 5s).
        """
        self.timeout_giay = timeout_giay

    def chay(self, ma_nguon: str) -> Tuple[bool, Any]:
        """
        Kiểm tra AST tĩnh và sau đó thực thi động.
        
        Args:
            ma_nguon: Mã Python.
            
        Returns:
            Tuple (Trạng_thái_thành_công_boolean, Kết_quả_chạy_hoặc_Lỗi_chuỗi).
        """
        # 1. Quét AST trước (Static Analysis)
        try:
            PhanTichAST.kiem_tra(ma_nguon)
        except LoiAnNinh as e:
            logger.warning(f"[Sandbox] Đã chặn mã độc: {e}")
            return False, f"Lỗi An Ninh: {e}"
        except SyntaxError as e:
            return False, f"Lỗi Cú Pháp: {e}"

        # 2. Thực thi động trong Process (Dynamic Execution)
        ctx = multiprocessing.get_context('spawn') # Khởi tạo Process sạch
        hang_doi = ctx.Queue()
        
        tien_trinh = ctx.Process(target=_tien_trinh_thuc_thi, args=(ma_nguon, hang_doi))
        tien_trinh.daemon = True # Tự hủy nếu framework bị tắt
        
        tien_trinh.start()
        tien_trinh.join(timeout=self.timeout_giay)
        
        # 3. Kiểm tra Timeout
        if tien_trinh.is_alive():
            logger.error(f"[Sandbox] Đã tiêu diệt Process (Vượt quá {self.timeout_giay}s)")
            tien_trinh.terminate()
            tien_trinh.join() # Chờ tiến trình chết hẳn
            return False, "Lỗi Timeout: Mã chạy quá thời gian cho phép hoặc dính vòng lặp vô tận."
            
        # 4. Lấy kết quả
        try:
            trang_thai, du_lieu = hang_doi.get_nowait()
            if trang_thai == "THANH_CONG":
                return True, du_lieu
            else:
                return False, f"Lỗi Thực Thi: {du_lieu}"
        except queue.Empty:
            # Tiến trình có thể đã crash mà không ném gì vào queue
            if tien_trinh.exitcode != 0:
                return False, f"Tiến trình bị lỗi (Mã thoát: {tien_trinh.exitcode})"
            return False, "Không nhận được kết quả (Tiến trình bị ngắt đột ngột)."
