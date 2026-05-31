import inspect
import math
import os
from typing import Any, Callable, Dict


class CongCu:
    """
    Lớp cơ sở cho mọi công cụ trong hệ thống tác tử.
    """

    def __init__(self, ten: str, mo_ta: str, ham_thuc_thi: Callable, yeu_cau_xac_nhan: bool = False):
        self.ten = ten
        self.mo_ta = mo_ta
        self._ham_thuc_thi = ham_thuc_thi
        self.yeu_cau_xac_nhan = yeu_cau_xac_nhan
        self.tham_so = self._lay_tham_so()

    def _lay_tham_so(self) -> Dict[str, Any]:
        sig = inspect.signature(self._ham_thuc_thi)
        tham_so = {}
        for ten, param in sig.parameters.items():
            if ten == "self":
                continue
            tham_so[ten] = {
                "kieu": param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation),
                "bat_buoc": param.default == inspect.Parameter.empty
            }
        return tham_so

    def chay(self, **kwargs) -> Any:
        """Thực thi công cụ với các tham số đầu vào."""
        try:
            return self._ham_thuc_thi(**kwargs)
        except Exception as e:
            return f"Lỗi khi chạy công cụ {self.ten}: {str(e)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ten": self.ten,
            "mo_ta": self.mo_ta,
            "tham_so": self.tham_so
        }


def cong_cu(ten: str, mo_ta: str, yeu_cau_xac_nhan: bool = False):
    """
    Decorator để biến một hàm Python thành một CongCu.

    Ví dụ:
    @cong_cu(ten="tinh_tong", mo_ta="Tính tổng hai số nguyên", yeu_cau_xac_nhan=False)
    def tinh_tong(a: int, b: int) -> int:
        return a + b
    """
    def decorator(func: Callable) -> CongCu:
        return CongCu(ten=ten, mo_ta=mo_ta, ham_thuc_thi=func, yeu_cau_xac_nhan=yeu_cau_xac_nhan)
    return decorator


# --- CÁC CÔNG CỤ TÍCH HỢP SẴN ---

@cong_cu(ten="may_tinh", mo_ta="Tính toán một biểu thức toán học. Tham số 'bieu_thuc' là một chuỗi (vd: '2 + 2'). Hỗ trợ các phép cộng, trừ, nhân, chia và math module.")
def cong_cu_may_tinh(bieu_thuc: str) -> str:
    # Cảnh báo: eval() có thể nguy hiểm, ở production nên dùng thư viện parse biểu thức an toàn.
    # Trong phiên bản này, sử dụng dict các hàm an toàn.
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        ket_qua = eval(bieu_thuc, {"__builtins__": {}}, allowed_names)
        return str(ket_qua)
    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"

@cong_cu(ten="doc_file", mo_ta="Đọc nội dung của một file văn bản. Tham số 'duong_dan' là đường dẫn tuyệt đối hoặc tương đối tới file.")
def cong_cu_doc_file(duong_dan: str) -> str:
    if not os.path.exists(duong_dan):
        return f"Lỗi: File '{duong_dan}' không tồn tại."
    try:
        with open(duong_dan, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Lỗi khi đọc file: {str(e)}"

@cong_cu(ten="tim_kiem_web", mo_ta="Tìm kiếm thông tin trên web (DuckDuckGo). Tham số 'tu_khoa' là từ cần tìm. Trả về đoạn trích dẫn của các trang web liên quan.")
def cong_cu_tim_kiem_web(tu_khoa: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return "Lỗi: Cần cài đặt requests và beautifulsoup4 (pip install requests beautifulsoup4)"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(tu_khoa)}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result'):
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')

            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                results.append(f"Tiêu đề: {title}\nNội dung: {snippet}\n---")

            if len(results) >= 3:
                break

        if not results:
            return "Không tìm thấy kết quả phù hợp."

        return "\n".join(results)
    except Exception as e:
        return f"Lỗi khi tìm kiếm web: {str(e)}"

@cong_cu(ten="python_repl", mo_ta="Thực thi mã Python. Trả về kết quả từ stdout (lệnh print). Tham số 'ma_nguon' là chuỗi code Python.")
def cong_cu_python_repl(ma_nguon: str) -> str:
    from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy
    return MoiTruongCachLy.thuc_thi(ma_nguon)

@cong_cu(ten="phan_tich_anh", mo_ta="Phân tích nội dung của một hình ảnh sử dụng Multi-modal RAG (v14). Tham số 'duong_dan_anh' là đường dẫn tuyệt đối tới file ảnh.")
def cong_cu_phan_tich_anh(duong_dan_anh: str) -> str:
    """Tích hợp Multi-modal RAG (v14) vào hệ thống Agent Swarm (v24)."""
    if not os.path.exists(duong_dan_anh):
        return f"Lỗi: Không tìm thấy file ảnh tại '{duong_dan_anh}'"
        
    try:
        # Ở đây ta import động để không làm nghẽn quá trình khởi tạo Tác tử nếu thư viện ảnh chưa được cài
        from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder
        embedder = ImageEmbedder()
        vector = embedder.nhung_hinh_anh([duong_dan_anh])[0]
        
        # Vì đây là ví dụ, ta giả lập kết quả mô tả ảnh sau khi nhúng
        return f"[Multi-modal RAG] Đã phân tích ảnh '{os.path.basename(duong_dan_anh)}'. Vector embedding được trích xuất thành công ({len(vector)} chiều). Bức ảnh chứa các thực thể có thể được tìm kiếm trong cơ sở dữ liệu vector."
    except Exception as e:
        return f"Lỗi khi phân tích ảnh: {str(e)}"

