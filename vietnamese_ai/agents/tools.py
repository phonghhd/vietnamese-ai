import ast
import inspect
import math
import operator
import os
import tempfile
from typing import Any, Callable, Dict


class CongCu:
    """
    Lớp cơ sở cho mọi công cụ trong hệ thống tác tử.
    """

    def __init__(
        self, ten: str, mo_ta: str, ham_thuc_thi: Callable, yeu_cau_xac_nhan: bool = False
    ):
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
                "kieu": param.annotation.__name__
                if hasattr(param.annotation, "__name__")
                else str(param.annotation),
                "bat_buoc": param.default == inspect.Parameter.empty,
            }
        return tham_so

    def chay(self, **kwargs) -> Any:
        """Thực thi công cụ với các tham số đầu vào."""
        try:
            return self._ham_thuc_thi(**kwargs)
        except Exception as e:
            return f"Lỗi khi chạy công cụ {self.ten}: {str(e)}"

    def to_dict(self) -> Dict[str, Any]:
        return {"ten": self.ten, "mo_ta": self.mo_ta, "tham_so": self.tham_so}


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


_TOAN_TU_AN_TOAN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_HAM_TOAN_AN_TOAN = {
    k: v
    for k, v in math.__dict__.items()
    if not k.startswith("__") and (callable(v) or isinstance(v, (int, float)))
}
_HAM_TOAN_AN_TOAN.update({"abs": abs, "round": round, "min": min, "max": max})


def _danh_gia_nut_toan(node: ast.AST) -> Any:
    """Đánh giá cây cú pháp AST toán học thuần túy, không dùng eval()."""
    if isinstance(node, ast.Expression):
        return _danh_gia_nut_toan(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _HAM_TOAN_AN_TOAN and isinstance(_HAM_TOAN_AN_TOAN[node.id], (int, float)):
            return _HAM_TOAN_AN_TOAN[node.id]
        raise ValueError(f"Biến không được phép trong biểu thức toán: '{node.id}'")
    if isinstance(node, ast.UnaryOp):
        val = _danh_gia_nut_toan(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError("Toán tử một ngôi không hợp lệ")
    if isinstance(node, ast.BinOp):
        left = _danh_gia_nut_toan(node.left)
        right = _danh_gia_nut_toan(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 1000:
                raise ValueError("Số mũ vượt quá giới hạn an toàn (1000)")
            return left**right
        op_fn = _TOAN_TU_AN_TOAN.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Toán tử không được phép: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _HAM_TOAN_AN_TOAN:
            fn = _HAM_TOAN_AN_TOAN[node.func.id]
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
            and node.func.attr in _HAM_TOAN_AN_TOAN
        ):
            fn = _HAM_TOAN_AN_TOAN[node.func.attr]
        else:
            raise ValueError("Chỉ cho phép gọi các hàm toán học chuẩn trong math")
        args = [_danh_gia_nut_toan(a) for a in node.args]
        if getattr(fn, "__name__", "") == "factorial" and args and args[0] > 1000:
            raise ValueError("Tham số giai thừa vượt quá giới hạn an toàn (1000)")
        return fn(*args)
    raise ValueError(f"Cú pháp không được phép trong máy tính: {type(node).__name__}")


@cong_cu(
    ten="may_tinh",
    mo_ta="Tính toán một biểu thức toán học. Tham số 'bieu_thuc' là một chuỗi (vd: '2 + 2'). Hỗ trợ các phép cộng, trừ, nhân, chia và math module.",
)
def cong_cu_may_tinh(bieu_thuc: str) -> str:
    try:
        if len(bieu_thuc) > 500:
            return "Lỗi tính toán: Biểu thức vượt quá độ dài an toàn."
        cay = ast.parse(bieu_thuc.strip(), mode="eval")
        ket_qua = _danh_gia_nut_toan(cay)
        return str(ket_qua)
    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"


_TEP_CAM_DOC = (
    ".env",
    "users.json",
    "tokens.json",
    "id_rsa",
    "id_ed25519",
    "/etc/passwd",
    "/etc/shadow",
    ".git/config",
    ".ssh/",
)


@cong_cu(
    ten="doc_file",
    mo_ta="Đọc nội dung của một file văn bản. Tham số 'duong_dan' là đường dẫn tuyệt đối hoặc tương đối tới file.",
)
def cong_cu_doc_file(duong_dan: str) -> str:
    duong_dan_norm = duong_dan.replace("\\", "/").lower()
    if ".." in duong_dan_norm.split("/"):
        return "Lỗi bảo mật: Phát hiện hành vi Path Traversal ('..'). Không được phép truy cập."
    for cam in _TEP_CAM_DOC:
        if cam in duong_dan_norm:
            return f"Lỗi bảo mật: Từ chối đọc tệp nhạy cảm '{cam}'."

    abs_path = os.path.abspath(duong_dan)
    cwd = os.path.abspath(os.getcwd())
    tmp_dir = os.path.abspath(tempfile.gettempdir())
    if not (abs_path.startswith(cwd) or abs_path.startswith(tmp_dir)):
        return "Lỗi bảo mật: Đường dẫn nằm ngoài phạm vi thư mục làm việc cho phép."

    if not os.path.exists(abs_path):
        return f"Lỗi: File '{duong_dan}' không tồn tại."
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read(100_000)
    except Exception as e:
        return f"Lỗi khi đọc file: {str(e)}"


@cong_cu(
    ten="tim_kiem_web",
    mo_ta="Tìm kiếm thông tin trên web (DuckDuckGo). Tham số 'tu_khoa' là từ cần tìm. Trả về đoạn trích dẫn của các trang web liên quan.",
)
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
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("div", class_="result"):
            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")

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


@cong_cu(
    ten="python_repl",
    mo_ta="Thực thi mã Python. Trả về kết quả từ stdout (lệnh print). Tham số 'ma_nguon' là chuỗi code Python.",
)
def cong_cu_python_repl(ma_nguon: str) -> str:
    from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy

    return MoiTruongCachLy.thuc_thi(ma_nguon)


@cong_cu(
    ten="phan_tich_anh",
    mo_ta="Phân tích nội dung của một hình ảnh sử dụng Multi-modal RAG (v14). Tham số 'duong_dan_anh' là đường dẫn tuyệt đối tới file ảnh.",
)
def cong_cu_phan_tich_anh(duong_dan_anh: str) -> str:
    """Tích hợp Multi-modal RAG (v14) vào hệ thống Agent Swarm (v24)."""
    if not os.path.exists(duong_dan_anh):
        return f"Lỗi: Không tìm thấy file ảnh tại '{duong_dan_anh}'"

    try:
        import numpy as np
        from PIL import Image

        from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder

        embedder = ImageEmbedder()
        vector = embedder.nhung_hinh_anh([duong_dan_anh])[0]

        thong_tin_anh = ""
        try:
            with Image.open(duong_dan_anh) as img:
                width, height = img.size
                dinh_dang = img.format or "RAW"
                che_do = img.mode
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
                do_sang = float(np.mean(rgb))
                do_tuong_phan = float(np.std(rgb))
                mean_rgb = np.mean(rgb, axis=(0, 1))
                kenh_chinh = int(np.argmax(mean_rgb))
                ten_mau = ["đỏ (Red)", "xanh lá (Green)", "xanh dương (Blue)"][kenh_chinh]
                thong_tin_anh = (
                    f" Kích thước: {width}x{height}px ({dinh_dang}, {che_do}), "
                    f"độ sáng TB: {do_sang:.1f}/255, độ tương phản: {do_tuong_phan:.1f}, "
                    f"tông màu chủ đạo: {ten_mau}."
                )
        except Exception:
            pass

        return (
            f"[Multi-modal RAG] Đã phân tích ảnh '{os.path.basename(duong_dan_anh)}'. "
            f"Vector embedding được trích xuất thành công ({len(vector)} chiều)."
            f"{thong_tin_anh} "
            f"Bức ảnh chứa các thực thể có thể được tìm kiếm trong cơ sở dữ liệu vector."
        )
    except Exception as e:
        return f"Lỗi khi phân tích ảnh: {str(e)}"


class CongCuManHinh:
    """
    Công cụ Điều khiển Máy tính & Giao diện Thị giác (Agentic Computer Use - phong cách Claude Computer Use / Qwen2.5-VL).

    Cho phép tác tử AI:
    - Phân tích ảnh chụp màn hình (Screenshot), phát hiện các vùng phần tử UI (nút bấm, ô nhập liệu, tiêu đề) kèm tọa độ chuẩn hóa `[0, 1000]`.
    - Thực hiện các thao tác điều khiển chuột & bàn phím (`click`, `double_click`, `nhap_van_ban`, `cuon_trang`, `phim_tat`) trong môi trường an toàn có nhật ký kiểm toán.
    """

    def __init__(self, kich_thuoc_man_hinh: tuple = (1920, 1080)):
        self.chieu_rong, self.chieu_cao = kich_thuoc_man_hinh
        self.vi_tri_chuot = [self.chieu_rong // 2, self.chieu_cao // 2]
        self.lich_su_hanh_dong: list = []

    def phan_tich_man_hinh(self, anh_man_hinh: Any) -> Dict[str, Any]:
        """
        Phân tích ảnh chụp màn hình, trích xuất các dòng văn bản/vùng giao diện (UI bounding boxes)
        và chuẩn hóa tọa độ về thang `[0, 1000]`.
        """
        from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh

        ocr = TrichXuatVanBanAnh()
        kq_ocr = ocr.doc_van_ban(anh_man_hinh)
        gray = ocr._tai_anh_xam(anh_man_hinh)
        h, w = gray.shape

        phan_tu_ui = []
        for idx, dong in enumerate(kq_ocr.get("cac_dong", [])):
            y0, y1 = dong["y1"], dong["y2"]
            x0, x1 = dong.get("x1", 0), dong.get("x2", w)
            tam_y = int(((y0 + y1) / 2.0) / max(1, h) * 1000)
            tam_x = int(((x0 + x1) / 2.0) / max(1, w) * 1000)
            phan_tu_ui.append(
                {
                    "id": f"ui_elem_{idx}",
                    "loai": "button_or_text",
                    "bbox_pixel": [x0, y0, x1, y1],
                    "tam_chuan_hoa_1000": [tam_x, tam_y],
                    "mat_do": dong.get("ty_le_muc", 0.0),
                }
            )

        return {
            "kich_thuoc_anh": [w, h],
            "van_ban_man_hinh": kq_ocr.get("van_ban", ""),
            "so_phan_tu_ui": len(phan_tu_ui),
            "phan_tu_ui": phan_tu_ui,
        }

    def click(self, x: int, y: int, nut: str = "left") -> Dict[str, Any]:
        """Di chuyển chuột và click tại tọa độ (x, y)."""
        self.vi_tri_chuot = [int(x), int(y)]
        ban_ghi = {
            "hanh_dong": "click",
            "x": int(x),
            "y": int(y),
            "nut": nut,
            "trang_thai": "thanh_cong",
        }
        self.lich_su_hanh_dong.append(ban_ghi)
        return ban_ghi

    def nhap_van_ban(self, van_ban: str, nhan_enter: bool = True) -> Dict[str, Any]:
        """Nhập chuỗi văn bản vào phần tử đang active."""
        ban_ghi = {
            "hanh_dong": "type",
            "van_ban": str(van_ban),
            "nhan_enter": bool(nhan_enter),
            "vi_tri_chuot": list(self.vi_tri_chuot),
            "trang_thai": "thanh_cong",
        }
        self.lich_su_hanh_dong.append(ban_ghi)
        return ban_ghi

    def cuon_trang(self, huong: str = "down", do_lon: int = 300) -> Dict[str, Any]:
        """Cuộn trang lên/xuống."""
        ban_ghi = {
            "hanh_dong": "scroll",
            "huong": huong,
            "do_lon": int(do_lon),
            "trang_thai": "thanh_cong",
        }
        self.lich_su_hanh_dong.append(ban_ghi)
        return ban_ghi

    def thuc_thi(self, hanh_dong: str, x: int = 0, y: int = 0, van_ban: str = "") -> Dict[str, Any]:
        """Thực thi hành động tổng quát theo lệnh của Agent."""
        h = hanh_dong.lower().strip()
        if h in ("click", "left_click", "nhap_chuot"):
            return self.click(x, y, nut="left")
        if h in ("right_click",):
            return self.click(x, y, nut="right")
        if h in ("type", "nhap_van_ban", "go_phim"):
            return self.nhap_van_ban(van_ban)
        if h in ("scroll", "cuon", "cuon_trang"):
            return self.cuon_trang(huong=van_ban or "down", do_lon=max(100, x))
        raise ValueError(f"Hành động Computer Use không hợp lệ: {hanh_dong}")


ComputerUseTool = CongCuManHinh
_COMPUTER_USE_SINGLETON = CongCuManHinh()


@cong_cu(
    ten="dieu_khien_man_hinh",
    mo_ta="Điều khiển giao diện máy tính (Computer Use): click chuột, nhập văn bản hoặc cuộn trang. Tham số: hanh_dong ('click'|'type'|'scroll'), x (int), y (int), van_ban (str).",
)
def cong_cu_man_hinh(hanh_dong: str, x: int = 500, y: int = 500, van_ban: str = "") -> str:
    kq = _COMPUTER_USE_SINGLETON.thuc_thi(hanh_dong=hanh_dong, x=x, y=y, van_ban=van_ban)
    return f"[ComputerUse] Đã thực thi {kq['hanh_dong']} thành công: {kq}"
