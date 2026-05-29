import threading
from typing import Any, Callable, Dict, List


class RAGEventBus:
    """
    Event Bus nội bộ (Observer Pattern) dùng để đồng bộ hóa
    dữ liệu RAG theo thời gian thực (Real-time).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEventBus, cls).__new__(cls)
            cls._instance.subscribers: Dict[str, List[Callable]] = {}
            cls._instance.lock = threading.Lock()
        return cls._instance

    def dang_ky_lang_nghe(self, ten_su_kien: str, callback: Callable[[Any], None]):
        """Đăng ký một hàm callback để lắng nghe sự kiện."""
        with self.lock:
            if ten_su_kien not in self.subscribers:
                self.subscribers[ten_su_kien] = []
            self.subscribers[ten_su_kien].append(callback)

    def phat_su_kien(self, ten_su_kien: str, payload: Any):
        """Phát một sự kiện tới tất cả các hàm lắng nghe (chạy ngầm)."""
        with self.lock:
            if ten_su_kien in self.subscribers:
                for callback in self.subscribers[ten_su_kien]:
                    # Chạy ngầm trong Thread để không block luồng chính
                    thread = threading.Thread(target=callback, args=(payload,))
                    thread.daemon = True
                    thread.start()

class DocumentWatcher:
    """
    Giả lập một watcher thư mục (ví dụ thư mục chứa PDF).
    Mỗi khi có tài liệu mới, nó bắn sự kiện 'NEW_DOCUMENT'.
    """
    def __init__(self, thu_muc: str):
        self.thu_muc = thu_muc
        self.event_bus = RAGEventBus()
        self.da_xu_ly = set()

    def phat_hien_file_moi(self, ma_tai_lieu: str, noi_dung: str):
        """Giả lập hàm báo cáo file mới."""
        if ma_tai_lieu not in self.da_xu_ly:
            self.da_xu_ly.add(ma_tai_lieu)
            print(f"[DocumentWatcher] Phát hiện tài liệu mới: {ma_tai_lieu}")
            self.event_bus.phat_su_kien("NEW_DOCUMENT", {
                "ma": ma_tai_lieu,
                "noi_dung": noi_dung
            })
