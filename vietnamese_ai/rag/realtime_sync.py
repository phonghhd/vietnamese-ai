import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional


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

    def phat_su_kien(self, ten_su_kien: str, payload: Any, dong_bo: bool = False):
        """Phát một sự kiện tới tất cả các hàm lắng nghe."""
        with self.lock:
            callbacks = list(self.subscribers.get(ten_su_kien, []))
        for callback in callbacks:
            if dong_bo:
                callback(payload)
            else:
                thread = threading.Thread(target=callback, args=(payload,))
                thread.daemon = True
                thread.start()


class DocumentWatcher:
    """
    Theo dõi thư mục tài liệu thực tế trên đĩa và phát sự kiện 'NEW_DOCUMENT'
    khi có file văn bản mới hoặc được cập nhật.
    """

    DUOI_HO_TRO = (".txt", ".md", ".json", ".csv", ".py", ".html")

    def __init__(self, thu_muc: str):
        self.thu_muc = thu_muc
        self.event_bus = RAGEventBus()
        self.da_xu_ly = set()
        self._mtime_map: Dict[str, float] = {}
        self._running = False
        self._watch_thread: Optional[threading.Thread] = None

    def phat_hien_file_moi(self, ma_tai_lieu: str, noi_dung: str, dong_bo: bool = False):
        """Báo cáo tài liệu mới và bắn sự kiện NEW_DOCUMENT qua RAGEventBus."""
        if ma_tai_lieu not in self.da_xu_ly:
            self.da_xu_ly.add(ma_tai_lieu)
            print(f"[DocumentWatcher] Phát hiện tài liệu mới: {ma_tai_lieu}")
            self.event_bus.phat_su_kien(
                "NEW_DOCUMENT",
                {"ma": ma_tai_lieu, "noi_dung": noi_dung},
                dong_bo=dong_bo,
            )

    def quet_thu_muc(self, dong_bo: bool = True) -> List[str]:
        """Quét thư mục thực tế trên đĩa, đọc các file mới/thay đổi và phát sự kiện."""
        phat_hien: List[str] = []
        if not self.thu_muc or not os.path.isdir(self.thu_muc):
            return phat_hien

        for root, _, files in os.walk(self.thu_muc):
            for fname in sorted(files):
                if not fname.lower().endswith(self.DUOI_HO_TRO):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    continue

                rel_id = os.path.relpath(fpath, self.thu_muc)
                if rel_id in self._mtime_map and self._mtime_map[rel_id] >= mtime:
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        noi_dung = f.read()
                except OSError:
                    continue

                self._mtime_map[rel_id] = mtime
                if rel_id in self.da_xu_ly:
                    self.da_xu_ly.discard(rel_id)
                self.phat_hien_file_moi(rel_id, noi_dung, dong_bo=dong_bo)
                phat_hien.append(rel_id)
        return phat_hien

    def bat_dau_theo_doi(self, chu_ky_giay: float = 0.5) -> None:
        """Bắt đầu luồng chạy ngầm quét thư mục định kỳ."""
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                self.quet_thu_muc(dong_bo=False)
                time.sleep(chu_ky_giay)

        self._watch_thread = threading.Thread(target=_loop, daemon=True)
        self._watch_thread.start()

    def dung_theo_doi(self) -> None:
        """Dừng luồng theo dõi thư mục."""
        self._running = False
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=2.0)
            self._watch_thread = None
