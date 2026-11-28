"""QuanLyPhienChatWorkspace (StudioWorkspaceManager) - Quản Lý Không Gian Làm Việc, Chat Đa Phiên Lưu SQLite & Trích Dẫn Gốc."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

_RE_MA_WORKSPACE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Quản lý đa không gian làm việc (Workspace theo phòng ban), lịch sử hội thoại đa phiên lưu bền vững "
        "trên SQLite kèm trích dẫn đoạn nguồn gốc, phân quyền `HeThongXacThuc` và kiểm toán `NhatKyHoatDong`."
    ),
    nhom_extra="core",
)
class QuanLyPhienChatWorkspace:
    """
    Bộ quản lý Không gian làm việc (Workspace) & Chat đa phiên cho Web Studio Doanh nghiệp.

    Tính năng:
    - Tách kho tri thức `.vaidb` riêng cho từng phòng ban / dự án (`mac_dinh`, `phap_che`, `nhan_su`, ...).
    - Lưu lịch sử chat đa phiên bền vững vào SQLite kèm trích dẫn gốc (`ten_tai_lieu`, `dieu`, `so_hieu`, `trich_doan`, `diem`).
    - Ghi nhận đánh giá Đúng/Sai của người dùng (`thumbs_up` / `thumbs_down`) phục vụ vòng lặp DPO.
    - Tùy chọn gắn `HeThongXacThuc` (RBAC) và `NhatKyHoatDong` (Audit Log).
    """

    def __init__(
        self,
        thu_muc_du_lieu: Union[str, Path] = "vai_studio_workspaces",
        bat_xac_thuc: bool = False,
        dung_vaidb_cho_workspace: bool = True,
    ) -> None:
        self.thu_muc = Path(thu_muc_du_lieu)
        self.thu_muc.mkdir(parents=True, exist_ok=True)
        self.bat_xac_thuc = bat_xac_thuc
        self.dung_vaidb_cho_workspace = dung_vaidb_cho_workspace

        self._db_path = self.thu_muc / "studio_chat.sqlite3"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._rag_cache: Dict[str, RAGPipeline] = {}

        self.xac_thuc: Optional[Any] = None
        self.nhat_ky: Optional[Any] = None
        if bat_xac_thuc:
            from vietnamese_ai.enterprise.audit import NhatKyHoatDong
            from vietnamese_ai.enterprise.auth import HeThongXacThuc

            self.xac_thuc = HeThongXacThuc(duong_dan=str(self.thu_muc / "auth"))
            self.nhat_ky = NhatKyHoatDong(duong_dan=str(self.thu_muc / "audit"))

        self._khoi_tao_bang()
        self._dam_bao_workspace_mac_dinh()

    def _khoi_tao_bang(self) -> None:
        """Tạo bảng SQLite cho workspaces, phiên chat và tin nhắn kèm trích dẫn."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA busy_timeout=5000;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    ma_workspace TEXT PRIMARY KEY,
                    ten TEXT NOT NULL,
                    mo_ta TEXT DEFAULT '',
                    chu_so_huu TEXT DEFAULT 'local',
                    duong_dan_vaidb TEXT NOT NULL,
                    tao_luc TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    ma_phien TEXT PRIMARY KEY,
                    ma_workspace TEXT NOT NULL,
                    tieu_de TEXT NOT NULL,
                    nguoi_tao TEXT DEFAULT 'local',
                    tao_luc TEXT NOT NULL,
                    cap_nhat_luc TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ma_phien TEXT NOT NULL,
                    vai_tro TEXT NOT NULL,
                    noi_dung TEXT NOT NULL,
                    trich_dan_json TEXT DEFAULT '[]',
                    hieu_luc_json TEXT DEFAULT '{}',
                    danh_gia TEXT DEFAULT NULL,
                    cau_tra_loi_sua TEXT DEFAULT NULL,
                    tao_luc TEXT NOT NULL
                )
                """
            )
            cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(chat_messages)").fetchall()}
            if "cau_tra_loi_sua" not in cols:
                cur.execute("ALTER TABLE chat_messages ADD COLUMN cau_tra_loi_sua TEXT DEFAULT NULL")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages(ma_phien)")
            self._conn.commit()

    def _dam_bao_workspace_mac_dinh(self) -> None:
        """Đảm bảo luôn có workspace `mac_dinh` sẵn sàng sử dụng ngay."""
        with self._lock:
            row = self._conn.execute(
                "SELECT ma_workspace FROM workspaces WHERE ma_workspace = 'mac_dinh'"
            ).fetchone()
            if row is None:
                self.tao_workspace(
                    ten="Không gian Chung",
                    mo_ta="Kho tri thức pháp lý & tài liệu mặc định của Web Studio",
                    ma_workspace="mac_dinh",
                    chu_so_huu="system",
                )

    def _chuan_hoa_ma_workspace(self, ma_workspace: str) -> str:
        """Xác thực mã workspace hợp lệ và đảm bảo đường dẫn tệp `.vaidb` nằm trong thư mục dữ liệu."""
        ma = str(ma_workspace or "").strip().lower()
        if not _RE_MA_WORKSPACE.match(ma):
            raise ValueError(
                f"Mã workspace không hợp lệ: '{ma_workspace}'. "
                "Chỉ chấp nhận chữ thường, số, gạch dưới hoặc gạch ngang (tối đa 64 ký tự)."
            )
        base_dir = self.thu_muc.resolve()
        target_path = (base_dir / f"{ma}.vaidb").resolve()
        if target_path.parent != base_dir:
            raise ValueError(f"Phát hiện đường dẫn workspace bất hợp lệ: '{ma_workspace}'.")
        return ma

    def _lay_thong_tin_token(self, token: Optional[str], quyen: str = "view") -> Tuple[str, str]:
        """Xác thực token nếu bật chế độ phân quyền doanh nghiệp và trả về `(ten_dang_nhap, vai_tro)`."""
        if not self.bat_xac_thuc or self.xac_thuc is None:
            return "local", "admin"
        if not token:
            raise PermissionError("Yêu cầu token xác thực để truy cập Workspace.")
        info = self.xac_thuc.xac_thuc(token)
        if not self.xac_thuc.kiem_tra_quyen(token, quyen):
            raise PermissionError(f"Tài khoản '{info['ten_dang_nhap']}' không có quyền '{quyen}'.")
        return str(info["ten_dang_nhap"]), str(info.get("vai_tro", "viewer"))

    def _kiem_tra_token(self, token: Optional[str], quyen: str = "view") -> str:
        """Xác thực token nếu bật chế độ phân quyền doanh nghiệp và trả về tên người dùng."""
        user, _ = self._lay_thong_tin_token(token, quyen=quyen)
        return user

    def _kiem_tra_quyen_phien(self, nguoi_tao: str, user: str, vai_tro: str, ma_phien: str) -> None:
        """Kiểm tra quyền sở hữu phiên chat để chặn IDOR giữa các tài khoản khi bật xác thực."""
        if not self.bat_xac_thuc:
            return
        if vai_tro == "admin" or nguoi_tao == user:
            return
        raise PermissionError(f"Tài khoản '{user}' không có quyền truy cập phiên chat '{ma_phien}'.")

    def _ghi_audit(self, hanh_dong: str, nguoi_dung: str, mo_ta: str, du_lieu: Optional[Dict[str, Any]] = None) -> None:
        """Ghi nhận hoạt động vào `NhatKyHoatDong` nếu đang bật."""
        if self.nhat_ky is not None:
            self.nhat_ky.ghi(hanh_dong, nguoi_dung=nguoi_dung, mo_ta=mo_ta, du_lieu=du_lieu)

    def tao_workspace(
        self,
        ten: str,
        mo_ta: str = "",
        ma_workspace: Optional[str] = None,
        chu_so_huu: str = "local",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo mới một Không gian làm việc có tệp `.vaidb` tách biệt."""
        if chu_so_huu == "system":
            user, vai_tro = "system", "admin"
        else:
            user, vai_tro = self._lay_thong_tin_token(token, quyen="manage")
        ma = self._chuan_hoa_ma_workspace(ma_workspace or f"ws_{uuid.uuid4().hex[:8]}")
        vaidb_file = self.thu_muc / f"{ma}.vaidb"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._conn.execute(
                "SELECT chu_so_huu, tao_luc FROM workspaces WHERE ma_workspace = ?",
                (ma,),
            ).fetchone()
            if existing is not None:
                chu_hien_tai = str(existing["chu_so_huu"])
                if self.bat_xac_thuc and vai_tro != "admin" and chu_hien_tai != user:
                    raise PermissionError(
                        f"Tài khoản '{user}' không được phép ghi đè workspace '{ma}' của '{chu_hien_tai}'."
                    )
                self._conn.execute(
                    """
                    UPDATE workspaces SET ten = ?, mo_ta = ? WHERE ma_workspace = ?
                    """,
                    (ten.strip() or ma, mo_ta, ma),
                )
                user_ket_qua = chu_hien_tai
                tao_luc_ket_qua = str(existing["tao_luc"])
            else:
                self._conn.execute(
                    """
                    INSERT INTO workspaces (ma_workspace, ten, mo_ta, chu_so_huu, duong_dan_vaidb, tao_luc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ma, ten.strip() or ma, mo_ta, user, str(vaidb_file), now_iso),
                )
                user_ket_qua = user
                tao_luc_ket_qua = now_iso
            self._conn.commit()
        self._ghi_audit("tao_workspace", user, f"Tạo workspace {ma}", {"ten": ten})
        return {
            "ma_workspace": ma,
            "ten": ten.strip() or ma,
            "mo_ta": mo_ta,
            "chu_so_huu": user_ket_qua,
            "duong_dan_vaidb": str(vaidb_file),
            "tao_luc": tao_luc_ket_qua,
        }

    def danh_sach_workspace(self, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liệt kê toàn bộ Workspace hiện có."""
        self._kiem_tra_token(token, quyen="view")
        with self._lock:
            rows = self._conn.execute(
                "SELECT ma_workspace, ten, mo_ta, chu_so_huu, duong_dan_vaidb, tao_luc FROM workspaces ORDER BY tao_luc ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def lay_rag_workspace(self, ma_workspace: str = "mac_dinh") -> RAGPipeline:
        """Lấy hoặc mở `RAGPipeline` gắn tệp `.vaidb` riêng của một Workspace."""
        ma_workspace = self._chuan_hoa_ma_workspace(ma_workspace)
        with self._lock:
            if ma_workspace in self._rag_cache:
                return self._rag_cache[ma_workspace]
            row = self._conn.execute(
                "SELECT duong_dan_vaidb FROM workspaces WHERE ma_workspace = ?",
                (ma_workspace,),
            ).fetchone()
            if row is None:
                ws = self.tao_workspace(ten=ma_workspace, ma_workspace=ma_workspace, chu_so_huu="system")
                db_path = ws["duong_dan_vaidb"]
            else:
                db_path = str(row["duong_dan_vaidb"])

            rag = RAGPipeline(
                nguong_diem=0.05,
                top_k=3,
                backend="vaidb" if self.dung_vaidb_cho_workspace else "memory",
                duong_dan_vaidb=db_path if self.dung_vaidb_cho_workspace else None,
                che_do_nhung="onnx_viet",
                che_do_rerank="onnx_viet",
                che_do_hanh_chinh=True,
            )
            self._rag_cache[ma_workspace] = rag
            return rag

    def tao_phien_chat(
        self,
        ma_workspace: str = "mac_dinh",
        tieu_de: str = "Phiên hỏi đáp mới",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo một phiên hội thoại mới gắn với Workspace."""
        user = self._kiem_tra_token(token, quyen="view")
        ma_workspace = self._chuan_hoa_ma_workspace(ma_workspace)
        ma_phien = f"chat_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chat_sessions (ma_phien, ma_workspace, tieu_de, nguoi_tao, tao_luc, cap_nhat_luc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ma_phien, ma_workspace, tieu_de.strip() or "Phiên hỏi đáp", user, now_iso, now_iso),
            )
            self._conn.commit()
        return {
            "ma_phien": ma_phien,
            "ma_workspace": ma_workspace,
            "tieu_de": tieu_de.strip() or "Phiên hỏi đáp",
            "nguoi_tao": user,
            "tao_luc": now_iso,
            "cap_nhat_luc": now_iso,
        }

    def danh_sach_phien_chat(
        self,
        ma_workspace: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Liệt kê các phiên chat đã lưu (tùy chọn lọc theo `ma_workspace`)."""
        user, vai_tro = self._lay_thong_tin_token(token, quyen="view")
        with self._lock:
            if ma_workspace:
                rows = self._conn.execute(
                    """
                    SELECT ma_phien, ma_workspace, tieu_de, nguoi_tao, tao_luc, cap_nhat_luc
                    FROM chat_sessions WHERE ma_workspace = ? ORDER BY cap_nhat_luc DESC
                    """,
                    (ma_workspace,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT ma_phien, ma_workspace, tieu_de, nguoi_tao, tao_luc, cap_nhat_luc
                    FROM chat_sessions ORDER BY cap_nhat_luc DESC
                    """
                ).fetchall()
            ds = [dict(r) for r in rows]
            if self.bat_xac_thuc and vai_tro != "admin":
                ds = [r for r in ds if r["nguoi_tao"] == user]
            return ds

    def gui_tin_nhan(
        self,
        ma_phien: str,
        cau_hoi: str,
        token: Optional[str] = None,
        rag_override: Optional[RAGPipeline] = None,
    ) -> Dict[str, Any]:
        """Gửi câu hỏi vào phiên chat, chạy truy xuất RAG kèm trích dẫn nổi bật và lưu bền vững hai chiều."""
        user, vai_tro = self._lay_thong_tin_token(token, quyen="predict")
        with self._lock:
            sess = self._conn.execute(
                "SELECT ma_workspace, nguoi_tao FROM chat_sessions WHERE ma_phien = ?",
                (ma_phien,),
            ).fetchone()
            if sess is None:
                raise KeyError(f"Không tìm thấy phiên chat '{ma_phien}'.")
            self._kiem_tra_quyen_phien(str(sess["nguoi_tao"]), user, vai_tro, ma_phien)
            ma_ws = str(sess["ma_workspace"])

        rag = rag_override if rag_override is not None else self.lay_rag_workspace(ma_ws)
        kq_rag = rag.hoi(cau_hoi, top_k=3, co_kem_nguon=True)
        tra_loi = str(kq_rag.get("tra_loi", ""))

        trich_dan_noi_bat: List[Dict[str, Any]] = []
        for ng in kq_rag.get("nguon") or []:
            meta = ng.get("metadata", {}) if isinstance(ng, dict) else {}
            nd = str(ng.get("noi_dung") or meta.get("noi_dung") or "")
            trich_dan_noi_bat.append(
                {
                    "ma_chunk": str(ng.get("ma", "")),
                    "tai_lieu": str(meta.get("tai_lieu_goc") or meta.get("van_ban") or "Tài liệu"),
                    "dieu": str(meta.get("dieu", "")),
                    "so_hieu": meta.get("so_hieu", ""),
                    "so_trang": meta.get("so_trang", 1),
                    "diem": round(float(ng.get("diem_rerank", ng.get("diem", 0.0))), 4),
                    "doan_trich": nd[:260] + ("..." if len(nd) > 260 else ""),
                }
            )

        hieu_luc_info = kq_rag.get("trich_dan_phap_ly") or {}
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_messages (ma_phien, vai_tro, noi_dung, trich_dan_json, hieu_luc_json, tao_luc)
                VALUES (?, 'user', ?, '[]', '{}', ?)
                """,
                (ma_phien, cau_hoi, now_iso),
            )
            cur.execute(
                """
                INSERT INTO chat_messages (ma_phien, vai_tro, noi_dung, trich_dan_json, hieu_luc_json, tao_luc)
                VALUES (?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    ma_phien,
                    tra_loi,
                    json.dumps(trich_dan_noi_bat, ensure_ascii=False, default=str),
                    json.dumps(hieu_luc_info, ensure_ascii=False, default=str),
                    now_iso,
                ),
            )
            assistant_msg_id = int(cur.lastrowid or 0)
            cur.execute(
                "UPDATE chat_sessions SET cap_nhat_luc = ? WHERE ma_phien = ?",
                (now_iso, ma_phien),
            )
            self._conn.commit()

        self._ghi_audit("chat_rag", user, f"Hỏi đáp trong phiên {ma_phien}", {"workspace": ma_ws})
        return {
            "id_tin_nhan": assistant_msg_id,
            "ma_phien": ma_phien,
            "ma_workspace": ma_ws,
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "trich_dan_noi_bat": trich_dan_noi_bat,
            "trich_dan_phap_ly": hieu_luc_info,
            "tao_luc": now_iso,
        }

    def lay_ma_workspace_phien(self, ma_phien: str) -> Optional[str]:
        """Trả về `ma_workspace` gắn với một phiên chat (hoặc `None` nếu không tồn tại)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT ma_workspace FROM chat_sessions WHERE ma_phien = ?",
                (ma_phien,),
            ).fetchone()
            return str(row["ma_workspace"]) if row is not None else None

    def ghi_luot_hoi_thoai_kenh(
        self,
        ma_phien: str,
        cau_hoi: str,
        tra_loi: str,
        ma_workspace: str = "mac_dinh",
        tieu_de_phien: Optional[str] = None,
        nguoi_tao: str = "connector",
        trich_dan_noi_bat: Optional[List[Dict[str, Any]]] = None,
        hieu_luc_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ghi bền vững một lượt hội thoại từ kênh ngoài (Zalo OA / Telegram) vào SQLite Workspace để đồng bộ lịch sử và gom DPO."""
        ma_workspace = self._chuan_hoa_ma_workspace(ma_workspace)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            sess = self._conn.execute(
                "SELECT ma_workspace FROM chat_sessions WHERE ma_phien = ?",
                (ma_phien,),
            ).fetchone()
            if sess is None:
                ws_row = self._conn.execute(
                    "SELECT ma_workspace FROM workspaces WHERE ma_workspace = ?",
                    (ma_workspace,),
                ).fetchone()
                if ws_row is None:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO workspaces (ma_workspace, ten, mo_ta, chu_so_huu, duong_dan_vaidb, tao_luc)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ma_workspace,
                            ma_workspace,
                            "Workspace kênh nhắn tin tự động",
                            nguoi_tao,
                            str(self.thu_muc / f"{ma_workspace}.vaidb"),
                            now_iso,
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO chat_sessions (ma_phien, ma_workspace, tieu_de, nguoi_tao, tao_luc, cap_nhat_luc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ma_phien,
                        ma_workspace,
                        (tieu_de_phien or ma_phien).strip(),
                        nguoi_tao,
                        now_iso,
                        now_iso,
                    ),
                )
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_messages (ma_phien, vai_tro, noi_dung, trich_dan_json, hieu_luc_json, tao_luc)
                VALUES (?, 'user', ?, '[]', '{}', ?)
                """,
                (ma_phien, cau_hoi, now_iso),
            )
            cur.execute(
                """
                INSERT INTO chat_messages (ma_phien, vai_tro, noi_dung, trich_dan_json, hieu_luc_json, tao_luc)
                VALUES (?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    ma_phien,
                    tra_loi,
                    json.dumps(trich_dan_noi_bat or [], ensure_ascii=False, default=str),
                    json.dumps(hieu_luc_info or {}, ensure_ascii=False, default=str),
                    now_iso,
                ),
            )
            assistant_msg_id = int(cur.lastrowid or 0)
            cur.execute(
                "UPDATE chat_sessions SET cap_nhat_luc = ? WHERE ma_phien = ?",
                (now_iso, ma_phien),
            )
            self._conn.commit()
        self._ghi_audit("kenh_chat", nguoi_tao, f"Ghi hội thoại kênh {ma_phien}", {"workspace": ma_workspace})
        return {
            "id_tin_nhan": assistant_msg_id,
            "ma_phien": ma_phien,
            "ma_workspace": ma_workspace,
            "tao_luc": now_iso,
        }

    def lay_lich_su_phien(self, ma_phien: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Đọc toàn bộ lịch sử tin nhắn cùng trích dẫn nguồn của một phiên chat."""
        user, vai_tro = self._lay_thong_tin_token(token, quyen="view")
        with self._lock:
            sess = self._conn.execute(
                "SELECT nguoi_tao FROM chat_sessions WHERE ma_phien = ?",
                (ma_phien,),
            ).fetchone()
            if sess is not None:
                self._kiem_tra_quyen_phien(str(sess["nguoi_tao"]), user, vai_tro, ma_phien)
            rows = self._conn.execute(
                """
                SELECT id, ma_phien, vai_tro, noi_dung, trich_dan_json, hieu_luc_json, danh_gia, cau_tra_loi_sua, tao_luc
                FROM chat_messages
                WHERE ma_phien = ?
                ORDER BY id ASC
                """,
                (ma_phien,),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "ma_phien": str(r["ma_phien"]),
                    "vai_tro": str(r["vai_tro"]),
                    "noi_dung": str(r["noi_dung"]),
                    "trich_dan_noi_bat": json.loads(str(r["trich_dan_json"] or "[]")),
                    "trich_dan_phap_ly": json.loads(str(r["hieu_luc_json"] or "{}")),
                    "danh_gia": r["danh_gia"],
                    "cau_tra_loi_sua": r["cau_tra_loi_sua"],
                    "tao_luc": str(r["tao_luc"]),
                }
                for r in rows
            ]

    def danh_gia_tin_nhan(
        self,
        id_tin_nhan: int,
        danh_gia: str,
        cau_tra_loi_sua: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ghi nhận phản hồi `thumbs_up` / `thumbs_down` (kèm câu trả lời đúng nếu sửa) để gom tập dữ liệu DPO."""
        if danh_gia not in ("thumbs_up", "thumbs_down"):
            raise ValueError("danh_gia phải là 'thumbs_up' hoặc 'thumbs_down'")
        user, vai_tro = self._lay_thong_tin_token(token, quyen="predict")
        with self._lock:
            owner_row = self._conn.execute(
                """
                SELECT m.ma_phien, s.nguoi_tao
                FROM chat_messages m
                JOIN chat_sessions s ON m.ma_phien = s.ma_phien
                WHERE m.id = ?
                """,
                (int(id_tin_nhan),),
            ).fetchone()
            if owner_row is not None:
                self._kiem_tra_quyen_phien(
                    str(owner_row["nguoi_tao"]), user, vai_tro, str(owner_row["ma_phien"])
                )
            cur = self._conn.execute(
                "UPDATE chat_messages SET danh_gia = ?, cau_tra_loi_sua = ? WHERE id = ?",
                (danh_gia, cau_tra_loi_sua, int(id_tin_nhan)),
            )
            self._conn.commit()
            da_cap_nhat = cur.rowcount > 0
        self._ghi_audit(
            "danh_gia_chat",
            user,
            f"Đánh giá tin nhắn #{id_tin_nhan}: {danh_gia}",
            {"co_sua_dap_an": bool(cau_tra_loi_sua)},
        )
        return {
            "cap_nhat": da_cap_nhat,
            "id_tin_nhan": int(id_tin_nhan),
            "danh_gia": danh_gia,
            "co_sua_dap_an": bool(cau_tra_loi_sua),
        }

    def xuat_tap_du_lieu_dpo(
        self,
        duong_dan_jsonl: Optional[Union[str, Path]] = None,
        ma_workspace: Optional[str] = None,
        an_pii: bool = True,
        ho_du_lieu: Optional[Any] = None,
        ten_dataset_dvc: str = "studio_dpo_feedback.jsonl",
        token: Optional[str] = None,
        loc_nhieu: bool = False,
        mo_hinh_tham_chieu: Optional[Any] = None,
        thu_muc_xuat_cho_phep: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Xuất toàn bộ phản hồi `thumbs_up` / `thumbs_down` (kèm đáp án sửa) trong Workspace thành tập dữ liệu
        huấn luyện DPO `(prompt, chosen, rejected)` đã ẩn PII, sẵn sàng đưa vào `DPOTrainer` hoặc `vai train-cloud`.
        """
        from vietnamese_ai.security.data_sanitizer import DataSanitizer

        user, vai_tro = self._lay_thong_tin_token(token, quyen="view")
        co_quyen_quan_tri = (not self.bat_xac_thuc) or vai_tro == "admin"
        if self.bat_xac_thuc and not co_quyen_quan_tri and self.xac_thuc is not None and token:
            co_quyen_quan_tri = bool(self.xac_thuc.kiem_tra_quyen(token, "manage"))

        if duong_dan_jsonl is not None and self.bat_xac_thuc and not co_quyen_quan_tri:
            raise PermissionError(
                f"Tài khoản '{user}' cần quyền 'manage' để xuất tập DPO ra tệp trên máy chủ."
            )

        p_out_resolved: Optional[Path] = None
        if duong_dan_jsonl is not None:
            raw_path = Path(duong_dan_jsonl)
            if thu_muc_xuat_cho_phep is not None:
                base_dir = Path(thu_muc_xuat_cho_phep).resolve()
                if ".." in str(duong_dan_jsonl).replace("\\", "/").split("/"):
                    raise PermissionError("Đường dẫn xuất DPO không được phép chứa '..'.")
                candidate = (base_dir / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
                try:
                    candidate.relative_to(base_dir)
                except ValueError as exc:
                    raise PermissionError(
                        "Đường dẫn xuất DPO nằm ngoài thư mục Workspace được phép."
                    ) from exc
                if candidate.suffix.lower() not in (".jsonl", ".json"):
                    raise PermissionError("Tệp xuất DPO chỉ chấp nhận phần mở rộng '.jsonl' hoặc '.json'.")
                p_out_resolved = candidate
            else:
                p_out_resolved = raw_path

        with self._lock:
            clauses = ["m.vai_tro = 'assistant'", "m.danh_gia IS NOT NULL"]
            params: List[Any] = []
            if ma_workspace:
                clauses.append("s.ma_workspace = ?")
                params.append(ma_workspace)
            if self.bat_xac_thuc and not co_quyen_quan_tri:
                clauses.append("s.nguoi_tao = ?")
                params.append(user)
            where_sql = " AND ".join(clauses)
            rows = self._conn.execute(
                f"""
                SELECT m.id, m.ma_phien, m.noi_dung, m.danh_gia, m.cau_tra_loi_sua, s.ma_workspace
                FROM chat_messages m
                JOIN chat_sessions s ON m.ma_phien = s.ma_phien
                WHERE {where_sql}
                ORDER BY m.id ASC
                """,
                tuple(params),
            ).fetchall()

            cac_mau: List[Dict[str, Any]] = []
            for r in rows:
                msg_id = int(r["id"])
                ma_phien = str(r["ma_phien"])
                row_user = self._conn.execute(
                    """
                    SELECT noi_dung FROM chat_messages
                    WHERE ma_phien = ? AND vai_tro = 'user' AND id < ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (ma_phien, msg_id),
                ).fetchone()
                if row_user is None:
                    continue

                prompt_raw = str(row_user["noi_dung"])
                ans_raw = str(r["noi_dung"])
                dg = str(r["danh_gia"])
                sua_raw = str(r["cau_tra_loi_sua"] or "").strip()

                if dg == "thumbs_down":
                    if not sua_raw:
                        continue
                    chosen_raw = sua_raw
                    rejected_raw = ans_raw
                else:
                    chosen_raw = ans_raw
                    rejected_raw = "Xin lỗi, tôi không có thông tin về câu hỏi này."

                if an_pii:
                    prompt_clean = DataSanitizer.lam_sach(prompt_raw)
                    chosen_clean = DataSanitizer.lam_sach(chosen_raw)
                    rejected_clean = DataSanitizer.lam_sach(rejected_raw)
                else:
                    prompt_clean, chosen_clean, rejected_clean = prompt_raw, chosen_raw, rejected_raw

                cac_mau.append(
                    {
                        "prompt": prompt_clean,
                        "instruction": prompt_clean,
                        "chosen": chosen_clean,
                        "rejected": rejected_clean,
                        "output": chosen_clean,
                        "nguon_danh_gia": dg,
                        "ma_workspace": str(r["ma_workspace"]),
                    }
                )

        bao_cao_loc_nhieu: Optional[Dict[str, Any]] = None
        if loc_nhieu:
            from vietnamese_ai.fine_tuning.preference_filter import BoLocNhieuCapUuTien

            bo_loc = BoLocNhieuCapUuTien()
            kq_loc = bo_loc.loc_tap_du_lieu(cac_mau, mo_hinh_tham_chieu=mo_hinh_tham_chieu)
            cac_mau = kq_loc.mau_sach
            bao_cao_loc_nhieu = {
                "tong_dau_vao": kq_loc.tong_dau_vao,
                "so_giu_lai": kq_loc.so_giu_lai,
                "so_cach_ly": kq_loc.so_cach_ly,
                "ty_le_giu_lai": kq_loc.ty_le_giu_lai,
                "thong_ke_ly_do_loai": kq_loc.thong_ke_ly_do_loai,
                "trong_so_trung_binh": kq_loc.trong_so_trung_binh,
            }

        duong_dan_xuat: Optional[str] = None
        if p_out_resolved is not None:
            p_out_resolved.parent.mkdir(parents=True, exist_ok=True)
            with open(p_out_resolved, "w", encoding="utf-8") as f:
                for rec in cac_mau:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            duong_dan_xuat = str(p_out_resolved)

        ma_bam_dvc: Optional[str] = None
        if ho_du_lieu is not None and cac_mau and hasattr(ho_du_lieu, "nhap_danh_sach"):
            ma_bam_dvc = str(
                ho_du_lieu.nhap_danh_sach(
                    cac_mau,
                    ten_dataset_dvc,
                    tin_nhan_commit="Xuất phản hồi Thumbs Up/Down từ Studio sang tập DPO",
                )
            )

        self._ghi_audit("xuat_dpo", user, f"Xuất {len(cac_mau)} mẫu DPO", {"duong_dan": duong_dan_xuat})
        ket_qua_xuat: Dict[str, Any] = {
            "so_mau_dpo": len(cac_mau),
            "da_an_pii": an_pii,
            "da_loc_nhieu": bool(loc_nhieu),
            "duong_dan_jsonl": duong_dan_xuat,
            "ma_bam_dvc": ma_bam_dvc,
            "mau_dpo": cac_mau,
        }
        if bao_cao_loc_nhieu is not None:
            ket_qua_xuat["bao_cao_loc_nhieu"] = bao_cao_loc_nhieu
        return ket_qua_xuat

    def dong(self) -> None:
        """Đóng toàn bộ kết nối SQLite và `.vaidb` an toàn."""
        with self._lock:
            for rag in self._rag_cache.values():
                rag.dong()
            self._rag_cache.clear()
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass


StudioWorkspaceManager = QuanLyPhienChatWorkspace

__all__ = [
    "QuanLyPhienChatWorkspace",
    "StudioWorkspaceManager",
]
