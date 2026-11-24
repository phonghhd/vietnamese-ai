"""KhoTriThucVAIDB (VAIDBStore) - Kho Tri Thức Bền Vững Một Tệp Duy Nhất `.vaidb` (SQLite + FTS5 + Vector Index)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.rag.vector_store import KetQuaTaiLieu
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

try:
    import vietnamese_ai_rust as _rust_core  # type: ignore
except ImportError:
    try:
        from vietnamese_ai import vietnamese_ai_rust as _rust_core  # type: ignore
    except ImportError:
        _rust_core = None


def _bo_dau_tieng_viet(van_ban: str) -> str:
    """Chuyển văn bản tiếng Việt sang dạng không dấu viết thường để hỗ trợ tìm kiếm FTS5 linh hoạt."""
    if not van_ban:
        return ""
    thay_d = van_ban.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", thay_d)
    khong_dau = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", khong_dau).lower()


def _tao_truy_van_fts5(cau_hoi: str) -> str:
    """Biến câu hỏi tự nhiên tiếng Việt thành biểu thức FTS5 an toàn (tránh lỗi cú pháp ký tự đặc biệt)."""
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹ_]+", cau_hoi.lower())
    unaccent_tokens = re.findall(r"[0-9a-z_]+", _bo_dau_tieng_viet(cau_hoi))
    hop_tu: List[str] = []
    for tok in tokens + unaccent_tokens:
        if len(tok) >= 2 and tok not in hop_tu:
            hop_tu.append(tok)
    if not hop_tu:
        return ""
    return " OR ".join(f'"{tok}"' for tok in hop_tu[:24])


def _tinh_sha256_chuoi(noi_dung: str) -> str:
    """Tính mã băm SHA-256 cho chuỗi nội dung văn bản."""
    return hashlib.sha256(noi_dung.encode("utf-8")).hexdigest()


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Kho tri thức bền vững một tệp duy nhất `.vaidb` kết hợp SQLite WAL, chỉ mục toàn văn FTS5 BM25 "
        "(có dấu & không dấu tiếng Việt), lưu trữ vector float32 và đồng bộ tăng dần theo SHA-256."
    ),
    nhom_extra="core",
)
class KhoTriThucVAIDB:
    """
    Cơ sở dữ liệu tri thức bền vững một tệp duy nhất (`.vaidb`) cho RAG tiếng Việt.

    Tương thích giao diện `CSDLVector`, hỗ trợ mở lại tức thì (~5–15 ms), tìm kiếm vector cosine/L2/IP,
    tìm kiếm toàn văn SQLite FTS5 BM25 (cả tiếng Việt có dấu lẫn không dấu), và cập nhật tăng dần
    theo mã băm SHA-256 mà không cần dựng máy chủ Vector DB rời.
    """

    PHIEN_BAN_DINH_DANG = "vaidb_v1"

    def __init__(
        self,
        duong_dan: Union[str, Path] = "vietnamese_ai_knowledge.vaidb",
        kich_thuoc: int = 384,
        khoang_cach: str = "cosine",
        suc_chua_toi_da: int = 500000,
    ) -> None:
        if khoang_cach not in ("cosine", "l2", "inner_product"):
            raise ValueError("khoang_cach phải là: cosine, l2, inner_product")

        self.duong_dan_str = str(duong_dan)
        self.la_bo_nho_tam = self.duong_dan_str == ":memory:"
        self.duong_dan = Path(self.duong_dan_str) if not self.la_bo_nho_tam else Path(":memory:")
        if not self.la_bo_nho_tam:
            self.duong_dan.parent.mkdir(parents=True, exist_ok=True)

        self.kich_thuoc = int(kich_thuoc)
        self.khoang_cach = khoang_cach
        self.suc_chua_toi_da = int(suc_chua_toi_da)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.duong_dan_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._co_fts5 = False

        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._vector_norms: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._bo_nhung_mac_dinh: Optional[Any] = None

        self._ivf_centroids: Optional[np.ndarray] = None
        self._ivf_buckets: Dict[int, np.ndarray] = {}
        self._ivf_nprobe: int = 3
        self._ivf_dirty: bool = True

        self._khoi_tao_luoc_do()
        self._nap_bo_nho_dem_vector()

    def _khoi_tao_luoc_do(self) -> None:
        """Khởi tạo cấu trúc bảng SQLite, WAL mode và bảng ảo FTS5."""
        with self._lock:
            cur = self._conn.cursor()
            if not self.la_bo_nho_tam:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vai_meta (
                    khoa TEXT PRIMARY KEY,
                    gia_tri TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vai_documents (
                    ma_tai_lieu TEXT PRIMARY KEY,
                    sha256_noi_dung TEXT NOT NULL,
                    duong_dan_nguon TEXT,
                    kich_thuoc_bytes INTEGER DEFAULT 0,
                    so_chunks INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    cap_nhat_luc TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vai_chunks (
                    ma_chunk TEXT PRIMARY KEY,
                    ma_tai_lieu TEXT NOT NULL,
                    noi_dung TEXT NOT NULL,
                    noi_dung_khong_dau TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    vector_norm REAL NOT NULL,
                    cap_nhat_luc TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vai_chunks_doc ON vai_chunks(ma_tai_lieu)")

            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vai_chunks_fts USING fts5(
                        ma_chunk UNINDEXED,
                        ma_tai_lieu UNINDEXED,
                        noi_dung,
                        noi_dung_khong_dau,
                        tokenize='unicode61'
                    )
                    """
                )
                self._co_fts5 = True
            except sqlite3.OperationalError:
                self._co_fts5 = False

            row_dim = cur.execute("SELECT gia_tri FROM vai_meta WHERE khoa='kich_thuoc'").fetchone()
            if row_dim is not None:
                self.kich_thuoc = int(row_dim["gia_tri"])
            else:
                cur.execute(
                    "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
                    ("phien_ban", self.PHIEN_BAN_DINH_DANG),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
                    ("kich_thuoc", str(self.kich_thuoc)),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
                    ("khoang_cach", self.khoang_cach),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
                    ("tao_luc", datetime.now(timezone.utc).isoformat()),
                )

            row_dist = cur.execute("SELECT gia_tri FROM vai_meta WHERE khoa='khoang_cach'").fetchone()
            if row_dist is not None:
                self.khoang_cach = str(row_dist["gia_tri"])

            self._conn.commit()

    def _nap_bo_nho_dem_vector(self) -> None:
        """Nạp toàn bộ ma trận vector float32 và metadata từ SQLite lên RAM để truy vấn siêu tốc."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ma_chunk, metadata_json, vector_blob, vector_norm FROM vai_chunks ORDER BY rowid ASC"
            ).fetchall()
            self._ids.clear()
            self._metadata.clear()
            self._id_to_idx.clear()
            self._ivf_dirty = True

            if not rows:
                self._vectors = None
                self._vector_norms = None
                return

            vecs: List[np.ndarray] = []
            norms: List[float] = []
            for idx, row in enumerate(rows):
                ma = str(row["ma_chunk"])
                vec = np.frombuffer(row["vector_blob"], dtype=np.float32).copy()
                if idx == 0 and vec.shape[0] != self.kich_thuoc:
                    self.kich_thuoc = int(vec.shape[0])
                self._ids.append(ma)
                self._metadata.append(json.loads(str(row["metadata_json"])))
                self._id_to_idx[ma] = idx
                vecs.append(vec)
                norms.append(float(row["vector_norm"]))

            self._vectors = np.vstack(vecs).astype(np.float32)
            self._vector_norms = np.asarray(norms, dtype=np.float32)

    def _cap_nhat_kich_thuoc_meta(self, kich_thuoc_moi: int) -> None:
        """Đồng bộ kích thước vector mới vào bảng `vai_meta` khi kho còn trống."""
        self.kich_thuoc = int(kich_thuoc_moi)
        self._conn.execute(
            "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES ('kich_thuoc', ?)",
            (str(self.kich_thuoc),),
        )

    def _luu_mot_ban_ghi_sql(
        self,
        ma: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        now_iso: str,
    ) -> None:
        """Ghi một bản ghi chunk + FTS5 trong cùng giao dịch SQLite hiện hành."""
        ma_tai_lieu = str(metadata.get("tai_lieu_goc") or ma.split("_chunk_")[0])
        noi_dung = str(metadata.get("noi_dung") or metadata.get("page_content") or metadata.get("text") or "")
        khong_dau = _bo_dau_tieng_viet(noi_dung)
        meta_json = json.dumps(metadata, ensure_ascii=False, default=str)
        vec_norm = float(np.linalg.norm(vector))

        self._conn.execute(
            """
            INSERT OR REPLACE INTO vai_chunks (
                ma_chunk, ma_tai_lieu, noi_dung, noi_dung_khong_dau,
                metadata_json, vector_blob, vector_norm, cap_nhat_luc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ma, ma_tai_lieu, noi_dung, khong_dau, meta_json, vector.tobytes(), vec_norm, now_iso),
        )
        if self._co_fts5:
            self._conn.execute("DELETE FROM vai_chunks_fts WHERE ma_chunk = ?", (ma,))
            self._conn.execute(
                """
                INSERT INTO vai_chunks_fts (ma_chunk, ma_tai_lieu, noi_dung, noi_dung_khong_dau)
                VALUES (?, ?, ?, ?)
                """,
                (ma, ma_tai_lieu, noi_dung, khong_dau),
            )

    def chen(
        self,
        ma: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Chèn hoặc cập nhật bền vững một vector và metadata vào tệp `.vaidb`."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        meta = dict(metadata) if metadata else {}
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if len(self._ids) == 0 and vec.shape[0] != self.kich_thuoc:
                self._cap_nhat_kich_thuoc_meta(int(vec.shape[0]))
                self._vectors = None
                self._vector_norms = None
            if vec.shape[0] != self.kich_thuoc:
                raise ValueError(f"Vector phải có {self.kich_thuoc} chiều, nhận được {vec.shape[0]}")
            if ma not in self._id_to_idx and len(self._ids) >= self.suc_chua_toi_da:
                raise RuntimeError(
                    f"Đã đạt giới hạn {self.suc_chua_toi_da} vectors trong KhoTriThucVAIDB."
                )

            self._luu_mot_ban_ghi_sql(ma, vec, meta, now_iso)
            self._conn.commit()

            vec_norm = float(np.linalg.norm(vec))
            if ma in self._id_to_idx:
                idx = self._id_to_idx[ma]
                if self._vectors is not None:
                    self._vectors[idx] = vec
                if self._vector_norms is not None:
                    self._vector_norms[idx] = vec_norm
                self._metadata[idx] = meta
                self._ivf_dirty = True
                return

            idx = len(self._ids)
            self._ids.append(ma)
            self._metadata.append(meta)
            self._id_to_idx[ma] = idx
            self._ivf_dirty = True

            if self._vectors is None:
                self._vectors = vec.reshape(1, -1)
                self._vector_norms = np.asarray([vec_norm], dtype=np.float32)
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])
                assert self._vector_norms is not None
                self._vector_norms = np.append(self._vector_norms, np.float32(vec_norm))

    def chen_batch(
        self,
        ma_list: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Chèn hàng loạt nhiều vector trong một giao dịch SQLite nguyên tử."""
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if metadata_list is None:
            metadata_list = [{}] * len(ma_list)
        if len(ma_list) == 0:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if len(self._ids) == 0 and vecs.shape[1] != self.kich_thuoc:
                self._cap_nhat_kich_thuoc_meta(int(vecs.shape[1]))
            if vecs.shape[1] != self.kich_thuoc:
                raise ValueError(f"Vector phải có {self.kich_thuoc} chiều, nhận được {vecs.shape[1]}")

            for ma, vec, meta in zip(ma_list, vecs, metadata_list):
                self._luu_mot_ban_ghi_sql(ma, vec, dict(meta or {}), now_iso)
            self._conn.commit()
            self._nap_bo_nho_dem_vector()

    def ghi_nhan_tai_lieu(
        self,
        ma_tai_lieu: str,
        sha256_noi_dung: str,
        so_chunks: int,
        duong_dan_nguon: Optional[str] = None,
        kich_thuoc_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ghi nhận hồ sơ tài liệu kèm mã băm SHA-256 để hỗ trợ đồng bộ tăng dần."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vai_documents (
                    ma_tai_lieu, sha256_noi_dung, duong_dan_nguon,
                    kich_thuoc_bytes, so_chunks, metadata_json, cap_nhat_luc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ma_tai_lieu,
                    sha256_noi_dung,
                    duong_dan_nguon,
                    int(kich_thuoc_bytes),
                    int(so_chunks),
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    now_iso,
                ),
            )
            self._conn.commit()

    def da_ton_tai_tai_lieu(self, ma_tai_lieu: str, sha256_noi_dung: Optional[str] = None) -> bool:
        """Kiểm tra một tài liệu đã được lập chỉ mục với đúng mã băm nội dung SHA-256 hay chưa."""
        with self._lock:
            row = self._conn.execute(
                "SELECT sha256_noi_dung FROM vai_documents WHERE ma_tai_lieu = ?",
                (ma_tai_lieu,),
            ).fetchone()
            if row is None:
                return False
            if sha256_noi_dung is None:
                return True
            return str(row["sha256_noi_dung"]) == sha256_noi_dung

    def danh_sach_tai_lieu(self) -> List[Dict[str, Any]]:
        """Trả về danh sách các tài liệu đã lập chỉ mục trong `.vaidb`."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ma_tai_lieu, sha256_noi_dung, duong_dan_nguon,
                       kich_thuoc_bytes, so_chunks, metadata_json, cap_nhat_luc
                FROM vai_documents
                ORDER BY ma_tai_lieu ASC
                """
            ).fetchall()
            if rows:
                return [
                    {
                        "ma_tai_lieu": str(r["ma_tai_lieu"]),
                        "sha256_noi_dung": str(r["sha256_noi_dung"]),
                        "duong_dan_nguon": r["duong_dan_nguon"],
                        "kich_thuoc_bytes": int(r["kich_thuoc_bytes"]),
                        "so_chunks": int(r["so_chunks"]),
                        "metadata": json.loads(str(r["metadata_json"] or "{}")),
                        "cap_nhat_luc": str(r["cap_nhat_luc"]),
                    }
                    for r in rows
                ]
            fallback_rows = self._conn.execute(
                """
                SELECT ma_tai_lieu, COUNT(*) AS so_chunks, MAX(cap_nhat_luc) AS cap_nhat_luc
                FROM vai_chunks
                GROUP BY ma_tai_lieu
                ORDER BY ma_tai_lieu ASC
                """
            ).fetchall()
            return [
                {
                    "ma_tai_lieu": str(r["ma_tai_lieu"]),
                    "sha256_noi_dung": "",
                    "duong_dan_nguon": None,
                    "kich_thuoc_bytes": 0,
                    "so_chunks": int(r["so_chunks"]),
                    "metadata": {},
                    "cap_nhat_luc": str(r["cap_nhat_luc"]),
                }
                for r in fallback_rows
            ]

    def xoa(self, ma: str) -> bool:
        """Xóa một chunk theo mã ID."""
        with self._lock:
            if ma not in self._id_to_idx:
                return False
            self._conn.execute("DELETE FROM vai_chunks WHERE ma_chunk = ?", (ma,))
            if self._co_fts5:
                self._conn.execute("DELETE FROM vai_chunks_fts WHERE ma_chunk = ?", (ma,))
            self._conn.commit()
            self._nap_bo_nho_dem_vector()
            return True

    def xoa_tai_lieu(self, ma_tai_lieu: str) -> int:
        """Xóa toàn bộ chunks và hồ sơ của một tài liệu theo `ma_tai_lieu`."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM vai_chunks WHERE ma_tai_lieu = ?",
                (ma_tai_lieu,),
            )
            da_xoa = int(cur.rowcount or 0)
            if self._co_fts5:
                self._conn.execute(
                    "DELETE FROM vai_chunks_fts WHERE ma_tai_lieu = ?",
                    (ma_tai_lieu,),
                )
            self._conn.execute(
                "DELETE FROM vai_documents WHERE ma_tai_lieu = ?",
                (ma_tai_lieu,),
            )
            self._conn.commit()
            if da_xoa > 0:
                self._nap_bo_nho_dem_vector()
            return da_xoa

    def xoa_tat_ca(self) -> None:
        """Xóa toàn bộ dữ liệu trong tệp `.vaidb`."""
        with self._lock:
            self._conn.execute("DELETE FROM vai_chunks")
            self._conn.execute("DELETE FROM vai_documents")
            if self._co_fts5:
                self._conn.execute("DELETE FROM vai_chunks_fts")
            self._conn.commit()
            self._ids.clear()
            self._vectors = None
            self._vector_norms = None
            self._metadata.clear()
            self._id_to_idx.clear()
            self._ivf_centroids = None
            self._ivf_buckets.clear()
            self._ivf_dirty = True

    def lay_metadata(self, ma: str) -> Optional[Dict[str, Any]]:
        """Lấy metadata của một chunk theo mã ID."""
        with self._lock:
            idx = self._id_to_idx.get(ma)
            if idx is None:
                return None
            return dict(self._metadata[idx])

    def so_luong(self) -> int:
        """Tổng số chunks hiện có trong kho `.vaidb`."""
        with self._lock:
            return len(self._ids)

    def so_tai_lieu(self) -> int:
        """Tổng số tài liệu gốc riêng biệt hiện có trong kho `.vaidb`."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM vai_documents").fetchone()
            if row is not None and int(row["n"]) > 0:
                return int(row["n"])
            row_c = self._conn.execute("SELECT COUNT(DISTINCT ma_tai_lieu) AS n FROM vai_chunks").fetchone()
            return int(row_c["n"]) if row_c is not None else 0

    def danh_sach_ma(self) -> List[str]:
        """Danh sách tất cả mã chunk trong kho."""
        with self._lock:
            return self._ids.copy()

    def _nhung_chuoi(self, van_ban: str) -> np.ndarray:
        """Nhúng chuỗi văn bản khi gọi trực tiếp `them_tai_lieu` hoặc `tim_kiem(str)`."""
        from vietnamese_ai.rag.onnx_semantic import BoNhungONNXTiengViet

        if (
            self._bo_nhung_mac_dinh is None
            or getattr(self._bo_nhung_mac_dinh, "kich_thuoc", None) != self.kich_thuoc
        ):
            self._bo_nhung_mac_dinh = BoNhungONNXTiengViet(kich_thuoc=self.kich_thuoc)
        return self._bo_nhung_mac_dinh.nhung_mot(van_ban)

    def them_tai_lieu(
        self,
        id_tai_lieu: str,
        noi_dung: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Nhúng và lưu trực tiếp một văn bản vào kho `.vaidb`."""
        meta = dict(metadata) if metadata else {}
        meta.setdefault("noi_dung", noi_dung)
        meta.setdefault("page_content", noi_dung)
        meta.setdefault("tai_lieu_goc", id_tai_lieu)
        vec = self._nhung_chuoi(noi_dung)
        self.chen(id_tai_lieu, vec, meta)
        self.ghi_nhan_tai_lieu(
            ma_tai_lieu=id_tai_lieu,
            sha256_noi_dung=_tinh_sha256_chuoi(noi_dung),
            so_chunks=1,
            kich_thuoc_bytes=len(noi_dung.encode("utf-8")),
            metadata=meta,
        )

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Thêm danh sách văn bản vào kho `.vaidb`."""
        ids: List[str] = []
        for i, text in enumerate(texts):
            doc_id = f"doc_{len(self._ids) + 1}_{uuid.uuid4().hex[:6]}"
            meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            self.them_tai_lieu(doc_id, text, meta)
            ids.append(doc_id)
        return ids

    def xay_dung_chi_muc_ivf(
        self,
        so_cum: int = 16,
        so_vong_lap: int = 8,
        nprobe: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Xây dựng chỉ mục gần đúng IVF-Flat trên bộ nhớ đệm vector của `.vaidb`."""
        with self._lock:
            if self._vectors is None or len(self._ids) == 0:
                self._ivf_centroids = None
                self._ivf_buckets = {}
                self._ivf_dirty = False
                return {"da_xay_dung": False, "so_cum": 0, "so_vector": 0}

            n_vecs = len(self._ids)
            k = max(1, min(int(so_cum), n_vecs))
            norms = np.maximum(np.linalg.norm(self._vectors, axis=1, keepdims=True), 1e-10)
            vecs_norm = (self._vectors / norms).astype(np.float32)

            rng = np.random.RandomState(seed)
            init_idx = rng.choice(n_vecs, size=k, replace=False)
            centroids = vecs_norm[init_idx].copy()

            labels = np.zeros(n_vecs, dtype=np.int32)
            for _ in range(max(1, int(so_vong_lap))):
                sims = vecs_norm @ centroids.T
                labels = np.argmax(sims, axis=1).astype(np.int32)
                for c in range(k):
                    members = vecs_norm[labels == c]
                    if len(members) > 0:
                        mean_c = np.mean(members, axis=0)
                        n_c = float(np.linalg.norm(mean_c))
                        centroids[c] = (mean_c / n_c if n_c > 1e-10 else mean_c).astype(np.float32)

            buckets: Dict[int, np.ndarray] = {}
            for c in range(k):
                idx_c = np.where(labels == c)[0]
                if len(idx_c) > 0:
                    buckets[c] = idx_c.astype(np.int32)

            self._ivf_centroids = centroids
            self._ivf_buckets = buckets
            self._ivf_nprobe = max(1, min(int(nprobe), k))
            self._ivf_dirty = False
            return {
                "da_xay_dung": True,
                "so_cum": k,
                "nprobe": self._ivf_nprobe,
                "so_vector": n_vecs,
            }

    @staticmethod
    def _khop_bo_loc(meta: Dict[str, Any], bo_loc: Optional[Dict[str, Any]]) -> bool:
        """Kiểm tra metadata có thỏa điều kiện lọc key-value hay không."""
        if not bo_loc:
            return True
        return all(meta.get(k) == v for k, v in bo_loc.items())

    def tim_kiem(
        self,
        query_vector: Union[np.ndarray, str],
        top_k: int = 5,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm vector gần nhất trên kho `.vaidb` (hỗ trợ tăng tốc Rust Core & IVF-Flat)."""
        with self._lock:
            if self._vectors is None or len(self._ids) == 0:
                return []

            if isinstance(query_vector, str):
                query = self._nhung_chuoi(query_vector).astype(np.float32).flatten()
            else:
                query = np.asarray(query_vector, dtype=np.float32).flatten()

            if query.shape[0] != self.kich_thuoc:
                raise ValueError(f"Query vector phải có {self.kich_thuoc} chiều, nhận được {query.shape[0]}")

            cand_indices: Optional[np.ndarray] = None
            if (
                not self._ivf_dirty
                and self._ivf_centroids is not None
                and self._ivf_buckets
                and bo_loc is None
                and self.khoang_cach == "cosine"
            ):
                q_norm = float(np.linalg.norm(query))
                q_unit = query / q_norm if q_norm > 1e-10 else query
                c_sims = self._ivf_centroids @ q_unit
                top_c = np.argsort(c_sims)[::-1][: self._ivf_nprobe]
                picked = [self._ivf_buckets[int(c)] for c in top_c if int(c) in self._ivf_buckets]
                if picked:
                    cand_indices = np.concatenate(picked)

            if cand_indices is not None and len(cand_indices) > 0:
                sub_vecs = self._vectors[cand_indices]
                q_norm = float(np.linalg.norm(query))
                if q_norm < 1e-10:
                    sub_scores = np.zeros(len(cand_indices), dtype=np.float32)
                else:
                    sub_norms = (
                        self._vector_norms[cand_indices]
                        if self._vector_norms is not None
                        else np.linalg.norm(sub_vecs, axis=1)
                    )
                    sub_norms = np.maximum(sub_norms, 1e-10)
                    sub_scores = (sub_vecs @ query) / (sub_norms * q_norm)
                order = np.argsort(sub_scores)[::-1]
                ranked_pairs = [(int(cand_indices[i]), float(sub_scores[i])) for i in order]
            else:
                if self.khoang_cach == "cosine":
                    if _rust_core is not None and hasattr(_rust_core, "tim_kiem_cosine_rust"):
                        try:
                            rust_res = _rust_core.tim_kiem_cosine_rust(
                                query.astype(np.float32).tolist(),
                                self._vectors.astype(np.float32).tolist(),
                                min(len(self._ids), max(top_k * 4, top_k)),
                            )
                            ranked_pairs = [(int(idx), float(sc)) for idx, sc in rust_res]
                        except Exception:
                            ranked_pairs = self._tinh_diem_toan_bo(query)
                    else:
                        ranked_pairs = self._tinh_diem_toan_bo(query)
                else:
                    ranked_pairs = self._tinh_diem_toan_bo(query)

            ket_qua: List[Dict[str, Any]] = []
            for idx, score in ranked_pairs:
                if nguong is not None and score < nguong:
                    continue
                meta = self._metadata[idx]
                if not self._khop_bo_loc(meta, bo_loc):
                    continue
                ket_qua.append(
                    {
                        "ma": self._ids[idx],
                        "diem": float(score),
                        "metadata": dict(meta),
                    }
                )
                if len(ket_qua) >= top_k:
                    break
            return ket_qua

    def _tinh_diem_toan_bo(self, query: np.ndarray) -> List[Tuple[int, float]]:
        """Tính điểm tương đồng vector đầy đủ bằng NumPy."""
        assert self._vectors is not None
        if self.khoang_cach == "cosine":
            q_norm = float(np.linalg.norm(query))
            if q_norm < 1e-10:
                scores = np.zeros(len(self._ids), dtype=np.float32)
            else:
                norms = (
                    self._vector_norms
                    if self._vector_norms is not None
                    else np.linalg.norm(self._vectors, axis=1)
                )
                norms = np.maximum(norms, 1e-10)
                scores = (self._vectors @ query) / (norms * q_norm)
        elif self.khoang_cach == "l2":
            diffs = self._vectors - query.reshape(1, -1)
            scores = 1.0 / (1.0 + np.linalg.norm(diffs, axis=1))
        else:
            scores = self._vectors @ query

        order = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in order]

    def tim_kiem_fts(
        self,
        cau_hoi: str,
        top_k: int = 5,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm toàn văn bằng SQLite FTS5 BM25 trên cả văn bản có dấu và không dấu tiếng Việt."""
        if not cau_hoi or not cau_hoi.strip():
            return []

        with self._lock:
            if not self._ids:
                return []

            if self._co_fts5:
                expr = _tao_truy_van_fts5(cau_hoi)
                if not expr:
                    return []
                try:
                    rows = self._conn.execute(
                        """
                        SELECT ma_chunk, bm25(vai_chunks_fts) AS rank_bm25
                        FROM vai_chunks_fts
                        WHERE vai_chunks_fts MATCH ?
                        ORDER BY rank_bm25 ASC
                        LIMIT ?
                        """,
                        (expr, max(top_k * 4, 20)),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

                if not rows:
                    return []

                raw_scores = [abs(float(r["rank_bm25"])) for r in rows]
                max_raw = max(raw_scores) if raw_scores else 1.0
                ket_qua: List[Dict[str, Any]] = []
                for row, raw_s in zip(rows, raw_scores):
                    ma = str(row["ma_chunk"])
                    meta = self.lay_metadata(ma)
                    if meta is None or not self._khop_bo_loc(meta, bo_loc):
                        continue
                    diem_chuan = float(1.0 / (1.0 + raw_s)) if max_raw <= 0 else float(max(0.05, 1.0 - (raw_s / (max_raw + 1.0))))
                    if nguong is not None and diem_chuan < nguong:
                        continue
                    ket_qua.append(
                        {
                            "ma": ma,
                            "diem": diem_chuan,
                            "metadata": meta,
                            "loai": "fts5_bm25",
                        }
                    )
                    if len(ket_qua) >= top_k:
                        break
                return ket_qua

            q_tokens = set(re.findall(r"[0-9a-z_]+", _bo_dau_tieng_viet(cau_hoi)))
            if not q_tokens:
                return []
            scored: List[Tuple[float, int]] = []
            for idx, meta in enumerate(self._metadata):
                if not self._khop_bo_loc(meta, bo_loc):
                    continue
                nd_kd = _bo_dau_tieng_viet(str(meta.get("noi_dung", "")))
                d_tokens = set(re.findall(r"[0-9a-z_]+", nd_kd))
                if not d_tokens:
                    continue
                ov = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
                if ov > 0:
                    scored.append((ov, idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            out: List[Dict[str, Any]] = []
            for sc, idx in scored[:top_k]:
                if nguong is not None and sc < nguong:
                    continue
                out.append(
                    {
                        "ma": self._ids[idx],
                        "diem": float(sc),
                        "metadata": dict(self._metadata[idx]),
                        "loai": "fts_fallback",
                    }
                )
            return out

    def tim_kiem_lai(
        self,
        cau_hoi: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        trong_so_semantic: float = 0.65,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm lai trực tiếp trên `.vaidb` (kết hợp Vector Cosine và SQLite FTS5 BM25)."""
        q_vec = query_vector if query_vector is not None else self._nhung_chuoi(cau_hoi)
        sem_res = self.tim_kiem(q_vec, top_k=top_k * 2, nguong=None, bo_loc=bo_loc)
        fts_res = self.tim_kiem_fts(cau_hoi, top_k=top_k * 2, nguong=None, bo_loc=bo_loc)

        hop: Dict[str, Dict[str, Any]] = {}
        for item in sem_res:
            hop[item["ma"]] = {
                "ma": item["ma"],
                "diem_semantic": float(item["diem"]),
                "diem_fts": 0.0,
                "metadata": item["metadata"],
            }
        for item in fts_res:
            ma = item["ma"]
            if ma in hop:
                hop[ma]["diem_fts"] = float(item["diem"])
            else:
                hop[ma] = {
                    "ma": ma,
                    "diem_semantic": 0.0,
                    "diem_fts": float(item["diem"]),
                    "metadata": item["metadata"],
                }

        w_sem = float(np.clip(trong_so_semantic, 0.0, 1.0))
        ket_qua: List[Dict[str, Any]] = []
        for info in hop.values():
            diem = w_sem * info["diem_semantic"] + (1.0 - w_sem) * info["diem_fts"]
            if nguong is not None and diem < nguong:
                continue
            ket_qua.append(
                {
                    "ma": info["ma"],
                    "diem": float(diem),
                    "metadata": info["metadata"],
                    "loai": "vaidb_hybrid",
                }
            )
        ket_qua.sort(key=lambda x: x["diem"], reverse=True)
        return ket_qua[:top_k]

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[KetQuaTaiLieu]:
        """Tìm kiếm tài liệu tương đồng trả về `KetQuaTaiLieu`."""
        kq = self.tim_kiem_lai(query, top_k=top_k, bo_loc=bo_loc)
        docs: List[KetQuaTaiLieu] = []
        for item in kq:
            meta = item.get("metadata", {})
            nd = str(meta.get("page_content") or meta.get("noi_dung") or meta.get("text") or "")
            docs.append(
                KetQuaTaiLieu(
                    page_content=nd,
                    metadata=meta,
                    diem=float(item.get("diem", 0.0)),
                    ma=str(item.get("ma", "")),
                )
            )
        return docs

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thống kê đầy đủ về tệp tri thức `.vaidb`."""
        with self._lock:
            kich_thuoc_tep = (
                self.duong_dan.stat().st_size
                if not self.la_bo_nho_tam and self.duong_dan.exists()
                else 0
            )
            return {
                "dinh_dang": self.PHIEN_BAN_DINH_DANG,
                "duong_dan": self.duong_dan_str,
                "so_tai_lieu": self.so_tai_lieu(),
                "so_luong": len(self._ids),
                "kich_thuoc": self.kich_thuoc,
                "khoang_cach": self.khoang_cach,
                "co_fts5": self._co_fts5,
                "chi_muc_ivf": self._ivf_centroids is not None and not self._ivf_dirty,
                "kich_thuoc_tep_bytes": int(kich_thuoc_tep),
            }

    def dong(self) -> None:
        """Đóng kết nối SQLite an toàn."""
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def close(self) -> None:
        """Alias tiếng Anh cho `dong()`."""
        self.dong()

    def __enter__(self) -> "KhoTriThucVAIDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dong()

    def __repr__(self) -> str:
        return (
            f"KhoTriThucVAIDB(duong_dan='{self.duong_dan_str}', "
            f"so_tai_lieu={self.so_tai_lieu()}, so_chunks={self.so_luong()}, "
            f"fts5={self._co_fts5})"
        )


VAIDBStore = KhoTriThucVAIDB
SQLiteVectorStore = KhoTriThucVAIDB

__all__ = [
    "KhoTriThucVAIDB",
    "VAIDBStore",
    "SQLiteVectorStore",
]
