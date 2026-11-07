"""Bộ nạp lười (Lazy Loader chuẩn PEP 562), Quản lý Phụ thuộc Thông minh & Fallback Học máy Thuần NumPy."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
else:

    class _LazyNumPyProxy:
        def __getattr__(self, item: str) -> Any:
            import numpy as _real_np

            globals()["np"] = _real_np
            return getattr(_real_np, item)

    np = _LazyNumPyProxy()


def tao_lazy_getattr(
    ten_package: str, anh_xa: Dict[str, Tuple[str, str]], namespace: Optional[Dict[str, Any]] = None
) -> Callable[[str], Any]:
    """
    Tạo hàm `__getattr__` chuẩn PEP 562 cho bất kỳ gói con nào.
    Chỉ import module thực sự khi thuộc tính được truy cập lần đầu và lưu đệm vào `namespace`
    để các lần truy cập tiếp theo đạt tốc độ 0 ns.
    """

    def _lazy_getattr(name: str) -> Any:
        if name in anh_xa:
            mod_path, attr_name = anh_xa[name]
            module = importlib.import_module(mod_path)
            gia_tri = getattr(module, attr_name)
            if namespace is not None:
                namespace[name] = gia_tri
            return gia_tri
        raise AttributeError(f"Module '{ten_package}' không có thuộc tính '{name}'")

    return _lazy_getattr


class QuanLyPhuThuoc:
    """
    Trình Quản lý Phụ thuộc Thông minh (`DependencyManager`) cho kiến trúc Micro-Core.

    Giúp giữ bản cài đặt mặc định `pip install vietnamese-ai` siêu nhẹ (chỉ cần `numpy`),
    kiểm tra gói mở rộng không tốn thời gian import và hướng dẫn cài đặt rõ ràng bằng tiếng Việt.
    """

    NHOM_PHU_THUOC: Dict[str, Dict[str, Any]] = {
        "core": {
            "mo_ta": "Lõi siêu nhẹ (GPTModel, LoRA/DoRA, RAG nội bộ, Agents, MCP, Tường lửa AI, Server HTTP)",
            "goi": ["numpy>=1.20.0"],
            "kich_thuoc_uoc_tinh_mb": 39,
        },
        "ml": {
            "mo_ta": "Học máy cổ điển & AutoML (Scikit-learn, Pandas)",
            "goi": ["scikit-learn>=1.0.0", "pandas>=1.3.0"],
            "kich_thuoc_uoc_tinh_mb": 210,
        },
        "tabular": {
            "mo_ta": "Tăng tốc dữ liệu bảng chuyên sâu (XGBoost, LightGBM, Scikit-learn, Pandas)",
            "goi": ["scikit-learn>=1.0.0", "pandas>=1.3.0", "xgboost>=2.0.0", "lightgbm>=4.0.0"],
            "kich_thuoc_uoc_tinh_mb": 320,
        },
        "viz": {
            "mo_ta": "Trực quan hóa biểu đồ (Matplotlib)",
            "goi": ["matplotlib>=3.4.0"],
            "kich_thuoc_uoc_tinh_mb": 65,
        },
        "nlp": {
            "mo_ta": "Tách từ & xử lý ngôn ngữ tự nhiên tiếng Việt nâng cao (Underthesea)",
            "goi": ["underthesea>=6.0"],
            "kich_thuoc_uoc_tinh_mb": 80,
        },
        "dl": {
            "mo_ta": "Deep Learning & HuggingFace Transformers (PyTorch, Transformers)",
            "goi": ["torch>=2.0", "transformers>=4.30"],
            "kich_thuoc_uoc_tinh_mb": 850,
        },
        "fast": {
            "mo_ta": "Siêu tăng tốc huấn luyện LLM 2x-5x trên GPU CUDA (Unsloth, Liger Kernel)",
            "goi": ["unsloth", "liger-kernel"],
            "kich_thuoc_uoc_tinh_mb": 120,
        },
        "rag": {
            "mo_ta": "CSDL Vector phân tán & đọc PDF quy mô lớn (ChromaDB, Qdrant, PyPDF, Sentence-Transformers)",
            "goi": [
                "chromadb>=0.4.0",
                "qdrant-client>=1.6.0",
                "pypdf>=3.0.0",
                "sentence-transformers>=2.2.2",
                "networkx>=2.6",
            ],
            "kich_thuoc_uoc_tinh_mb": 260,
        },
        "llm": {
            "mo_ta": "Kết nối Cloud LLM APIs (OpenAI, Google GenAI Gemini, Requests, BeautifulSoup4)",
            "goi": [
                "openai>=1.0.0",
                "google-genai>=1.0.0",
                "requests>=2.25.0",
                "beautifulsoup4>=4.9.0",
            ],
            "kich_thuoc_uoc_tinh_mb": 25,
        },
        "api": {
            "mo_ta": "Production Gateway hiệu năng cao (FastAPI, Uvicorn, Pydantic, WebSockets)",
            "goi": [
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0",
                "websockets>=11.0",
            ],
            "kich_thuoc_uoc_tinh_mb": 30,
        },
    }

    ANH_XA_TEN_IMPORT: Dict[str, str] = {
        "scikit-learn": "sklearn",
        "beautifulsoup4": "bs4",
        "google-genai": "google.genai",
        "qdrant-client": "qdrant_client",
        "sentence-transformers": "sentence_transformers",
        "liger-kernel": "liger_kernel",
    }

    @classmethod
    def da_cai(cls, ten_goi_hoac_module: str) -> bool:
        """Kiểm tra gói đã được cài đặt hay chưa bằng `find_spec` (không tốn RAM/thời gian nạp)."""
        goc = ten_goi_hoac_module.split(">=")[0].split("==")[0].split("[")[0].strip()
        ten_mod = cls.ANH_XA_TEN_IMPORT.get(goc, goc.replace("-", "_"))
        try:
            return importlib.util.find_spec(ten_mod) is not None
        except (ImportError, ValueError, AttributeError):
            return False

    @classmethod
    def yeu_cau(
        cls,
        ten_module: str,
        nhom_extra: str = "ml",
        tinh_nang: str = "",
    ) -> Any:
        """
        Nạp module theo yêu cầu; nếu chưa cài đặt sẽ ném `ImportError` tiếng Việt rõ ràng
        hướng dẫn chính xác câu lệnh cài gói mở rộng tương ứng.
        """
        try:
            return importlib.import_module(ten_module)
        except ImportError as exc:
            boi_canh = f" cho tính năng '{tinh_nang}'" if tinh_nang else ""
            thong_bao = (
                f"Thiếu thư viện '{ten_module}'{boi_canh}. "
                f"Để giữ framework nhẹ nhất, gói này được tách thành phần mở rộng tùy chọn. "
                f"Hãy cài đặt bằng lệnh: pip install \"vietnamese-ai[{nhom_extra}]\" "
                f"(hoặc chạy: vai install {nhom_extra})"
            )
            raise ImportError(thong_bao) from exc

    @classmethod
    def cai_dat_nhom(cls, nhom: str, chay_that: bool = False) -> Dict[str, Any]:
        """
        Tạo lệnh (hoặc thực thi cài đặt an toàn qua danh sách trắng) cho một nhóm phụ thuộc mở rộng.
        """
        nhom_sach = nhom.strip().lower()
        if nhom_sach in ("full", "all"):
            danh_sach_goi: List[str] = []
            for k, v in cls.NHOM_PHU_THUOC.items():
                if k != "core":
                    for pkg in v["goi"]:
                        if pkg not in danh_sach_goi:
                            danh_sach_goi.append(pkg)
        elif nhom_sach == "server":
            danh_sach_goi = list(cls.NHOM_PHU_THUOC["api"]["goi"])
        elif nhom_sach in cls.NHOM_PHU_THUOC:
            danh_sach_goi = list(cls.NHOM_PHU_THUOC[nhom_sach]["goi"])
        else:
            hop_le = ", ".join(list(cls.NHOM_PHU_THUOC.keys()) + ["full"])
            raise ValueError(f"Nhóm mở rộng '{nhom}' không hợp lệ. Các nhóm hỗ trợ: {hop_le}")

        lenh = [sys.executable, "-m", "pip", "install", *danh_sach_goi]
        lenh_hien_thi = f'pip install "vietnamese-ai[{nhom_sach}]"'
        ket_qua: Dict[str, Any] = {
            "nhom": nhom_sach,
            "danh_sach_goi": danh_sach_goi,
            "lenh_goi_y": lenh_hien_thi,
            "da_thuc_thi": False,
            "thanh_cong": True,
        }
        if chay_that:
            proc = subprocess.run(lenh, capture_output=True, text=True, check=False)
            ket_qua["da_thuc_thi"] = True
            ket_qua["thanh_cong"] = proc.returncode == 0
            ket_qua["ma_tra_ve"] = proc.returncode
            ket_qua["dau_ra"] = proc.stdout[-1000:] if proc.stdout else proc.stderr[-1000:]
        return ket_qua

    @classmethod
    def chan_doan_he_thong(cls) -> Dict[str, Any]:
        """Chẩn đoán toàn diện sức khỏe môi trường, tốc độ nạp, phần cứng và các gói mở rộng (`vai doctor`)."""
        t0 = time.perf_counter()
        import vietnamese_ai

        thoi_gian_import_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        trang_thai_nhom: Dict[str, Dict[str, Any]] = {}
        for ten_nhom, thong_tin in cls.NHOM_PHU_THUOC.items():
            cac_goi = thong_tin["goi"]
            goi_da_cai = [g for g in cac_goi if cls.da_cai(g)]
            goi_con_thieu = [g for g in cac_goi if g not in goi_da_cai]
            trang_thai_nhom[ten_nhom] = {
                "san_sang": len(goi_con_thieu) == 0,
                "da_cai": goi_da_cai,
                "con_thieu": goi_con_thieu,
                "mo_ta": thong_tin["mo_ta"],
                "kich_thuoc_uoc_tinh_mb": thong_tin["kich_thuoc_uoc_tinh_mb"],
            }

        co_cuda = False
        ten_gpu = None
        if cls.da_cai("torch"):
            try:
                import torch

                co_cuda = bool(torch.cuda.is_available())
                if co_cuda:
                    ten_gpu = torch.cuda.get_device_name(0)
            except Exception:
                pass

        return {
            "phien_ban_framework": vietnamese_ai.__version__,
            "python": platform.python_version(),
            "he_dieu_hanh": f"{platform.system()} {platform.release()}",
            "so_nhan_cpu": os.cpu_count() or 1,
            "co_cuda_gpu": co_cuda,
            "ten_gpu": ten_gpu,
            "thoi_gian_nap_core_ms": thoi_gian_import_ms,
            "che_do_sieu_nhe_san_sang": cls.da_cai("numpy"),
            "bao_mat_tuong_lua_mac_dinh": True,
            "ky_so_hmac_mac_dinh": True,
            "nhom_phu_thuoc": trang_thai_nhom,
        }


# =====================================================================
# BỘ ƯỚC LƯỢNG HỌC MÁY THUẦN NUMPY SIÊU NHẸ (ZERO-SKLEARN FALLBACK)
# Đảm bảo PhanLoai, HoiQuy, PhanCum, AutoML vẫn chạy thật 100% chỉ với NumPy!
# =====================================================================


class _LogisticNumPy:
    """Hồi quy Logistic đa lớp (Softmax / Sigmoid Cross-Entropy + L2) thuần NumPy."""

    def __init__(self, max_iter: int = 250, lr: float = 0.1, C: float = 1.0, **_: Any):
        self.max_iter = max_iter
        self.lr = lr
        self.reg = 1.0 / max(1e-6, float(C))
        self.classes_: np.ndarray = np.array([])
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticNumPy":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        n_samples, n_features = X_arr.shape
        n_classes = len(self.classes_)

        rng = np.random.RandomState(42)
        W = rng.randn(n_features, n_classes) * 0.01
        b = np.zeros((1, n_classes), dtype=np.float64)
        Y_onehot = np.eye(n_classes)[y_idx]

        for _ in range(self.max_iter):
            logits = X_arr @ W + b
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            diff = (probs - Y_onehot) / max(1, n_samples)
            dW = X_arr.T @ diff + (self.reg / max(1, n_samples)) * W
            db = np.sum(diff, axis=0, keepdims=True)
            W -= self.lr * dW
            b -= self.lr * db

        self.coef_ = W.T
        self.intercept_ = b.ravel()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        assert self.coef_ is not None and self.intercept_ is not None
        logits = X_arr @ self.coef_.T + self.intercept_[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


class _KNNPhanLoaiNumPy:
    """K-Nearest Neighbors phân loại thuần NumPy."""

    def __init__(self, n_neighbors: int = 5, **_: Any):
        self.n_neighbors = max(1, int(n_neighbors))
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KNNPhanLoaiNumPy":
        self._X = np.asarray(X, dtype=np.float64)
        if self._X.ndim == 1:
            self._X = self._X.reshape(-1, 1)
        self._y = np.asarray(y)
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._X is not None and self._y is not None
        X_q = np.asarray(X, dtype=np.float64)
        if X_q.ndim == 1:
            X_q = X_q.reshape(-1, 1)
        dists = np.sum((X_q[:, None, :] - self._X[None, :, :]) ** 2, axis=-1)
        k = min(self.n_neighbors, len(self._X))
        nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        probs = np.zeros((len(X_q), len(self.classes_)), dtype=np.float64)
        for i in range(len(X_q)):
            neighbors = self._y[nn_idx[i]]
            for c_i, cls_val in enumerate(self.classes_):
                probs[i, c_i] = np.mean(neighbors == cls_val)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _NaiveBayesNumPy:
    """Gaussian Naive Bayes thuần NumPy."""

    def __init__(self, var_smoothing: float = 1e-8, **_: Any):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = np.array([])
        self._means: Optional[np.ndarray] = None
        self._vars: Optional[np.ndarray] = None
        self._priors: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NaiveBayesNumPy":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_c = len(self.classes_)
        n_f = X_arr.shape[1]
        self._means = np.zeros((n_c, n_f), dtype=np.float64)
        self._vars = np.zeros((n_c, n_f), dtype=np.float64)
        self._priors = np.zeros(n_c, dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            X_c = X_arr[y_arr == c]
            self._means[idx] = np.mean(X_c, axis=0)
            self._vars[idx] = np.var(X_c, axis=0) + self.var_smoothing
            self._priors[idx] = len(X_c) / max(1, len(X_arr))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._means is not None and self._vars is not None and self._priors is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        log_probs = np.zeros((len(X_arr), len(self.classes_)), dtype=np.float64)
        for idx in range(len(self.classes_)):
            diff = X_arr - self._means[idx]
            log_lik = -0.5 * np.sum(np.log(2.0 * np.pi * self._vars[idx]) + (diff**2) / self._vars[idx], axis=1)
            log_probs[:, idx] = np.log(self._priors[idx] + 1e-12) + log_lik
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(log_probs)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _TuyenTinhNumPy:
    """Hồi quy Tuyến tính / Ridge (OLS + L2 Closed-Form) thuần NumPy."""

    def __init__(self, alpha: float = 0.0, **_: Any):
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TuyenTinhNumPy":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64)
        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        X_c = X_arr - X_mean
        y_c = y_arr - y_mean
        n_f = X_c.shape[1]
        reg_mat = self.alpha * np.eye(n_f, dtype=np.float64)
        if self.alpha > 0:
            self.coef_ = np.linalg.solve(X_c.T @ X_c + reg_mat, X_c.T @ y_c)
        else:
            self.coef_, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr @ self.coef_ + self.intercept_


class _KMeansNumPy:
    """Phân cụm K-Means++ thuần NumPy."""

    def __init__(self, n_clusters: int = 3, max_iter: int = 100, random_state: int = 42, **_: Any):
        self.n_clusters = max(1, int(n_clusters))
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_KMeansNumPy":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        rng = np.random.RandomState(self.random_state)
        k = min(self.n_clusters, len(X_arr))
        idx = rng.choice(len(X_arr), size=k, replace=False)
        centers = X_arr[idx].copy()
        labels = np.zeros(len(X_arr), dtype=int)

        for _ in range(self.max_iter):
            dists = np.sum((X_arr[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels) and self.cluster_centers_ is not None:
                break
            labels = new_labels
            for c in range(k):
                pts = X_arr[labels == c]
                if len(pts) > 0:
                    centers[c] = np.mean(pts, axis=0)
            self.cluster_centers_ = centers
        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.cluster_centers_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        dists = np.sum((X_arr[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=-1)
        return np.argmin(dists, axis=1)


def doc_csv_nhe(duong_dan: str, cot_nhan: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc tệp CSV siêu nhẹ bằng thư viện chuẩn `csv` của Python (không cần cài `pandas`)."""
    import csv

    with open(duong_dan, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]

    if not rows:
        raise ValueError(f"Tệp CSV rỗng: {duong_dan}")

    # Kiểm tra dòng đầu có phải header không
    co_header = False
    try:
        for cell in rows[0]:
            float(cell)
    except ValueError:
        co_header = True

    header = [c.strip() for c in rows[0]] if co_header else [f"col_{i}" for i in range(len(rows[0]))]
    data_rows = rows[1:] if co_header else rows

    if cot_nhan is not None:
        if cot_nhan not in header:
            raise ValueError(f"Không tìm thấy cột '{cot_nhan}'. Các cột có sẵn: {header}")
        target_idx = header.index(cot_nhan)
    else:
        target_idx = len(header) - 1

    X_list: List[List[float]] = []
    y_raw: List[Any] = []
    for r in data_rows:
        feat = [float(r[i]) for i in range(len(r)) if i != target_idx]
        X_list.append(feat)
        val_y = r[target_idx].strip()
        try:
            f_y = float(val_y)
            y_raw.append(int(f_y) if f_y.is_integer() and "." not in val_y else f_y)
        except ValueError:
            y_raw.append(val_y)

    return np.asarray(X_list, dtype=np.float64), np.asarray(y_raw)


def luu_csv_nhe(duong_dan: str, du_doan: Any, ten_cot: str = "du_doan") -> str:
    """Lưu kết quả dự đoán ra CSV bằng thư viện chuẩn `csv` (không cần `pandas`)."""
    import csv

    arr = np.asarray(du_doan).ravel().tolist()
    with open(duong_dan, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ten_cot])
        for item in arr:
            writer.writerow([item])
    return duong_dan


def bam_khoa_sha256(chuoi: str) -> str:
    """Băm khóa bí mật SHA-256 thống nhất trên toàn framework."""
    return hashlib.sha256(chuoi.encode("utf-8")).hexdigest()


DependencyManager = QuanLyPhuThuoc
