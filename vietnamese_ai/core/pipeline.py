"""Pipeline - Quy trình học máy tự động."""

import pickle
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger


class Pipeline:
    """
    Quy trình học máy tự động (Pipeline).

    Kết hợp nhiều bước tiền xử lý và mô hình thành một chuỗi thống nhất.
    Hỗ trợ fit -> predict -> evaluate -> save/load trong một luồng duy nhất.

    Sử dụng:
        >>> pipe = Pipeline()
        >>> pipe.them_buoc("chuan_hoa", normalizer)
        >>> pipe.them_buoc("mo_hinh", classifier)
        >>> pipe.fit(X_train, y_train)
        >>> du_doan = pipe.predict(X_test)
        >>> pipe.luu("pipeline.pkl")
        >>> pipe2 = Pipeline.tai("pipeline.pkl")
    """

    def __init__(self, ten: str = "Pipeline"):
        self.ten = ten
        self.cac_buoc: List[Tuple[str, Any]] = []
        self.da_fit = False
        self.logger = Logger(ten)

    def them_buoc(self, ten: str, bo_xu_ly: Any) -> "Pipeline":
        """
        Thêm một bước vào pipeline.

        Args:
            ten: Tên của bước
            bo_xu_ly: Đối tượng xử lý (phương thức fit_transform/transform
                      cho bước tiền xử lý, hoặc huan_luyen/du_doan cho mô hình)

        Returns:
            self (để hỗ trợ chaining)
        """
        self.cac_buoc.append((ten, bo_xu_ly))
        self.logger.info(f"Đã thêm bước: {ten}")
        return self

    @staticmethod
    def _ap_dung_bien_doi(buoc: Any, du_lieu: Any, la_fit: bool = False) -> Any:
        """Áp dụng bước biến đổi cho cả ma trận số, danh sách văn bản, Guardrails và Embedder."""
        if la_fit and hasattr(buoc, "fit_transform"):
            return buoc.fit_transform(du_lieu)
        if hasattr(buoc, "transform"):
            return buoc.transform(du_lieu)
        if hasattr(buoc, "nhung") and isinstance(du_lieu, (list, tuple)):
            return buoc.nhung(list(du_lieu))
        if hasattr(buoc, "ma_hoa") and isinstance(du_lieu, (list, tuple)):
            return buoc.ma_hoa(list(du_lieu))
        if hasattr(buoc, "lam_sach"):
            if isinstance(du_lieu, str):
                return buoc.lam_sach(du_lieu)
            if isinstance(du_lieu, (list, tuple)):
                return [buoc.lam_sach(str(x)) for x in du_lieu]
        if hasattr(buoc, "kiem_tra"):
            if isinstance(du_lieu, str):
                kq = buoc.kiem_tra(du_lieu)
                if not kq.get("an_toan", True) and hasattr(buoc, "loc_tu_cam"):
                    return buoc.loc_tu_cam(du_lieu)
                return du_lieu
            if isinstance(du_lieu, (list, tuple)):
                out = []
                for x in du_lieu:
                    s = str(x)
                    kq = buoc.kiem_tra(s)
                    if not kq.get("an_toan", True) and hasattr(buoc, "loc_tu_cam"):
                        s = buoc.loc_tu_cam(s)
                    out.append(s)
                return out
        if callable(buoc):
            return buoc(du_lieu)
        return du_lieu

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "Pipeline":
        """
        Huấn luyện toàn bộ pipeline (hỗ trợ cả ma trận số X, y, văn bản thô, và dữ liệu Fine-Tuning).

        Args:
            X: Dữ liệu đầu vào (np.ndarray, List[str], hoặc List[Dict])
            y: Nhãn mục tiêu (tùy chọn với mô hình không giám sát hoặc Fine-Tuning SFT/DPO)

        Returns:
            self
        """
        self.logger.info(f"Bắt đầu fit pipeline ({len(self.cac_buoc)} bước)")

        du_lieu_hien_tai = X
        for i, (ten, buoc) in enumerate(self.cac_buoc):
            la_buoc_cuoi = i == len(self.cac_buoc) - 1

            if la_buoc_cuoi and hasattr(buoc, "huan_luyen"):
                if y is not None:
                    try:
                        buoc.huan_luyen(du_lieu_hien_tai, y, **kwargs)
                    except TypeError:
                        buoc.huan_luyen((du_lieu_hien_tai, y), **kwargs)
                else:
                    buoc.huan_luyen(du_lieu_hien_tai, **kwargs)
                self.logger.info(f"Bước [{ten}]: Đã huấn luyện mô hình")
            elif la_buoc_cuoi and hasattr(buoc, "fit") and (hasattr(buoc, "predict") or hasattr(buoc, "du_doan")):
                if y is not None:
                    try:
                        buoc.fit(du_lieu_hien_tai, y)
                    except TypeError:
                        buoc.fit(du_lieu_hien_tai)
                else:
                    buoc.fit(du_lieu_hien_tai)
                self.logger.info(f"Bước [{ten}]: Đã fit mô hình")
            else:
                du_lieu_hien_tai = self._ap_dung_bien_doi(buoc, du_lieu_hien_tai, la_fit=True)
                self.logger.info(f"Bước [{ten}]: Biến đổi dữ liệu hoàn tất")

        self.da_fit = True
        self.logger.info("Pipeline fit hoàn tất")
        return self

    def huan_luyen(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "Pipeline":
        """Alias tiếng Việt cho `fit()` để tương thích đồng nhất với `Engine`."""
        return self.fit(X, y, **kwargs)

    def predict(self, X: Any) -> Any:
        """
        Dự đoán với toàn bộ pipeline (hỗ trợ cả dữ liệu bảng, văn bản, RAG và Agent).

        Args:
            X: Dữ liệu đầu vào

        Returns:
            Kết quả dự đoán
        """
        if not self.da_fit:
            raise RuntimeError("Pipeline chưa được fit. Gọi fit() trước.")

        du_lieu_hien_tai = X
        for i, (ten, buoc) in enumerate(self.cac_buoc):
            la_buoc_cuoi = i == len(self.cac_buoc) - 1

            if la_buoc_cuoi:
                if hasattr(buoc, "du_doan"):
                    return buoc.du_doan(du_lieu_hien_tai)
                if hasattr(buoc, "hoi"):
                    if isinstance(du_lieu_hien_tai, str):
                        return buoc.hoi(du_lieu_hien_tai)
                    return [buoc.hoi(str(q)) for q in du_lieu_hien_tai]
                if hasattr(buoc, "chay"):
                    if isinstance(du_lieu_hien_tai, str):
                        return buoc.chay(du_lieu_hien_tai)
                    return [buoc.chay(str(q)) for q in du_lieu_hien_tai]
                if hasattr(buoc, "sinh_van_ban"):
                    if isinstance(du_lieu_hien_tai, str):
                        return buoc.sinh_van_ban(du_lieu_hien_tai)
                    return [buoc.sinh_van_ban(str(q)) for q in du_lieu_hien_tai]
            du_lieu_hien_tai = self._ap_dung_bien_doi(buoc, du_lieu_hien_tai, la_fit=False)

        return du_lieu_hien_tai

    def du_doan(self, X: Any) -> Any:
        """Alias tiếng Việt cho `predict()` để tương thích đồng nhất với `Engine` và `MayChuBatch`."""
        return self.predict(X)

    def danh_sach_buoc(self) -> List[str]:
        """Trả về danh sách tên các bước trong pipeline."""
        return [ten for ten, _ in self.cac_buoc]

    def lay_buoc(self, ten: str) -> Any:
        """Lấy một bước theo tên."""
        for ten_buoc, buoc in self.cac_buoc:
            if ten_buoc == ten:
                return buoc
        raise KeyError(f"Không tìm thấy bước: {ten}")

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        """Đánh giá pipeline (biến đổi dữ liệu qua các bước tiền xử lý rồi gọi danh_gia() của bước cuối)."""
        if not self.da_fit:
            raise RuntimeError("Pipeline chưa được fit.")

        du_lieu_hien_tai: Any = X
        for _, buoc in self.cac_buoc[:-1]:
            du_lieu_hien_tai = self._ap_dung_bien_doi(buoc, du_lieu_hien_tai, la_fit=False)

        buoc_cuoi = self.cac_buoc[-1][1]
        if hasattr(buoc_cuoi, "danh_gia"):
            return buoc_cuoi.danh_gia(du_lieu_hien_tai, y)
        return 0.0

    def luu(self, duong_dan: str) -> None:
        """Lưu pipeline ra file pickle.

        Cảnh báo: File pickle có thể chứa mã độc. Chỉ tải pipeline từ nguồn tin cậy.
        """
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"Đã lưu pipeline tại: {duong_dan_path}")

    @classmethod
    def tai(cls, duong_dan: str, cho_phep_pickle_tho: bool = False) -> "Pipeline":
        """Tải pipeline từ file pickle kèm kiểm tra `.sig` và `RestrictedUnpickler` mặc định."""
        from vietnamese_ai.utils.io_utils import LuuTai, tai_an_toan

        p = Path(duong_dan)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        sig_path = p.with_suffix(p.suffix + ".sig")
        da_ky_hop_le = False
        if sig_path.exists():
            if not LuuTai.xac_thuc_file(str(p)):
                raise ValueError(
                    f"CẢNH BÁO BẢO MẬT: Chữ ký toàn vẹn HMAC-SHA256 của pipeline '{duong_dan}' không hợp lệ!"
                )
            da_ky_hop_le = True
        if cho_phep_pickle_tho:
            warnings.warn(
                "Đang tải pipeline từ pickle thô. Chỉ tải file từ nguồn tin cậy "
                "để tránh mã độc (remote code execution).",
                stacklevel=2,
            )
            try:
                with open(duong_dan, "rb") as f:
                    pipe = pickle.load(f)
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")
        else:
            try:
                pipe = tai_an_toan(
                    duong_dan, cho_phep_mo_hinh=True, cho_phep_lop_da_ky=da_ky_hop_le
                )
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(f"Lỗi đọc file pickle an toàn: {e}")
        pipe.logger.info(f"Đã tải pipeline từ: {duong_dan}")
        return pipe

    def __repr__(self) -> str:
        buoc_str = " -> ".join(self.danh_sach_buoc())
        trang_thai = "đã fit" if self.da_fit else "chưa fit"
        return f"Pipeline('{self.ten}': {buoc_str} [{trang_thai}])"
