"""QuanLyMoHinh - Quản lý và lưu trữ phiên bản mô hình."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.io_utils import LuuTai, tai_an_toan
from vietnamese_ai.utils.logger import Logger


class QuanLyMoHinh:
    """
    Hệ thống quản lý phiên bản mô hình (Model Registry) tích hợp xác thực chữ ký HMAC-SHA256.

    Tính năng:
    - Đăng ký mô hình với version
    - Lưu trữ metadata (params, metrics, tags)
    - Tải mô hình theo version hoặc alias (latest, production)
    - So sánh các phiên bản
    - Promote model (staging -> production)

    Sử dụng:
        >>> ql = QuanLyMoHinh("models/")
        >>> ql.dang_ky(mo_hinh, ten="phan_loai", version="1.0.0", metrics={"accuracy": 0.95})
        >>> ql.dang_ky(mo_hinh2, ten="phan_loai", version="1.1.0", metrics={"accuracy": 0.97})
        >>> ql.promote("phan_loai", "1.1.0", "production")
        >>> mo_hinh = ql.tai("phan_loai", "production")
    """

    def __init__(self, duong_dan: str = "model_registry", xac_thuc_chu_ky: bool = True):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.xac_thuc_chu_ky = xac_thuc_chu_ky
        self.logger = Logger("QuanLyMoHinh")
        self._metadata_file = self.duong_dan / "registry.json"
        self._metadata = self._tai_metadata()

    def _tai_metadata(self) -> Dict:
        if self._metadata_file.exists():
            with open(self._metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _luu_metadata(self) -> None:
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def dang_ky(
        self,
        mo_hinh: Any,
        ten: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        ghi_chu: str = "",
    ) -> str:
        """
        Đăng ký một mô hình mới.

        Args:
            mo_hinh: Đối tượng mô hình
            ten: Tên mô hình
            version: Phiên bản (tự động nếu None)
            metrics: Các chỉ số đánh giá
            params: Các tham số
            tags: Thẻ gắn nhãn
            ghi_chu: Ghi chú

        Returns:
            Version đã đăng ký
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ten not in self._metadata:
            self._metadata[ten] = {"versions": {}, "aliases": {}}

        duong_dan_model = self.duong_dan / ten / version
        duong_dan_model.mkdir(parents=True, exist_ok=True)

        loai_mo_hinh = mo_hinh.__class__.__name__
        duong_dan_luu = str(duong_dan_model / "model.pkl")

        if loai_mo_hinh in ("BoTinhChinhMoHinh", "AutoFineTuner"):
            import pickle

            with open(duong_dan_luu, "wb") as f:
                pickle.dump(mo_hinh, f)
        elif loai_mo_hinh == "HuggingFaceWrapper":
            import pickle

            with open(duong_dan_luu, "wb") as f:
                pickle.dump(mo_hinh, f)
        elif hasattr(mo_hinh, "luu"):
            res = mo_hinh.luu(duong_dan_luu)
            if isinstance(res, str) and Path(res).exists():
                duong_dan_luu = res
        else:
            import pickle

            with open(duong_dan_luu, "wb") as f:
                pickle.dump(mo_hinh, f)

        chu_ky_hmac = ""
        if Path(duong_dan_luu).is_file():
            chu_ky_hmac = LuuTai.ky_file(duong_dan_luu)

        ban_ghi = {
            "version": version,
            "loai_mo_hinh": loai_mo_hinh,
            "thoi_gian": datetime.now().isoformat(),
            "metrics": metrics or {},
            "params": params or {},
            "tags": tags or [],
            "ghi_chu": ghi_chu,
            "duong_dan": duong_dan_luu,
            "hmac_sha256": chu_ky_hmac,
        }

        self._metadata[ten]["versions"][version] = ban_ghi
        self._luu_metadata()

        self.logger.info(f"Đã đăng ký: {ten} v{version} ({loai_mo_hinh})")
        return version

    def tai(self, ten: str, version: Optional[str] = None) -> Any:
        """
        Tải mô hình theo tên và version (kèm kiểm tra chữ ký toàn vẹn HMAC-SHA256 và RestrictedUnpickler).

        Args:
            ten: Tên mô hình
            version: Version hoặc alias (None = latest)
        """
        if ten not in self._metadata:
            raise KeyError(f"Không tìm thấy mô hình: {ten}")

        if version is None:
            version = self._lay_version_moi_nhat(ten)
        elif version in self._metadata[ten].get("aliases", {}):
            version = self._metadata[ten]["aliases"][version]

        if version not in self._metadata[ten]["versions"]:
            raise KeyError(f"Không tìm thấy version {version} cho {ten}")

        info = self._metadata[ten]["versions"][version]
        duong_dan = info["duong_dan"]
        loai_mo_hinh = info.get("loai_mo_hinh", "BaseModel")

        if self.xac_thuc_chu_ky and Path(duong_dan).is_file():
            co_chu_ky_meta = bool(info.get("hmac_sha256"))
            sig_path = Path(duong_dan).with_suffix(Path(duong_dan).suffix + ".sig")
            if co_chu_ky_meta or sig_path.exists():
                if not LuuTai.xac_thuc_file(duong_dan):
                    raise ValueError(
                        f"Bảo mật: Phát hiện file mô hình '{duong_dan}' đã bị can thiệp trái phép "
                        "hoặc thiếu chữ ký toàn vẹn HMAC-SHA256!"
                    )

        if loai_mo_hinh == "GPTModel":
            from vietnamese_ai.transformer.gpt_model import GPTModel

            return GPTModel.tai(duong_dan)
        if loai_mo_hinh == "Pipeline":
            from vietnamese_ai.core.pipeline import Pipeline

            return Pipeline.tai(duong_dan)
        if loai_mo_hinh in ("BoTinhChinhMoHinh", "AutoFineTuner", "HuggingFaceWrapper"):
            return tai_an_toan(duong_dan, cho_phep_mo_hinh=True)

        from vietnamese_ai.models.base import BaseModel

        try:
            return BaseModel.tai(duong_dan)
        except Exception:
            return tai_an_toan(duong_dan, cho_phep_mo_hinh=True)

    def promote(self, ten: str, version: str, alias: str) -> None:
        """
        Gán alias cho một version (VD: production, staging).

        Args:
            ten: Tên mô hình
            version: Version cần promote
            alias: Alias mới
        """
        if ten not in self._metadata:
            raise KeyError(f"Không tìm thấy mô hình: {ten}")
        if version not in self._metadata[ten]["versions"]:
            raise KeyError(f"Không tìm thấy version {version}")

        self._metadata[ten]["aliases"][alias] = version
        self._luu_metadata()
        self.logger.info(f"Promote: {ten} v{version} -> {alias}")

    def danh_sach(self, ten: Optional[str] = None) -> List[Dict]:
        """Liệt kê tất cả mô hình hoặc các version của một mô hình."""
        if ten:
            if ten not in self._metadata:
                return []
            versions = self._metadata[ten]["versions"]
            aliases = self._metadata[ten].get("aliases", {})
            ket_qua = []
            for ver, info in versions.items():
                alias_list = [a for a, v in aliases.items() if v == ver]
                ket_qua.append({**info, "aliases": alias_list})
            return ket_qua
        else:
            ket_qua = []
            for ten_model, info in self._metadata.items():
                so_version = len(info.get("versions", {}))
                latest = self._lay_version_moi_nhat(ten_model) if so_version > 0 else None
                ket_qua.append({"ten": ten_model, "so_version": so_version, "latest": latest})
            return ket_qua

    def so_sanh(self, ten: str) -> Dict:
        """So sánh metrics giữa các version."""
        if ten not in self._metadata:
            raise KeyError(f"Không tìm thấy mô hình: {ten}")

        versions = self._metadata[ten]["versions"]
        so_sanh = {}
        for ver, info in versions.items():
            so_sanh[ver] = info.get("metrics", {})
        return so_sanh

    def xoa(self, ten: str, version: Optional[str] = None) -> None:
        """Xóa mô hình hoặc một version cụ thể."""
        if ten not in self._metadata:
            raise KeyError(f"Không tìm thấy mô hình: {ten}")

        if version:
            if version in self._metadata[ten]["versions"]:
                duong_dan = Path(self._metadata[ten]["versions"][version]["duong_dan"]).parent
                if duong_dan.exists():
                    shutil.rmtree(duong_dan)
                del self._metadata[ten]["versions"][version]
                self.logger.info(f"Đã xóa: {ten} v{version}")
        else:
            duong_dan_model = self.duong_dan / ten
            if duong_dan_model.exists():
                shutil.rmtree(duong_dan_model)
            del self._metadata[ten]
            self.logger.info(f"Đã xóa mô hình: {ten}")

        self._luu_metadata()

    def _lay_version_moi_nhat(self, ten: str) -> str:
        versions = list(self._metadata[ten]["versions"].keys())
        if not versions:
            raise KeyError(f"Mô hình {ten} chưa có version nào")
        return versions[-1]
