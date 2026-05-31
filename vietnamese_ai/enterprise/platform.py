"""NenTangDichVu - Nền tảng dịch vụ AI đám mây (SaaS)."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from vietnamese_ai.utils.logger import Logger


class Workspace:
    """
    Workspace - Không gian làm việc của một tổ chức/người dùng.

    Quản lý models, deployments, API keys và usage trong một workspace.
    """

    def __init__(self, ma: str, ten: str, chu_so_huu: str, goi_dich_vu: str = "free"):
        self.ma = ma
        self.ten = ten
        self.chu_so_huu = chu_so_huu
        self.goi_dich_vu = goi_dich_vu
        self.thoi_gian_tao = time.time()
        self.models: Dict[str, Dict] = {}
        self.deployments: Dict[str, Dict] = {}
        self.api_keys: Dict[str, Dict] = {}
        self.usage: Dict[str, int] = {
            "api_calls": 0,
            "models_created": 0,
            "predictions": 0,
            "storage_bytes": 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ma": self.ma,
            "ten": self.ten,
            "chu_so_huu": self.chu_so_huu,
            "goi_dich_vu": self.goi_dich_vu,
            "thoi_gian_tao": self.thoi_gian_tao,
            "so_models": len(self.models),
            "so_deployments": len(self.deployments),
            "so_api_keys": len(self.api_keys),
            "usage": self.usage.copy(),
        }


GOI_DICH_VU = {
    "free": {
        "ten": "Miễn phí",
        "max_models": 3,
        "max_deployments": 1,
        "max_api_keys": 1,
        "max_predictions_per_day": 100,
        "max_storage_mb": 100,
    },
    "starter": {
        "ten": "Starter",
        "max_models": 10,
        "max_deployments": 5,
        "max_api_keys": 3,
        "max_predictions_per_day": 5000,
        "max_storage_mb": 1000,
    },
    "pro": {
        "ten": "Professional",
        "max_models": 50,
        "max_deployments": 20,
        "max_api_keys": 10,
        "max_predictions_per_day": 100000,
        "max_storage_mb": 10000,
    },
    "enterprise": {
        "ten": "Enterprise",
        "max_models": -1,
        "max_deployments": -1,
        "max_api_keys": -1,
        "max_predictions_per_day": -1,
        "max_storage_mb": -1,
    },
}


class NenTangDichVu:
    """
    Nền tảng dịch vụ AI đám mây (SaaS Platform).

    Tính năng:
    - Quản lý workspace (multi-tenant)
    - Quản lý models trong workspace
    - Triển khai mô hình (deployments)
    - Quản lý API keys
    - Theo dõi sử dụng (usage tracking)
    - Kiểm tra giới hạn gói dịch vụ (quota)

    Sử dụng:
        >>> ntdv = NenTangDichVu()
        >>> ws = ntdv.tao_workspace("my_org", "admin_user")
        >>> ntdv.tao_api_key(ws["ma"])
        >>> ntdv.dang_ky_model(ws["ma"], "sentiment", mo_hinh)
    """

    def __init__(self, duong_dan: str = "saas_data"):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("NenTangDichVu")

        self._workspaces: Dict[str, Workspace] = {}
        self._api_key_to_workspace: Dict[str, str] = {}
        self._tai_du_lieu()

    def _luu_du_lieu(self) -> None:
        """Lưu dữ liệu ra file JSON."""
        data = {}
        for ma, ws in self._workspaces.items():
            data[ma] = ws.to_dict()
        duong_dan = self.duong_dan / "platform.json"
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _tai_du_lieu(self) -> None:
        """Tải dữ liệu từ file."""
        duong_dan = self.duong_dan / "platform.json"
        if duong_dan.exists():
            try:
                with open(duong_dan, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ma, ws_data in data.items():
                    ws = Workspace(
                        ma=ws_data["ma"],
                        ten=ws_data["ten"],
                        chu_so_huu=ws_data["chu_so_huu"],
                        goi_dich_vu=ws_data.get("goi_dich_vu", "free"),
                    )
                    ws.usage = ws_data.get("usage", ws.usage)
                    self._workspaces[ma] = ws
            except (json.JSONDecodeError, IOError, KeyError):
                pass

    @staticmethod
    def _tao_ma() -> str:
        """Tạo mã định danh ngẫu nhiên."""
        return hashlib.sha256(os.urandom(16)).hexdigest()[:12]

    @staticmethod
    def _tao_api_key() -> str:
        """Tạo API key ngẫu nhiên."""
        return f"vai_{hashlib.sha256(os.urandom(32)).hexdigest()[:32]}"

    def _kiem_tra_quota(self, ws: Workspace, tai_nguyen: str) -> bool:
        """Kiểm tra workspace có vượt quota không."""
        goi = GOI_DICH_VU.get(ws.goi_dich_vu, GOI_DICH_VU["free"])
        gioi_han = goi.get(f"max_{tai_nguyen}", 0)
        if gioi_han == -1:
            return True

        if tai_nguyen == "models":
            return len(ws.models) < gioi_han
        elif tai_nguyen == "deployments":
            return len(ws.deployments) < gioi_han
        elif tai_nguyen == "api_keys":
            return len(ws.api_keys) < gioi_han
        return True

    def tao_workspace(
        self,
        ten: str,
        chu_so_huu: str,
        goi_dich_vu: str = "free",
    ) -> Dict[str, Any]:
        """
        Tạo workspace mới.

        Args:
            ten: Tên workspace
            chu_so_huu: Chủ sở hữu
            goi_dich_vu: Gói dịch vụ (free, starter, pro, enterprise)

        Returns:
            Dict chứa thông tin workspace
        """
        if goi_dich_vu not in GOI_DICH_VU:
            raise ValueError(
                f"Gói '{goi_dich_vu}' không hợp lệ. Chọn: {', '.join(GOI_DICH_VU.keys())}"
            )

        if not ten or not isinstance(ten, str):
            raise ValueError("Tên workspace phải là chuỗi không rỗng")

        ma = self._tao_ma()
        ws = Workspace(ma, ten, chu_so_huu, goi_dich_vu)
        self._workspaces[ma] = ws
        self._luu_du_lieu()

        self.logger.info(f"Đã tạo workspace: {ten} ({ma}) [{goi_dich_vu}]")
        return ws.to_dict()

    def lay_workspace(self, ma: str) -> Dict[str, Any]:
        """Lấy thông tin workspace."""
        if ma not in self._workspaces:
            raise KeyError(f"Workspace '{ma}' không tồn tại")
        return self._workspaces[ma].to_dict()

    def danh_sach_workspaces(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả workspaces."""
        return [ws.to_dict() for ws in self._workspaces.values()]

    def cap_nhat_goi(self, ma: str, goi_moi: str) -> Dict[str, Any]:
        """Nâng cấp/hạ cấp gói dịch vụ."""
        if ma not in self._workspaces:
            raise KeyError(f"Workspace '{ma}' không tồn tại")
        if goi_moi not in GOI_DICH_VU:
            raise ValueError(f"Gói '{goi_moi}' không hợp lệ")

        ws = self._workspaces[ma]
        cu = ws.goi_dich_vu
        ws.goi_dich_vu = goi_moi
        self._luu_du_lieu()

        self.logger.info(f"Workspace {ma}: {cu} -> {goi_moi}")
        return ws.to_dict()

    def tao_api_key(self, ma_workspace: str) -> str:
        """Tạo API key cho workspace."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if not self._kiem_tra_quota(ws, "api_keys"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã đạt giới hạn API keys ({goi['max_api_keys']}). Nâng cấp gói để thêm."
            )

        api_key = self._tao_api_key()
        ws.api_keys[api_key] = {
            "key": api_key,
            "thoi_gian_tao": time.time(),
            "su_dung_cuoi": None,
            "so_lan_su_dung": 0,
        }
        self._api_key_to_workspace[api_key] = ma_workspace
        self._luu_du_lieu()

        self.logger.info(f"Đã tạo API key cho workspace {ma_workspace}")
        return api_key

    def xac_thuc_api_key(self, api_key: str) -> Dict[str, Any]:
        """Xác thực API key và trả về workspace."""
        if api_key not in self._api_key_to_workspace:
            raise PermissionError("API key không hợp lệ")

        ma_ws = self._api_key_to_workspace[api_key]
        ws = self._workspaces[ma_ws]

        ws.api_keys[api_key]["su_dung_cuoi"] = time.time()
        ws.api_keys[api_key]["so_lan_su_dung"] += 1
        ws.usage["api_calls"] += 1

        return ws.to_dict()

    def dang_ky_model(
        self,
        ma_workspace: str,
        ten_model: str,
        mo_hinh: Any,
        mo_ta: str = "",
    ) -> Dict[str, Any]:
        """
        Đăng ký mô hình trong workspace.

        Args:
            ma_workspace: Mã workspace
            ten_model: Tên mô hình
            mo_hinh: Mô hình đã huấn luyện
            mo_ta: Mô tả

        Returns:
            Dict chứa thông tin model
        """
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if not self._kiem_tra_quota(ws, "models"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã đạt giới hạn models ({goi['max_models']}). Nâng cấp gói để thêm."
            )

        ma_model = self._tao_ma()
        ws.models[ma_model] = {
            "ma": ma_model,
            "ten": ten_model,
            "mo_ta": mo_ta,
            "mo_hinh": mo_hinh,
            "thoi_gian_tao": time.time(),
            "da_deploy": False,
        }
        ws.usage["models_created"] += 1
        self._luu_du_lieu()

        self.logger.info(f"Đã đăng ký model: {ten_model} ({ma_model})")
        return {
            "ma": ma_model,
            "ten": ten_model,
            "mo_ta": mo_ta,
            "thoi_gian_tao": time.time(),
        }

    def lay_model(self, ma_workspace: str, ma_model: str) -> Any:
        """Lấy mô hình từ workspace."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")
        ws = self._workspaces[ma_workspace]
        if ma_model not in ws.models:
            raise KeyError(f"Model '{ma_model}' không tồn tại")
        return ws.models[ma_model]["mo_hinh"]

    def deploy_model(
        self,
        ma_workspace: str,
        ma_model: str,
        ten_deployment: str = "",
    ) -> Dict[str, Any]:
        """Triển khai mô hình."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if not self._kiem_tra_quota(ws, "deployments"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã đạt giới hạn deployments ({goi['max_deployments']}). Nâng cấp gói để thêm."
            )

        if ma_model not in ws.models:
            raise KeyError(f"Model '{ma_model}' không tồn tại")

        ma_deployment = self._tao_ma()
        ten = ten_deployment or f"deploy_{ma_model[:6]}"
        ws.deployments[ma_deployment] = {
            "ma": ma_deployment,
            "ten": ten,
            "ma_model": ma_model,
            "trang_thai": "hoat_dong",
            "thoi_gian_tao": time.time(),
            "so_requests": 0,
        }
        ws.models[ma_model]["da_deploy"] = True
        self._luu_du_lieu()

        self.logger.info(f"Đã deploy: {ten} ({ma_deployment})")
        return ws.deployments[ma_deployment]

    def du_doan(
        self,
        ma_workspace: str,
        ma_deployment: str,
        du_lieu: Any,
    ) -> Dict[str, Any]:
        """Dự đoán qua deployment."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if ma_deployment not in ws.deployments:
            raise KeyError(f"Deployment '{ma_deployment}' không tồn tại")

        deployment = ws.deployments[ma_deployment]
        if deployment["trang_thai"] != "hoat_dong":
            raise RuntimeError(f"Deployment '{ma_deployment}' không hoạt động")

        ma_model = deployment["ma_model"]
        mo_hinh = ws.models[ma_model]["mo_hinh"]

        import numpy as np

        du_lieu_np = np.asarray(du_lieu)
        if du_lieu_np.ndim == 1:
            du_lieu_np = du_lieu_np.reshape(1, -1)

        ket_qua = mo_hinh.du_doan(du_lieu_np)
        deployment["so_requests"] += 1
        ws.usage["predictions"] += 1

        return {
            "ket_qua": ket_qua.tolist(),
            "deployment": ma_deployment,
            "so_requests": deployment["so_requests"],
        }

    def thong_ke_usage(self, ma_workspace: str) -> Dict[str, Any]:
        """Thống kê sử dụng của workspace."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        goi = GOI_DICH_VU[ws.goi_dich_vu]

        return {
            "workspace": ma_workspace,
            "goi_dich_vu": ws.goi_dich_vu,
            "usage": ws.usage.copy(),
            "gioi_han": {
                "max_models": goi["max_models"],
                "max_deployments": goi["max_deployments"],
                "max_api_keys": goi["max_api_keys"],
                "max_predictions_per_day": goi["max_predictions_per_day"],
            },
        }

    def xoa_workspace(self, ma: str) -> None:
        """Xóa workspace."""
        if ma not in self._workspaces:
            raise KeyError(f"Workspace '{ma}' không tồn tại")

        ws = self._workspaces[ma]
        for api_key in ws.api_keys:
            self._api_key_to_workspace.pop(api_key, None)

        del self._workspaces[ma]
        self._luu_du_lieu()
        self.logger.info(f"Đã xóa workspace: {ma}")

    def lay_goi_dich_vu(self) -> Dict[str, Dict]:
        """Lấy thông tin tất cả gói dịch vụ."""
        return GOI_DICH_VU.copy()
