"""NenTangDichVu - Nền tảng dịch vụ AI đám mây (SaaS) liên kết Xác thực Băm Khóa, Ký số Mô hình & Tường lửa AI."""

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.enterprise.audit import NhatKyHoatDong
from vietnamese_ai.registry.model_registry import QuanLyMoHinh
from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework
from vietnamese_ai.utils.logger import Logger


class Workspace:
    """
    Workspace - Không gian làm việc của một tổ chức/người dùng.

    Quản lý models, deployments, API keys (lưu mã băm SHA-256) và usage trong một workspace.
    """

    def __init__(self, ma: str, ten: str, chu_so_huu: str, goi_dich_vu: str = "free"):
        self.ma = ma
        self.ten = ten
        self.chu_so_huu = chu_so_huu
        self.goi_dich_vu = goi_dich_vu
        self.thoi_gian_tao = time.time()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
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

    def to_storage_dict(self) -> Dict[str, Any]:
        """Xuất dữ liệu bền vững xuống đĩa (không lưu API key thô hay object RAM không tuần tự hóa được)."""
        models_meta = {}
        for m_id, m_info in self.models.items():
            models_meta[m_id] = {
                k: v for k, v in m_info.items() if k != "mo_hinh"
            }
        return {
            **self.to_dict(),
            "api_keys": self.api_keys,
            "deployments": self.deployments,
            "models_meta": models_meta,
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

    Đã liên kết chặt chẽ với hệ sinh thái bảo mật & quản trị của Framework:
    - Băm khóa API bằng SHA-256 và xác thực hằng thời gian (`hmac.compare_digest`)
    - Lưu trữ mô hình bền vững có chữ ký toàn vẹn HMAC-SHA256 qua `QuanLyMoHinh`
    - Ghi vết kiểm toán mọi tác vụ qua `NhatKyHoatDong`
    - Bảo vệ suy luận đầu vào/đầu ra qua `TuongLuaBaoVeFramework`
    """

    def __init__(self, duong_dan: str = "saas_data", bat_tuong_lua: bool = True):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("NenTangDichVu")

        self.quan_ly_mo_hinh = QuanLyMoHinh(duong_dan=str(self.duong_dan / "registry"))
        self.nhat_ky = NhatKyHoatDong(duong_dan=str(self.duong_dan / "audit"))
        self.bat_tuong_lua = bat_tuong_lua
        self.tuong_lua: Optional[TuongLuaBaoVeFramework] = (
            TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000) if bat_tuong_lua else None
        )

        self._workspaces: Dict[str, Workspace] = {}
        self._key_hash_to_workspace: Dict[str, str] = {}
        self._tai_du_lieu()

    @staticmethod
    def _bam_api_key(api_key: str) -> str:
        """Băm API key bằng SHA-256 để không bao giờ lưu khóa thô trên đĩa."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def _luu_du_lieu(self) -> None:
        """Lưu dữ liệu bền vững ra file JSON (chỉ lưu mã băm API key và tham chiếu Registry)."""
        data = {ma: ws.to_storage_dict() for ma, ws in self._workspaces.items()}
        duong_dan = self.duong_dan / "platform.json"
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _tai_du_lieu(self) -> None:
        """Khôi phục workspaces, mã băm API keys, deployments và metadata mô hình từ đĩa."""
        duong_dan = self.duong_dan / "platform.json"
        if not duong_dan.exists():
            return
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
                ws.thoi_gian_tao = ws_data.get("thoi_gian_tao", ws.thoi_gian_tao)
                ws.usage = ws_data.get("usage", ws.usage)
                ws.api_keys = ws_data.get("api_keys", {})
                ws.deployments = ws_data.get("deployments", {})
                for m_id, m_meta in ws_data.get("models_meta", {}).items():
                    ws.models[m_id] = {**m_meta, "mo_hinh": None}
                for k_hash in ws.api_keys:
                    self._key_hash_to_workspace[k_hash] = ma
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
        if tai_nguyen == "deployments":
            return len(ws.deployments) < gioi_han
        if tai_nguyen == "api_keys":
            return len(ws.api_keys) < gioi_han
        if tai_nguyen == "predictions_per_day":
            return ws.usage.get("predictions", 0) < gioi_han
        return True

    def tao_workspace(
        self,
        ten: str,
        chu_so_huu: str,
        goi_dich_vu: str = "free",
    ) -> Dict[str, Any]:
        """Tạo workspace mới và ghi nhận vào nhật ký kiểm toán."""
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
        self.nhat_ky.ghi(
            "tao_workspace",
            nguoi_dung=chu_so_huu,
            mo_ta=f"Khởi tạo workspace '{ten}' ({ma}) gói {goi_dich_vu}",
        )
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
        self.nhat_ky.ghi(
            "cap_nhat_goi",
            nguoi_dung=ws.chu_so_huu,
            mo_ta=f"Đổi gói workspace {ma}: {cu} -> {goi_moi}",
        )
        self.logger.info(f"Workspace {ma}: {cu} -> {goi_moi}")
        return ws.to_dict()

    def tao_api_key(self, ma_workspace: str) -> str:
        """Tạo API key mới cho workspace (chỉ lưu mã băm SHA-256 trên đĩa, trả khóa gốc 1 lần)."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if not self._kiem_tra_quota(ws, "api_keys"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã đạt giới hạn API keys ({goi['max_api_keys']}). Nâng cấp gói để thêm."
            )

        api_key = self._tao_api_key()
        key_hash = self._bam_api_key(api_key)
        ws.api_keys[key_hash] = {
            "key_hash": key_hash,
            "key_prefix": f"{api_key[:10]}...",
            "thoi_gian_tao": time.time(),
            "su_dung_cuoi": None,
            "so_lan_su_dung": 0,
        }
        self._key_hash_to_workspace[key_hash] = ma_workspace
        self._luu_du_lieu()
        self.nhat_ky.ghi(
            "tao_api_key",
            nguoi_dung=ws.chu_so_huu,
            mo_ta=f"Tạo API key ({api_key[:10]}...) cho workspace {ma_workspace}",
        )
        self.logger.info(f"Đã tạo API key cho workspace {ma_workspace}")
        return api_key

    def xac_thuc_api_key(self, api_key: str) -> Dict[str, Any]:
        """Xác thực API key bằng SHA-256 + `hmac.compare_digest` chống tấn công định thời."""
        if not api_key or not isinstance(api_key, str):
            raise PermissionError("API key không hợp lệ")

        ung_vien_hash = self._bam_api_key(api_key)
        ma_ws_khop: Optional[str] = None
        hash_khop: Optional[str] = None
        for stored_hash, ws_id in self._key_hash_to_workspace.items():
            if hmac.compare_digest(ung_vien_hash, stored_hash):
                ma_ws_khop = ws_id
                hash_khop = stored_hash
                break

        if ma_ws_khop is None or hash_khop is None:
            self.nhat_ky.ghi(
                "xac_thuc_that_bai",
                nguoi_dung="anonymous",
                mo_ta="Từ chối API key không hợp lệ",
                muc_do="warning",
            )
            raise PermissionError("API key không hợp lệ")

        ws = self._workspaces[ma_ws_khop]
        ws.api_keys[hash_khop]["su_dung_cuoi"] = time.time()
        ws.api_keys[hash_khop]["so_lan_su_dung"] += 1
        ws.usage["api_calls"] += 1
        self._luu_du_lieu()
        return ws.to_dict()

    def dang_ky_model(
        self,
        ma_workspace: str,
        ten_model: str,
        mo_hinh: Any,
        mo_ta: str = "",
    ) -> Dict[str, Any]:
        """Đăng ký mô hình trong workspace và lưu ký số HMAC-SHA256 vào `QuanLyMoHinh`."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if not self._kiem_tra_quota(ws, "models"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã đạt giới hạn models ({goi['max_models']}). Nâng cấp gói để thêm."
            )

        ma_model = self._tao_ma()
        ten_registry = f"{ma_workspace}_{ma_model}"
        da_luu_registry = False
        try:
            self.quan_ly_mo_hinh.dang_ky(ten=ten_registry, mo_hinh=mo_hinh, ghi_chu=mo_ta)
            da_luu_registry = True
        except Exception:
            # Đối với các mock/lambda cục bộ trong unit test không pickle được, giữ trong RAM
            da_luu_registry = False

        ws.models[ma_model] = {
            "ma": ma_model,
            "ten": ten_model,
            "mo_ta": mo_ta,
            "ten_registry": ten_registry if da_luu_registry else None,
            "mo_hinh": mo_hinh,
            "thoi_gian_tao": time.time(),
            "da_deploy": False,
        }
        ws.usage["models_created"] += 1
        self._luu_du_lieu()
        self.nhat_ky.ghi(
            "dang_ky_model",
            nguoi_dung=ws.chu_so_huu,
            mo_ta=f"Đăng ký model '{ten_model}' ({ma_model}) trong workspace {ma_workspace}",
        )
        self.logger.info(f"Đã đăng ký model: {ten_model} ({ma_model})")
        return {
            "ma": ma_model,
            "ten": ten_model,
            "mo_ta": mo_ta,
            "da_ky_so_registry": da_luu_registry,
            "thoi_gian_tao": ws.models[ma_model]["thoi_gian_tao"],
        }

    def lay_model(self, ma_workspace: str, ma_model: str) -> Any:
        """Lấy mô hình từ workspace (tự động khôi phục và xác thực chữ ký từ `QuanLyMoHinh` nếu cần)."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")
        ws = self._workspaces[ma_workspace]
        if ma_model not in ws.models:
            raise KeyError(f"Model '{ma_model}' không tồn tại")

        info = ws.models[ma_model]
        if info.get("mo_hinh") is None and info.get("ten_registry"):
            info["mo_hinh"] = self.quan_ly_mo_hinh.tai(info["ten_registry"])
        return info["mo_hinh"]

    def deploy_model(
        self,
        ma_workspace: str,
        ma_model: str,
        ten_deployment: str = "",
    ) -> Dict[str, Any]:
        """Triển khai mô hình lên endpoint của workspace."""
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
        self.nhat_ky.ghi(
            "deploy_model",
            nguoi_dung=ws.chu_so_huu,
            mo_ta=f"Triển khai deployment '{ten}' ({ma_deployment})",
        )
        self.logger.info(f"Đã deploy: {ten} ({ma_deployment})")
        return ws.deployments[ma_deployment]

    def du_doan(
        self,
        ma_workspace: str,
        ma_deployment: str,
        du_lieu: Any,
    ) -> Dict[str, Any]:
        """Dự đoán qua deployment kèm kiểm tra Quota và Tường lửa AI đầu vào/đầu ra."""
        if ma_workspace not in self._workspaces:
            raise KeyError(f"Workspace '{ma_workspace}' không tồn tại")

        ws = self._workspaces[ma_workspace]
        if ma_deployment not in ws.deployments:
            raise KeyError(f"Deployment '{ma_deployment}' không tồn tại")

        if not self._kiem_tra_quota(ws, "predictions_per_day"):
            goi = GOI_DICH_VU[ws.goi_dich_vu]
            raise PermissionError(
                f"Đã vượt hạn mức dự đoán trong ngày ({goi['max_predictions_per_day']})."
            )

        deployment = ws.deployments[ma_deployment]
        if deployment["trang_thai"] != "hoat_dong":
            raise RuntimeError(f"Deployment '{ma_deployment}' không hoạt động")

        ma_model = deployment["ma_model"]
        mo_hinh = self.lay_model(ma_workspace, ma_model)

        # Nhánh văn bản / LLM / RAG / Agent (bảo vệ trực tiếp bởi TuongLuaBaoVeFramework)
        if isinstance(du_lieu, str):
            if self.tuong_lua is not None:
                kq_fw = self.tuong_lua.kiem_tra_dau_vao(du_lieu, client_id=ma_workspace)
                if not kq_fw["an_toan"]:
                    ly_do = kq_fw["ly_do"]
                    self.nhat_ky.ghi(
                        "chan_tan_cong_prompt",
                        nguoi_dung=ws.chu_so_huu,
                        mo_ta=ly_do,
                        muc_do="warning",
                    )
                    raise PermissionError(f"Tường lửa AI chặn truy vấn: {ly_do}")

            if hasattr(mo_hinh, "sinh_van_ban"):
                van_ban_ra = str(mo_hinh.sinh_van_ban(du_lieu))
            elif hasattr(mo_hinh, "hoi"):
                kq_hoi = mo_hinh.hoi(du_lieu)
                van_ban_ra = str(kq_hoi.get("tra_loi", kq_hoi) if isinstance(kq_hoi, dict) else kq_hoi)
            elif hasattr(mo_hinh, "du_doan"):
                kq_dd = mo_hinh.du_doan([du_lieu])
                van_ban_ra = str(kq_dd[0] if len(kq_dd) > 0 else "")
            else:
                van_ban_ra = str(du_lieu)

            if self.tuong_lua is not None:
                van_ban_ra = self.tuong_lua.loc_dau_ra(van_ban_ra)

            deployment["so_requests"] += 1
            ws.usage["predictions"] += 1
            self._luu_du_lieu()
            return {
                "ket_qua": van_ban_ra,
                "deployment": ma_deployment,
                "so_requests": deployment["so_requests"],
            }

        import numpy as np

        du_lieu_np = np.asarray(du_lieu)
        if du_lieu_np.ndim == 1:
            du_lieu_np = du_lieu_np.reshape(1, -1)

        ket_qua = mo_hinh.du_doan(du_lieu_np)
        deployment["so_requests"] += 1
        ws.usage["predictions"] += 1
        self._luu_du_lieu()

        return {
            "ket_qua": ket_qua.tolist() if hasattr(ket_qua, "tolist") else list(ket_qua),
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
        for key_hash in ws.api_keys:
            self._key_hash_to_workspace.pop(key_hash, None)

        del self._workspaces[ma]
        self._luu_du_lieu()
        self.nhat_ky.ghi("xoa_workspace", nguoi_dung=ws.chu_so_huu, mo_ta=f"Đã xóa workspace {ma}")
        self.logger.info(f"Đã xóa workspace: {ma}")

    def lay_goi_dich_vu(self) -> Dict[str, Dict]:
        """Lấy thông tin tất cả gói dịch vụ."""
        return GOI_DICH_VU.copy()
