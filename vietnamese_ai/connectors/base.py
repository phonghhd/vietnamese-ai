"""KetNoiKenhCoSo (BaseChannelConnector) - Lớp Nền Thống Nhất Cho Zalo OA, Telegram & Các Kênh Nhắn Tin Doanh Nghiệp."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework


@dataclass
class TinNhanKenh:
    """Biểu diễn chuẩn hóa của một tin nhắn đến từ kênh ngoài (Zalo OA, Telegram...)."""

    kenh: str
    nguoi_gui_id: str
    noi_dung: str
    ma_tin_nhan: str = ""
    ten_nguoi_gui: str = ""
    cuoc_hoi_thoai_id: str = ""
    du_lieu_goc: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi tin nhắn kênh sang dict thuần JSON."""
        return asdict(self)


@dataclass
class PhanHoiKenh:
    """Kết quả phản hồi chuẩn hóa sau khi đi qua Tường lửa AI và bộ xử lý nghiệp vụ."""

    kenh: str
    nguoi_nhan_id: str
    noi_dung: str
    an_toan: bool = True
    da_gui: bool = False
    payload_gui: Dict[str, Any] = field(default_factory=dict)
    chi_tiet_gui: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển phản hồi kênh sang dict thuần JSON."""
        return asdict(self)


class KetNoiKenhCoSo(ABC):
    """
    Lớp cơ sở cho mọi bộ kết nối kênh chat doanh nghiệp (`KetNoiZaloOA`, `KetNoiTelegram`).

    Tự động gắn:
    - Tường lửa AI đa tầng (`TuongLuaBaoVeFramework`) chống Prompt Injection và ẩn PII đầu ra.
    - Điều hướng lệnh nhanh `/hieuluc <so_hieu>`, `/luat <so_hieu>`, `/help`.
    - Gắn linh hoạt với `TroLyPhapLy`, `TongDaiCSKH`, `RAGPipeline`, `DongCoSuyLuanEdge` hoặc hàm Python bất kỳ.
    """

    TEN_KENH = "co_so"

    def __init__(
        self,
        bo_xu_ly: Optional[Union[Any, Callable[..., str]]] = None,
        bat_tuong_lua: bool = True,
        che_do_gui_that: bool = False,
        quan_ly_workspace: Optional[Any] = None,
        ma_workspace: str = "mac_dinh",
    ) -> None:
        self.bo_xu_ly = bo_xu_ly
        self.bat_tuong_lua = bat_tuong_lua
        self.che_do_gui_that = che_do_gui_that
        self.quan_ly_workspace = quan_ly_workspace
        self.ma_workspace = ma_workspace
        self.tuong_lua = (
            TuongLuaBaoVeFramework(chan_khi_vi_pham=False, gioi_han_req_moi_phut=600)
            if bat_tuong_lua
            else None
        )
        self._lich_su: Dict[str, List[Dict[str, str]]] = {}

    def _dieu_phoi_bo_xu_ly(self, tin_nhan: TinNhanKenh) -> str:
        """Gọi lệnh nhanh hoặc chuyển tiếp tin nhắn tới bộ xử lý AI đã gắn."""
        text = tin_nhan.noi_dung.strip()
        if text.lower() in ("/start", "/help", "trợ giúp"):
            return (
                "Xin chào! Tôi là Trợ lý Vietnamese AI.\n"
                "- Gửi câu hỏi bất kỳ để tra cứu tri thức / pháp lý / CSKH.\n"
                "- Gõ `/hieuluc 45/2013/QH13` để kiểm tra hiệu lực văn bản luật.\n"
                "- Gõ `/luat 123/2020/NĐ-CP` để xem Nghị định/Thông tư hướng dẫn & sửa đổi.\n"
                "- Gõ `/hopdong <điều khoản hợp đồng>` để rà soát rủi ro pháp lý nhanh."
            )

        if text.lower().startswith("/hopdong "):
            from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong

            noi_dung_hd = text.split(None, 1)[1].strip()
            kq_hd = BoSoSanhRaSoatHopDong().so_sanh_va_ra_soat(
                hop_dong_doi_tac=noi_dung_hd, cho_phep_doc_tep=False
            )
            lines_hd = [
                f"Rà soát nhanh hợp đồng (Mức rủi ro: {kq_hd['xep_loai_rui_ro']} | Điểm an toàn: {kq_hd['diem_an_toan_phap_ly']}/100):"
            ]
            for rr in kq_hd.get("danh_sach_canh_bao", [])[:4]:
                lines_hd.append(f"- [{str(rr['muc_do']).upper()}] {rr['tieu_de']}: {rr['khuyen_nghi']}")
            for cb in (kq_hd.get("kiem_toan_hieu_luc") or {}).get("canh_bao", [])[:3]:
                lines_hd.append(f"- [HIỆU LỰC] {cb}")
            if len(lines_hd) == 1:
                lines_hd.append("- Chưa phát hiện điều khoản bất lợi hoặc trích dẫn luật hết hiệu lực.")
            return "\n".join(lines_hd)

        if text.lower().startswith("/hieuluc "):
            from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy

            so_hieu = text.split(None, 1)[1].strip()
            kq = BoTheoDoiHieuLucPhapLy().kiem_tra_hieu_luc(so_hieu)
            if not kq["tim_thay"]:
                return f"Chưa có dữ liệu hiệu lực cho số hiệu `{so_hieu}`."
            dong = [
                f"Văn bản: {kq['ten_van_ban']} ({kq['so_hieu']})",
                f"Trạng thái: {kq['trang_thai']} (hiệu lực từ {kq['ngay_hieu_luc']})",
            ]
            if kq.get("canh_bao"):
                dong.append(str(kq["canh_bao"]))
            return "\n".join(dong)

        if text.lower().startswith("/luat "):
            from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy

            so_hieu = text.split(None, 1)[1].strip()
            kq = DoThiThamChieuPhapLy().truy_van_van_ban_lien_quan(so_hieu, chi_lay_con_hieu_luc=True)
            lines = [f"Đồ thị pháp lý cho {kq['so_hieu']} ({kq['hieu_luc_goc']['trang_thai']}):"]
            if kq["van_ban_thay_the_hien_hanh"]:
                lines.append(
                    "- Thay thế hiện hành: "
                    + ", ".join(x["so_hieu"] for x in kq["van_ban_thay_the_hien_hanh"])
                )
            if kq["van_ban_huong_dan"]:
                lines.append(
                    "- Hướng dẫn thi hành: "
                    + ", ".join(x["so_hieu"] for x in kq["van_ban_huong_dan"])
                )
            if kq["van_ban_sua_doi_bo_sung"]:
                lines.append(
                    "- Sửa đổi, bổ sung: "
                    + ", ".join(x["so_hieu"] for x in kq["van_ban_sua_doi_bo_sung"])
                )
            return "\n".join(lines)

        handler = self.bo_xu_ly
        if handler is None:
            from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

            self.bo_xu_ly = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True)
            handler = self.bo_xu_ly

        if hasattr(handler, "tra_cuu"):
            res = handler.tra_cuu(text)
            return str(res.get("tra_loi", res) if isinstance(res, dict) else res)
        if hasattr(handler, "xu_ly_yeu_cau"):
            try:
                res = handler.xu_ly_yeu_cau(text, cho_phep_doc_tep=False)
            except TypeError:
                res = handler.xu_ly_yeu_cau(text)
            return str(res.get("tra_loi", res) if isinstance(res, dict) else res)
        if hasattr(handler, "hoi"):
            res = handler.hoi(text)
            return str(res.get("tra_loi", res) if isinstance(res, dict) else res)
        if hasattr(handler, "sinh_kem_thong_ke"):
            res = handler.sinh_kem_thong_ke(text)
            return str(res.get("van_ban", res))
        if callable(handler):
            try:
                return str(handler(text, tin_nhan))
            except TypeError:
                return str(handler(text))
        return f"Đã nhận tin nhắn: {text}"

    def xu_ly_tin_nhan(self, tin_nhan: TinNhanKenh) -> PhanHoiKenh:
        """Xử lý trọn gói một `TinNhanKenh` (Tường lửa đầu vào → AI → Lọc DLP đầu ra → Lưu SQLite Workspace nếu gắn → Tạo/gửi payload)."""
        an_toan = True
        if self.tuong_lua is not None:
            kt = self.tuong_lua.kiem_tra_dau_vao(tin_nhan.noi_dung, client_id=f"{self.TEN_KENH}:{tin_nhan.nguoi_gui_id}")
            if not kt["an_toan"]:
                an_toan = False
                noi_dung_tra = f"Xin lỗi, yêu cầu bị chặn vì lý do an toàn: {kt['ly_do']}"
            else:
                noi_dung_tra = self._dieu_phoi_bo_xu_ly(tin_nhan)
                noi_dung_tra = self.tuong_lua.loc_dau_ra(noi_dung_tra)
        else:
            noi_dung_tra = self._dieu_phoi_bo_xu_ly(tin_nhan)

        hist = self._lich_su.setdefault(tin_nhan.nguoi_gui_id, [])
        hist.append({"vai_tro": "user", "noi_dung": tin_nhan.noi_dung})
        hist.append({"vai_tro": "assistant", "noi_dung": noi_dung_tra})
        if len(hist) > 40:
            del hist[:-40]

        if self.quan_ly_workspace is not None and hasattr(self.quan_ly_workspace, "ghi_luot_hoi_thoai_kenh"):
            ma_phien_kenh = f"{self.TEN_KENH}:{tin_nhan.nguoi_gui_id}"
            self.quan_ly_workspace.ghi_luot_hoi_thoai_kenh(
                ma_phien=ma_phien_kenh,
                cau_hoi=tin_nhan.noi_dung,
                tra_loi=noi_dung_tra,
                ma_workspace=self.ma_workspace,
                tieu_de_phien=f"{self.TEN_KENH.upper()} - {tin_nhan.ten_nguoi_gui or tin_nhan.nguoi_gui_id}",
                nguoi_tao=f"{self.TEN_KENH}:{tin_nhan.nguoi_gui_id}",
            )

        nguoi_nhan = tin_nhan.cuoc_hoi_thoai_id or tin_nhan.nguoi_gui_id
        payload = self.tao_payload_tra_loi(nguoi_nhan, noi_dung_tra)
        da_gui = False
        chi_tiet: Dict[str, Any] = {"che_do": "dry_run"}
        if self.che_do_gui_that:
            chi_tiet = self.gui_tra_loi(nguoi_nhan, noi_dung_tra)
            da_gui = bool(chi_tiet.get("thanh_cong", False))

        return PhanHoiKenh(
            kenh=self.TEN_KENH,
            nguoi_nhan_id=nguoi_nhan,
            noi_dung=noi_dung_tra,
            an_toan=an_toan,
            da_gui=da_gui,
            payload_gui=payload,
            chi_tiet_gui=chi_tiet,
        )

    def lay_lich_su_nguoi_dung(self, nguoi_gui_id: str) -> List[Dict[str, str]]:
        """Lấy lịch sử hội thoại gần nhất của một người dùng trên kênh (ưu tiên đọc từ SQLite Workspace nếu có)."""
        if self.quan_ly_workspace is not None and hasattr(self.quan_ly_workspace, "lay_lich_su_phien"):
            rows = self.quan_ly_workspace.lay_lich_su_phien(f"{self.TEN_KENH}:{nguoi_gui_id}")
            if rows:
                return [{"vai_tro": str(r["vai_tro"]), "noi_dung": str(r["noi_dung"])} for r in rows]
        return list(self._lich_su.get(nguoi_gui_id, []))

    @abstractmethod
    def phan_tich_webhook(self, payload: Dict[str, Any]) -> Optional[TinNhanKenh]:
        """Chuyển đổi payload JSON Webhook của nền tảng sang `TinNhanKenh` chuẩn."""

    @abstractmethod
    def tao_payload_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Tạo payload JSON đúng đặc tả API gửi tin nhắn của nền tảng."""

    @abstractmethod
    def gui_tra_loi(self, nguoi_nhan_id: str, noi_dung: str) -> Dict[str, Any]:
        """Gửi phản hồi thực tế qua HTTPS tới API nền tảng."""


BaseChannelConnector = KetNoiKenhCoSo

__all__ = [
    "TinNhanKenh",
    "PhanHoiKenh",
    "KetNoiKenhCoSo",
    "BaseChannelConnector",
]
