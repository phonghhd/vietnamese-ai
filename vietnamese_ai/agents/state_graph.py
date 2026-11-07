"""Đồ thị Trạng thái Đa Agent (StateGraph) - Vietnamese AI Framework."""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

START = "__start__"
END = "__end__"


class BoLuuDiemKiemTra:
    """
    Bộ lưu vết điểm kiểm tra (Checkpointer / MemorySaver) cho Đồ thị Trạng thái:
    - Lưu lại toàn bộ lịch sử snapshot trạng thái qua từng bước thực thi.
    - Hỗ trợ quay ngược thời gian (Time-Travel Debugging) hoặc rẽ nhánh thực thi từ điểm bất kỳ.
    """

    def __init__(self):
        self._cac_snapshot: Dict[str, List[Dict[str, Any]]] = {}

    def luu_snapshot(self, session_id: str, buoc: int, nut: str, state: Dict[str, Any]) -> str:
        snap_id = f"snap_{uuid.uuid4().hex[:8]}"
        ds = self._cac_snapshot.setdefault(session_id, [])
        ds.append(
            {
                "snapshot_id": snap_id,
                "buoc": buoc,
                "nut": nut,
                "state": copy.deepcopy(state),
            }
        )
        return snap_id

    def lay_lich_su(self, session_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._cac_snapshot.get(session_id, []))

    def lay_snapshot_gan_nhat(self, session_id: str) -> Optional[Dict[str, Any]]:
        ds = self._cac_snapshot.get(session_id, [])
        return copy.deepcopy(ds[-1]) if ds else None

    def lay_trang_thai(self, session_id: str) -> Optional[Dict[str, Any]]:
        snap = self.lay_snapshot_gan_nhat(session_id)
        if snap is None:
            return None
        return {
            "snapshot_id": snap["snapshot_id"],
            "buoc": snap["buoc"],
            "nut": snap["nut"],
            "du_lieu": snap["state"],
            "state": snap["state"],
        }


MemorySaver = BoLuuDiemKiemTra
Checkpointer = BoLuuDiemKiemTra


class DoThiThucThi:
    """Đồ thị trạng thái đã biên dịch và sẵn sàng thực thi (Compiled State Graph)."""

    def __init__(
        self,
        cac_nut: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        cac_canh: Dict[str, str],
        cac_canh_dieu_kien: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Optional[Dict[str, str]]]],
        diem_bat_dau: str,
        kenh_cap_nhat: Dict[str, Callable[[Any, Any], Any]],
        checkpointer: Optional[BoLuuDiemKiemTra] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
    ):
        self.cac_nut = cac_nut
        self.cac_canh = cac_canh
        self.cac_canh_dieu_kien = cac_canh_dieu_kien
        self.diem_bat_dau = diem_bat_dau
        self.kenh_cap_nhat = kenh_cap_nhat
        self.checkpointer = checkpointer or BoLuuDiemKiemTra()
        self.interrupt_before: Set[str] = set(interrupt_before or [])
        self.interrupt_after: Set[str] = set(interrupt_after or [])

        # Trạng thái phiên hiện tại
        self._trang_thai_hien_tai: Dict[str, Any] = {}
        self._nut_hien_tai: Optional[str] = None
        self._session_id: str = f"sess_{uuid.uuid4().hex[:8]}"
        self._so_buoc: int = 0
        self._dang_tam_dung: bool = False
        self._nut_bi_dung: Optional[str] = None

    def lay_trang_thai(
        self, session_id: Optional[str] = None, thread_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Lấy snapshot trạng thái gần nhất từ Checkpointer."""
        sid = thread_id or session_id or self._session_id
        return self.checkpointer.lay_trang_thai(sid)

    def _cap_nhat_state(self, state_goc: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        """Cập nhật state bằng các kênh Reducer đã khai báo."""
        state_moi = copy.deepcopy(state_goc)
        if not isinstance(delta, dict):
            return state_moi

        for k, v in delta.items():
            if k in self.kenh_cap_nhat:
                reducer = self.kenh_cap_nhat[k]
                state_moi[k] = reducer(state_moi.get(k), v)
            else:
                state_moi[k] = v
        return state_moi

    def _tim_nut_tiep_theo(self, nut_hien_tai: str, state: Dict[str, Any]) -> str:
        """Xác định nút kế tiếp dựa trên cạnh trực tiếp hoặc cạnh điều kiện."""
        if nut_hien_tai in self.cac_canh_dieu_kien:
            ham_dk, ban_do = self.cac_canh_dieu_kien[nut_hien_tai]
            gia_tri_dk = ham_dk(state)
            if ban_do and gia_tri_dk in ban_do:
                return ban_do[gia_tri_dk]
            return str(gia_tri_dk)

        if nut_hien_tai in self.cac_canh:
            return self.cac_canh[nut_hien_tai]

        return END

    def chay(
        self,
        trang_thai_khoi_tao: Dict[str, Any],
        session_id: Optional[str] = None,
        gioi_han_buoc: int = 25,
    ) -> Dict[str, Any]:
        """
        Khởi chạy đồ thị trạng thái từ điểm bắt đầu.
        """
        self._session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self._trang_thai_hien_tai = copy.deepcopy(trang_thai_khoi_tao)
        self._so_buoc = 0
        self._dang_tam_dung = False
        self._nut_bi_dung = None

        # Lưu snapshot điểm bắt đầu (buoc 0)
        self.checkpointer.luu_snapshot(
            self._session_id, 0, START, self._trang_thai_hien_tai
        )

        nut_ke_tiep = self.diem_bat_dau
        nhat_ky_buoc: List[Dict[str, Any]] = []

        while nut_ke_tiep != END and self._so_buoc < gioi_han_buoc:
            # 1. Kiểm tra ngắt trước nút (Human-In-The-Loop interrupt_before)
            if nut_ke_tiep in self.interrupt_before and self._so_buoc > 0:
                self._dang_tam_dung = True
                self._nut_bi_dung = nut_ke_tiep
                self._nut_hien_tai = nut_ke_tiep
                return {
                    "trang_thai": self._trang_thai_hien_tai,
                    "dang_tam_dung": True,
                    "nut_tam_dung": nut_ke_tiep,
                    "loai_tam_dung": "interrupt_before",
                    "so_buoc": self._so_buoc,
                    "session_id": self._session_id,
                    "nhat_ky_buoc": nhat_ky_buoc,
                }

            if nut_ke_tiep not in self.cac_nut:
                raise ValueError(f"Nút '{nut_ke_tiep}' không tồn tại trong Đồ thị Trạng thái.")

            self._so_buoc += 1
            ham_nut = self.cac_nut[nut_ke_tiep]
            delta = ham_nut(self._trang_thai_hien_tai)
            self._trang_thai_hien_tai = self._cap_nhat_state(self._trang_thai_hien_tai, delta)

            # Lưu snapshot trạng thái vào Checkpointer
            self.checkpointer.luu_snapshot(
                self._session_id, self._so_buoc, nut_ke_tiep, self._trang_thai_hien_tai
            )

            nhat_ky_buoc.append(
                {
                    "buoc": self._so_buoc,
                    "nut": nut_ke_tiep,
                    "ket_qua": delta,
                }
            )

            # 2. Kiểm tra ngắt sau nút (Human-In-The-Loop interrupt_after)
            if nut_ke_tiep in self.interrupt_after:
                nut_sau_do = self._tim_nut_tiep_theo(nut_ke_tiep, self._trang_thai_hien_tai)
                self._dang_tam_dung = True
                self._nut_bi_dung = nut_ke_tiep
                self._nut_hien_tai = nut_sau_do
                return {
                    "trang_thai": self._trang_thai_hien_tai,
                    "dang_tam_dung": True,
                    "nut_tam_dung": nut_ke_tiep,
                    "nut_tiep_theo": nut_sau_do,
                    "loai_tam_dung": "interrupt_after",
                    "so_buoc": self._so_buoc,
                    "session_id": self._session_id,
                    "nhat_ky_buoc": nhat_ky_buoc,
                }

            # 3. Chuyển sang nút kế tiếp
            nut_ke_tiep = self._tim_nut_tiep_theo(nut_ke_tiep, self._trang_thai_hien_tai)

        return {
            "trang_thai": self._trang_thai_hien_tai,
            "dang_tam_dung": False,
            "nut_tam_dung": None,
            "so_buoc": self._so_buoc,
            "session_id": self._session_id,
            "nhat_ky_buoc": nhat_ky_buoc,
            "hoan_tat": nut_ke_tiep == END,
        }

    def thuc_thi(
        self,
        trang_thai_khoi_tao: Dict[str, Any],
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        gioi_han_buoc: int = 25,
    ) -> Dict[str, Any]:
        """Thực thi đồ thị trạng thái và trả về kết quả hợp nhất (chuẩn LangGraph / StateGraph)."""
        sid = thread_id or session_id
        res = self.chay(trang_thai_khoi_tao, session_id=sid, gioi_han_buoc=gioi_han_buoc)
        out = copy.deepcopy(res["trang_thai"])
        out["__dang_tam_dung__"] = res["dang_tam_dung"]
        out["__tam_dung__"] = res["dang_tam_dung"]
        out["__nut_hien_tai__"] = (
            res["nut_tam_dung"]
            if res["dang_tam_dung"]
            else (END if res.get("hoan_tat") else self._nut_hien_tai)
        )
        out["__buoc__"] = res["so_buoc"]
        out["__session_id__"] = res["session_id"]
        out["__thread_id__"] = res["session_id"]
        out["trang_thai"] = res["trang_thai"]
        out["dang_tam_dung"] = res["dang_tam_dung"]
        out["nut_tam_dung"] = res["nut_tam_dung"]
        return out

    invoke = thuc_thi

    def tiep_tuc(
        self,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        trang_thai_cap_nhat: Optional[Dict[str, Any]] = None,
        gioi_han_buoc: int = 25,
    ) -> Dict[str, Any]:
        """
        Tiếp tục thực thi đồ thị sau khi con người can thiệp hoặc điều chỉnh State.
        """
        if not self._dang_tam_dung or not self._nut_hien_tai:
            raise RuntimeError("Đồ thị không ở trạng thái tạm dừng để có thể tiếp tục.")

        if thread_id or session_id:
            self._session_id = thread_id or session_id or self._session_id

        if trang_thai_cap_nhat:
            self._trang_thai_hien_tai = self._cap_nhat_state(
                self._trang_thai_hien_tai, trang_thai_cap_nhat
            )

        self._dang_tam_dung = False
        self._nut_bi_dung = None
        nut_ke_tiep = self._nut_hien_tai
        nhat_ky_buoc: List[Dict[str, Any]] = []
        da_bo_qua_ngat_dau = False

        while nut_ke_tiep != END and self._so_buoc < gioi_han_buoc:
            if nut_ke_tiep in self.interrupt_before and da_bo_qua_ngat_dau:
                self._dang_tam_dung = True
                self._nut_bi_dung = nut_ke_tiep
                self._nut_hien_tai = nut_ke_tiep
                out = copy.deepcopy(self._trang_thai_hien_tai)
                out["__dang_tam_dung__"] = True
                out["__tam_dung__"] = True
                out["__nut_hien_tai__"] = nut_ke_tiep
                out["__buoc__"] = self._so_buoc
                out["trang_thai"] = self._trang_thai_hien_tai
                return out

            da_bo_qua_ngat_dau = True

            if nut_ke_tiep not in self.cac_nut:
                raise ValueError(f"Nút '{nut_ke_tiep}' không tồn tại trong Đồ thị.")

            self._so_buoc += 1
            ham_nut = self.cac_nut[nut_ke_tiep]
            delta = ham_nut(self._trang_thai_hien_tai)
            self._trang_thai_hien_tai = self._cap_nhat_state(self._trang_thai_hien_tai, delta)

            self.checkpointer.luu_snapshot(
                self._session_id, self._so_buoc, nut_ke_tiep, self._trang_thai_hien_tai
            )

            nhat_ky_buoc.append({"buoc": self._so_buoc, "nut": nut_ke_tiep, "ket_qua": delta})

            if nut_ke_tiep in self.interrupt_after:
                nut_sau_do = self._tim_nut_tiep_theo(nut_ke_tiep, self._trang_thai_hien_tai)
                self._dang_tam_dung = True
                self._nut_bi_dung = nut_ke_tiep
                self._nut_hien_tai = nut_sau_do
                out = copy.deepcopy(self._trang_thai_hien_tai)
                out["__dang_tam_dung__"] = True
                out["__tam_dung__"] = True
                out["__nut_hien_tai__"] = nut_sau_do
                out["__buoc__"] = self._so_buoc
                out["trang_thai"] = self._trang_thai_hien_tai
                return out

            nut_ke_tiep = self._tim_nut_tiep_theo(nut_ke_tiep, self._trang_thai_hien_tai)

        out = copy.deepcopy(self._trang_thai_hien_tai)
        out["__dang_tam_dung__"] = False
        out["__tam_dung__"] = False
        out["__nut_hien_tai__"] = END
        out["__buoc__"] = self._so_buoc
        out["__session_id__"] = self._session_id
        out["trang_thai"] = self._trang_thai_hien_tai
        out["hoan_tat"] = nut_ke_tiep == END
        return out


class DoThiTrangThai:
    """
    Đồ thị Trạng thái Đa Tác tử (StateGraph / LangGraph Tiếng Việt):
    - Hỗ trợ xây dựng luồng tác tử tự chủ dạng đồ thị có hướng (hỗ trợ cả chu trình / cyclic loops).
    - Hỗ trợ định tuyến có điều kiện (Conditional Edges).
    - Hỗ trợ Human-In-The-Loop (HITL) gián đoạn trước/sau nút.
    - Hỗ trợ Checkpointer lưu vết và Time-Travel Debugging.
    """

    def __init__(
        self,
        kenh_trang_thai: Optional[Dict[str, Callable[[Any, Any], Any]]] = None,
        kenh_cap_nhat: Optional[Dict[str, Callable[[Any, Any], Any]]] = None,
        bo_luu_diem: Optional[BoLuuDiemKiemTra] = None,
        checkpointer: Optional[BoLuuDiemKiemTra] = None,
    ):
        self.cac_nut: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.cac_canh: Dict[str, str] = {}
        self.cac_canh_dieu_kien: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Optional[Dict[str, str]]]] = {}
        self.diem_bat_dau: Optional[str] = None
        self.kenh_cap_nhat = kenh_cap_nhat or kenh_trang_thai or {}
        self.bo_luu_diem = checkpointer or bo_luu_diem

    def them_nut(self, ten: str, ham_thuc_thi: Callable[[Dict[str, Any]], Dict[str, Any]]) -> DoThiTrangThai:
        """Thêm một nút tác tử vào đồ thị."""
        if ten in (START, END):
            raise ValueError(f"Tên nút không được trùng với điểm mốc đặc biệt '{ten}'.")
        self.cac_nut[ten] = ham_thuc_thi
        return self

    def them_canh(self, nut_nguon: str, nut_dich: str) -> DoThiTrangThai:
        """Thêm một cạnh trực tiếp giữa hai nút."""
        if nut_nguon == START:
            self.diem_bat_dau = nut_dich
        else:
            self.cac_canh[nut_nguon] = nut_dich
        return self

    def them_canh_dieu_kien(
        self,
        nut_nguon: str,
        ham_dieu_kien: Callable[[Dict[str, Any]], str],
        ban_do_dich: Optional[Dict[str, str]] = None,
    ) -> DoThiTrangThai:
        """
        Thêm cạnh điều kiện: `ham_dieu_kien(state)` trả về nhãn nhánh hoặc tên nút đích.
        """
        self.cac_canh_dieu_kien[nut_nguon] = (ham_dieu_kien, ban_do_dich)
        return self

    def dat_diem_bat_dau(self, ten_nut: str) -> DoThiTrangThai:
        """Thiết lập nút khởi động của đồ thị."""
        self.diem_bat_dau = ten_nut
        return self

    def bien_dich(
        self,
        checkpointer: Optional[BoLuuDiemKiemTra] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
        gioi_han_buoc: int = 25,
    ) -> DoThiThucThi:
        """
        Biên dịch đồ thị thành đối tượng `DoThiThucThi` sẵn sàng chạy.
        """
        if not self.diem_bat_dau:
            raise ValueError("Đồ thị chưa có điểm bắt đầu. Hãy gọi dat_diem_bat_dau() hoặc them_canh(START, ...).")

        return DoThiThucThi(
            cac_nut=self.cac_nut,
            cac_canh=self.cac_canh,
            cac_canh_dieu_kien=self.cac_canh_dieu_kien,
            diem_bat_dau=self.diem_bat_dau,
            kenh_cap_nhat=self.kenh_cap_nhat,
            checkpointer=checkpointer or self.bo_luu_diem,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )


# Các hàm Reducer chuẩn
def kenh_ghi_de(gia_tri_cu: Any, gia_tri_moi: Any) -> Any:
    """Reducer ghi đè giá trị mới lên giá trị cũ."""
    return gia_tri_moi


def kenh_them_danh_sach(danh_sach_cu: Optional[List[Any]], gia_tri_moi: Any) -> List[Any]:
    """Reducer nối phần tử mới vào danh sách."""
    res = list(danh_sach_cu or [])
    if isinstance(gia_tri_moi, list):
        res.extend(gia_tri_moi)
    else:
        res.append(gia_tri_moi)
    return res


def kenh_gop_dict(dict_cu: Optional[Dict[str, Any]], gia_tri_moi: Any) -> Dict[str, Any]:
    """Reducer hợp nhất từ điển."""
    res = dict(dict_cu or {})
    if isinstance(gia_tri_moi, dict):
        res.update(gia_tri_moi)
    return res


# English Aliases
StateGraph = DoThiTrangThai
CompiledGraph = DoThiThucThi
