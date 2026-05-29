import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agent import TacTu
from .memory import BoNhoTacTu
from .tools import CongCu


@dataclass
class KetQuaSwarm:
    """Kết quả trả về từ một Tác tử trong Swarm."""
    tin_nhan: str
    tac_tu_tiep_theo: Optional['TacTuSwarm'] = None
    du_lieu_bo_sung: Optional[Dict[str, Any]] = None

class TacTuSwarm(TacTu):
    """
    Tác tử chuyên dụng cho kiến trúc Bầy đàn (Swarm).
    Có khả năng chuyển giao (hand-off) công việc cho tác tử khác.
    """
    def __init__(
        self,
        ten: str,
        vai_tro: str,
        llm: Any,
        danh_sach_cong_cu: Optional[List[CongCu]] = None,
        max_iterations: int = 5,
        huong_dan: str = ""
    ):
        self.ten = ten
        self.vai_tro = vai_tro
        self.huong_dan = huong_dan

        # Thêm hướng dẫn cụ thể của Tác tử Swarm vào System Prompt
        self._chuan_bi_cong_cu_chuyen_giao(danh_sach_cong_cu or [])

        super().__init__(llm=llm, danh_sach_cong_cu=self.danh_sach_cong_cu_day_du, max_iterations=max_iterations)

        # Override system prompt để nhấn mạnh vai trò
        self.bo_nho.system_prompt += f"\n\nTên của bạn: {self.ten}\nVai trò của bạn: {self.vai_tro}\n{self.huong_dan}"

    def _chuan_bi_cong_cu_chuyen_giao(self, danh_sach_cong_cu: List[CongCu]):
        """Chuẩn bị danh sách công cụ hiện có"""
        self.danh_sach_cong_cu_day_du = danh_sach_cong_cu.copy()

    def them_cong_cu_chuyen_giao(self, agent_dich: 'TacTuSwarm', ten_cong_cu: str, mo_ta: str):
        """Thêm một công cụ để chuyển giao cho agent khác."""
        def handoff_func() -> str:
            # Trả về tín hiệu đặc biệt để orchestrator bắt được
            return f"__HANDOFF_TO__:{agent_dich.ten}"

        cc_chuyen_giao = CongCu(
            ten=ten_cong_cu,
            mo_ta=mo_ta,
            ham_thuc_thi=handoff_func
        )
        self.danh_sach_cong_cu_day_du.append(cc_chuyen_giao)
        # Cập nhật lại dict công cụ của class cha
        self.cong_cu[cc_chuyen_giao.ten] = cc_chuyen_giao

        # Cập nhật lại system prompt để hiện công cụ mới
        tools_desc = ""
        for name, cc in self.cong_cu.items():
            tools_desc += f"- {name}: {cc.mo_ta}. Tham số: {json.dumps(cc.tham_so, ensure_ascii=False)}\n"

        # Cần extract prompt gốc từ super()
        from .agent import REACT_SYSTEM_PROMPT
        self.bo_nho.system_prompt = REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc)
        self.bo_nho.system_prompt += f"\n\nTên của bạn: {self.ten}\nVai trò của bạn: {self.vai_tro}\n{self.huong_dan}"


class HeThongSwarm:
    """
    Bộ điều phối (Orchestrator) cho hệ thống Bầy đàn.
    Quản lý luồng giao tiếp và chuyển giao (hand-off) giữa các TacTuSwarm.
    """
    def __init__(self, agent_khoi_tao: TacTuSwarm):
        self.agent_khoi_tao = agent_khoi_tao
        self.cac_agent: Dict[str, TacTuSwarm] = {}
        self.dang_ky_agent(agent_khoi_tao)

    def dang_ky_agent(self, agent: TacTuSwarm):
        self.cac_agent[agent.ten] = agent

    def tao_lien_ket(self, agent_nguon: TacTuSwarm, agent_dich: TacTuSwarm, ly_do: str):
        """Tạo công cụ chuyển giao từ agent nguồn sang agent đích."""
        self.dang_ky_agent(agent_nguon)
        self.dang_ky_agent(agent_dich)

        ten_cong_cu = f"chuyen_giao_cho_{agent_dich.ten.lower().replace(' ', '_')}"
        agent_nguon.them_cong_cu_chuyen_giao(
            agent_dich=agent_dich,
            ten_cong_cu=ten_cong_cu,
            mo_ta=f"Chuyển giao quyền điều khiển cho {agent_dich.ten} khi cần: {ly_do}"
        )

    def chay(self, truy_van: str, bo_nho_chung: Optional[BoNhoTacTu] = None) -> KetQuaSwarm:
        """
        Khởi chạy luồng Swarm bắt đầu từ agent_khoi_tao.
        """
        agent_hien_tai = self.agent_khoi_tao
        truy_van_hien_tai = truy_van

        # Nếu có bộ nhớ chung (ví dụ context của cả session)
        if bo_nho_chung:
            agent_hien_tai.bo_nho.lich_su = bo_nho_chung.lich_su.copy()

        lich_su_handoff = []

        # Giới hạn số lần hand-off tối đa để tránh lặp vô hạn
        max_handoffs = 10
        handoff_count = 0

        while handoff_count < max_handoffs:
            # Chạy agent hiện tại
            ket_qua_str = agent_hien_tai.chay(truy_van_hien_tai)

            # Kiểm tra xem kết quả có phải là lệnh hand-off không
            # Lệnh hand-off được trả về trong "Quan sát" nếu agent gọi công cụ handoff,
            # tuy nhiên `agent.chay` của chúng ta trả về string khi xong.
            # Do cơ chế `TacTu.chay` ẩn tool call bên trong, nên ta cần sửa lại logic nếu agent kết luận bằng __HANDOFF_TO__.

            # Ta cần kiểm tra trong bộ nhớ của agent hiện tại xem có lệnh handoff nào được thực hiện không
            handoff_target = None
            for msg in reversed(agent_hien_tai.bo_nho.lich_su):
                if msg["role"] == "tool" and "__HANDOFF_TO__:" in msg["content"]:
                    handoff_target = msg["content"].split("__HANDOFF_TO__:")[1].strip()
                    break

            if handoff_target and handoff_target in self.cac_agent:
                # Đã có chuyển giao
                agent_tiep_theo = self.cac_agent[handoff_target]
                lich_su_handoff.append(f"{agent_hien_tai.ten} -> {agent_tiep_theo.ten}")

                # Copy bộ nhớ sang agent tiếp theo
                agent_tiep_theo.bo_nho.lich_su = agent_hien_tai.bo_nho.lich_su.copy()

                # Chuyển agent hiện tại thành agent tiếp theo và lặp lại
                agent_hien_tai = agent_tiep_theo
                # Cập nhật truy vấn thành chỉ thị tiếp tục
                truy_van_hien_tai = "Hệ thống đã chuyển giao luồng cho bạn. Hãy tiếp tục giải quyết yêu cầu dựa trên lịch sử."
                handoff_count += 1
            else:
                # Không có chuyển giao, tác vụ đã hoàn thành
                return KetQuaSwarm(
                    tin_nhan=ket_qua_str,
                    tac_tu_tiep_theo=None,
                    du_lieu_bo_sung={"lich_su_handoff": lich_su_handoff}
                )

        return KetQuaSwarm(
            tin_nhan="Lỗi: Hệ thống đạt giới hạn số lần chuyển giao tối đa.",
            tac_tu_tiep_theo=agent_hien_tai
        )
