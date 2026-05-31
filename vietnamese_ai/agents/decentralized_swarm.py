"""Decentralized Swarm - Bầy đàn phân tán trên mạng lưới DePIN."""

import json
import logging
from typing import Optional, Dict, Any, List
import concurrent.futures

from vietnamese_ai.agents.swarm import TacTuSwarm, HeThongSwarm, KetQuaSwarm
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.edge.p2p_network import P2PTracker, TokenLedger

logger = logging.getLogger("DecentralizedSwarm")

class P2PSwarmOrchestrator(HeThongSwarm):
    """
    Orchestrator mạng lưới Swarm phân tán.
    Cho phép tác tử tự động "đẻ" (spawn) các tác tử con lên các Node khác trong mạng P2P.
    Hỗ trợ Tự động hồi sinh (Fault Tolerance) và Đồng thuận Bầy đàn (Consensus).
    """
    def __init__(self, agent_khoi_tao: TacTuSwarm, tracker: P2PTracker, ledger: Optional[TokenLedger] = None):
        super().__init__(agent_khoi_tao)
        self.tracker = tracker
        self.ledger = ledger
        self.active_tasks: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        # Cung cấp công cụ Spawn cho agent khởi tạo
        self._cap_quyen_spawn(self.agent_khoi_tao)

    def dang_ky_agent(self, agent: TacTuSwarm):
        super().dang_ky_agent(agent)
        self._cap_quyen_spawn(agent)

    def _cap_quyen_spawn(self, agent: TacTuSwarm):
        """Thêm công cụ 'spawn_sub_agent' cho tác tử."""
        def spawn_func(nhiem_vu: str, ten_sub_agent: str, vai_tro: str) -> str:
            # Tìm Node rảnh rỗi trên mạng DePIN
            best_node = self.tracker.tim_node_tot_nhat()
            
            if best_node is None:
                return f"[Spawn Failed] Không có Node nào trên mạng P2P đang rảnh rỗi để xử lý nhiệm vụ: {nhiem_vu}"
                
            logger.info(f"Đang offload Tác tử con '{ten_sub_agent}' ({vai_tro}) sang Node từ xa.")
            
            sub_agent = TacTuSwarm(
                ten=ten_sub_agent,
                vai_tro=vai_tro,
                llm=agent.llm,
                huong_dan="Hãy hoàn thành nhiệm vụ và báo cáo lại kết quả."
            )
            
            # Đẩy vào luồng chạy song song, truyền theo metadata để retry
            future = self.executor.submit(self._chay_node_tu_xa, best_node, sub_agent, nhiem_vu)
            self.active_tasks[future] = {
                "nhiem_vu": nhiem_vu,
                "ten_sub_agent": ten_sub_agent,
                "vai_tro": vai_tro,
                "sub_agent": sub_agent,
                "retries": 3 # Tối đa hồi sinh 3 lần
            }
            
            return f"[Spawn Success] Đã giao nhiệm vụ '{nhiem_vu}' cho tác tử '{ten_sub_agent}'. Kết quả sẽ được tổng hợp tự động."

        tool_spawn = CongCu(
            ten="spawn_sub_agent",
            mo_ta="Tạo ra một Tác tử con trên mạng DePIN để giải quyết song song một bài toán phụ.",
            ham_thuc_thi=spawn_func
        )
        
        if "spawn_sub_agent" not in agent.cong_cu:
            agent.danh_sach_cong_cu_day_du.append(tool_spawn)
            agent.cong_cu[tool_spawn.ten] = tool_spawn
            
            from vietnamese_ai.agents.agent import REACT_SYSTEM_PROMPT
            tools_desc = ""
            for name, cc in agent.cong_cu.items():
                tools_desc += f"- {name}: {cc.mo_ta}. Tham số: {json.dumps(cc.tham_so, ensure_ascii=False)}\n"
            agent.bo_nho.system_prompt = REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc)
            agent.bo_nho.system_prompt += f"\n\nTên của bạn: {agent.ten}\nVai trò của bạn: {agent.vai_tro}\n{agent.huong_dan}"

    def _chay_node_tu_xa(self, node_engine: Any, sub_agent: TacTuSwarm, nhiem_vu: str) -> str:
        """Hàm mô phỏng việc chạy Agent trên Edge Node thông qua P2P RPC."""
        ket_qua = sub_agent.chay(nhiem_vu)
        
        if self.ledger is not None:
            node_id = getattr(node_engine, "node_id", "UnknownNode")
            self.ledger.thuong_token(node_id, 5.0)
            
        return ket_qua

    def _chay_dong_thuan_bft(self, danh_sach_ket_qua: List[str]) -> str:
        """
        Giao thức Đồng thuận Bầy đàn (Consensus Protocol).
        Nếu có nhiều hơn 1 kết quả, ta dùng LLM của Agent Mẹ để chốt đáp án cuối cùng.
        """
        if not danh_sach_ket_qua:
            return "Không có kết quả từ mạng Swarm."
        if len(danh_sach_ket_qua) == 1:
            return danh_sach_ket_qua[0]
            
        # Dùng LLM của Agent Khởi tạo làm Trọng tài
        prompt_trong_tai = "Dưới đây là các câu trả lời từ nhiều Agent con khác nhau cho cùng một bài toán:\n\n"
        for i, res in enumerate(danh_sach_ket_qua):
            prompt_trong_tai += f"--- Ý kiến {i+1} ---\n{res}\n"
        prompt_trong_tai += "\nHãy đóng vai trò Trọng tài (Byzantine Fault Tolerance). So sánh, tìm ra đáp án đúng nhất theo đa số hoặc độ hợp lý, loại bỏ các đáp án sai/ảo giác và đưa ra Kết Luận Cuối Cùng."
        
        # Mô phỏng quá trình LLM chốt đáp án
        if hasattr(self.agent_khoi_tao.llm, "sinh_van_ban"):
            ket_luan = self.agent_khoi_tao.llm.sinh_van_ban(prompt_trong_tai, do_dai=256)
        elif callable(self.agent_khoi_tao.llm):
            ket_luan = self.agent_khoi_tao.llm(prompt_trong_tai)
        else:
            ket_luan = "[Lỗi Trọng Tài] Không thể đánh giá đồng thuận."
            
        return f"\n--- KẾT LUẬN ĐỒNG THUẬN BẦY ĐÀN (CONSENSUS) ---\n{ket_luan}"
        
    def chay(self, truy_van: str, bo_nho_chung=None) -> KetQuaSwarm:
        ket_qua_chinh = super().chay(truy_van, bo_nho_chung)
        
        if self.active_tasks:
            logger.info("Đang chờ các Tác tử con hoàn thành...")
            results = []
            
            # Sử dụng danh sách động để có thể chèn thêm task Retry
            while self.active_tasks:
                done, not_done = concurrent.futures.wait(
                    self.active_tasks.keys(), 
                    timeout=5.0, # Time out mỗi vòng kiểm tra
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    meta = self.active_tasks.pop(future)
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        logger.warning(f"Tác tử '{meta['ten_sub_agent']}' bị lỗi (Node Crash): {str(e)}")
                        if meta["retries"] > 0:
                            logger.info(f"Đang HỒI SINH (Auto-Respawn) tác tử '{meta['ten_sub_agent']}' sang Node khác... (Còn {meta['retries']} lần thử)")
                            
                            new_node = self.tracker.tim_node_tot_nhat()
                            if new_node:
                                new_future = self.executor.submit(self._chay_node_tu_xa, new_node, meta["sub_agent"], meta["nhiem_vu"])
                                meta["retries"] -= 1
                                self.active_tasks[new_future] = meta
                            else:
                                results.append(f"Lỗi: Không còn Node rảnh để hồi sinh tác tử '{meta['ten_sub_agent']}'.")
                        else:
                            results.append(f"Tác tử '{meta['ten_sub_agent']}' đã CHẾT hẳn sau 3 lần thử.")

            # Chạy giao thức Đồng Thuận Bầy Đàn
            ket_qua_dong_thuan = self._chay_dong_thuan_bft(results)
            ket_qua_chinh.tin_nhan += ket_qua_dong_thuan
            
        return ket_qua_chinh
