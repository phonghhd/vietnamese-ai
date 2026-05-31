import concurrent.futures
from typing import Any, Dict, List

from .agent import TacTu
from .tools import CongCu


class WorkerAgent(TacTu):
    """
    Tác tử cấp dưới (Worker), chuyên thực thi một tác vụ cụ thể.
    """

    def __init__(self, ten: str, mo_ta_chuyen_mon: str, llm: Any, danh_sach_cong_cu: List[CongCu]):
        super().__init__(llm=llm, danh_sach_cong_cu=danh_sach_cong_cu, max_iterations=5)
        self.ten = ten
        self.mo_ta_chuyen_mon = mo_ta_chuyen_mon

        # Override System Prompt để Worker biết rõ nhiệm vụ
        self.bo_nho.system_prompt += f"\n\nBẠN LÀ WORKER: {self.ten}\nCHUYÊN MÔN: {self.mo_ta_chuyen_mon}\nHãy hoàn thành tác vụ được giao một cách tốt nhất và báo cáo lại kết quả cuối cùng."


class ManagerAgent(TacTu):
    """
    Tác tử quản lý (Manager), có khả năng phân rã tác vụ và điều phối các WorkerAgent.
    """

    def __init__(self, llm: Any, workers: List[WorkerAgent]):
        # ManagerAgent không cần tool thông thường, nó dùng tool "giao_viec" để gọi Worker
        self.workers = {w.ten: w for w in workers}
        danh_sach_cong_cu = self._tao_cong_cu_quan_ly()
        super().__init__(llm=llm, danh_sach_cong_cu=danh_sach_cong_cu, max_iterations=10)

        self.bo_nho.system_prompt += "\n\nBẠN LÀ MANAGER AGENT. Nhiệm vụ của bạn là phân tích yêu cầu phức tạp của người dùng, chia nhỏ nó ra, và giao việc cho các Worker Agent phù hợp thông qua các công cụ. Sau khi thu thập đủ kết quả từ các Worker, hãy tổng hợp thành một câu trả lời cuối cùng."

    def _tao_cong_cu_quan_ly(self) -> List[CongCu]:
        cong_cu_list = []
        for worker_name, worker in self.workers.items():
            # Tạo scope bằng default arg để tránh late binding
            def tao_ham_giao_viec(w=worker):
                def giao_viec(nhiem_vu: str) -> str:
                    print(f"[Manager] Giao việc cho Worker '{w.ten}': {nhiem_vu}")
                    ket_qua = w.chay(nhiem_vu)
                    # Reset memory của worker sau khi xong để sạch sẽ cho tác vụ sau
                    w.bo_nho.lich_su = []
                    return f"Kết quả từ {w.ten}: {ket_qua}"

                return giao_viec

            cc = CongCu(
                ten=f"giao_viec_cho_{worker_name.lower().replace(' ', '_')}",
                mo_ta=f"Giao nhiệm vụ cho chuyên gia: {worker.mo_ta_chuyen_mon}. Tham số 'nhiem_vu' mô tả chi tiết việc cần làm.",
                ham_thuc_thi=tao_ham_giao_viec(),
            )
            cong_cu_list.append(cc)
        return cong_cu_list

    def chay_song_song(self, nhiem_vu_list: List[Dict[str, str]]) -> str:
        """
        Giao nhiều việc cho nhiều worker chạy song song (Thread pool).
        nhiem_vu_list format: [{"worker": "TenWorker", "nhiem_vu": "Noi dung"}]
        """
        ket_qua_tong_hop = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_worker = {}
            for nv in nhiem_vu_list:
                ten_worker = nv["worker"]
                nhiem_vu = nv["nhiem_vu"]
                if ten_worker in self.workers:
                    worker = self.workers[ten_worker]
                    future = executor.submit(worker.chay, nhiem_vu)
                    future_to_worker[future] = ten_worker
                else:
                    ket_qua_tong_hop.append(f"Lỗi: Không tìm thấy worker '{ten_worker}'.")

            for future in concurrent.futures.as_completed(future_to_worker):
                worker_name = future_to_worker[future]
                try:
                    res = future.result()
                    ket_qua_tong_hop.append(f"Kết quả từ {worker_name}:\n{res}")
                except Exception as exc:
                    ket_qua_tong_hop.append(f"Worker {worker_name} sinh ra lỗi: {exc}")

        return "\n\n".join(ket_qua_tong_hop)
