from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from .agent import TacTu


class MoA:
    """
    Kiến trúc Mixture of Agents (MoA).
    Sử dụng nhiều Proposer Agents để đưa ra các phương án giải quyết khác nhau,
    sau đó dùng Aggregator Agent để tổng hợp thành câu trả lời tốt nhất.
    """

    def __init__(
        self,
        danh_sach_proposers: List[TacTu],
        aggregator: TacTu,
        enable_semantic_routing: bool = True,
    ):
        """
        Khởi tạo hệ thống MoA.

        Args:
            danh_sach_proposers: Danh sách các Tác tử đóng vai trò đưa ra ý tưởng/câu trả lời ban đầu.
            aggregator: Tác tử đóng vai trò tổng hợp, đánh giá và đưa ra kết luận cuối cùng.
        """
        if not danh_sach_proposers:
            raise ValueError("Cần ít nhất một Proposer Agent.")
        if not aggregator:
            raise ValueError("Cần có một Aggregator Agent.")

        self.proposers = danh_sach_proposers
        self.aggregator = aggregator

        # Thiết lập Prompt cho Aggregator để nó biết nhiệm vụ của mình
        original_prompt = self.aggregator.bo_nho.system_prompt
        moa_instructions = """
Bạn là một Tác tử Tổng hợp (Aggregator Agent) trong hệ thống Mixture of Agents (MoA).
Nhiệm vụ của bạn là nhận các câu trả lời/ý tưởng từ nhiều Tác tử Đề xuất (Proposers) khác nhau cho cùng một câu hỏi của người dùng.
Hãy phân tích, đánh giá điểm mạnh/yếu của từng câu trả lời, đối chiếu thông tin, và tổng hợp lại thành MỘT CÂU TRẢ LỜI DUY NHẤT, chính xác và toàn diện nhất.
Đừng chỉ nối các câu trả lời lại với nhau. Hãy viết lại một cách tự nhiên như thể đó là câu trả lời của chính bạn.
"""
        self.aggregator.bo_nho.system_prompt = original_prompt + "\n\n" + moa_instructions
        self.enable_semantic_routing = enable_semantic_routing

    def _chon_proposers(self, truy_van: str) -> List[TacTu]:
        """Tối ưu v27.0.1: Semantic Routing (O(1)) thay vì chạy tất cả (O(N))."""
        if not self.enable_semantic_routing or len(self.proposers) <= 2:
            return self.proposers

        tu_khoa = set(truy_van.lower().split())
        diem_so = []
        for proposer in self.proposers:
            mota = proposer.bo_nho.system_prompt.lower()
            # Tính Jaccard similarity cơ bản làm Semantic Router
            tu_khoa_mota = set(mota.split())
            chung = len(tu_khoa.intersection(tu_khoa_mota))
            diem_so.append((chung, proposer))

        # Chỉ gọi tối đa 2 Agent tốt nhất
        diem_so.sort(key=lambda x: x[0], reverse=True)
        return [p for diem, p in diem_so[:2]]

    def chay(self, truy_van: str) -> str:
        """
        Chạy kiến trúc MoA cho một truy vấn.
        """
        ket_qua_proposers = []

        # Semantic Routing v27
        proposers_duoc_chon = self._chon_proposers(truy_van)

        # Chạy song song các proposer được chọn
        with ThreadPoolExecutor(max_workers=len(proposers_duoc_chon)) as executor:
            future_to_proposer = {
                executor.submit(proposer.chay, truy_van): idx
                for idx, proposer in enumerate(proposers_duoc_chon)
            }

            # Thu thập kết quả
            for future in as_completed(future_to_proposer):
                idx = future_to_proposer[future]
                try:
                    res = future.result()
                    ket_qua_proposers.append((idx, res))
                except Exception as exc:
                    ket_qua_proposers.append((idx, f"Lỗi từ Proposer {idx + 1}: {exc}"))

        # Sắp xếp lại theo thứ tự ban đầu để dễ theo dõi (tùy chọn)
        ket_qua_proposers.sort(key=lambda x: x[0])

        # Xây dựng prompt cho Aggregator
        aggregator_prompt = f"Câu hỏi gốc của người dùng: {truy_van}\n\nCác câu trả lời đề xuất:\n"
        for idx, (_, res) in enumerate(ket_qua_proposers):
            aggregator_prompt += f"\n--- Proposer {idx + 1} ---\n{res}\n"

        aggregator_prompt += (
            "\nDựa trên các đề xuất trên, hãy đưa ra câu trả lời tổng hợp cuối cùng."
        )

        # Chạy aggregator
        ket_qua_cuoi = self.aggregator.chay(aggregator_prompt)

        return ket_qua_cuoi
