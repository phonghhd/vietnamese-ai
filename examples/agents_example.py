from vietnamese_ai.agents import TacTu, HeThongDaTacTu, cong_cu, cong_cu_may_tinh, cong_cu_doc_file
from vietnamese_ai.llm import VietnameseLLM
import sys

# Khởi tạo mô hình LLM (Ở đây có thể dùng VietnameseLLM hoặc HuggingFace)
# Vì đây là ví dụ, chúng ta sẽ giả lập một LLM trả về chuỗi ReAct tĩnh hoặc dùng LLM thật nếu có RAM.
print("Đang khởi tạo hệ thống tác tử...")

# Định nghĩa một công cụ tùy chỉnh
@cong_cu(ten="lay_thoi_tiet", mo_ta="Lấy thông tin thời tiết. Tham số: dia_diem (tên thành phố)")
def lay_thoi_tiet(dia_diem: str) -> str:
    # Mô phỏng gọi API thời tiết
    if "Hà Nội" in dia_diem:
        return "Trời nắng, 32 độ C."
    elif "Hồ Chí Minh" in dia_diem:
        return "Trời mưa rào, 28 độ C."
    else:
        return "Không có dữ liệu thời tiết cho khu vực này."

# Mock LLM đơn giản để ví dụ chạy được ngay mà không cần tải model nặng
class MockLLMChoViDu:
    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        if "Thời tiết" in prompt or "Hà Nội" in prompt:
            return "Suy nghĩ: Tôi cần dùng công cụ lay_thoi_tiet để biết thời tiết Hà Nội.\nHành động: lay_thoi_tiet\nTham số: {\"dia_diem\": \"Hà Nội\"}"
        elif "Quan sát: Trời nắng" in prompt:
            return "Suy nghĩ: Tôi đã biết thời tiết.\nTrả lời: Thời tiết ở Hà Nội hiện tại là trời nắng, 32 độ C nhé bạn!"
        elif "toán học" in prompt or "15 * 8" in prompt:
            return "Suy nghĩ: Tôi cần tính 15 * 8.\nHành động: may_tinh\nTham số: {\"bieu_thuc\": \"15 * 8\"}"
        elif "Quan sát: 120" in prompt:
            return "Suy nghĩ: Tôi đã có kết quả.\nTrả lời: Kết quả của 15 * 8 là 120."
        return "Suy nghĩ: Tôi chưa hiểu.\nTrả lời: Xin lỗi, tôi chưa hiểu yêu cầu của bạn."

llm = MockLLMChoViDu()

# 1. Tác tử đơn (Single Agent with Tools)
print("\n--- 1. Tác tử đơn với Công cụ ---")
tac_tu_thoi_tiet = TacTu(
    llm=llm,
    danh_sach_cong_cu=[lay_thoi_tiet, cong_cu_may_tinh]
)

ket_qua_1 = tac_tu_thoi_tiet.chay("Thời tiết ở Hà Nội hôm nay thế nào?")
print(f"Người dùng: Thời tiết ở Hà Nội hôm nay thế nào?")
print(f"Trợ lý: {ket_qua_1}")

ket_qua_2 = tac_tu_thoi_tiet.chay("Tính giúp tôi 15 * 8 bằng bao nhiêu?")
print(f"Người dùng: Tính giúp tôi 15 * 8 bằng bao nhiêu?")
print(f"Trợ lý: {ket_qua_2}")

# 2. Hệ thống Đa Tác tử (Multi-Agent System)
print("\n--- 2. Hệ thống Đa Tác tử (Tuần tự) ---")
tac_tu_toan_hoc = TacTu(llm=llm, danh_sach_cong_cu=[cong_cu_may_tinh])

he_thong = HeThongDaTacTu(
    danh_sach_tac_tu={
        "ChuyenGiaThoiTiet": tac_tu_thoi_tiet,
        "ChuyenGiaToanHoc": tac_tu_toan_hoc
    },
    loai_dieu_phoi="sequential"
)

ket_qua_nhom = he_thong.chay("Hãy cho tôi biết thời tiết Hà Nội và tính 15 * 8.")
print("Quá trình kết hợp nhiều tác tử hoàn tất. Lịch sử:")
for msg in he_thong.lich_su_chung.lay_lich_su():
    print(f"[{msg['role'].upper()}] {msg['content']}")
