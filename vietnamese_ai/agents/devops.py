import ast
import os
import subprocess
from typing import Any, Optional

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu


def viet_code_an_toan(file_name: str, ma_nguon: str, ma_test: Optional[str] = None) -> str:
    """
    Công cụ ghi mã nguồn an toàn với AST Validation và Dry-Run Pipeline.
    """
    # 1. Kiểm tra Safe Zone
    base_dir = "/home/phong/V-Neural/scratch"
    os.makedirs(base_dir, exist_ok=True)

    # Chỉ cho phép lưu vào scratch
    if ".." in file_name or "/" in file_name:
        return "Lỗi Bảo mật: Tên file không được chứa đường dẫn."

    file_path = os.path.join(base_dir, file_name)
    test_path = os.path.join(base_dir, f"test_{file_name}")

    # 2. Bước 1: AST Validation
    try:
        ast.parse(ma_nguon)
    except SyntaxError as e:
        return f"Lỗi Cú pháp (SyntaxError) trong ma_nguon:\n{e.msg} tại dòng {e.lineno}"

    if ma_test:
        try:
            ast.parse(ma_test)
        except SyntaxError as e:
            return f"Lỗi Cú pháp (SyntaxError) trong ma_test:\n{e.msg} tại dòng {e.lineno}"

    # 3. Bước 2: Dry Run Pipeline
    # Tạm thời ghi file ra đĩa
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(ma_nguon)

    if ma_test:
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(ma_test)

        # Chạy Pytest trong thư mục scratch
        try:
            # Need to run with the venv python to have pytest
            python_path = "/home/phong/V-Neural/venv/bin/python"
            if not os.path.exists(python_path):
                python_path = "python"

            result = subprocess.run(
                [python_path, "-m", "pytest", test_path, "-v"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=base_dir,
            )

            if result.returncode != 0:
                # Test Fail -> Hoàn tác và trả lỗi
                os.remove(file_path)
                os.remove(test_path)
                return f"Lỗi Dry-Run (Test Failed). Sửa code lại nhé:\n{result.stdout}\n{result.stderr}"
        except subprocess.TimeoutExpired:
            os.remove(file_path)
            os.remove(test_path)
            return "Lỗi Dry-Run: Quá thời gian (Timeout). Có vòng lặp vô hạn không?"
        except Exception as e:
            os.remove(file_path)
            if os.path.exists(test_path):
                os.remove(test_path)
            return f"Lỗi hệ thống khi chạy test: {str(e)}"

    return f"Thành công! Mã nguồn đã được kiểm duyệt và lưu an toàn tại: {file_path}"


cong_cu_viet_code_an_toan = CongCu(
    ten="viet_code_an_toan",
    mo_ta="Công cụ để viết code Python. Bắt buộc cung cấp ma_nguon và (khuyến nghị) ma_test (pytest). Công cụ sẽ tự động check AST và chạy Test trước khi lưu.",
    ham_thuc_thi=viet_code_an_toan,
)


class DevOpsAgent(TacTu):
    """
    Tác tử chuyên trách Lập trình Tự động (Auto-Coding).
    Sử dụng vòng lặp Self-Correction để hoàn thiện mã nguồn cho tới khi Pass mọi Test.
    """

    def __init__(self, llm: Any, max_iterations: int = 5):
        super().__init__(
            llm=llm, danh_sach_cong_cu=[cong_cu_viet_code_an_toan], max_iterations=max_iterations
        )

        # Override System Prompt
        devops_prompt = (
            "Bạn là một Kỹ sư DevOps AI (DevOps Agent). Nhiệm vụ của bạn là lập trình tự động.\n"
            "Khi người dùng yêu cầu viết code, BẠN PHẢI SỬ DỤNG công cụ 'viet_code_an_toan'.\n"
            "Hãy luôn viết mã chính (ma_nguon) và mã kiểm thử (ma_test sử dụng pytest).\n"
            "LƯU Ý QUAN TRỌNG: Nếu công cụ trả về lỗi cú pháp (SyntaxError) hoặc lỗi Test Fail, "
            "hãy KIÊN NHẪN đọc lỗi đó, suy luận tìm nguyên nhân và gọi lại công cụ với phiên bản code đã sửa.\n"
            "Hãy làm đến khi nào nhận được thông báo 'Thành công!' thì mới dừng lại."
        )
        self.bo_nho.system_prompt += f"\n\n{devops_prompt}"
