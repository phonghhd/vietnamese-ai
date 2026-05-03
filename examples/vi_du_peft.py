"""Ví dụ: PEFT & Instruction Tuning."""

from vietnamese_ai.fine_tuning.dataset import InstructionDataset
from vietnamese_ai.fine_tuning.peft_config import PEFTConfig


def vi_du_peft_config():
    """Cấu hình PEFT."""
    config = PEFTConfig.lora(rank=16, alpha=32.0)
    print(f"Phương pháp: {config.phuong_phap}")
    print(f"Rank: {config.rank}")
    print(f"Scaling: {config.scaling}")
    print(f"Target modules: {config.target_modules}")

    config_qlora = PEFTConfig.qlora(rank=8, bits=4)
    print(f"\nQLoRA config: {config_qlora}")


def vi_du_instruction_dataset():
    """Chuẩn bị dữ liệu instruction tuning."""
    dataset = InstructionDataset(che_do="alpaca")
    dataset.tai_tu_list([
        {
            "instruction": "Tóm tắt văn bản sau",
            "input": "Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập.",
            "output": "AI đang thay đổi nhiều lĩnh vực trong cuộc sống.",
        },
        {
            "instruction": "Dịch sang tiếng Anh",
            "input": "Học máy rất thú vị",
            "output": "Machine learning is very interesting",
        },
        {
            "instruction": "Giải thích khái niệm",
            "input": "",
            "output": "Học máy là một nhánh của trí tuệ nhân tạo.",
        },
    ])

    print(f"Số mẫu: {len(dataset)}")
    print(f"Thống kê: {dataset.thong_ke()}")

    formatted = dataset.format_tat_ca()
    print(f"\nFormatted sample:\n{formatted[0][:200]}...")

    dataset.chia_du_lieu(ty_le_val=0.3)
    print(f"\nTrain: {len(dataset.train)}, Val: {len(dataset.val)}")


if __name__ == "__main__":
    print("=== PEFT Config ===")
    vi_du_peft_config()
    print("\n=== Instruction Dataset ===")
    vi_du_instruction_dataset()
