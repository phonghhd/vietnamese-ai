"""
Tutorial 5: Self-Adapting Language Models (SALM)
=================================================

Hướng dẫn: Self-Refine, Self-Consistency, Adaptive LoRA, Self-Generated Data, TTT.
"""

import numpy as np

# === 1. Self-Refinement ===
from vietnamese_ai import SelfRefine


def mock_generate(prompt):
    return f"Đây là output cho: {prompt[:30]}... AI đang phát triển mạnh mẽ tại Việt Nam."

refine = SelfRefine(
    ham_sinh=mock_generate,
    so_vong_toi_da=3,
    nguong_chat_luong=0.9,
)

ket_qua = refine.chay("Viết đoạn văn về AI tại Việt Nam")
print("Self-Refine:")
print(f"  Output: {ket_qua['output_cuoi'][:80]}...")
print(f"  Số vòng: {ket_qua['so_vong']}")
print(f"  Điểm: {ket_qua['diem_cuoi']:.3f}")
print(f"  Đạt ngưỡng: {ket_qua['dat_nguong']}")

# === 2. Self-Consistency ===

from vietnamese_ai import SelfConsistency


def mock_generate_with_variation(prompt):
    import random
    answers = ["42", "42", "43", "42", "44"]
    return random.choice(answers)

sc = SelfConsistency(
    ham_sinh=mock_generate_with_variation,
    so_luong=7,
)

ket_qua = sc.chay("2 + 2 = ?")
print("\nSelf-Consistency:")
print(f"  Đáp án: {ket_qua['dap_an']}")
print(f"  Đồng nhất: {ket_qua['ty_le_dong_nhat']:.0%}")
print(f"  Phân phối: {ket_qua['phan_phoi']}")

# Chain-of-Thought mode
ket_qua_cot = sc.chay("Phân tích ưu nhược điểm của AI", che_do="cot")
print(f"  CoT paths: {ket_qua_cot['so_luong_paths']}")

# === 3. Adaptive LoRA ===

from vietnamese_ai import AdaptiveLoRA

adaptive = AdaptiveLoRA(che_do="keyword")

# Đăng ký adapters
adaptive.dang_ky_adapter(
    "math", "lora_math_adapter",
    keywords=["tính", "cộng", "trừ", "nhân", "chia", "phương trình", "số"]
)
adaptive.dang_ky_adapter(
    "code", "lora_code_adapter",
    keywords=["code", "function", "class", "python", "algorithm", "program"]
)
adaptive.dang_ky_adapter(
    "translate", "lora_translate_adapter",
    keywords=["dịch", "translate", "english", "vietnamese", "ngôn ngữ"]
)

# Chọn adapter
for text in ["Tính phương trình bậc 2", "Viết function Python", "Dịch sang tiếng Anh"]:
    chon = adaptive.chon_adapter(text)
    print(f"\n  '{text}' → {chon[0]['ten']} (điểm: {chon[0]['diem']:.2f})")

# Kết hợp adapters
trong_so = adaptive.ket_hop_adapters("Dịch code Python sang Java")
print(f"\n  Kết hợp: {trong_so}")

print(f"\n  Stats: {adaptive.thong_ke_su_dung()}")

# === 4. Self-Generated Data ===

from vietnamese_ai import SinhDuLieuTuDong

sinh = SinhDuLieuTuDong(
    ham_sinh=mock_generate,
    nguong_chat_luong=0.3,
)

# Thêm seed examples
sinh.them_giong_mau("Tóm tắt văn bản về AI", "AI là lĩnh vực khoa học máy tính...")
sinh.them_giong_mau("Dịch câu sang tiếng Anh", "Hello world")
sinh.them_giong_mau("Phân tích ưu nhược điểm", "Ưu điểm: hiệu quả. Nhược điểm: tốn kém.")

# Sinh dữ liệu
du_lieu = sinh.sinh(5, loai="instruction")
print("\nSelf-Generated Data:")
print(f"  Đã sinh: {len(du_lieu)} mẫu")
for mau in du_lieu[:2]:
    print(f"  - {mau['instruction'][:50]}...")

print(f"\n  Stats: {sinh.thong_ke()}")

# === 5. Test-Time Training ===

from vietnamese_ai import PhanLoai, TestTimeTraining

# Train model
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] > 0).astype(int)
model = PhanLoai(thuat_toan="logistic")
model.huan_luyen(X_train, y_train)

# TTT
ttt = TestTimeTraining(model, che_do="entropy_minimization", so_buoc_mac_dinh=5)

# Lưu trọng số gốc
ttt.luu_trong_so_goc()

# Adapt trên test data
X_test = np.random.randn(20, 4)
ket_qua = ttt.thich_ung(X_test)
print("\nTest-Time Training:")
print(f"  Loss: {ket_qua['loss_dau']:.4f} → {ket_qua['loss_cuoi']:.4f}")
print(f"  Giảm loss: {ket_qua['giam_loss']:.4f}")
print(f"  Số bước: {ket_qua['so_buoc']}")

# Dự đoán
du_doan = ttt.du_doan(X_test[:5])
print(f"  Predictions: {du_doan}")

# Phục hồi trọng số gốc
ttt.phuc_hoi_trong_so()
print(f"  Restored: {not ttt.da_thich_ung}")

print(f"\n  Stats: {ttt.thong_ke()}")

print("\n✓ Tutorial 5 hoàn tất!")
