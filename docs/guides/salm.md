# Hướng dẫn: Self-Adapting Language Models (SALM)

## Giới thiệu

SALM là paradigm mới nơi model **tự thích ứng** tại inference time mà không cần fine-tuning truyền thống. Vietnamese AI Framework triển khai 5 components:

| Component | Mô tả |
|---|---|
| `SelfRefine` | Iterative refinement: model tự đánh giá và cải thiện output |
| `SelfConsistency` | Multiple reasoning paths → majority voting |
| `AdaptiveLoRA` | Tự chọn LoRA adapter phù hợp theo task |
| `SinhDuLieuTuDong` | Tự sinh training data từ seed examples |
| `TestTimeTraining` | Adapt weights ngay tại inference time |

---

## Self-Refinement

Model tự đánh giá output qua nhiều vòng lặp: **Generate → Evaluate → Feedback → Refine**

```python
from vietnamese_ai import SelfRefine

def generate(prompt):
    return your_llm.generate(prompt)

refine = SelfRefine(
    ham_sinh=generate,
    so_vong_toi_da=5,
    nguong_chat_luong=0.8,
)

ket_qua = refine.chay("Viết đoạn văn về AI tại Việt Nam")
print(ket_qua["output_cuoi"])
print(f"Hoàn thành sau {ket_qua['so_vong']} vòng")
```

### Custom evaluator

```python
def my_evaluator(prompt, output):
    score = 0.0
    if len(output.split()) > 50:
        score += 0.3
    if "Việt Nam" in output:
        score += 0.3
    if any(c.isupper() for c in output):
        score += 0.2
    return {"diem": score, "chi_tiet": {"do_dai": len(output)}}

refine = SelfRefine(
    ham_sinh=generate,
    ham_danh_gia=my_evaluator,
    so_vong_toi_da=3,
)
```

---

## Self-Consistency

Tạo nhiều reasoning paths khác nhau rồi chọn answer xuất hiện nhiều nhất (majority voting).

```python
from vietnamese_ai import SelfConsistency

sc = SelfConsistency(
    ham_sinh=generate,
    so_luong=7,  # Số paths
)

# Direct mode
ket_qua = sc.chay("2 + 2 = ?")
print(f"Đáp án: {ket_qua['dap_an']}")  # "42"
print(f"Đồng nhất: {ket_qua['ty_le_dong_nhat']:.0%}")

# Chain-of-Thought mode
ket_qua = sc.chay("Phân tích ưu nhược điểm của deep learning", che_do="cot")
print(f"Phân phối: {ket_qua['phan_phoi']}")
```

---

## Adaptive LoRA

Tự động chọn LoRA adapter phù hợp dựa trên nội dung input.

```python
from vietnamese_ai import AdaptiveLoRA

adaptive = AdaptiveLoRA(che_do="keyword")

# Đăng ký adapters cho từng task
adaptive.dang_ky_adapter(
    "math", lora_math,
    keywords=["tính", "cộng", "trừ", "nhân", "chia", "phương trình"]
)
adaptive.dang_ky_adapter(
    "code", lora_code,
    keywords=["code", "function", "class", "python", "algorithm"]
)
adaptive.dang_ky_adapter(
    "translate", lora_translate,
    keywords=["dịch", "translate", "english", "vietnamese"]
)

# Chọn adapter phù hợp
chon = adaptive.chon_adapter("Tính phương trình bậc 2")
print(chon[0]["ten"])  # "math"

# Kết hợp nhiều adapters
trong_so = adaptive.ket_hop_adapters("Dịch code Python sang Java")
# {"code": 0.6, "translate": 0.4}
```

---

## Self-Generated Data

Tự sinh training data từ seed examples (Self-Instruct approach).

```python
from vietnamese_ai import SinhDuLieuTuDong

sinh = SinhDuLieuTuDong(
    ham_sinh=generate,
    nguong_chat_luong=0.5,
)

# Thêm seed examples
sinh.them_giong_mau(
    instruction="Tóm tắt văn bản về AI",
    output="AI là lĩnh vực khoa học máy tính..."
)
sinh.them_giong_mau(
    instruction="Dịch câu sau sang tiếng Anh",
    output="Hello world"
)

# Sinh 50 mẫu instruction-following data
du_lieu = sinh.sinh(50, loai="instruction", chu_de="công nghệ")

for mau in du_lieu[:3]:
    print(f"Instruction: {mau['instruction']}")
    print(f"Output: {mau['output']}")
    print(f"Điểm: {mau['diem']}")
    print()
```

### Các loại data supported

| loai | Mô tả |
|---|---|
| `instruction` | Instruction-following (Alpaca format) |
| `qa` | Question-Answer pairs |
| `classification` | Text + label pairs |
| `completion` | Text completion |

---

## Test-Time Training

Adapt model weights ngay tại inference time.

```python
from vietnamese_ai import TestTimeTraining

ttt = TestTimeTraining(
    model=your_model,
    che_do="entropy_minimization",
    toc_do_hoc=0.001,
    so_buoc_mac_dinh=10,
)

# Lưu trọng số gốc
ttt.luu_trong_so_goc()

# Adapt trên unlabeled test data
ket_qua = ttt.thich_ung(X_test)
print(f"Loss: {ket_qua['loss_dau']:.4f} → {ket_qua['loss_cuoi']:.4f}")

# Dự đoán với model đã adapt
du_doan = ttt.du_doan(X_test)

# Phục hồi trọng số gốc
ttt.phuc_hoi_trong_so()
```

### Các chiến lược TTT

| che_do | Mô tả | Khi nào dùng |
|---|---|---|
| `entropy_minimization` | Giảm entropy predictions | Classification |
| `contrastive` | Consistency với augmented data | Robust predictions |
| `masked_prediction` | Predict masked features | NLP / tabular |

---

## Kết hợp tất cả

```python
from vietnamese_ai import (
    SelfRefine, SelfConsistency, AdaptiveLoRA,
    SinhDuLieuTuDong, TestTimeTraining
)

# 1. Sinh dữ liệu
sinh = SinhDuLieuTuDong(ham_sinh=generate)
sinh.them_giong_mau("Câu hỏi mẫu", "Trả lời mẫu")
du_lieu = sinh.sinh(100, loai="instruction")

# 2. Fine-tune với dữ liệu sinh
# model.huan_luyen(du_lieu)

# 3. Adapt tại test time
ttt = TestTimeTraining(model, che_do="entropy_minimization")
ttt.thich_ung(X_test)

# 4. Chọn adapter phù hợp
adaptive = AdaptiveLoRA(che_do="keyword")
adaptive.dang_ky_adapter("task_a", lora_a, keywords=["keyword1"])
adapter = adaptive.chon_adapter(input_text)

# 5. Self-consistency cho answer
sc = SelfConsistency(ham_sinh=generate, so_luong=5)
ket_qua = sc.chay(cau_hoi)

# 6. Self-refine để cải thiện
refine = SelfRefine(ham_sinh=generate, so_vong_toi_da=3)
final = refine.chay(ket_qua["dap_an"])
```
