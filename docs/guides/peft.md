# PEFT & Instruction Tuning

## Tổng quan

PEFT (Parameter-Efficient Fine-Tuning) cho phép fine-tune LLM với ít tham số hơn, tiết kiệm bộ nhớ và thời gian huấn luyện.

Vietnamese AI hỗ trợ:

- **LoRA** - Low-Rank Adaptation
- **QLoRA** - Quantized LoRA (4-bit/8-bit)
- **Instruction Tuning** - Huấn luyện trên dữ liệu Alpaca/ShareGPT

---

## PEFTConfig

Cấu hình unified cho tất cả phương pháp PEFT.

```python
from vietnamese_ai import PEFTConfig

# Cách 1: Constructor
config = PEFTConfig(
    phuong_phap="lora",
    rank=16,
    alpha=16.0,
    dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Cách 2: Preset
config = PEFTConfig.lora(rank=16, alpha=16.0)
config = PEFTConfig.qlora(rank=16, bits=4)
```

### Các tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `phuong_phap` | `"lora"` | Phương pháp: `lora`, `qlora`, `prefix_tuning`, `prompt_tuning` |
| `rank` | `16` | Rank của LoRA (8, 16, 32, 64) |
| `alpha` | `16.0` | Scaling factor (`alpha/rank`) |
| `dropout` | `0.05` | Dropout rate |
| `target_modules` | Auto | Các Linear layers áp dụng LoRA |
| `bits` | `4` | Số bit quantization (4, 8, 16) |

---

## LoRAPeft

Áp dụng LoRA lên PyTorch model một cách tự động.

### Cơ bản

```python
import torch.nn as nn
from vietnamese_ai import LoRAPeft, PEFTConfig

# Tạo model gốc
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
)

# Cấu hình LoRA
config = PEFTConfig.lora(rank=16, alpha=32.0)

# Áp dụng LoRA
peft = LoRAPeft(config)
model = peft.ap_dung(model)

# Chỉ train LoRA params
peft.chi_trainable(model)

# Thống kê
print(peft.thong_ke(model))
```

### Merge weights

Sau khi train xong, gộp LoRA weights vào model gốc để inference nhanh hơn:

```python
peft.gop_trong_so(model)
```

### Save/Load adapter

Lưu chỉ LoRA weights (rất nhỏ so với model gốc):

```python
# Lưu
peft.luu(model, "lora_adapter.json")

# Tải
peft = LoRAPeft.tai("lora_adapter.json", model)
```

---

## Instruction Tuning

Huấn luyện mô hình trên dữ liệu instruction-following.

### Chuẩn bị dữ liệu

```python
from vietnamese_ai import InstructionTuningTrainer
from vietnamese_ai.fine_tuning.dataset import InstructionDataset

# Format Alpaca
dataset = InstructionDataset(che_do="alpaca")
dataset.tai_file("data/alpaca_vi.json")

# Hoặc từ list
dataset.tai_tu_list([
    {
        "instruction": "Tóm tắt văn bản sau",
        "input": "Trí tuệ nhân tạo đang thay đổi...",
        "output": "AI đang thay đổi nhiều lĩnh vực.",
    },
    {
        "instruction": "Dịch sang tiếng Anh",
        "input": "Học máy rất thú vị",
        "output": "Machine learning is very interesting",
    },
])

# Chia train/val
dataset.chia_du_lieu(ty_le_val=0.1)
```

### Format ShareGPT

```python
dataset = InstructionDataset(che_do="sharegpt")
dataset.tai_tu_list([
    {
        "conversations": [
            {"from": "human", "value": "AI là gì?"},
            {"from": "gpt", "value": "AI là trí tuệ nhân tạo..."},
        ]
    }
])
```

### Huấn luyện

```python
trainer = InstructionTuningTrainer(
    so_vong=3,
    kich_thuoc_batch=4,
    toc_do_hoc=2e-5,
    gradient_accumulation=4,
    max_seq_length=512,
)

ket_qua = trainer.huan_luyen(model, tokenizer, dataset)
print(f"Loss: {ket_qua['train_loss_min']:.4f}")
print(f"Thời gian: {ket_qua['tong_thoi_gian']:.1f}s")
```

### Với Callback

```python
def on_step(step, loss):
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}")

trainer.huan_luyen(model, tokenizer, dataset, callback=on_step)
```

---

## Kết hợp LoRA + Instruction Tuning

```python
from vietnamese_ai import (
    PEFTConfig, LoRAPeft, InstructionTuningTrainer,
    InstructionDataset,
)

# 1. Cấu hình LoRA
config = PEFTConfig.lora(rank=16, alpha=32.0)

# 2. Áp dụng LoRA lên model
peft = LoRAPeft(config)
model = peft.ap_dung(model)
peft.chi_trainable(model)

# 3. Chuẩn bị dataset
dataset = InstructionDataset(che_do="alpaca")
dataset.tai_file("data/alpaca_vi.json")

# 4. Huấn luyện
trainer = InstructionTuningTrainer(so_vong=3)
ket_qua = trainer.huan_luyen(model, tokenizer, dataset)

# 5. Lưu adapter
peft.luu(model, "outputs/lora_adapter.json")
```

---

## So sánh LoRA vs QLoRA

| Đặc điểm | LoRA | QLoRA |
|---|---|---|
| Base weights | FP16/BF16 | 4-bit NF4 |
| LoRA weights | FP16/BF16 | FP16/BF16 |
| VRAM | Trung bình | Thấp (tiết kiệm ~50%) |
| Tốc độ | Nhanh | Chậm hơn một chút |
| Chất lượng | Tốt | Tương đương |

```python
# QLoRA - tiết kiệm VRAM hơn
config = PEFTConfig.qlora(rank=16, bits=4)
```
