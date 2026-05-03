# GPT Pre-training

## Tổng quan

Pre-training GPT-style decoder-only transformer từ đầu cho tiếng Việt.

Vietnamese AI cung cấp:

- **GPTModel** - Kiến trúc GPT (causal attention, GELU, pre-norm)
- **PreTrainer** - Training loop với warmup + cosine decay
- **TextDataset** - Text corpus processing, sliding window, batch iteration

---

## GPTModel

Kiến trúc GPT decoder-only transformer (NumPy).

### Tạo model

```python
from vietnamese_ai import GPTModel

model = GPTModel(
    d_model=128,       # Hidden dimension
    so_dau=8,          # Số attention heads
    d_ff=512,          # Feed-forward dimension
    so_block=6,        # Số transformer blocks
    so_tu_vung=10000,  # Vocabulary size
    do_dai_toi_da=1024, # Max sequence length
    dropout=0.1,
)

print(model.thong_ke())
```

### Kiến trúc

```
Input Token IDs
      ↓
┌─────────────────┐
│  Token Embedding │  (so_tu_vung × d_model)
│  + Positional    │  (sinusoidal)
└────────┬────────┘
         ▼
┌─────────────────┐
│  GPT Block × N   │
│  ┌─────────────┐ │
│  │ LayerNorm   │ │
│  │ Causal Attn │ │  (masked self-attention)
│  │ + Residual  │ │
│  ├─────────────┤ │
│  │ LayerNorm   │ │
│  │ FFN (GELU)  │ │
│  │ + Residual  │ │
│  └─────────────┘ │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Final LayerNorm │
│  LM Head (tied)  │  (weight tying với embedding)
└─────────────────┘
```

### Forward pass

```python
import numpy as np

input_ids = np.array([[1, 2, 3, 4, 5]])  # (batch, seq_len)
logits = model.tien(input_ids)            # (batch, seq_len, vocab)
```

### Text generation

```python
# Sinh text tự hồi quy
input_ids = np.array([[1, 2, 3]])
generated = model.sinh_tiep(
    input_ids,
    so_token=50,     # Số token cần sinh
    nhiet_do=0.8,    # Temperature
    top_k=50,        # Top-k sampling
    top_p=0.9,       # Nucleus sampling
)
```

### Tính loss

```python
input_ids = np.array([[1, 2, 3, 4]])
targets = np.array([[2, 3, 4, 5]])
loss = model.tinh_loss(input_ids, targets)
```

---

## TextDataset

Chuẩn bị dữ liệu text cho pre-training.

### Tạo dataset

```python
from vietnamese_ai import TextDataset

dataset = TextDataset(
    do_dai_window=512,  # Độ dài mỗi chunk
    buoc_nhay=256,      # Bước nhảy (sliding window)
)

# Load corpus
corpus = [
    "Trí tuệ nhân tạo đang thay đổi cách con người làm việc.",
    "Học máy là một nhánh quan trọng của AI.",
    # ... thêm nhiều văn bản
]

ket_qua = dataset.tai_corpus(corpus, vocab_size=5000)
print(f"Tokens: {ket_qua['so_tokens']}")
print(f"Chunks: {ket_qua['so_chunks']}")
print(f"Vocab: {ket_qua['vocab_size']}")
```

### Chia dữ liệu

```python
dataset.chia_du_lieu(ty_le_val=0.1)
```

### Batch iteration

```python
for input_ids, targets in dataset.iter_batches(batch_size=32, che_do="train"):
    # input_ids: (batch, window-1)
    # targets:   (batch, window-1)
    loss = model.tinh_loss(input_ids, targets)
```

### Encode/Decode

```python
ids = dataset.ma_hoa("học máy")    # [45, 12, 67, ...]
text = dataset.giai_ma(ids)         # "học máy"
```

---

## PreTrainer

Training loop cho pre-training.

### Cơ bản

```python
from vietnamese_ai import PreTrainer, GPTModel, TextDataset

# 1. Tạo model
model = GPTModel(d_model=128, so_dau=8, so_block=4, so_tu_vung=5000)

# 2. Chuẩn bị dataset
dataset = TextDataset(do_dai_window=256, buoc_nhay=128)
dataset.tai_corpus(cac_van_ban, vocab_size=5000)

# 3. Train
trainer = PreTrainer(
    so_vong=10,
    kich_thuoc_batch=32,
    toc_do_hoc=3e-4,
    warmup_steps=100,
)

ket_qua = trainer.huan_luyen(model, dataset)
```

### Kết quả

```python
print(f"Thời gian: {ket_qua['tong_thoi_gian']:.1f}s")
print(f"Train loss: {ket_qua['train_loss_min']:.4f}")
print(f"Eval loss: {ket_qua['eval_loss_min']:.4f}")
print(f"Perplexity: {ket_qua['final_perplexity']:.2f}")
```

### Callback

```python
def on_step(step, loss):
    if step % 50 == 0:
        print(f"Step {step}: loss={loss:.4f}")

trainer.huan_luyen(model, dataset, callback=on_step)
```

---

## Sử dụng với ModelConfig

Kết hợp với ModelConfig để tạo model đúng kích thước:

```python
from vietnamese_ai import ModelConfig, GPTModel, PreTrainer, TextDataset

# Sử dụng preset
config = ModelConfig.from_preset("vnlm-tiny")  # 10M params
model = GPTModel(**config.to_dict())

# Hoặc custom
config = ModelConfig(d_model=256, so_dau=8, d_ff=1024, so_block=6)
model = GPTModel(**config.to_dict())

# Pre-train
dataset = TextDataset(do_dai_window=config.do_dai_toi_da)
dataset.tai_corpus(corpus, vocab_size=config.so_tu_vung)

trainer = PreTrainer(so_vong=10)
ket_qua = trainer.huan_luyen(model, dataset)

# Sinh text
generated = model.sinh_tiep(
    np.array([[dataset.ma_hoa("học máy")[0]]]),
    so_token=100,
    nhiet_do=0.8,
)
print(dataset.giai_ma(generated[0].tolist()))
```

---

## Lưu ý khi pre-training

### Dữ liệu

- Cần corpus lớn (ít nhất hàng triệu tokens)
- Đa dạng thể loại: tin tức, sách, Wikipedia, forum
- Làm sạch dữ liệu: loại bỏ HTML, ký tự đặc biệt

### Hyperparameters

| Parameter | Khuyến nghị | Mô tả |
|---|---|---|
| `d_model` | 128-4096 |越大越好 nhưng cần更多 VRAM |
| `so_block` | 4-32 | Nhiều blocks = mạnh hơn |
| `toc_do_hoc` | 3e-4 → 1e-5 | Warmup rồi decay |
| `kich_thuoc_batch` | 32-256 |越大越好 |
| `do_dai_window` | 256-2048 | Tùy độ dài văn bản |

### GPU Memory ước tính

| Config | Params | GPU RAM (FP16) |
|---|---|---|
| vnlm-tiny | 10M | ~100 MB |
| vnlm-small | 125M | ~500 MB |
| vnlm-medium | 350M | ~1.5 GB |
| vnlm-large | 1.3B | ~5 GB |
| vnlm-7b | 6.8B | ~14 GB |
