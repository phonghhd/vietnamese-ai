# SFT, DPO & RLHF

## Tổng quan

Bộ công cụ huấn luyện LLM nâng cao:

- **SFT** - Supervised Fine-Tuning (tinh chỉnh có giám sát)
- **DPO** - Direct Preference Optimization (tối ưu hóa trực tiếp từ preference)
- **RLHF** - Reinforcement Learning from Human Feedback (pipeline đầy đủ)

---

## SFT (Supervised Fine-Tuning)

Huấn luyện mô hình trên dữ liệu instruction-input-output.

### Cơ bản

```python
from vietnamese_ai import SFTTrainer

trainer = SFTTrainer(
    so_vong=3,
    kich_thuoc_batch=4,
    toc_do_hoc=2e-5,
    gradient_accumulation=4,
)

# Dữ liệu dạng list of dicts
du_lieu = [
    {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
    {"input_ids": [6, 7, 8], "labels": [-100, 7, 8]},
]

ket_qua = trainer.huan_luyen(model, du_lieu, du_lieu_val=val_data)
print(f"Train loss: {ket_qua['train_loss_min']:.4f}")
print(f"Eval loss: {ket_qua['eval_loss_min']:.4f}")
```

### Tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `so_vong` | `3` | Số epochs |
| `kich_thuoc_batch` | `4` | Batch size |
| `toc_do_hoc` | `2e-5` | Learning rate |
| `gradient_accumulation` | `4` | Gradient accumulation steps |
| `gradient_clip` | `1.0` | Gradient clipping |
| `warmup_ratio` | `0.1` | Tỷ lệ warmup steps |

---

## DPO (Direct Preference Optimization)

Tối ưu hóa trực tiếp từ preference pairs mà không cần reward model.

### Lý thuyết

DPO sử dụng Bradley-Terry loss:

```
L = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x)) - log π(y_l|x)/π_ref(y_l|x)))]
```

Trong đó:

- `π` là policy model
- `π_ref` là reference model (frozen)
- `y_w` là response tốt (chosen)
- `y_l` là response xấu (rejected)
- `β` là KL regularization coefficient

### Sử dụng

```python
from vietnamese_ai import DPOTrainer

trainer = DPOTrainer(
    so_vong=1,
    beta=0.1,
    toc_do_hoc=5e-7,
    label_smoothing=0.0,
)

# Dữ liệu preference
preference_data = [
    {
        "prompt": "Giải thích AI là gì?",
        "chosen": "AI là trí tuệ nhân tạo, giúp máy tính học và ra quyết định.",
        "rejected": "AI là robot biết suy nghĩ như con người.",
    },
    {
        "prompt": "Viết code Python in Hello World",
        "chosen": "print('Hello World')",
        "rejected": "Bạn cần cài Python trước, sau đó mở terminal...",
    },
]

ket_qua = trainer.huan_luyen(model, ref_model, preference_data)

print(f"Loss: {ket_qua['train_loss_min']:.4f}")
print(f"Reward margin: {ket_qua['final_reward_margin']:.4f}")
```

### Giải thích kết quả

- **`train_loss`**: Giảm dần = model học tốt hơn
- **`chosen_rewards`**: Reward cho response tốt (nên tăng)
- **`rejected_rewards`**: Reward cho response xấu (nên giảm)
- **`reward_margin`**: Chosen - Rejected (nên dương và tăng)

---

## Reward Model

Huấn luyện mô hình đánh giá chất lượng response.

```python
from vietnamese_ai import RewardModel

rm = RewardModel(toc_do_hoc=1e-5)

# Train từ preference pairs
preference_data = [
    {"chosen": "Trả lời chính xác và đầy đủ", "rejected": "Trả lời sai"},
    {"chosen": "Giải thích rõ ràng", "rejected": "Mơ hồ không rõ"},
]

ket_qua = rm.huan_luyen(model, preference_data, so_vong=3)
print(f"Accuracy: {ket_qua['final_accuracy']:.4f}")

# Đánh giá scores
scores = rm.diem_danh_gia(model, ["văn bản 1", "văn bản 2"])
for s in scores:
    print(f"{s['van_ban']}: score={s['score']:.4f}")
```

---

## RLHF Pipeline

Pipeline đầy đủ: SFT → Reward Model → DPO.

### Cách 1: Chạy từng bước

```python
from vietnamese_ai import RLHFPipeline

pipeline = RLHFPipeline()

# Bước 1: SFT
sft_data = [{"input_ids": [...], "labels": [...]}]
pipeline.sft(model, sft_data)

# Bước 2: Train Reward Model
preference_data = [{"chosen": "...", "rejected": "..."}]
pipeline.train_reward_model(reward_model, preference_data)

# Bước 3: DPO
dpo_data = [{"prompt": "...", "chosen": "...", "rejected": "..."}]
pipeline.rlhf(model, ref_model, dpo_data)
```

### Cách 2: Chạy toàn bộ

```python
pipeline = RLHFPipeline()

ket_qua = pipeline.chay_day_du(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    sft_data=sft_data,
    preference_data=preference_data,
)

# Xem kết quả
print(pipeline.thong_ke())
```

### Custom config

```python
pipeline = RLHFPipeline(
    sft_config={"so_vong": 3, "toc_do_hoc": 2e-5},
    dpo_config={"beta": 0.2, "so_vong": 1},
    reward_config={"toc_do_hoc": 1e-5},
)
```

---

## Quy trình RLHF hoàn chỉnh

```
┌─────────────────┐
│  1. SFT Data     │  (instruction, response)
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. SFT Training │  Fine-tune model trên SFT data
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Preference   │  (prompt, chosen, rejected)
│     Data         │
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Reward Model │  Train RM từ preference pairs
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. DPO/RLHF     │  Tối ưu policy với RM signal
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Final Model  │  Model aligned với human preferences
└─────────────────┘
```
