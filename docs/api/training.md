# Training API Reference

## SFTTrainer

```python
class SFTTrainer:
    """Supervised Fine-Tuning Trainer."""

    def __init__(
        so_vong=3,
        kich_thuoc_batch=4,
        toc_do_hoc=2e-5,
        gradient_accumulation=4,
        gradient_clip=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_seq_length=512,
        logging_steps=10,
        seed=42,
    )

    def huan_luyen(model, du_lieu_train, du_lieu_val=None, tokenizer=None, callback=None) -> dict
    def lay_lich_su() -> dict
    def thong_ke() -> dict
```

---

## DPOTrainer

```python
class DPOTrainer:
    """Direct Preference Optimization Trainer."""

    def __init__(
        so_vong=1,
        kich_thuoc_batch=2,
        toc_do_hoc=5e-7,
        beta=0.1,                 # KL regularization
        label_smoothing=0.0,
        gradient_accumulation=4,
        gradient_clip=1.0,
        max_seq_length=512,
        logging_steps=10,
        seed=42,
    )

    def huan_luyen(model, ref_model, preference_data, tokenizer=None, callback=None) -> dict
    def lay_lich_su() -> dict
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `beta` | float | `0.1` | KL regularization coefficient |
| `label_smoothing` | float | `0.0` | Label smoothing (0-0.5) |
| `so_vong` | int | `1` | Số epochs |

### Preference Data Format

```python
[
    {
        "prompt": "câu hỏi hoặc context",
        "chosen": "response tốt",
        "rejected": "response xấu",
    },
]
```

### Returns

```python
{
    "tong_thoi_gian": float,
    "so_epoch": int,
    "global_step": int,
    "train_loss_min": float,
    "final_reward_margin": float,  # chosen_reward - rejected_reward
    "history": {
        "train_loss": list,
        "chosen_rewards": list,
        "rejected_rewards": list,
        "reward_margin": list,
    },
}
```

---

## RewardModel

```python
class RewardModel:
    """Mô hình phần thưởng cho RLHF."""

    def __init__(toc_do_hoc=1e-5, seed=42)

    def huan_luyen(model, preference_data, so_vong=1, callback=None) -> dict
    def diem_danh_gia(model, cac_van_ban) -> list
    def thong_ke() -> dict
```

### `diem_danh_gia` Returns

```python
[
    {
        "van_ban": str,
        "score": float,              # Raw score
        "score_normalized": float,   # Z-score normalized
    },
]
```

---

## RLHFPipeline

```python
class RLHFPipeline:
    """Complete RLHF Pipeline."""

    def __init__(sft_config=None, dpo_config=None, reward_config=None)

    def sft(model, du_lieu, du_lieu_val=None) -> dict
    def train_reward_model(reward_model, preference_data, so_vong=1) -> dict
    def rlhf(model, ref_model, preference_data) -> dict
    def chay_day_du(model, ref_model, reward_model, sft_data, preference_data) -> dict
    def lay_ket_qua() -> dict
    def thong_ke() -> dict
```

### `thong_ke()` Returns

```python
{
    "sft_done": bool,
    "reward_done": bool,
    "rlhf_done": bool,
    "sft_trainer": dict,
    "dpo_trainer": dict,
    "reward_model": dict,
}
```
