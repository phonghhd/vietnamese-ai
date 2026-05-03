# PEFT API Reference

## PEFTConfig

```python
class PEFTConfig:
    """Cấu hình cho các phương pháp PEFT."""

    def __init__(
        phuong_phap="lora",     # "lora", "qlora", "prefix_tuning", "prompt_tuning"
        rank=16,                 # LoRA rank
        alpha=16.0,              # Scaling factor
        dropout=0.05,            # Dropout rate
        target_modules=None,     # List[str] - target layers
        bits=4,                  # Quantization bits (4, 8, 16)
        bias="none",             # "none", "all", "lora_only"
        task_type="causal_lm",   # Task type
        fan_in_fan_out=False,
        init_weights="gaussian",
    )

    @property
    def scaling(self) -> float           # alpha / rank
    def to_dict(self) -> dict
    def from_dict(data) -> PEFTConfig    # classmethod
    def lora(rank, alpha, **kwargs) -> PEFTConfig    # classmethod
    def qlora(rank, alpha, bits, **kwargs) -> PEFTConfig  # classmethod
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `phuong_phap` | str | `"lora"` | Phương pháp PEFT |
| `rank` | int | `16` | Rank của LoRA matrices |
| `alpha` | float | `16.0` | Scaling factor |
| `dropout` | float | `0.05` | Dropout rate |
| `target_modules` | list | Auto | Các module áp dụng LoRA |
| `bits` | int | `4` | Số bit quantization |

---

## LoRAPeft

```python
class LoRAPeft:
    """Enhanced LoRA PEFT adapter cho PyTorch models."""

    def __init__(config: PEFTConfig = None)
    def ap_dung(model) -> model              # Áp dụng LoRA lên model
    def gop_trong_so(model) -> model          # Merge LoRA weights
    def chi_trainable(model) -> list          # Freeze non-LoRA params
    def luu(model, duong_dan) -> str          # Lưu adapter
    def tai(duong_dan, model) -> LoRAPeft     # classmethod - Tải adapter
    def thong_ke(model=None) -> dict
```

### Methods

#### `ap_dung(model) -> model`

Áp dụng LoRA lên tất cả Linear layers matching `target_modules`.

```python
peft = LoRAPeft(PEFTConfig.lora(rank=16))
model = peft.ap_dung(model)
```

#### `gop_trong_so(model) -> model`

Gộp LoRA weights vào model gốc (dùng sau khi train).

#### `chi_trainable(model) -> list`

Freeze tất cả parameters trừ LoRA layers. Trả về danh sách trainable param names.

#### `luu(model, duong_dan) -> str`

Lưu chỉ LoRA adapter weights (nhỏ hơn nhiều so với full model).

#### `tai(duong_dan, model) -> LoRAPeft`

Tải LoRA adapter từ file và apply vào model.

---

## LoRAAdapter (NumPy)

```python
class LoRAAdapter:
    """LoRA adapter cho NumPy-based models."""

    def __init__(rank=16, alpha=16.0, dropout=0.0, target_modules=None)
    def them_layer(ten, dau_vao, dau_ra) -> None
    def tien(ten, X) -> np.ndarray
    def gop_trong_so() -> dict
    def so_tham_so() -> int
    def ty_le_trainable(tong_tham_so_goc) -> float
    def luu(duong_dan) -> str
    def tai(duong_dan) -> LoRAAdapter       # classmethod
    def thong_ke() -> dict
```

---

## QLoRAAdapter (NumPy)

```python
class QLoRAAdapter(LoRAAdapter):
    """QLoRA - Quantized LoRA."""

    def __init__(rank=16, alpha=16.0, dropout=0.0, bits=4)
    def quantize_weights(weights) -> dict   # Quantize base weights
    def thong_ke() -> dict
```

---

## InstructionTuningTrainer

```python
class InstructionTuningTrainer:
    """Trainer cho instruction tuning."""

    def __init__(
        so_vong=3,
        kich_thuoc_batch=4,
        toc_do_hoc=2e-5,
        gradient_accumulation=4,
        max_seq_length=512,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_clip=1.0,
        logging_steps=10,
        eval_steps=100,
        save_steps=500,
        output_dir="outputs/instruction",
        seed=42,
    )

    def huan_luyen(model, tokenizer, dataset, eval_dataset=None, callback=None, data_collator=None) -> dict
    def lay_lich_su() -> dict
    def thong_ke() -> dict
```

### Returns

```python
{
    "tong_thoi_gian": float,    # Thời gian huấn luyện (giây)
    "so_epoch": int,            # Số epochs đã train
    "global_step": int,         # Tổng steps
    "train_loss_min": float,    # Loss thấp nhất
    "history": dict,            # Lịch sử train_loss, eval_loss, learning_rate
}
```
