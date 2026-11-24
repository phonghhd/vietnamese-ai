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

## Tinh chỉnh GPU Từ Xa Đa Nhà Cung Cấp (`vietnamese_ai.fine_tuning.remote_gpu`)

Vòng lặp **Local-First ↔ Cloud GPU Burst Flywheel** hỗ trợ cắm nóng nhiều nhà cung cấp (`modal`, `ssh`, `local` hoặc tùy chỉnh), đi qua lớp lọc PII (`DataSanitizer`) + chống đầu độc (`PhongThuDuLieu`), trần ngân sách (`MayTinhTien`), xác minh SHA-256 + ký số HMAC-SHA256 (`LuuTai`) và cổng kiểm định `Eval Gate` trước khi nạp vào `DongCoSuyLuanEdge`:

```python
class CauHinhTinhChinhXa:
    mo_hinh: str = "vgpt2"
    nhiem_vu: str = "sft"
    loai_gpu: str = "T4"
    so_vong: int = 3
    rank_lora: int = 8
    quantization: str = "q8_0"
    ngan_sach_toi_da_usd: float = 2.0
    cho_phep_gui_cloud: bool = False

class NhaCungCapModal(NhaCungCapGPUXa): ...
class NhaCungCapSSH(NhaCungCapGPUXa): ...
class NhaCungCapCucBo(NhaCungCapGPUXa): ...

class BoDieuPhoiTinhChinhXa:
    def dang_ky_nha_cung_cap(ten: str, provider: NhaCungCapGPUXa) -> None
    def danh_sach_nha_cung_cap() -> Dict[str, Dict[str, Any]]
    def chuan_bi_du_lieu_an_toan(nguon_du_lieu, cau_hinh, thu_muc_dich, ma_tac_vu)
    def kiem_dinh_eval_gate(ket_qua, bo_cau_hoi_kiem_thu=None, diem_toi_thieu=0.5, ham_danh_gia=None) -> Dict[str, Any]
    def chay_vong_lap_flywheel(
        nguon_du_lieu,
        ten_nha_cung_cap: str = "modal",
        cau_hinh: Optional[CauHinhTinhChinhXa] = None,
        chay_that: bool = False,
        chay_eval_gate: bool = True,
        quan_ly_mo_hinh: Optional[Any] = None,
    ) -> KetQuaTinhChinhXa
```

Cài đặt phụ thuộc tùy chọn cho Modal: `pip install "vietnamese-ai[cloud-gpu]"`.
CLI tương ứng: `vai train-cloud --data data.jsonl --provider modal --gpu T4 --allow-cloud --apply`.

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
