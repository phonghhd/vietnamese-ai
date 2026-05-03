# GPT & PreTrainer API Reference

## GPTModel

```python
class GPTModel:
    """GPT-style Decoder-Only Transformer."""

    def __init__(
        d_model=128,          # Hidden dimension
        so_dau=4,             # Số attention heads
        d_ff=512,             # Feed-forward dimension
        so_block=4,           # Số transformer blocks
        so_tu_vung=5000,      # Vocabulary size
        do_dai_toi_da=1024,   # Max sequence length
        dropout=0.1,          # Dropout rate
    )

    def tien(input_ids) -> np.ndarray           # Forward: (batch, seq_len, vocab)
    def sinh_tiep(input_ids, so_token, nhiet_do, top_k, top_p) -> np.ndarray
    def tinh_loss(input_ids, targets) -> float
    def thong_ke() -> dict
```

### Methods

#### `tien(input_ids) -> np.ndarray`

Forward pass. Trả về logits cho next token prediction.

- **input_ids**: `(batch, seq_len)` - Token IDs
- **Returns**: `(batch, seq_len, so_tu_vung)` - Logits

#### `sinh_tiep(input_ids, so_token=50, nhiet_do=1.0, top_k=0, top_p=0.0)`

Sinh token tiếp theo tự hồi quy.

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `input_ids` | np.ndarray | required | Token IDs ban đầu |
| `so_token` | int | `50` | Số token cần sinh |
| `nhiet_do` | float | `1.0` | Temperature (0.1=conservative, 2.0=creative) |
| `top_k` | int | `0` | Top-k sampling (0=disabled) |
| `top_p` | float | `0.0` | Nucleus sampling (0=disabled) |

#### `tinh_loss(input_ids, targets) -> float`

Tính cross-entropy loss.

- **input_ids**: `(batch, seq_len)` - Input tokens
- **targets**: `(batch, seq_len)` - Target tokens

---

## CausalSelfAttention

```python
class CausalSelfAttention:
    """Causal (masked) self-attention cho decoder."""

    def __init__(d_model, so_dau, dropout=0.1)
    def tien(X) -> np.ndarray
```

---

## GPTBlock

```python
class GPTBlock:
    """GPT Transformer Block: CausalAttention + FFN + LayerNorm."""

    def __init__(d_model, so_dau, d_ff, dropout=0.1)
    def tien(X) -> np.ndarray
```

---

## TextDataset

```python
class TextDataset:
    """Dataset cho pre-training."""

    def __init__(do_dai_window=512, buoc_nhay=256, seed=42)

    def tai_corpus(cac_van_ban, vocab_size=5000) -> dict
    def chia_du_lieu(ty_le_val=0.1, seed=None) -> dict
    def iter_batches(batch_size=32, che_do="train") -> generator
    def ma_hoa(text) -> list
    def giai_ma(ids) -> str
    def thong_ke() -> dict

    @property
    def vocab_size(self) -> int
    @property
    def so_chunks(self) -> int
```

### Methods

#### `tai_corpus(cac_van_ban, vocab_size=5000) -> dict`

Load corpus và tokenize.

```python
ket_qua = dataset.tai_corpus(
    ["văn bản 1", "văn bản 2"],
    vocab_size=5000,
)
# Returns: {"so_van_ban": 2, "so_tokens": 150, "so_chunks": 10, "vocab_size": 200}
```

#### `iter_batches(batch_size, che_do) -> generator`

Yield `(input_ids, targets)` batches.

```python
for input_ids, targets in dataset.iter_batches(32):
    # input_ids: (32, window-1)
    # targets:   (32, window-1)
    loss = model.tinh_loss(input_ids, targets)
```

---

## PreTrainer

```python
class PreTrainer:
    """Pre-training trainer cho GPT-style models."""

    def __init__(
        so_vong=10,
        kich_thuoc_batch=32,
        toc_do_hoc=3e-4,
        gradient_accumulation=1,
        warmup_steps=100,
        weight_decay=0.1,
        gradient_clip=1.0,
        logging_steps=10,
        eval_steps=500,
        seed=42,
    )

    def huan_luyen(model, dataset, callback=None) -> dict
    def lay_lich_su() -> dict
    def thong_ke() -> dict
```

### Returns

```python
{
    "tong_thoi_gian": float,
    "so_epoch": int,
    "global_step": int,
    "train_loss_min": float,
    "eval_loss_min": float,
    "final_perplexity": float,
    "history": {
        "train_loss": list,
        "eval_loss": list,
        "perplexity": list,
        "learning_rate": list,
    },
}
```
