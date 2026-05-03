# Evaluation API Reference

## ModelConfig

```python
class ModelConfig:
    """Cấu hình Vietnamese LLM từ 125M đến 7B."""

    def __init__(
        d_model=768,
        so_dau=12,
        d_ff=3072,
        so_block=12,
        so_tu_vung=32000,
        do_dai_toi_da=2048,
        dropout=0.1,
        ten="custom",
        mo_ta="",
    )

    def from_preset(preset) -> ModelConfig       # classmethod
    def danh_sach_presets() -> dict               # classmethod
    def to_dict() -> dict

    @property
    def so_tham_so(self) -> int
    @property
    def so_tham_so_str(self) -> str               # "125M", "1.3B", ...
```

### Presets

| Preset | d_model | so_dau | so_block | so_tu_vung | Params |
|---|---|---|---|---|---|
| `vnlm-tiny` | 64 | 2 | 2 | 1,000 | ~10M |
| `vnlm-small` | 768 | 12 | 12 | 32,000 | ~125M |
| `vnlm-medium` | 1,024 | 16 | 24 | 32,000 | ~350M |
| `vnlm-large` | 2,048 | 32 | 24 | 32,000 | ~1.3B |
| `vnlm-xl` | 2,560 | 32 | 32 | 32,000 | ~2.7B |
| `vnlm-7b` | 4,096 | 32 | 32 | 32,000 | ~6.8B |

---

## LMEvalHarness

```python
class LMEvalHarness:
    """LM Evaluation Framework."""

    def __init__()

    def dang_ky_task(task: EvalTask) -> None
    def danh_sach_tasks() -> dict
    def danh_gia(model, cac_task=None, so_shot=0, limit=None) -> dict
    def bao_cao() -> str
    def thong_ke() -> dict
```

### `danh_gia()` Returns

```python
{
    "so_tasks": int,
    "so_shot": int,
    "tong_thoi_gian": float,
    "ket_qua": {
        "task_name": {
            "accuracy": float,
            "perplexity": float,
            # ... metrics tùy task type
        },
    },
    "tong_hop": dict,  # Aggregated metrics
}
```

---

## EvalTask

```python
class EvalTask:
    """Định nghĩa một evaluation task."""

    def __init__(
        ten: str,
        loai: str = "text_generation",  # classification, perplexity, qa, cloze
        du_lieu: list = None,           # [{text, label}, ...]
        metrics: list = None,           # ["accuracy", "f1", ...]
        mo_ta: str = "",
    )
    def to_dict() -> dict
```

---

## BenchmarkRunner

```python
class BenchmarkRunner:
    """Benchmark Runner cho Vietnamese LLM."""

    def __init__()

    def chay(model, benchmarks=None, so_shot=0) -> dict
    def bao_cao() -> str
    def thong_ke() -> dict
```

### Benchmarks

| Tên | Mô tả | Metrics |
|---|---|---|
| `perplexity` | Perplexity trên corpus tiếng Việt | perplexity_mean, perplexity_std |
| `generation` | Sinh văn bản | do_dai_trung_binh, thoi_gian |
| `sentiment` | Phân loại cảm xúc | accuracy |
| `speed` | Tốc độ inference | latency_mean_ms, latency_p95_ms |
| `qa` | Hỏi đáp | accuracy |

### `chay()` Returns

```python
{
    "perplexity": {
        "perplexity_mean": float,
        "perplexity_std": float,
        "perplexity_min": float,
        "perplexity_max": float,
        "so_mau": int,
    },
    "generation": {
        "so_prompts": int,
        "do_dai_trung_binh": float,
        "chi_tiet": list,
    },
    "sentiment": {
        "accuracy": float,
        "so_mau": int,
    },
    "speed": {
        "latency_mean_ms": float,
        "latency_p50_ms": float,
        "latency_p95_ms": float,
        "latency_p99_ms": float,
        "so_lan_chay": int,
    },
    "tong_thoi_gian": float,
    "benchmarks": list,
}
```
