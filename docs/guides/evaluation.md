# Evaluation & Benchmarks

## Tổng quan

Đánh giá Vietnamese LLM với:

- **ModelConfig** - Cấu hình model từ 10M đến 7B params
- **LMEvalHarness** - Multi-task evaluation framework
- **BenchmarkRunner** - Vietnamese-specific benchmarks

---

## ModelConfig

Cấu hình model presets cho Vietnamese LLM.

### Danh sách presets

```python
from vietnamese_ai import ModelConfig

# Xem tất cả presets
print(ModelConfig.danh_sach_presets())
```

| Preset | Params | d_model | Blocks | Heads | Vocab |
|---|---|---|---|---|---|
| `vnlm-tiny` | ~10M | 64 | 2 | 2 | 1,000 |
| `vnlm-small` | ~125M | 768 | 12 | 12 | 32,000 |
| `vnlm-medium` | ~350M | 1,024 | 24 | 16 | 32,000 |
| `vnlm-large` | ~1.3B | 2,048 | 24 | 32 | 32,000 |
| `vnlm-xl` | ~2.7B | 2,560 | 32 | 32 | 32,000 |
| `vnlm-7b` | ~6.8B | 4,096 | 32 | 32 | 32,000 |

### Sử dụng

```python
from vietnamese_ai import ModelConfig, GPTModel

# Load preset
config = ModelConfig.from_preset("vnlm-small")
print(config.so_tham_so_str)  # "125M"

# Tạo model từ config
model = GPTModel(**config.to_dict())

# Custom config
config = ModelConfig(
    d_model=512,
    so_dau=8,
    d_ff=2048,
    so_block=8,
    so_tu_vung=10000,
)
```

---

## LMEvalHarness

Framework đánh giá LLM tương thích với phong cách lm-eval-harness.

### Tasks có sẵn

```python
from vietnamese_ai import LMEvalHarness

harness = LMEvalHarness()
print(harness.danh_sach_tasks())
```

| Task | Loại | Metrics |
|---|---|---|
| `vie_perplexity` | Perplexity | perplexity |
| `vie_sentiment` | Classification | accuracy, f1 |
| `vie_text_generation` | Generation | bleu, rouge |
| `vie_qa` | Q&A | accuracy, f1 |
| `vie_cloze` | Cloze test | accuracy |

### Đánh giá model

```python
from vietnamese_ai import VietnameseLLM, LMEvalHarness

# Train model
llm = VietnameseLLM(bac=3, toi_thieu_dem=1)
llm.huan_luyen(cac_van_ban)

# Đánh giá
harness = LMEvalHarness()
ket_qua = harness.danh_gia(
    llm,
    cac_task=["vie_perplexity", "vie_sentiment"],
    so_shot=0,
)

print(f"Số tasks: {ket_qua['so_tasks']}")
print(f"Thời gian: {ket_qua['tong_thoi_gian']:.1f}s")

# Xem chi tiết
for task, result in ket_qua["ket_qua"].items():
    print(f"  {task}: {result}")
```

### Đăng ký custom task

```python
from vietnamese_ai.llm.lm_eval import EvalTask

task = EvalTask(
    ten="my_custom_task",
    loai="classification",
    du_lieu=[
        {"text": "Sản phẩm tốt", "label": "positive"},
        {"text": "Dịch vụ kém", "label": "negative"},
    ],
    metrics=["accuracy", "f1"],
    mo_ta="Custom sentiment task",
)

harness.dang_ky_task(task)
ket_qua = harness.danh_gia(model, ["my_custom_task"])
```

### Báo cáo

```python
print(harness.bao_cao())
```

```
============================================================
LM EVALUATION REPORT
============================================================

[vie_perplexity]
  perplexity: 45.2300
  so_mau: 50

[vie_sentiment]
  accuracy: 0.7800
  f1: 0.7410
  so_mau: 20

============================================================
```

---

## BenchmarkRunner

Chạy benchmarks toàn diện cho Vietnamese LLM.

### Chạy tất cả benchmarks

```python
from vietnamese_ai import BenchmarkRunner, VietnameseLLM

llm = VietnameseLLM(bac=3, toi_thieu_dem=1)
llm.huan_luyen(cac_van_ban)

runner = BenchmarkRunner()
ket_qua = runner.chay(llm)

print(runner.bao_cao())
```

### Chạy benchmark cụ thể

```python
ket_qua = runner.chay(
    llm,
    benchmarks=["perplexity", "generation", "sentiment", "speed", "qa"],
)
```

### Các benchmark

| Benchmark | Mô tả | Metrics |
|---|---|---|
| `perplexity` | Perplexity trên corpus tiếng Việt | mean, std, min, max |
| `generation` | Chất lượng sinh văn bản | do_dai_trung_binh, thoi_gian |
| `sentiment` | Phân loại cảm xúc | accuracy |
| `speed` | Tốc độ inference | latency p50/p95/p99 |
| `qa` | Hỏi đáp | accuracy |

### Báo cáo mẫu

```
============================================================
VIETNAMESE LLM BENCHMARK REPORT
============================================================

📊 Perplexity:
  Mean: 42.1500
  Std:  12.3400
  Min:  18.5000

📝 Generation:
  Avg length: 8.5 words

😊 Sentiment:
  Accuracy: 0.7500

⚡ Speed:
  Mean latency: 2.35 ms
  P95 latency:  3.12 ms

❓ Q&A:
  Accuracy: 0.4000

⏱️  Total time: 5.2s
============================================================
```

---

## Quy trình đánh giá hoàn chỉnh

```python
from vietnamese_ai import (
    ModelConfig, GPTModel, PreTrainer, TextDataset,
    VietnameseLLM, LMEvalHarness, BenchmarkRunner,
)

# 1. Tạo model từ config
config = ModelConfig.from_preset("vnlm-tiny")
model = GPTModel(**config.to_dict())

# 2. Pre-train
dataset = TextDataset(do_dai_window=128)
dataset.tai_corpus(corpus, vocab_size=config.so_tu_vung)
trainer = PreTrainer(so_vong=5)
trainer.huan_luyen(model, dataset)

# 3. Eval với LMEvalHarness
harness = LMEvalHarness()
eval_results = harness.danh_gia(model, ["vie_perplexity"])
print(harness.bao_cao())

# 4. Benchmark
runner = BenchmarkRunner()
bench_results = runner.chay(model, benchmarks=["perplexity", "speed"])
print(runner.bao_cao())
```
