# Vietnamese AI Framework

<p align="center">
  <strong>Framework AI thuần tiếng Việt cho Python</strong><br>
  <em>Học máy đơn giản. API tiếng Việt. Production-ready.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-9.0.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-352%2F352-brightgreen.svg" alt="tests">
</p>

---

## Tại sao chọn Vietnamese AI?

| Vấn đề | Giải pháp |
|---|---|
| Framework AI lớn quá phức tạp | API đơn giản, học trong 5 phút |
| Tài liệu toàn tiếng Anh | Tài liệu 100% tiếng Việt |
| Thiếu toolkit NLP tiếng Việt | Tích hợp underthesea, TF-IDF, Word2Vec |
| Không có framework all-in-one | Models + NLP + AutoML + Mobile + FL + SaaS + Studio + LLM + PEFT + RLHF + GPT |

## Tính năng chính

### Machine Learning

- **6 mô hình học máy**: Phân loại, hồi quy, phân cụm, mạng nơ-ron, ensemble
- **NLP tiếng Việt**: Tách từ, sentiment, Word2Vec, FastText, PhoBERT
- **AutoML + NAS**: Tự động chọn mô hình và kiến trúc tốt nhất

### LLM & Fine-tuning

- **PEFT**: LoRA, QLoRA, Instruction Tuning (Alpaca/ShareGPT)
- **SFT/DPO/RLHF**: Supervised Fine-Tuning, Direct Preference Optimization, Full RLHF Pipeline
- **GPT Pre-training**: Decoder-only transformer, pre-train từ đầu
- **Vietnamese LLM**: N-gram LM, text generation, templates

### Evaluation

- **Model Configs**: Presets từ 10M đến 7B parameters
- **LM Eval Harness**: Multi-task evaluation framework
- **Benchmarks**: Perplexity, generation, sentiment, speed, QA

### Production

- **Mobile/Edge**: Xuất TFLite, CoreML, ONNX Mobile + quantization
- **Federated Learning**: FedAvg, Differential Privacy
- **Real-time Pipeline**: Message Queue, Feature Store, latency tracking
- **Cloud SaaS**: Multi-tenant workspace, API keys, quota, deploy
- **No-code Studio**: Kéo thả xây dựng ML pipeline
- **CLI + Docker**: Deploy production-ready

## Bắt đầu nhanh

```bash
pip install vietnamese-ai
```

```python
from vietnamese_ai import PhanLoai, XuLySo, DuLieuMau

X, y = DuLieuMau.phan_loai_don_gian(so_mau=400)
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)
print(pl.bao_cao(X_test, y_test))
```

## Tài liệu

### Bắt đầu

- [Cài đặt](getting-started/installation.md) - Hướng dẫn cài đặt
- [Sử dụng nhanh](getting-started/quickstart.md) - Ví dụ nhanh

### Hướng dẫn

- [Phân loại](guides/classification.md) - Phân loại dữ liệu
- [Hồi quy](guides/regression.md) - Dự đoán giá trị
- [Xử lý văn bản](guides/text-processing.md) - NLP tiếng Việt
- [AutoML](guides/automl.md) - Tự động hóa ML
- [Pipeline](guides/pipeline.md) - Xây dựng pipeline
- [PEFT & Instruction Tuning](guides/peft.md) - Fine-tune hiệu quả
- [SFT, DPO & RLHF](guides/sft-dpo-rlhf.md) - Training nâng cao
- [GPT Pre-training](guides/gpt-pretraining.md) - Pre-train từ đầu
- [Evaluation & Benchmarks](guides/evaluation.md) - Đánh giá mô hình

### API Reference

- [Models](api/models.md) - Các mô hình ML
- [PEFT](api/peft.md) - PEFTConfig, LoRAPeft
- [Training](api/training.md) - SFT, DPO, RLHF
- [GPT](api/gpt.md) - GPTModel, PreTrainer
- [Evaluation](api/evaluation.md) - ModelConfig, LMEval, Benchmark

---

*Được xây dựng với ❤️ cho cộng đồng AI Việt Nam*
