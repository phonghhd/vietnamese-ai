# Vietnamese AI Framework

<p align="center">
  <strong>Framework AI thuần tiếng Việt cho Python</strong><br>
  <em>Học máy đơn giản. API tiếng Việt. Production-ready.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-230%2F230-brightgreen.svg" alt="tests">
</p>

---

## Tại sao chọn Vietnamese AI?

| Vấn đề | Giải pháp |
|---|---|
| Framework AI lớn quá phức tạp | API đơn giản, học trong 5 phút |
| Tài liệu toàn tiếng Anh | Tài liệu 100% tiếng Việt |
| Thiếu toolkit NLP tiếng Việt | Tích hợp underthesea, TF-IDF, Word2Vec |
| Không có framework all-in-one | Models + NLP + AutoML + Mobile + FL + SaaS + Studio + LLM |

## Tính năng chính

- **6 mô hình học máy**: Phân loại, hồi quy, phân cụm, mạng nơ-ron, ensemble
- **NLP tiếng Việt**: Tách từ, sentiment, Word2Vec, FastText, PhoBERT
- **AutoML + NAS**: Tự động chọn mô hình và kiến trúc tốt nhất
- **Mobile/Edge**: Xuất TFLite, CoreML, ONNX Mobile + quantization
- **Federated Learning**: FedAvg, Differential Privacy
- **Real-time Pipeline**: Message Queue, Feature Store, latency tracking
- **Cloud SaaS**: Multi-tenant workspace, API keys, quota, deploy
- **No-code Studio**: Kéo thả xây dựng ML pipeline
- **Vietnamese LLM**: N-gram language model, text generation
- **Interpretability**: Feature Importance, Permutation, LIME
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

- [Cài đặt](getting-started/installation.md) - Hướng dẫn cài đặt
- [Sử dụng nhanh](getting-started/quickstart.md) - Ví dụ nhanh
- [Hướng dẫn](guides/classification.md) - Hướng dẫn chi tiết
- [API Reference](api/models.md) - Tham chiếu API

---

*Được xây dựng với ❤️ cho cộng đồng AI Việt Nam*
