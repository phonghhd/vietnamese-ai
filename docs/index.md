# Vietnamese AI Framework

<p align="center">
  <strong>Framework AI thuần tiếng Việt cho Python</strong><br>
  <em>Học máy đơn giản. API tiếng Việt. Production-ready.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-57%2F57-brightgreen.svg" alt="tests">
</p>

---

## Tại sao chọn Vietnamese AI?

| Vấn đề | Giải pháp |
|---|---|
| Framework AI lớn quá phức tạp | API đơn giản, học trong 5 phút |
| Tài liệu toàn tiếng Anh | Tài liệu 100% tiếng Việt |
| Thiếu toolkit NLP tiếng Việt | Tích hợp underthesea, TF-IDF, Word2Vec |
| Không có framework all-in-one | Models + NLP + AutoML + API + CLI + Docker |

## Tính năng chính

- **6 mô hình học máy**: Phân loại, hồi quy, phân cụm, mạng nơ-ron, ensemble
- **NLP tiếng Việt**: Tách từ, sentiment, Word2Vec, FastText
- **AutoML**: Tự động chọn mô hình tốt nhất
- **Interpretability**: Giải thích mô hình (Feature Importance, LIME)
- **Experiment Tracking**: Theo dõi thí nghiệm (MLflow compatible)
- **CLI + Docker**: Deploy production-ready

## Bắt đầu nhanh

```bash
pip install vietnamese-ai
```

```python
from vietnamese_ai import AutoML

auto = AutoML()
auto.fit(X_train, y_train)
du_doan = auto.predict(X_test)
```

---

*Được xây dựng với ❤️ cho cộng đồng AI Việt Nam*
