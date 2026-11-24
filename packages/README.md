# Phân phối Gói Độc lập & Meta-Package (Standalone Sub-Packages & SemVer 2.0.0)

Thư mục `packages/` chứa cấu hình PEP 621 (`pyproject.toml`) cho các gói phân phối PyPI tách rời theo lộ trình **Giai đoạn 3 — Modularization hoàn thiện**:

| Gói phân phối | Vai trò | Namespaces đóng gói | Phụ thuộc lõi |
|---|---|---|---|
| **`vietnamese-ai`** (tại root) | Meta-package chính (`pip install vietnamese-ai` / `vietnamese-ai[all]`) | `vietnamese_ai.*` | `numpy>=1.20.0,<3.0.0` |
| **`vietnamese-ai-nlp`** | Lõi NLP tiếng Việt (Tokenizer, Embedder, POS, NER, PhoBERT) | `vietnamese_ai.nlp`, `vietnamese_ai.preprocessing.text` | `numpy>=1.20.0,<3.0.0` |
| **`vietnamese-ai-edge`** | Zero-GPU Edge CPU Inference (GGUF mmap, Q2_K/Q3_K/Q4_K/Q4_0/Q8_0, Swarm LAN) | `vietnamese_ai.edge`, `vietnamese_ai.core.edge` | `numpy>=1.20.0,<3.0.0` |
| **`vietnamese-ai-legal`** | RAG Hành chính & Pháp lý Việt Nam (Hierarchy Chunking, OCR/PDF Parser, Prompt Library) | `vietnamese_ai.rag.legal`, `vietnamese_ai.legal` | `numpy>=1.20.0,<3.0.0` |
| **`vietnamese-ai-zkml`** | Xác thực suy luận mật mã học (Merkle Tree, Schnorr/Fiat-Shamir RFC 5114, Self-Audit) | `vietnamese_ai.crypto`, `vietnamese_ai.zkml` | `numpy>=1.20.0,<3.0.0` |
| **`vietnamese-ai-classic-ml`** | Học máy cổ điển tùy chọn (`PhanLoai`, `HoiQuy`, `PhanCum`, `MangNron`) | `vietnamese_ai.classic_ml`, `vietnamese_ai.models` | `numpy`, `scikit-learn`, `pandas` |

## Quy tắc quản lý phiên bản (Semantic Versioning 2.0.0)

Tất cả gói tuân thủ nghiêm ngặt **SemVer 2.0.0** (`MAJOR.MINOR.PATCH`):
- **PATCH** (`31.0.x`): Vá lỗi bảo mật, sửa lỗi hồi quy, cải tiến tài liệu/hiệu năng không đổi API.
- **MINOR** (`31.x.0`): Bổ sung tính năng/sub-package/kernel mới, giữ 100% tương thích ngược.
- **MAJOR** (`x.0.0`): Thay đổi phá vỡ tương thích ngược API công khai hoặc gỡ bỏ module lỗi thời.

Kiểm tra tự động sự đồng bộ phiên bản bằng Python:

```python
from vietnamese_ai.utils import QuanLyPhatHanhGoi

bao_cao = QuanLyPhatHanhGoi.kiem_tra_tuan_thu_semver()
assert bao_cao["hop_le_semver"] and bao_cao["dong_bo_phien_ban"]
```
