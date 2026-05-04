# RAG Pipeline

## Tổng quan

RAG (Retrieval-Augmented Generation) kết hợp tìm kiếm tài liệu với sinh ngôn ngữ, giúp LLM trả lời dựa trên dữ liệu thực tế.

Vietnamese AI cung cấp pipeline RAG hoàn chỉnh:

- **CSDLVector** - Cơ sở dữ liệu vector in-memory
- **CatVanBan** - Chia văn bản thành chunks
- **TrichXuat** - Trích xuất thông tin (semantic, keyword, hybrid)
- **SapXepLai** - Reranking kết quả (MMR, cross-encoder, keyword)
- **RAGPipeline** - Pipeline end-to-end

---

## CSDLVector

Lưu trữ và tìm kiếm vector embeddings.

```python
from vietnamese_ai import CSDLVector
import numpy as np

csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine", suc_chua_toi_da=100000)

vector_1 = np.random.randn(128).astype(np.float32)
vector_2 = np.random.randn(128).astype(np.float32)

csdl.chen("doc_1", vector_1, {"noi_dung": "Học máy là gì?"})
csdl.chen("doc_2", vector_2, {"noi_dung": "Giới thiệu về AI"})

ket_qua = csdl.tim_kiem(query_vector, top_k=5, nguong=0.3)
```

### Các tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `kich_thuoc` | `128` | Số chiều vector |
| `khoang_cach` | `"cosine"` | Metric: `cosine`, `l2`, `inner_product` |
| `suc_chua_toi_da` | `100000` | Giới hạn số vector |

### Batch insert và metadata filter

```python
csdl.chen_batch(
    ma_list=["doc_3", "doc_4"],
    vectors=np.random.randn(2, 128).astype(np.float32),
    metadata_list=[
        {"noi_dung": "Python cơ bản", "chu_de": "lap_trinh"},
        {"noi_dung": "TensorFlow nâng cao", "chu_de": "deep_learning"},
    ],
)

ket_qua = csdl.tim_kiem(query_vector, top_k=3, bo_loc={"chu_de": "deep_learning"})
```

### Lưu/tải

```python
csdl.luu("data/vector_store.pkl")
csdl = CSDLVector.tai("data/vector_store.pkl")
```

---

## CatVanBan

Chia văn bản thành các đoạn nhỏ (chunks) để đưa vào vector store.

```python
from vietnamese_ai import CatVanBan

cat = CatVanBan(
    kich_thuoc=200,
    chong_chong=50,
    chien_luoc="tu",
    toi_thieu_kich_thuoc=20,
)

van_ban = "Trí tuệ nhân tạo đang thay đổi cách con người làm việc..."
cac_doan = cat.chia(van_ban, metadata={"nguon": "bai_viet_1"})
```

### Chiến lược chia

| Chiến lược | Mô tả |
|---|---|
| `"tu"` | Theo số từ với sliding window |
| `"cau"` | Theo câu (dấu `.`, `!`, `?`) |
| `"doan"` | Theo đoạn (double newline) |
| `"ky_tu"` | Theo số ký tự |

### Chia nhiều văn bản

```python
cac_doan = cat.chia_nhieu(
    van_ban_list=["Văn bản 1...", "Văn bản 2..."],
    metadata_list=[{"nguon": "file1"}, {"nguon": "file2"}],
)
```

---

## TrichXuat

Trích xuất thông tin kết hợp vector store và text chunker.

```python
from vietnamese_ai import TrichXuat, CSDLVector

csdl = CSDLVector(kich_thuoc=128)
trich_xuat = TrichXuat(
    csdl_vector=csdl,
    ham_embed=my_embed_function,
    che_do="hybrid",
    trong_so_semantic=0.7,
)

trich_xuat.them_tai_lieu("doc1", "Nội dung tài liệu về AI...")
trich_xuat.them_tai_lieu("doc2", "Nội dung về học máy...")

ket_qua = trich_xuat.tim_kiem("AI là gì?", top_k=5)
```

### Chế độ tìm kiếm

| Chế độ | Mô tả |
|---|---|
| `"semantic"` | Tìm kiếm bằng vector similarity |
| `"keyword"` | Tìm kiếm BM25-like |
| `"hybrid"` | Kết hợp semantic + keyword |

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `che_do` | `"hybrid"` | Chế độ tìm kiếm |
| `trong_so_semantic` | `0.7` | Trọng số semantic trong hybrid (0-1) |

---

## SapXepLai

Reranking kết quả tìm kiếm để cải thiện chất lượng.

```python
from vietnamese_ai import SapXepLai

reranker = SapXepLai(che_do="mmr", trong_so_da_dang=0.5)
ket_qua_moi = reranker.sap_xep_lai("AI là gì?", ket_qua, top_k=3)
```

### Chế độ reranking

| Chế độ | Mô tả |
|---|---|
| `"mmr"` | Maximal Marginal Relevance - cân bằng relevance và diversity |
| `"cross_encoder"` | Cross-encoder scoring (cần custom `ham_score`) |
| `"keyword"` | Keyword overlap + original score |
| `"position"` | Ưu tiên kết quả xuất hiện sớm trong tài liệu |

### Custom cross-encoder score

```python
def my_score(query, text):
    return len(set(query.split()) & set(text.split())) / max(len(query.split()), 1)

reranker = SapXepLai(che_do="cross_encoder", ham_score=my_score)
```

---

## RAGPipeline

Pipeline RAG hoàn chỉnh: chunking → vector store → retrieval → reranking → generation.

### Cơ bản

```python
from vietnamese_ai import RAGPipeline

rag = RAGPipeline(
    ham_embed=my_embed_fn,
    ham_sinh=my_generate_fn,
    kich_thuoc_vector=128,
    kich_thuoc_chunk=200,
    chong_chong_chunk=50,
    chien_luoc_chunk="tu",
    che_do_tim_kiem="hybrid",
    che_do_rerank="mmr",
    top_k=5,
)

rag.them_tai_lieu("doc1", "Nội dung tài liệu...")
rag.them_tai_lieu("doc2", "Nội dung khác...")

ket_qua = rag.hoi("Câu hỏi về tài liệu?")
print(ket_qua["tra_loi"])
print(ket_qua["nguon"])
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `kich_thuoc_vector` | `128` | Số chiều vector |
| `kich_thuoc_chunk` | `200` | Kích thước chunk (số từ) |
| `chong_chong_chunk` | `50` | Overlap giữa các chunk |
| `che_do_tim_kiem` | `"hybrid"` | Chế độ retrieval |
| `che_do_rerank` | `"mmr"` | Chế độ reranking |
| `top_k` | `5` | Số kết quả trả về |
| `nguong_diem` | `None` | Ngưỡng điểm tối thiểu |

### Tìm kiếm không sinh

```python
ket_qua = rag.tim_kiem("Câu hỏi?", top_k=3, rerank=True)
```

### Quản lý tài liệu

```python
rag.xoa_tai_lieu("doc1")
rag.xoa_tat_ca()

rag.luu("data/rag_store.pkl")
rag.tai("data/rag_store.pkl")

print(rag.thong_ke())
```

---

## End-to-End Example

```python
import numpy as np
from vietnamese_ai import RAGPipeline

def embed_fn(text):
    vector = np.zeros(128, dtype=np.float32)
    for i, c in enumerate(text[:128]):
        vector[i % 128] += ord(c) % 10 / 10.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def generate_fn(question, sources):
    noi_dung = "\n".join(s.get("metadata", {}).get("noi_dung", "") for s in sources)
    return f"Dựa trên tài liệu, câu trả lời cho '{question}': {noi_dung[:200]}"

rag = RAGPipeline(
    ham_embed=embed_fn,
    ham_sinh=generate_fn,
    kich_thuoc_vector=128,
    che_do_tim_kiem="hybrid",
    che_do_rerank="mmr",
    top_k=3,
)

tai_lieu = [
    {"ma": "ai", "van_ban": "Trí tuệ nhân tạo (AI) là ngành học tạo máy thông minh."},
    {"ma": "ml", "van_ban": "Học máy là nhánh con của AI, giúp máy học từ dữ liệu."},
    {"ma": "dl", "van_ban": "Học sâu sử dụng mạng nơ-ron nhiều lớp để học biểu diễn."},
]
rag.them_nhieu_tai_lieu(tai_lieu)

ket_qua = rag.hoi("Học máy là gì?")
print(ket_qua["tra_loi"])
print(f"Số nguồn: {ket_qua['so_luong_nguon']}")
```
