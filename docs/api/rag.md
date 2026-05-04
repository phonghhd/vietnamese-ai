# RAG API Reference

## CSDLVector

```python
class CSDLVector:
    """Cơ sở dữ liệu vector in-memory cho RAG."""

    def __init__(
        kich_thuoc=128,             # Số chiều vector
        khoang_cach="cosine",       # "cosine", "l2", "inner_product"
        suc_chua_toi_da=100000,     # Số vector tối đa
    )

    def chen(ma, vector, metadata=None) -> None
    def chen_batch(ma_list, vectors, metadata_list=None) -> None
    def xoa(ma) -> bool
    def tim_kiem(query_vector, top_k=5, nguong=None, bo_loc=None) -> list
    def lay_vector(ma) -> Optional[np.ndarray]
    def lay_metadata(ma) -> Optional[dict]
    def so_luong() -> int
    def danh_sach_ma() -> list
    def xoa_tat_ca() -> None
    def luu(duong_dan) -> None
    def tai(duong_dan) -> CSDLVector          # classmethod
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `kich_thuoc` | int | `128` | Số chiều của vector |
| `khoang_cach` | str | `"cosine"` | Phương pháp tính khoảng cách |
| `suc_chua_toi_da` | int | `100000` | Số lượng vector tối đa |

### `tim_kiem` Returns

```python
[
    {
        "ma": str,              # ID vector
        "diem": float,          # Điểm tương đồng
        "metadata": dict,       # Metadata đính kèm
    },
]
```

---

## CatVanBan

```python
class CatVanBan:
    """Chia văn bản thành các đoạn (chunks) cho RAG."""

    def __init__(
        kich_thuoc=200,              # Kích thước chunk (số từ/ky_tu)
        chong_chong=50,              # Overlap giữa các chunk
        chien_luoc="tu",             # "cau", "doan", "tu", "ky_tu"
        toi_thieu_kich_thuoc=20,     # Kích thước tối thiểu
        bo_qua_khoang_trang=True,    # Strip whitespace
    )

    def chia(van_ban, metadata=None) -> list
    def chia_nhieu(van_ban_list, metadata_list=None) -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `kich_thuoc` | int | `200` | Kích thước mỗi chunk |
| `chong_chong` | int | `50` | Số từ/ky_tu overlap giữa chunks |
| `chien_luoc` | str | `"tu"` | Chiến lược chia văn bản |
| `toi_thieu_kich_thuoc` | int | `20` | Chunk nhỏ hơn sẽ bị gộp |

### `chia` Returns

```python
[
    {
        "noi_dung": str,           # Nội dung chunk
        "vi_tri_bat_dau": int,     # Vị trí bắt đầu trong văn bản gốc
        "vi_tri_ket_thuc": int,    # Vị trí kết thúc
        "metadata": dict,          # Metadata (loai, so_tu, ...)
    },
]
```

---

## TrichXuat

```python
class TrichXuat:
    """Trích xuất thông tin (retriever) kết hợp vector store và text chunker."""

    def __init__(
        csdl_vector=None,               # CSDLVector instance
        ham_embed=None,                  # Callable[[str], np.ndarray]
        cat_van_ban=None,                # CatVanBan instance
        che_do="hybrid",                # "semantic", "keyword", "hybrid"
        trong_so_semantic=0.7,          # Trọng số semantic trong hybrid
    )

    def them_tai_lieu(ma, van_ban, metadata=None) -> int
    def them_nhieu_tai_lieu(tai_lieu) -> int
    def tim_kiem(cau_hoi, top_k=5, nguong=None) -> list
    def xoa_tai_lieu(ma) -> int
    def so_tai_lieu() -> int
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `csdl_vector` | CSDLVector | `None` | Vector store (tự tạo nếu None) |
| `ham_embed` | Callable | `None` | Hàm embedding (fallback TF-IDF nếu None) |
| `che_do` | str | `"hybrid"` | Chế độ tìm kiếm |
| `trong_so_semantic` | float | `0.7` | Trọng số semantic vs keyword |

### `them_tai_lieu` Arguments

| Parameter | Type | Mô tả |
|---|---|---|
| `ma` | str | ID tài liệu |
| `van_ban` | str | Nội dung tài liệu |
| `metadata` | dict | Metadata bổ sung |

**Returns:** `int` — Số chunks được tạo

### `tim_kiem` Returns

```python
[
    {
        "ma": str,               # Chunk ID
        "diem": float,           # Điểm liên quan
        "metadata": dict,        # Metadata
        "loai": str,             # "semantic" | "keyword" | "hybrid"
    },
]
```

---

## SapXepLai

```python
class SapXepLai:
    """Reranking kết quả tìm kiếm để cải thiện chất lượng RAG."""

    def __init__(
        che_do="mmr",                    # "mmr", "cross_encoder", "keyword", "position"
        ham_score=None,                  # Callable[[str, str], float]
        trong_so_da_dang=0.5,           # Trọng số đa dạng (MMR)
    )

    def sap_xep_lai(cau_hoi, ket_qua, top_k=5) -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `che_do` | str | `"mmr"` | Chiến lược reranking |
| `ham_score` | Callable | `None` | Custom scoring function cho cross_encoder |
| `trong_so_da_dang` | float | `0.5` | Trọng số đa dạng cho MMR (0-1) |

### `sap_xep_lai` Returns

Danh sách kết quả đã sắp xếp lại, mỗi phần tử thêm trường `diem_rerank`.

---

## RAGPipeline

```python
class RAGPipeline:
    """Pipeline RAG (Retrieval-Augmented Generation) hoàn chỉnh."""

    def __init__(
        ham_embed=None,                  # Callable[[str], np.ndarray]
        ham_sinh=None,                   # Callable[[str, list], str]
        kich_thuoc_vector=128,
        kich_thuoc_chunk=200,
        chong_chong_chunk=50,
        chien_luoc_chunk="tu",
        che_do_tim_kiem="hybrid",
        che_do_rerank="mmr",
        trong_so_semantic=0.7,
        trong_so_da_dang=0.5,
        top_k=5,
        nguong_diem=None,
        toi_da_tu_vung=3000,
    )

    def them_tai_lieu(ma, van_ban, metadata=None) -> int
    def them_nhieu_tai_lieu(tai_lieu) -> int
    def hoi(cau_hoi, top_k=None, rerank=True, co_kem_nguon=True) -> dict
    def tim_kiem(cau_hoi, top_k=None, rerank=True) -> list
    def xoa_tai_lieu(ma) -> int
    def xoa_tat_ca() -> None
    def lay_lich_su() -> list
    def xoa_lich_su() -> None
    def luu(duong_dan) -> None
    def tai(duong_dan) -> None
    def thong_ke() -> dict
```

### `hoi` Returns

```python
{
    "cau_hoi": str,             # Câu hỏi gốc
    "tra_loi": str,             # Câu trả lời
    "nguon": list,              # Danh sách nguồn tham khảo
    "so_luong_nguon": int,      # Số nguồn
}
```
