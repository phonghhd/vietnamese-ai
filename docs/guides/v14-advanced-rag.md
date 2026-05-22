# Hướng dẫn RAG Thế Hệ Mới (v14.0)

Version 14 cung cấp kiến trúc GraphRAG (Tri thức đồ thị) và Multi-modal RAG (Đa phương thức), bổ sung cho Vector RAG truyền thống.

## 1. GraphRAG
Dùng `GraphExtractor` để bóc tách thông tin thành các điểm nút, và `NetworkXStore` để lưu trữ.

```python
from vietnamese_ai import GraphExtractor, NetworkXStore, GraphRetriever

# 1. Trích xuất
extractor = GraphExtractor(llm=my_llm)
bo_ba = extractor.trich_xuat("Hà Nội là thủ đô của Việt Nam.") 

# 2. Lưu trữ
store = NetworkXStore()
store.them_nhieu(bo_ba)

# 3. Truy xuất theo vùng lân cận (Neighborhood Search)
retriever = GraphRetriever(graph_store=store, llm=my_llm)
ngu_canh = retriever.truy_xuat("Hà Nội nằm ở đâu?")
```

## 2. Multi-modal RAG
Truy xuất cả văn bản lẫn hình ảnh.

```python
from vietnamese_ai import ImageEmbedder, MultimodalStore

embedder = ImageEmbedder(model_name="clip-ViT-B-32")
store = MultimodalStore(text_store=my_vector_store, image_embedder=embedder)

# Lưu trữ ảnh
store.them_hinh_anh(["bieu_do_1.png", "bieu_do_2.png"], metadatas=[{"ten": "Doanh thu Q1"}, {"ten": "Doanh thu Q2"}])

# Tìm kiếm ảnh bằng ảnh (Image-to-Image similarity)
ket_qua = store.tim_kiem_anh_tu_anh("truy_van.png", top_k=1)
print(ket_qua)
```
