from unittest.mock import MagicMock

from vietnamese_ai.rag.graph import GraphExtractor, GraphRetriever, NetworkXStore
from vietnamese_ai.rag.multimodal import ImageEmbedder, MultimodalStore


class DummyLLM:
    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        if "Trích xuất" in prompt:
            return "Hà Nội | thủ đô | Việt Nam\nHồ Chí Minh | thành phố lớn nhất | Việt Nam"
        if "Xác định các thực thể" in prompt:
            return "Hà Nội, Việt Nam"
        return ""

def test_graph_rag():
    llm = DummyLLM()
    extractor = GraphExtractor(llm=llm)

    # Test trích xuất
    bo_ba = extractor.trich_xuat("Hà Nội là thủ đô của Việt Nam.")
    assert len(bo_ba) == 2
    assert bo_ba[0] == ("Hà Nội", "thủ đô", "Việt Nam")

    # Test Graph Store
    store = NetworkXStore()
    store.them_nhieu(bo_ba)

    lan_can = store.lay_vung_lan_can("hà nội")
    assert len(lan_can) == 1

    # Test Retriever
    retriever = GraphRetriever(graph_store=store, llm=llm)
    ngu_canh = retriever.truy_xuat("Hà Nội ở đâu?")
    assert "hà nội thủ đô việt nam" in ngu_canh or "Hà Nội" in ngu_canh or "hà nội" in ngu_canh

def test_multimodal_rag():
    # Test Image Embedder
    embedder = ImageEmbedder()
    vectors = embedder.nhung_hinh_anh(["test1.jpg", "test2.jpg"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 512

    # Test Store
    mock_text_store = MagicMock()
    store = MultimodalStore(text_store=mock_text_store, image_embedder=embedder)

    store.them_hinh_anh(
        image_paths=["test1.jpg", "test2.jpg"],
        metadatas=[{"name": "Ảnh 1"}, {"name": "Ảnh 2"}]
    )
    assert len(store.image_vectors) == 2

    # Test tìm kiếm ảnh
    ket_qua = store.tim_kiem_anh_tu_anh("query.jpg", top_k=1)
    assert len(ket_qua) == 1
    assert "score" in ket_qua[0]
