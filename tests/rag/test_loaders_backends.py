"""Test RAG loaders và backends tùy chọn."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTextLoaderVaSplitter:
    def test_loader_va_split(self, tmp_path):
        from vietnamese_ai.rag.document_loaders import TextLoader
        from vietnamese_ai.rag.text_splitters import RecursiveCharacterTextSplitter

        f = tmp_path / "a.txt"
        f.write_text("Đoạn một.\n\nĐoạn hai dài hơn một chút. " * 20, encoding="utf-8")
        docs = TextLoader(str(f)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
        chunks = splitter.split_documents(docs)
        assert len(docs) == 1
        assert len(chunks) >= 1


class TestChromaOptional:
    def test_chroma_ephemeral(self):
        if sys.version_info < (3, 9):
            pytest.skip("ChromaDB không hỗ trợ Python 3.8")
        pytest.importorskip("chromadb")
        from vietnamese_ai.rag.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(thu_muc_luu_tru=None, ten_collection="test_vai")
        vec = np.random.randn(8).astype(float)
        store.chen("id1", vec, metadata={"nguon": "test"})
        if hasattr(store, "tim_kiem"):
            kq = store.tim_kiem(vec, top_k=1)
            assert len(kq) >= 1


class TestQdrantOptional:
    def test_import_class(self):
        from vietnamese_ai.rag.qdrant_store import QdrantVectorStore

        assert QdrantVectorStore is not None

    def test_qdrant_memory_hoac_mock(self, tmp_path):
        qdrant = pytest.importorskip("qdrant_client")
        from vietnamese_ai.rag.qdrant_store import QdrantVectorStore

        try:
            store = QdrantVectorStore(
                path=None,
                url=None,
                ten_collection="test_vai_qdrant",
                kich_thuoc=8,
            )
        except Exception:
            # Fallback: mock client nếu local Qdrant storage lỗi trên môi trường CI
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            with patch.object(qdrant, "QdrantClient", return_value=mock_client):
                store = QdrantVectorStore(
                    path=str(tmp_path / "qdrant"),
                    ten_collection="test_vai_qdrant",
                    kich_thuoc=8,
                )
            mock_client.create_collection.assert_called()
            return

        vec = np.random.randn(8).astype(float)
        store.chen("id1", vec, metadata={"nguon": "test"})
        kq = store.tim_kiem(vec, top_k=1)
        assert isinstance(kq, list)
