"""Embeddings - Các lớp chuyển đổi văn bản thành vector."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddings(ABC):
    """Lớp trừu tượng cho Embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Tạo embeddings cho danh sách văn bản."""
        pass
        
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Tạo embedding cho một câu truy vấn."""
        pass

class OpenAIEmbeddings(BaseEmbeddings):
    """
    Sử dụng OpenAI API để tạo embeddings.
    Yêu cầu: pip install openai
    """
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt openai: pip install openai")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        # OpenAI cho phép gửi batch
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # Sắp xếp lại theo thứ tự (OpenAI trả về theo thứ tự, nhưng đề phòng)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(item.embedding) for item in sorted_data]
        
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

class HuggingFaceEmbeddings(BaseEmbeddings):
    """
    Sử dụng các mô hình mã nguồn mở qua thư viện sentence-transformers.
    Rất tốt cho các mô hình tiếng Việt như keepitreal/vietnamese-sbert.
    Yêu cầu: pip install sentence-transformers
    """
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Vui lòng cài đặt: pip install sentence-transformers")
            
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [np.array(emb) for emb in embeddings]
        
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
