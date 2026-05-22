"""Document Loaders - Đọc dữ liệu từ nhiều nguồn khác nhau."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class Document:
    """Đại diện cho một đoạn tài liệu văn bản cùng với metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        
    def __repr__(self):
        return f"Document(length={len(self.page_content)}, metadata={self.metadata})"

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        """Tải và trả về danh sách Document."""
        pass

class TextLoader(BaseLoader):
    """Đọc file .txt."""
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
        
    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file {self.file_path}")
            
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            text = f.read()
            
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

class PyPDFLoader(BaseLoader):
    """Đọc file .pdf sử dụng pypdf."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError("Vui lòng cài đặt: pip install pypdf")
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file {self.file_path}")
            
        docs = []
        with open(self.file_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    metadata = {"source": self.file_path, "page": i + 1}
                    docs.append(Document(page_content=text, metadata=metadata))
                    
        return docs
