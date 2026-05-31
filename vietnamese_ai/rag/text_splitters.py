"""Text Splitters - Chia nhỏ văn bản cho RAG."""

from typing import List

from vietnamese_ai.rag.document_loaders import Document


class RecursiveCharacterTextSplitter:
    """
    Chia văn bản dựa trên một mảng các ký tự phân cách (separators).
    Thích hợp cho tiếng Việt để giữ lại các câu hoàn chỉnh.
    """

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Ưu tiên cắt theo đoạn văn kép, đoạn văn đơn, dấu chấm, dấu phẩy, khoảng trắng
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Đệ quy chia nhỏ văn bản."""
        if len(text) <= self.chunk_size:
            return [text]

        # Tìm separator phù hợp nhất
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s in text:
                separator = _s
                new_separators = separators[i + 1 :]
                break

        # Split text bằng separator đó
        splits = text.split(separator)

        # Merge các đoạn nhỏ lại với nhau (có tính toán overlap)
        good_splits = []
        _separator = separator if separator is not None else ""

        current_chunk = ""
        for s in splits:
            if not s:
                continue

            # Nếu s tự thân đã lớn hơn chunk_size, cần chia nhỏ nó ra tiếp
            if len(s) > self.chunk_size:
                if current_chunk:
                    good_splits.append(current_chunk)
                    current_chunk = ""
                # Gọi đệ quy cho chính khúc lớn này
                if new_separators:
                    good_splits.extend(self._split_text(s, new_separators))
                else:
                    # Hết cách, đành cắt cứng
                    for i in range(0, len(s), self.chunk_size):
                        good_splits.append(s[i : i + self.chunk_size])
                continue

            # Gộp vào current_chunk nếu độ dài cho phép
            if len(current_chunk) + len(s) + len(_separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += _separator + s
                else:
                    current_chunk = s
            else:
                if current_chunk:
                    good_splits.append(current_chunk)

                # Tính overlap để bắt đầu chunk mới
                # Chỉ lấy một phần của current_chunk cũ
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    # Lùi lại tìm dấu cách đầu tiên để tránh cắt ngang từ
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx + 1 :]
                    current_chunk = overlap_text + _separator + s if overlap_text else s
                else:
                    current_chunk = s

        if current_chunk:
            good_splits.append(current_chunk)

        return good_splits

    def split_text(self, text: str) -> List[str]:
        """Chia một chuỗi văn bản."""
        return self._split_text(text, self.separators)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Chia danh sách các Document."""
        new_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                # Copy metadata
                new_meta = doc.metadata.copy()
                new_meta["chunk"] = i
                new_docs.append(Document(page_content=chunk, metadata=new_meta))
        return new_docs
