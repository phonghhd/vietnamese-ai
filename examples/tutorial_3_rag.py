"""
Tutorial 3: RAG Pipeline - Retrieval-Augmented Generation
==========================================================

Hướng dẫn: xây dựng hệ thống hỏi đáp dựa trên tài liệu.
"""

import numpy as np

# === 1. Vector Store ===
from vietnamese_ai import CSDLVector

csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine")

# Thêm vectors
for i in range(10):
    vector = np.random.randn(128)
    vector = vector / np.linalg.norm(vector)
    csdl.chen(f"doc_{i}", vector, {"noi_dung": f"Nội dung tài liệu {i}"})

print(f"CSDL: {csdl.so_luong()} vectors")

# Tìm kiếm
query = np.random.randn(128)
query = query / np.linalg.norm(query)
ket_qua = csdl.tim_kiem(query, top_k=3)
for kq in ket_qua:
    print(f"  {kq['ma']}: {kq['diem']:.3f}")

# === 2. Text Chunking ===

from vietnamese_ai import CatVanBan

cat = CatVanBan(kich_thuoc=50, chong_chong=10, chien_luoc="tu")
van_ban = " ".join([f"tu{i}" for i in range(100)])
chunks = cat.chia(van_ban)
print(f"\nChia {len(van_ban.split())} từ thành {len(chunks)} chunks")

# === 3. Full RAG Pipeline ===

from vietnamese_ai import RAGPipeline

# Tạo pipeline (không cần embedding model - dùng fallback)
rag = RAGPipeline(
    kich_thuoc_vector=64,
    kich_thuoc_chunk=200,
    chong_chong_chunk=50,
    che_do_tim_kiem="keyword",
    che_do_rerank="keyword",
)

# Thêm tài liệu
tai_lieu = [
    {"ma": "ai", "van_ban": "Trí tuệ nhân tạo (AI) là ngành khoa học máy tính nghiên cứu cách tạo ra máy tính thông minh. AI bao gồm học máy, học sâu, và xử lý ngôn ngữ tự nhiên."},
    {"ma": "ml", "van_ban": "Học máy (Machine Learning) là một nhánh của AI cho phép máy tính học từ dữ liệu mà không cần lập trình rõ ràng. Các thuật toán phổ biến bao gồm decision tree, SVM, và neural network."},
    {"ma": "dl", "van_ban": "Học sâu (Deep Learning) sử dụng mạng neural nhiều lớp để học representations phức tạp. CNN được dùng cho hình ảnh, RNN cho chuỗi thời gian, và Transformer cho văn bản."},
    {"ma": "nlp", "van_ban": "Xử lý ngôn ngữ tự nhiên (NLP) giúp máy tính hiểu và sinh ngôn ngữ người. Các tác vụ bao gồm phân loại văn bản, tóm tắt, dịch thuật, và hỏi đáp."},
]

for tl in tai_lieu:
    rag.them_tai_lieu(tl["ma"], tl["van_ban"])

print(f"\nRAG: {rag.thong_ke()['so_tai_lieu']} tài liệu, {rag.thong_ke()['so_chunks']} chunks")

# Hỏi đáp
for cau_hoi in ["AI là gì?", "Học máy hoạt động như thế nào?", "Deep learning dùng cho gì?"]:
    ket_qua = rag.hoi(cau_hoi, top_k=2)
    print(f"\nQ: {cau_hoi}")
    print(f"A: {ket_qua['tra_loi'][:100]}...")
    print(f"   Nguồn: {ket_qua['so_luong_nguon']} tài liệu")

# === 4. Save/Load ===

import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    duong_dan = os.path.join(tmpdir, "rag.pkl")
    rag.luu(duong_dan)
    rag2 = RAGPipeline()
    rag2.tai(duong_dan)
    print(f"\nLoaded RAG: {rag2.thong_ke()['so_chunks']} chunks")

print("\n✓ Tutorial 3 hoàn tất!")
