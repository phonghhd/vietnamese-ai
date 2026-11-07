from vietnamese_ai.rag.retriever import IdentityAwareRetriever
from vietnamese_ai.rag.vector_store import CSDLVector
import numpy as np

csdl = CSDLVector(kich_thuoc=3)
retriever = IdentityAwareRetriever(csdl_vector=csdl, che_do="semantic")
retriever.ham_embed = lambda x: np.array([0.1, 0.2, 0.3], dtype=np.float32)

retriever.them_tai_lieu("doc_1", "Quy định công ty chung", metadata={"allowed_roles": ["user", "admin"]})
print("So luong tai lieu CSDL:", csdl.so_luong())
print("So luong chunks:", retriever.so_tai_lieu())
kq = retriever.tim_kiem("quy định")
print("Ket qua tim kiem tho:", kq)
