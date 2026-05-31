from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.rag.multimodal.image_embedder import ImageEmbedder
from vietnamese_ai.rag.multimodal.multimodal_store import MultimodalStore
from vietnamese_ai.rag.retriever import MultiModalRetriever
from vietnamese_ai.rag.vector_store import CSDLVector
from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy
from vietnamese_ai.security.red_team import RedTeamAgent


class MockLLM:
    def __init__(self, mode="triplets"):
        self.mode = mode

    def sinh_van_ban(self, prompt, **kwargs):
        if "Hacker" in prompt or "Red Team" in prompt:
            return 'Suy nghĩ: Mình sẽ dùng getattr để bypass AST.\nHành động: sandbox_thuc_thi\nTham số: {"ma_nguon": "print(\'Bypass success\')"}\n'
        return "Trả lời: Thành công."

    def __call__(self, prompt):
        return self.sinh_van_ban(prompt)


def test_multimodal_rag():
    text_store = CSDLVector(kich_thuoc=512)
    embedder = ImageEmbedder()
    store = MultimodalStore(text_store, embedder)

    retriever = MultiModalRetriever(store, ham_embed=lambda x: [0.1] * 512)

    # Test methods exist and don't crash
    assert hasattr(retriever, "tim_kiem_da_phuong_thuc")

    # Note: tim_kiem_da_phuong_thuc returns a dict
    res = retriever.tim_kiem_da_phuong_thuc("Bản vẽ kiến trúc", la_hinh_anh=False, top_k=1)
    assert "anh" in res
    assert "van_ban" in res


def test_red_team_agent():
    llm = MockLLM()

    def sandbox_wrapper(ma_nguon: str) -> str:
        return MoiTruongCachLy.thuc_thi(ma_nguon)

    cong_cu_sandbox = CongCu(
        ten="sandbox_thuc_thi",
        mo_ta="Thực thi mã Python trong môi trường cách ly",
        ham_thuc_thi=sandbox_wrapper,
    )

    agent = RedTeamAgent(llm=llm, muc_tieu=cong_cu_sandbox, max_iterations=2)

    # The MockLLM will use the tool, we check if it parses correctly
    res = agent.bat_dau_tan_cong()
    assert res is not None
