import pytest
import time
from vietnamese_ai.agents.advanced_memory import GraphMemory
from vietnamese_ai.rag.graph.graph_store import NetworkXStore

class MockLLM:
    def __init__(self, mode="triplets"):
        self.mode = mode

    def sinh_van_ban(self, prompt, **kwargs):
        if "phân tích câu nói sau" in prompt:
            return "phong, làm việc tại, evonet\nevonet, là, dự án ai"
        else:
            # Keyword extraction
            return "phong, evonet"

def test_graph_memory():
    store = NetworkXStore()
    llm = MockLLM()
    memory = GraphMemory(llm=llm, graph_store=store)

    # 1. Test adding to graph
    memory.them("user", "Tôi là Phong, tôi làm việc tại dự án AI tên là EvoNet.")
    
    # Wait briefly for background thread to finish
    time.sleep(0.5)

    assert store.graph.has_node("phong")
    assert store.graph.has_node("evonet")
    
    # 2. Test extracting context
    context = memory.lay_ngu_canh("Phong làm ở đâu?")
    assert "phong" in context.lower()
    assert "evonet" in context.lower()
    assert "dự án ai" in context.lower()
    
    # 3. Test clean up
    memory.lam_sach()
    assert len(store.graph.nodes) == 0
