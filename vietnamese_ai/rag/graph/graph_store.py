import json
from typing import List, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

class NetworkXStore:
    """
    Graph Store sử dụng NetworkX chạy trực tiếp trên bộ nhớ Local (hoặc lưu file).
    Lưu trữ cấu trúc Knowledge Graph.
    """
    def __init__(self, file_path: Optional[str] = None):
        if nx is None:
            raise ImportError("Cần cài đặt networkx: pip install networkx")

        self.file_path = file_path
        self.graph = nx.DiGraph()

        if self.file_path:
            self._tai_do_thi()

    def them_bo_ba(self, chu_the: str, quan_he: str, doi_tuong: str, **metadata):
        """Thêm một mối quan hệ vào đồ thị."""
        # Chuẩn hóa
        chu_the = chu_the.lower().strip()
        doi_tuong = doi_tuong.lower().strip()
        quan_he = quan_he.lower().strip()

        # Thêm Node nếu chưa có
        if not self.graph.has_node(chu_the):
            self.graph.add_node(chu_the)
        if not self.graph.has_node(doi_tuong):
            self.graph.add_node(doi_tuong)

        # Thêm Edge
        self.graph.add_edge(chu_the, doi_tuong, relation=quan_he, **metadata)

    def them_nhieu(self, danh_sach_bo_ba: List[Tuple[str, str, str]]):
        """Thêm nhiều bộ ba cùng lúc."""
        for subject, rel, obj in danh_sach_bo_ba:
            self.them_bo_ba(subject, rel, obj)

    def lay_vung_lan_can(self, thuc_the: str, do_sau: int = 1) -> List[Tuple[str, str, str]]:
        """
        Lấy các mối quan hệ lân cận của một thực thể (Neighborhood Search).
        Giúp tìm kiếm ngữ cảnh xung quanh thực thể đó.
        """
        thuc_the = thuc_the.lower().strip()
        if not self.graph.has_node(thuc_the):
            return []

        # Dùng BFS để tìm các node trong do_sau
        nodes_lan_can = set([thuc_the])
        frontier = [thuc_the]

        for _ in range(do_sau):
            next_frontier = []
            for n in frontier:
                # Node kề (outgoing)
                for ngh in self.graph.successors(n):
                    if ngh not in nodes_lan_can:
                        nodes_lan_can.add(ngh)
                        next_frontier.append(ngh)
                # Node kề (incoming)
                for ngh in self.graph.predecessors(n):
                    if ngh not in nodes_lan_can:
                        nodes_lan_can.add(ngh)
                        next_frontier.append(ngh)
            frontier = next_frontier

        # Lấy tất cả các cạnh (edges) giữa các node trong vùng lân cận
        ket_qua = []
        subgraph = self.graph.subgraph(nodes_lan_can)
        for u, v, data in subgraph.edges(data=True):
            ket_qua.append((u, data.get("relation", ""), v))

        return ket_qua

    def _tai_do_thi(self):
        """Tải đồ thị từ file JSON (Node-Link format)."""
        import os
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)

    def luu_do_thi(self):
        """Lưu đồ thị ra file JSON."""
        if self.file_path:
            data = nx.node_link_data(self.graph)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
