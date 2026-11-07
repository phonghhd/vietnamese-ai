"""Ring Attention - Phân rã Ngữ cảnh ra nhiều máy chủ Swarm."""

from typing import List, Optional, Tuple

import numpy as np


class RingAttentionNode:
    """
    Ring Attention Node (Dành cho Cụm GPU/Swarm).
    Vượt qua giới hạn Context Length của 1 máy bằng cách chia khối K, V dọc theo Seq Dimension.
    Các Node sẽ truyền khối (Block) cho nhau theo vòng tròn.
    """

    def __init__(self, d_model: int, node_id: int, total_nodes: int):
        self.d_model = d_model
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.peer_nodes: Optional[List["RingAttentionNode"]] = None
        self.k_buffer: Optional[np.ndarray] = None
        self.v_buffer: Optional[np.ndarray] = None

    def _local_attention_step(
        self,
        Q_local: np.ndarray,
        K_block: np.ndarray,
        V_block: np.ndarray,
        m_prev: np.ndarray,
        l_prev: np.ndarray,
        O_prev: np.ndarray,
    ) -> tuple:
        """Thực thi FlashAttention 1 bước trên khối K, V hiện tại."""
        batch_size, seq_len, d_model = Q_local.shape
        d_k = d_model

        S_ij = (Q_local @ K_block.transpose(0, 2, 1)) / np.sqrt(d_k)

        m_ij = np.max(S_ij, axis=-1, keepdims=True)
        m_new = np.maximum(m_prev, m_ij)

        P_ij = np.exp(S_ij - m_new)

        factor = np.exp(m_prev - m_new)
        factor = np.where(m_prev == -np.inf, 0.0, factor)

        l_new = factor * l_prev + np.sum(P_ij, axis=-1, keepdims=True)

        O_new = (factor * O_prev * l_prev + P_ij @ V_block) / (l_new + 1e-10)

        return m_new, l_new, O_new

    def forward_ring(
        self,
        Q_local: np.ndarray,
        K_local: np.ndarray,
        V_local: np.ndarray,
        peer_kv_blocks: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> np.ndarray:
        """
        Thực thi Ring Attention qua vòng tròn các khối (K_block, V_block).
        Nếu `peer_kv_blocks` được cung cấp (hoặc `K_local` được chia thành `total_nodes` phân đoạn),
        Node sẽ xoay vòng qua từng phân đoạn K, V theo đúng topo vòng tròn.
        """
        batch_size, local_seq_len, d_model = Q_local.shape

        O = np.zeros_like(Q_local, dtype=np.float64)  # noqa: E741
        m = np.full((batch_size, local_seq_len, 1), -np.inf, dtype=np.float64)
        l = np.zeros((batch_size, local_seq_len, 1), dtype=np.float64)  # noqa: E741

        if peer_kv_blocks is not None and len(peer_kv_blocks) > 0:
            ring_blocks = [
                peer_kv_blocks[(self.node_id - step) % len(peer_kv_blocks)]
                for step in range(len(peer_kv_blocks))
            ]
        elif (
            self.peer_nodes is not None
            and len(self.peer_nodes) == self.total_nodes
            and all(n.k_buffer is not None and n.v_buffer is not None for n in self.peer_nodes)
        ):
            ring_blocks = [
                (
                    self.peer_nodes[(self.node_id - step) % self.total_nodes].k_buffer,
                    self.peer_nodes[(self.node_id - step) % self.total_nodes].v_buffer,
                )
                for step in range(self.total_nodes)
            ]
        elif (
            self.total_nodes > 1
            and K_local.shape[1] >= self.total_nodes
            and K_local.shape[1] % self.total_nodes == 0
        ):
            k_splits = np.array_split(K_local, self.total_nodes, axis=1)
            v_splits = np.array_split(V_local, self.total_nodes, axis=1)
            ring_blocks = [
                (
                    k_splits[(self.node_id - step) % self.total_nodes],
                    v_splits[(self.node_id - step) % self.total_nodes],
                )
                for step in range(self.total_nodes)
            ]
        else:
            ring_blocks = [(K_local, V_local)]

        for current_K, current_V in ring_blocks:
            m, l, O = self._local_attention_step(Q_local, current_K, current_V, m, l, O)  # noqa: E741

        return O.astype(Q_local.dtype)


class RingAttentionCluster:
    """Cụm Ring Attention kết nối N node xoay vòng K, V thực tế."""

    def __init__(self, d_model: int, total_nodes: int):
        self.d_model = d_model
        self.total_nodes = total_nodes
        self.nodes = [RingAttentionNode(d_model, i, total_nodes) for i in range(total_nodes)]
        for node in self.nodes:
            node.peer_nodes = self.nodes

    def forward(self, Q_full: np.ndarray, K_full: np.ndarray, V_full: np.ndarray) -> np.ndarray:
        """Chia Q, K, V cho các node trong vòng tròn và hợp nhất kết quả."""
        q_shards = np.array_split(Q_full, self.total_nodes, axis=1)
        k_shards = np.array_split(K_full, self.total_nodes, axis=1)
        v_shards = np.array_split(V_full, self.total_nodes, axis=1)

        for i, node in enumerate(self.nodes):
            node.k_buffer = k_shards[i]
            node.v_buffer = v_shards[i]

        outputs = [
            node.forward_ring(q_shards[i], k_shards[i], v_shards[i])
            for i, node in enumerate(self.nodes)
        ]
        return np.concatenate(outputs, axis=1)
