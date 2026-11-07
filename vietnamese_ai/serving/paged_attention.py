from typing import Any, Dict, List, Optional

import numpy as np


class LogicalTokenBlock:
    """Đại diện cho một khối token logic trong câu request của người dùng."""

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.tokens: List[int] = []

    def add_token(self, token_id: int) -> bool:
        if len(self.tokens) < self.block_size:
            self.tokens.append(token_id)
            return True
        return False

    def is_full(self) -> bool:
        return len(self.tokens) == self.block_size


class PhysicalMemoryBlock:
    """Đại diện cho một khối bộ nhớ KV Cache thực tế trên VRAM/RAM."""

    def __init__(self, block_id: int, block_size: int, d_model: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.d_model = d_model
        self.ref_count = 0
        self.k_cache = np.zeros((block_size, d_model), dtype=np.float32)
        self.v_cache = np.zeros((block_size, d_model), dtype=np.float32)
        self.kv_cache = self.k_cache


class BlockManager:
    """
    Quản lý việc ánh xạ (mapping) giữa Logical Blocks và Physical Blocks.
    Thực thi kiến trúc PagedAttention của vLLM kèm Copy-on-Write (CoW).
    """

    def __init__(self, num_blocks: int, block_size: int = 16, d_model: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.d_model = d_model

        self.physical_blocks: List[PhysicalMemoryBlock] = [
            PhysicalMemoryBlock(i, block_size, d_model=d_model) for i in range(num_blocks)
        ]
        self.free_blocks = set(range(num_blocks))

        self.logical_blocks: Dict[str, List[LogicalTokenBlock]] = {}
        self.block_mapping: Dict[int, int] = {}

        self.logical_block_counter = 0

    def _tao_kv_mac_dinh(self, token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Sinh cặp vector (k, v) xác định từ token_id khi người gọi không truyền trực tiếp."""
        angles = (np.arange(self.d_model, dtype=np.float32) + 1.0) * float(token_id + 1) * 0.1
        k_vec = np.sin(angles).astype(np.float32)
        v_vec = np.cos(angles).astype(np.float32)
        return k_vec, v_vec

    def allocate_sequence(self, seq_id: str):
        """Khởi tạo không gian cho một request/sequence mới."""
        self.logical_blocks[seq_id] = []
        self._allocate_new_block(seq_id)

    def _allocate_new_block(self, seq_id: str):
        """Cấp phát một Physical Block mới cho sequence."""
        if not self.free_blocks:
            raise MemoryError("Hết bộ nhớ VRAM vật lý (OOM)!")

        phys_id = self.free_blocks.pop()
        phys_block = self.physical_blocks[phys_id]
        phys_block.ref_count += 1
        phys_block.k_cache.fill(0.0)
        phys_block.v_cache.fill(0.0)

        logical_id = self.logical_block_counter
        self.logical_block_counter += 1

        logical_block = LogicalTokenBlock(logical_id, self.block_size)
        self.logical_blocks[seq_id].append(logical_block)

        self.block_mapping[logical_id] = phys_id

    def fork_sequence(self, src_seq_id: str, dst_seq_id: str) -> None:
        """Nhân bản sequence theo cơ chế Copy-on-Write (chia sẻ chung Physical Blocks)."""
        if src_seq_id not in self.logical_blocks:
            raise KeyError(f"Không tìm thấy sequence nguồn: {src_seq_id}")

        self.logical_blocks[dst_seq_id] = []
        for src_logical in self.logical_blocks[src_seq_id]:
            phys_id = self.block_mapping[src_logical.block_id]
            self.physical_blocks[phys_id].ref_count += 1

            new_logical_id = self.logical_block_counter
            self.logical_block_counter += 1
            dst_logical = LogicalTokenBlock(new_logical_id, self.block_size)
            dst_logical.tokens = list(src_logical.tokens)
            self.logical_blocks[dst_seq_id].append(dst_logical)
            self.block_mapping[new_logical_id] = phys_id

    def append_token(
        self,
        seq_id: str,
        token_id: int,
        k_vec: Optional[np.ndarray] = None,
        v_vec: Optional[np.ndarray] = None,
    ):
        """Thêm một token kèm vector KV vào khối vật lý (hỗ trợ Copy-on-Write)."""
        if seq_id not in self.logical_blocks:
            self.allocate_sequence(seq_id)

        last_logical_block = self.logical_blocks[seq_id][-1]
        if last_logical_block.is_full():
            self._allocate_new_block(seq_id)
            last_logical_block = self.logical_blocks[seq_id][-1]

        phys_id = self.block_mapping[last_logical_block.block_id]
        phys_block = self.physical_blocks[phys_id]

        # Copy-on-Write nếu block vật lý đang được chia sẻ bởi nhiều sequence
        if phys_block.ref_count > 1:
            if not self.free_blocks:
                raise MemoryError("Hết bộ nhớ VRAM vật lý khi thực hiện Copy-on-Write!")
            new_phys_id = self.free_blocks.pop()
            new_phys_block = self.physical_blocks[new_phys_id]
            new_phys_block.ref_count = 1
            np.copyto(new_phys_block.k_cache, phys_block.k_cache)
            np.copyto(new_phys_block.v_cache, phys_block.v_cache)
            phys_block.ref_count -= 1
            self.block_mapping[last_logical_block.block_id] = new_phys_id
            phys_block = new_phys_block

        slot_idx = len(last_logical_block.tokens)
        last_logical_block.add_token(token_id)

        if k_vec is None or v_vec is None:
            k_default, v_default = self._tao_kv_mac_dinh(token_id)
            k_arr = k_default if k_vec is None else np.asarray(k_vec, dtype=np.float32).ravel()
            v_arr = v_default if v_vec is None else np.asarray(v_vec, dtype=np.float32).ravel()
        else:
            k_arr = np.asarray(k_vec, dtype=np.float32).ravel()
            v_arr = np.asarray(v_vec, dtype=np.float32).ravel()

        phys_block.k_cache[slot_idx, : min(self.d_model, k_arr.shape[0])] = k_arr[: self.d_model]
        phys_block.v_cache[slot_idx, : min(self.d_model, v_arr.shape[0])] = v_arr[: self.d_model]

    def gather_kv(self, seq_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Thu thập toàn bộ K và V thực tế từ các Physical Blocks không liên tiếp."""
        if seq_id not in self.logical_blocks:
            raise KeyError(f"Sequence '{seq_id}' chưa được cấp phát.")

        k_list: List[np.ndarray] = []
        v_list: List[np.ndarray] = []
        for logical_block in self.logical_blocks[seq_id]:
            n_tok = len(logical_block.tokens)
            if n_tok == 0:
                continue
            phys_id = self.block_mapping[logical_block.block_id]
            phys_block = self.physical_blocks[phys_id]
            k_list.append(phys_block.k_cache[:n_tok])
            v_list.append(phys_block.v_cache[:n_tok])

        if not k_list:
            empty = np.zeros((0, self.d_model), dtype=np.float32)
            return empty, empty

        return np.vstack(k_list), np.vstack(v_list)

    def free_sequence(self, seq_id: str):
        """Giải phóng bộ nhớ khi request hoàn thành."""
        if seq_id not in self.logical_blocks:
            return

        for logical_block in self.logical_blocks[seq_id]:
            phys_id = self.block_mapping[logical_block.block_id]
            phys_block = self.physical_blocks[phys_id]
            phys_block.ref_count -= 1

            if phys_block.ref_count == 0:
                self.free_blocks.add(phys_id)

            del self.block_mapping[logical_block.block_id]

        del self.logical_blocks[seq_id]

    def print_memory_status(self) -> str:
        """Kiểm tra tình trạng phân mảnh."""
        used = self.num_blocks - len(self.free_blocks)
        usage_pct = (used / self.num_blocks) * 100
        return f"VRAM Blocks Used: {used}/{self.num_blocks} ({usage_pct:.1f}%). No external fragmentation."


class PagedAttentionSimulation:
    """Thực thi tính toán Scaled Dot-Product Attention trên các Physical Blocks không liên tục."""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.last_attention_output: Optional[np.ndarray] = None
        self.last_attention_weights: Optional[np.ndarray] = None

    def forward_tensor(self, query_tensor: Optional[np.ndarray], seq_id: str) -> np.ndarray:
        """Tính toán Scaled Dot-Product Attention thật trên KV cache phân trang."""
        k_mat, v_mat = self.block_manager.gather_kv(seq_id)
        if k_mat.shape[0] == 0:
            out = np.zeros((1, self.block_manager.d_model), dtype=np.float32)
            self.last_attention_output = out
            return out

        if query_tensor is None:
            q_mat = k_mat[-1:, :]
        else:
            q_mat = np.asarray(query_tensor, dtype=np.float32)
            if q_mat.ndim == 1:
                q_mat = q_mat.reshape(1, -1)

        d_k = float(q_mat.shape[-1])
        scores = (q_mat @ k_mat.T) / np.sqrt(max(d_k, 1.0))
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores_shifted)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        output = weights @ v_mat

        self.last_attention_weights = weights
        self.last_attention_output = output
        return output

    def forward(self, query_tensor: Any, seq_id: str, return_tensor: bool = False):
        """
        Thực thi Attention bằng cách thu thập K, V từ các Physical Blocks không liên tục.
        Đặt `return_tensor=True` để nhận trực tiếp ma trận numpy output.
        """
        if seq_id not in self.block_manager.logical_blocks:
            return None

        physical_pointers = []
        for logical_block in self.block_manager.logical_blocks[seq_id]:
            phys_id = self.block_manager.block_mapping[logical_block.block_id]
            physical_pointers.append(phys_id)

        out = self.forward_tensor(
            query_tensor if isinstance(query_tensor, np.ndarray) else None, seq_id
        )
        if return_tensor:
            return out
        return f"Attention computed using physical blocks: {physical_pointers}"
