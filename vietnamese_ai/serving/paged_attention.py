from typing import Any, Dict, List


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
    """Đại diện cho một khối bộ nhớ KV Cache thực tế trên VRAM."""

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        # Mô phỏng tensor KV Cache
        self.kv_cache = [0.0] * block_size


class BlockManager:
    """
    Quản lý việc ánh xạ (mapping) giữa Logical Blocks và Physical Blocks.
    Mô phỏng kiến trúc PagedAttention của vLLM giúp loại bỏ phân mảnh VRAM.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Khởi tạo Pool bộ nhớ vật lý
        self.physical_blocks: List[PhysicalMemoryBlock] = [
            PhysicalMemoryBlock(i, block_size) for i in range(num_blocks)
        ]
        self.free_blocks = set(range(num_blocks))

        # Ánh xạ: sequence_id -> List[LogicalTokenBlock]
        self.logical_blocks: Dict[str, List[LogicalTokenBlock]] = {}
        # Ánh xạ: logical_block_id -> physical_block_id
        self.block_mapping: Dict[int, int] = {}

        self.logical_block_counter = 0

    def allocate_sequence(self, seq_id: str):
        """Khởi tạo không gian cho một request/sequence mới."""
        self.logical_blocks[seq_id] = []
        self._allocate_new_block(seq_id)

    def _allocate_new_block(self, seq_id: str):
        """Cấp phát một Physical Block mới cho sequence."""
        if not self.free_blocks:
            raise MemoryError("Hết bộ nhớ VRAM vật lý (OOM)!")

        phys_id = self.free_blocks.pop()
        self.physical_blocks[phys_id].ref_count += 1

        logical_id = self.logical_block_counter
        self.logical_block_counter += 1

        logical_block = LogicalTokenBlock(logical_id, self.block_size)
        self.logical_blocks[seq_id].append(logical_block)

        # Ánh xạ
        self.block_mapping[logical_id] = phys_id

    def append_token(self, seq_id: str, token_id: int):
        """Thêm một token vào quá trình sinh (generation)."""
        if seq_id not in self.logical_blocks:
            self.allocate_sequence(seq_id)

        last_logical_block = self.logical_blocks[seq_id][-1]
        if last_logical_block.is_full():
            self._allocate_new_block(seq_id)
            last_logical_block = self.logical_blocks[seq_id][-1]

        last_logical_block.add_token(token_id)

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
    """Mô phỏng lớp PagedAttention dùng BlockManager."""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager

    def forward(self, query_tensor: Any, seq_id: str):
        """
        Mô phỏng Attention computation bằng cách fetch từ Physical Blocks không liên tục.
        (Thực tế sẽ gọi CUDA kernel trỏ tới các blocks này).
        """
        if seq_id not in self.block_manager.logical_blocks:
            return None

        # Thu thập các pointers vật lý
        physical_pointers = []
        for logical_block in self.block_manager.logical_blocks[seq_id]:
            phys_id = self.block_manager.block_mapping[logical_block.block_id]
            physical_pointers.append(phys_id)

        # Ở đây vLLM sẽ thực thi FlashAttention trên các physical pointers.
        return f"Attention computed using physical blocks: {physical_pointers}"
