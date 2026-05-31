"""Continuous Batching - Tối đa hóa Thông lượng Server."""

from typing import List, Dict, Any
import uuid

class RequestState:
    def __init__(self, prompt: str, max_tokens: int):
        self.req_id = str(uuid.uuid4())
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generated_tokens = []
        self.is_finished = False

class ContinuousBatcher:
    """
    Continuous Batching (Phân lô liên tục ở mức Token).
    Khác với Static Batching (đợi tất cả req trong batch xong mới chuyển batch mới),
    Continuous Batching sẽ loại bỏ request đã xong (Early Finish) và chèn ngay Request mới vào slot trống.
    Giúp tăng thông lượng (Throughput) GPU lên gấp 10-20 lần.
    """
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.waiting_queue: List[RequestState] = []
        self.running_batch: List[RequestState] = []
        
    def add_request(self, prompt: str, max_tokens: int = 10) -> str:
        req = RequestState(prompt, max_tokens)
        self.waiting_queue.append(req)
        return req.req_id

    def _fill_batch(self):
        """Lấp đầy các Slot trống trong Running Batch từ Waiting Queue."""
        while len(self.running_batch) < self.batch_size and self.waiting_queue:
            next_req = self.waiting_queue.pop(0)
            self.running_batch.append(next_req)

    def step(self, mock_model_func: Any) -> List[Dict[str, Any]]:
        """
        Thực thi 1 bước suy luận (1 Token).
        Trả về danh sách các kết quả ĐÃ HOÀN THÀNH.
        """
        self._fill_batch()
        
        if not self.running_batch:
            return []
            
        # Giả lập suy luận 1 token cho toàn bộ Batch hiện tại
        # Trong thực tế, dữ liệu sẽ được pack thành tensor 2D: (batch_size, seq_len)
        finished_results = []
        
        # Duyệt qua các request đang chạy
        for req in self.running_batch:
            # Sinh 1 token (Mô phỏng)
            new_token = mock_model_func(req.prompt, req.generated_tokens)
            req.generated_tokens.append(new_token)
            
            # Kiểm tra hoàn thành (Sinh đủ số lượng hoặc gặp EOS)
            if len(req.generated_tokens) >= req.max_tokens or new_token == "<EOS>":
                req.is_finished = True
                finished_results.append({
                    "req_id": req.req_id,
                    "prompt": req.prompt,
                    "output": " ".join(req.generated_tokens)
                })
                
        # Lọc bỏ các request đã hoàn thành (Trống Slot)
        self.running_batch = [req for req in self.running_batch if not req.is_finished]
        
        # Lập tức lấp đầy slot trống (Kích hoạt Continuous Batching)
        self._fill_batch()
        
        return finished_results
