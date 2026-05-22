# Hướng dẫn sử dụng Edge AI & DePIN (v13.0)

Kiến trúc Edge-Cloud Continuum cho phép ứng dụng của bạn linh hoạt xử lý trên thiết bị cục bộ (Local) để bảo mật hoặc offload lên Cloud (EvoNet) khi câu hỏi quá khó.

## 1. Khởi chạy Local Edge Node
Hệ thống ngầm sử dụng `node-llama-cpp` để chạy file GGUF cực kỳ hiệu quả.

```python
from vietnamese_ai import NodeLlamaEngine

# Engine sẽ tự động kích hoạt npx node-llama-cpp server ở background
edge_engine = NodeLlamaEngine(
    model_path="~/.vietnamese_ai/models/Llama-3-8B-Instruct.Q4_K_M.gguf",
    port=8080,
    gpu_layers=35
)

tra_loi = edge_engine.sinh_van_ban("Chào bạn!")
print(tra_loi)
```

## 2. Intelligent Routing (Định tuyến thông minh)
Tránh quá tải Edge bằng cách dùng `EdgeRouter`.

```python
from vietnamese_ai import EdgeRouter

router = EdgeRouter(edge_engine=edge_engine, cloud_api_key="your_evonet_key")

# Câu đơn giản -> Chạy ở Edge (Local)
print(router.sinh_van_ban("1+1 bằng mấy?"))

# Câu phức tạp -> Chạy ở Cloud (Data Center)
print(router.sinh_van_ban("Giải thích cấu trúc mạng LSTM chi tiết"))
```
