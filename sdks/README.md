# SDK Đa Nền Tảng (`sdks/`) — Trạng Thái Thử Nghiệm / Deprecated (`v31.2.0`)

> **Thông báo lỗi thời (Soft Deprecation — `v31.2.0`)**: Các bản phác thảo SDK trong thư mục `sdks/` (`nodejs/`, `php/`, `wasm/`) hiện là bản mẫu thử nghiệm/vỏ bọc giao diện chưa đóng gói bộ biên dịch chính thức và được đánh dấu **deprecated** từ `v31.2.0` (dự kiến tách sang kho riêng ở `v32.0.0`).

Để tích hợp **Vietnamese AI Framework** từ Node.js, PHP, trình duyệt hoặc bất kỳ ngôn ngữ nào khác trong môi trường production, hãy sử dụng một trong hai giao diện chuẩn được bảo trì chính thức:

1. **REST / OpenAI-Compatible HTTP API** (`FastAPIServer` / `vai serve`): hỗ trợ SSE streaming, Tool Calling và kiểm định an toàn đa tầng.
2. **Giao thức MCP chuẩn (`stdio` / `SSE`)** (`MayChuMCPThucTe` / `vai mcp-server`): tích hợp trực tiếp với IDE và client đa ngôn ngữ.
