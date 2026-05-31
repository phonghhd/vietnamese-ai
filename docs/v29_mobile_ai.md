# Tác tử Di động (Mobile AI) - EvoNet v29

Phiên bản v29 là bước chuyển mình để đưa EvoNet AI lên hàng tỷ thiết bị di động (Smartphones).

## 1. Tối ưu RAM (Context Compression)
Điện thoại thông minh có lượng RAM rất nhỏ so với Server. v29 tích hợp bộ `BrowserCopilot` với thuật toán nén ngữ cảnh (Token Pruning), tự động cắt bỏ từ nối (stop-words) và giới hạn độ dài văn bản (chống tràn bộ nhớ OOM) trước khi đưa vào NPU xử lý.

## 2. Tri giác Di động (Mobile OS Tools)
Tác tử v29 không còn bị "mù". Thông qua `mobile_tools.py`, Tác tử có thể giả lập gọi API hệ điều hành để:
- Lấy tọa độ GPS (`@cong_cu_lay_toa_do_gps`)
- Đọc tin nhắn SMS/Push (`@cong_cu_doc_thong_bao_sms`)
- Chụp ảnh Camera (`@cong_cu_chup_anh_camera`)

## 3. Triển khai GGUF & ExecuTorch
Lớp `TriKhaiDiDong` được nâng cấp để hỗ trợ xuất thẳng mô hình Pytorch/ONNX sang định dạng GGUF (dùng cho llama.cpp) và ExecuTorch (chuẩn Mobile mới nhất của Meta), giúp chạy LLM offline trên iOS và Android với độ trễ bằng 0.
