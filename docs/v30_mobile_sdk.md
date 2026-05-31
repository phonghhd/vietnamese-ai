# Cầu nối Ứng dụng (Mobile SDK Bridge) - EvoNet v30

Phiên bản v30 tập trung vào việc biến phần Backend Python (Lõi EvoNet) thành một Hệ sinh thái mà các lập trình viên Frontend (React Native, Flutter) có thể dễ dàng sử dụng.

## 1. WebSockets & Push Notification
Sử dụng `MobileAppBridge`, EvoNet duy trì một kết nối liên tục với điện thoại. Tác tử có thể Streaming từng ký tự (như ChatGPT) xuống màn hình điện thoại. 
Đặc biệt, v30 hỗ trợ gửi thông báo (Push Notification) giả lập nếu Tác tử hoàn thành công việc khi điện thoại đang ở trạng thái chạy ngầm (Background).

## 2. Tối ưu Phần cứng từ UI
Khắc phục điểm yếu của v27 (Dùng thư viện `psutil` không hoạt động trên Mobile), v30 yêu cầu App UI đọc chỉ số Pin và Nhiệt độ **thực tế**, sau đó gửi về cho Backend qua WebSocket. Nhờ đó, tính năng san tải (Offload) của DePIN hoạt động chính xác 100%.

## 3. Mã hóa E2EE và Hàng đợi Offline
Bảo mật tuyệt đối:
- **E2EE**: Toàn bộ gói tin JSON qua WebSocket được mã hóa bằng AES/Base64.
- **Offline Queue**: Khi mất kết nối mạng, App tự động lưu tin nhắn vào hàng đợi và đồng bộ lại ngay khi có sóng Wi-Fi/4G. 

Lập trình viên chỉ cần chạy `SDKGenerator.generate_flutter()` hoặc `SDKGenerator.generate_react_native()` để nhận ngay file mã nguồn kết nối (Boilerplate) tích hợp sẵn bảo mật!
