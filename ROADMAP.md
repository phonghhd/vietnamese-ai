# Tầm nhìn & Lộ trình Phát triển (Roadmap) - Kỷ nguyên EvoNet Thực tiễn (v31 - v35)

Hệ sinh thái EvoNet đã đạt đến độ chín muồi về mặt kiến trúc lõi (v1 - v27), tích hợp không gian vật lý (v28) và thiết bị di động (v29 - v30). 

Thay vì hướng tới những khái niệm viễn tưởng xa xôi cần hàng trăm kỹ sư, Lộ trình tiếp theo (v31 - v35) được thiết kế **đặc biệt thực dụng**, bám sát xu hướng công nghệ lõi và **hoàn toàn khả thi** để phát triển bởi một đội ngũ tinh gọn (Solo Developer & AI Assistant).

---

## 🎯 Giai đoạn 1: Tương tác Đa phương thức (v31 - v32)

### v31.0: Real-time Voice AI (Trợ lý Giọng nói Thời gian thực)
Giao tiếp bằng văn bản (Text) trên di động vẫn còn nhiều hạn chế. v31 sẽ mang lại khả năng "Nghe - Nói" trực tiếp.
- **WebRTC Streaming**: Tích hợp giao thức truyền giọng nói siêu tốc độ thấp (độ trễ < 300ms).
- **Speech-to-Speech**: Kết hợp mô hình Whisper (Nhận diện giọng nói) và XTTS (Tổng hợp giọng nói tiếng Việt tự nhiên) thẳng vào lõi `MobileAppBridge` (v30).
- **Mục tiêu**: Người dùng có thể "gọi điện thoại" cho Tác tử EvoNet để nhờ tổng hợp báo cáo thay vì phải nhắn tin.

### v32.0: Computer Use & Desktop Integration (Thao tác Máy tính Tự trị)
Đưa Tác tử ra khỏi môi trường Terminal/Web và cấp cho nó quyền điều khiển màn hình máy tính (Giống chức năng Computer Use của Claude).
- **OS Automation API**: Tác tử có thể phân tích ảnh chụp màn hình (Multi-modal RAG), tự động di chuyển chuột, nhấp chuột và gõ bàn phím.
- **Mục tiêu**: Bạn có thể ra lệnh: *"Hãy mở Excel, tổng hợp doanh thu tháng này từ phần mềm Kế toán và gửi email cho sếp"*, Tác tử sẽ tự động làm mọi thao tác trên màn hình.

---

## 🌍 Giai đoạn 2: Kỹ sư Phần mềm Tự trị (v33)

### v33.0: Autonomous SWE-Agent (Tác tử Lập trình Tự trị)
Nâng cấp toàn diện cho tính năng Sandbox (v21) và AST AST (Abstract Syntax Tree). 
- **Git & Workspace Integration**: Tác tử có quyền clone mã nguồn từ GitHub vào một môi trường hộp cát (Sandbox).
- **Self-Correction Loop**: Tác tử đọc Github Issue -> Phân tích lỗi -> Viết code sửa lỗi -> Tự động chạy Unit Test (Pytest) -> Nếu lỗi, tự đọc log lỗi và sửa lại -> Tạo Pull Request.
- **Mục tiêu**: Bạn sẽ có một "Trợ lý Lập trình" tự động dọn dẹp các bug nhỏ trong hệ sinh thái EvoNet, giúp chúng ta tăng tốc phát triển các version tiếp theo.

---

## 🚀 Giai đoạn 3: Phân tích Video & Edge-to-Edge (v34 - v35)

### v34.0: Local Video Intelligence (Phân tích Video Cục bộ)
Nâng cấp hệ thống Multi-modal RAG (v14) từ phân tích Ảnh tĩnh sang Video động.
- **Frame-by-frame RAG**: Tự động chia nhỏ video thành các khung hình, trích xuất âm thanh và lập chỉ mục Vector.
- **Mục tiêu**: Bạn ném một đoạn video camera an ninh hoặc bài giảng dài 2 tiếng vào EvoNet, Tác tử có thể trả lời chính xác: *"Tên trộm xuất hiện ở phút thứ mấy, mặc áo màu gì?"* hoàn toàn chạy trên máy cá nhân mà không cần upload lên Cloud.

### v35.0: Edge-to-Edge Swarm (Mạng lưới Tác tử Cục bộ)
Phát triển mạng lưới DePIN (v13) lên mức cao nhất: Giao tiếp giữa các thiết bị nội bộ mà không cần Server trung tâm.
- **P2P Seamless Handoff**: Tác tử trên điện thoại (v29) có thể kết nối thẳng với Tác tử trên Laptop qua Wi-Fi Direct/Bluetooth. 
- **Mục tiêu**: Bạn đang yêu cầu điện thoại xử lý một file dữ liệu nặng, điện thoại báo pin yếu (áp dụng `PowerManager` v27). Tác vụ sẽ tự động được "ném" sang chiếc Laptop đang mở trên bàn của bạn để xử lý tiếp mà bạn không hề nhận ra sự gián đoạn.

---

> *"Lộ trình này không phải là viễn tưởng. Nó sử dụng chính những viên gạch (VectorDB, RAG, WebSockets, Agents, PowerManager) mà chúng ta đã xây dựng tỉ mỉ từ v1 đến v30. Với bộ khung đã quá vững chắc, v31-v35 là mục tiêu hoàn toàn trong tầm tay của chúng ta!"*
