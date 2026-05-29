# THE ZERO-DEPENDENCY OATH (LỜI THỀ ZERO-DEPENDENCY)

Là một lập trình viên đóng góp (Contributor) cho lõi **V-UI (Vietnamese Micro UI Framework)**, bạn phải tuyên thệ tuân thủ các nguyên tắc sau đây trước khi tạo Pull Request:

## 1. KHÔNG SỬ DỤNG `pip install` CHO UI
Lõi V-UI sinh ra với sứ mệnh giữ cho V-Neural siêu nhẹ. **Tuyệt đối cấm** việc thêm các thư viện backend UI như `fastapi`, `gradio`, `streamlit`, `flask` vào danh sách dependencies (`requirements.txt` hoặc `setup.py`) chỉ để phục vụ V-UI. V-UI phải chạy được chỉ với các thư viện chuẩn (Standard Library) của Python như `http.server`.

## 2. VẮT KIỆT SỨC MẠNH CLIENT (TRÌNH DUYỆT)
Nếu bạn cần một tính năng nặng (Render 3D, Nhận diện Giọng nói, Lưu trữ Vector tạm thời, Quay màn hình), bạn **PHẢI** tìm cách đẩy gánh nặng đó xuống Trình duyệt Web của người dùng thông qua các HTML5 Web APIs nguyên bản:
- Cần Database? Dùng `IndexedDB`.
- Cần Voice AI? Dùng `Web Speech API`.
- Cần Vision / Screen Capture? Dùng `WebRTC / getDisplayMedia`.
- Cần Định vị? Dùng `Geolocation API`.

## 3. CHỈ DÙNG CDN CHO FRONTEND
Không dùng Node.js, không Webpack, không npm install. Nếu bạn cần một thư viện Javascript hoặc CSS (như Tailwind, Three.js, Vis-network), hãy nhúng thẳng link CDN vào thẻ `<script>` trong code Python. V-UI phải là một Single Page Application (SPA) được sinh ra từ chuỗi (String) Python tĩnh.

---
*“Mọi dòng code Python thêm vào V-UI phải nhẹ tựa lông hồng, mọi hiệu ứng đồ họa sinh ra phải nặng tựa Thái Sơn (do Client gánh).”*

Ký tên: **Vietnamese AI Team**
