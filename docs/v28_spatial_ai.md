# Không gian Trí tuệ (Spatial AI) & Robotics - EvoNet v28

Phiên bản v28 mở rộng EvoNet AI ra khỏi không gian ảo của dữ liệu văn bản và hình ảnh 2D, tiến thẳng vào thế giới thực với **Spatial AI** và **Robotics**.

## 1. Hệ tọa độ 3D (Spatial Vector Store)
VectorDB của EvoNet giờ đây không chỉ lưu trữ embedding ngữ nghĩa, mà còn lưu trữ tọa độ Không gian (X, Y, Z).
- **Hybrid Search**: Hệ thống cho phép tìm kiếm kết hợp giữa "Ý nghĩa văn bản" và "Khoảng cách vật lý".
- **Ứng dụng**: "Tìm cho tôi cuốn sách về Trí tuệ nhân tạo (Ngữ nghĩa) nằm gần tôi nhất trong bán kính 2 mét (Tọa độ)".

## 2. Công cụ Robotics & LiDAR
Thư viện `vietnamese_ai/agents/spatial_tools.py` cung cấp các API để Tác tử tương tác với phần cứng:
- `@cong_cu_di_chuyen_robot`: Lệnh điều khiển cánh tay robot hoặc xe tự hành (AGV) dựa trên Inverse Kinematics.
- `@cong_cu_quet_radar_3d`: Kích hoạt LiDAR quét và thu thập Point Cloud, tự động nạp vào Spatial Vector Store.

## 3. WebXR Edge Node
EvoNet v28 biến kính thực tế ảo (AR/VR) thành một Node trong mạng lưới DePIN. Cung cấp băng thông siêu thấp cho các thiết bị như Apple Vision Pro hay Meta Quest.
