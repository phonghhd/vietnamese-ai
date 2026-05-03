# No-code Studio

## Kéo thả xây dựng ML pipeline

`StudioKeoTha` cho phép xây dựng pipeline bằng cách kéo thả nodes.

## Sử dụng template

```python
from vietnamese_ai import StudioKeoTha

studio = StudioKeoTha()
studio.tai_template("phan_loai_co_ban")
ket_qua = studio.chay()
print(ket_qua['trang_thai'])  # 'thanh_cong'
```

## Tự xây dựng pipeline

```python
studio = StudioKeoTha()

# Thêm nodes
studio.them_node("du_lieu", "Dữ liệu mẫu", tham_so={"so_mau": 200})
studio.them_node("tien_xu_ly", "Chuẩn hóa", tham_so={"phuong_phap": "zscore"})
studio.them_node("mo_hinh", "Random Forest", tham_so={"thuat_toan": "rung_ngau_nhien"})
studio.them_node("danh_gia", "Đánh giá")

# Kết nối
canvas = studio.lay_canvas()
node_ids = list(canvas['nodes'].keys())
for i in range(len(node_ids) - 1):
    studio.ket_noi(node_ids[i], node_ids[i + 1])

# Chạy
ket_qua = studio.chay()
```

## Các loại node

| Loại | Mô tả | Đầu vào | Đầu ra |
|---|---|---|---|
| `du_lieu` | Nguồn dữ liệu | - | du_lieu |
| `tien_xu_ly` | Tiền xử lý | du_lieu | du_lieu |
| `mo_hinh` | Huấn luyện mô hình | du_lieu | mo_hinh |
| `danh_gia` | Đánh giá | mo_hinh, du_lieu | ket_qua |
| `truc_quan_hoa` | Biểu đồ | ket_qua | bieu_do |
| `xuat` | Lưu kết quả | mo_hinh, ket_qua | - |

## Templates có sẵn

- `phan_loai_co_ban`: Pipeline phân loại cơ bản
- `hoi_quy_nang_cao`: Pipeline hồi quy với feature engineering
- `so_sanh_nhieu_mo_hinh`: So sánh nhiều thuật toán

## Lưu/Tải

```python
studio.luu("pipeline.json")
studio2 = StudioKeoTha.tai("pipeline.json")
```
