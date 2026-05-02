# AutoML

Tự động hóa toàn bộ quy trình học máy.

## Sử dụng cơ bản

```python
from vietnamese_ai import AutoML

auto = AutoML(so_fold=5)
ket_qua = auto.fit(X_train, y_train)

# Xem báo cáo
print(auto.bao_cao())

# Dự đoán
du_doan = auto.predict(X_test)

# Đánh giá
diem = auto.danh_gia(X_test, y_test)
```

## Quy trình AutoML

1. **Phát hiện nhiệm vụ**: Tự động phân loại hồi quy/phân loại
2. **Tiền xử lý**: Chuẩn hóa Z-Score
3. **Thử thuật算法**: logistic, knn, rung_ngau_nhien, gradient_boosting, naive_bayes
4. **Cross-validation**: 5-Fold cho mỗi thuật toán
5. **Chọn tốt nhất**: Mô hình có điểm CV cao nhất
6. **Huấn luyện cuối**: Train trên toàn bộ dữ liệu

## Output

```python
ket_qua = auto.fit(X, y)
# {
#   'mo_hinh_tot_nhat': PhanLoai(rung_ngau_nhien),
#   'thuat_toan_tot_nhat': 'rung_ngau_nhien',
#   'diem_tot_nhat': 0.95,
#   'nhiem_vu': 'phan_loai',
#   'tat_ca_ket_qua': [...]
# }
```
