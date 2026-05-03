# Federated Learning

## Học liên kết phân tán

`HocLienKet` triển khai thuật toán FedAvg cho federated learning.

## Cơ bản

```python
from vietnamese_ai import HocLienKet, PhanLoai

hl = HocLienKet(so_client=5, so_vong=10)
ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")

print(f"Điểm global: {ket_qua['diem_toan_cuc']:.4f}")
print(f"Số rounds: {ket_qua['so_vong']}")
```

## Differential Privacy

Bảo vệ dữ liệu bằng cách thêm nhiễu Gaussian:

```python
hl = HocLienKet(
    so_client=5,
    so_vong=10,
    rieng_tu_differntial=0.1  # Mức độ riêng tư
)
ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
```

## Client Sampling

Chỉ chọn subset clients mỗi round:

```python
hl = HocLienKet(so_client=10, so_vong=20, ty_le_client=0.6)
# Chỉ 60% clients tham gia mỗi round
```

## Dự đoán

```python
du_doan = hl.du_doan(PhanLoai, X_test, thuat_toan="logistic")
```

## Báo cáo

```python
print(hl.bao_cao())
```
