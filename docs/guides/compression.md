# Model Compression

## Tổng quan

Bộ công cụ nén và tối ưu mô hình ML:

- **HocRutGon** - Knowledge Distillation (huấn luyện mô hình nhỏ từ mô hình lớn)
- **CatTiaMoHinh** - Pruning (cắt tỉa trọng số không quan trọng)

---

## HocRutGon (Knowledge Distillation)

Huấn luyện mô hình nhỏ (student) từ mô hình lớn (teacher) bằng soft labels.

### Cơ bản

```python
from vietnamese_ai import HocRutGon

teacher = PhanLoai(thuat_toan="rung_ngau_nhien")
teacher.huan_luyen(X_train, y_train)

distiller = HocRutGon(
    teacher=teacher,
    nhiet_do=3.0,
    alpha=0.5,
    ham_loss="kl_divergence",
)

student = PhanLoai(thuat_toan="logistic")
ket_qua = distiller.huan_luyen(student, X_train, y_train, X_val, y_val)

print(f"Teacher accuracy: {ket_qua['teacher_acc']:.4f}")
print(f"Student accuracy: {ket_qua['student_acc']:.4f}")
print(f"Tỷ lệ nén: {ket_qua['ty_le_nen']:.1f}x")
```

### Soft Labels và Temperature

Temperature scaling điều khiển mức độ "mịn" của xác suất teacher:

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

- `T = 1.0`: Xác suất gốc (sharp)
- `T > 1.0`: Xác suất mịn hơn (soft), student học được nhiều thông tin hơn
- `T < 1.0`: Xác suất sắc nét hơn

```python
# Temperature cao = soft labels mịn hơn
distiller = HocRutGon(teacher=teacher, nhiet_do=5.0)

# Temperature thấp = soft labels sắc nét
distiller = HocRutGon(teacher=teacher, nhiet_do=1.5)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `nhiet_do` | `3.0` | Temperature scaling |
| `alpha` | `0.5` | Trọng số soft labels vs hard labels |
| `so_vong` | `10` | Số epochs |
| `ham_loss` | `"kl_divergence"` | Loss function: `kl_divergence`, `mse`, `cross_entropy` |

### Ensemble Distillation

Distillation từ nhiều teachers:

```python
teacher_1 = PhanLoai(thuat_toan="rung_ngau_nhien")
teacher_2 = PhanLoai(thuat_toan="svm")
teacher_3 = PhanLoai(thuat_toan="gradient_boosting")

for t in [teacher_1, teacher_2, teacher_3]:
    t.huan_luyen(X_train, y_train)

distiller = HocRutGon(teacher=teacher_1, nhiet_do=3.0)
student = PhanLoai(thuat_toan="logistic")

ket_qua = distiller.huan_luyen_ensemble(
    student=student,
    teachers=[teacher_1, teacher_2, teacher_3],
    X=X_train,
    y=y_train,
)

print(f"Student accuracy: {ket_qua['student_acc']:.4f}")
print(f"Số teachers: {ket_qua['so_teachers']}")
```

### Thống kê

```python
print(distiller.thong_ke())
# {"nhiet_do": 3.0, "alpha": 0.5, "ham_loss": "kl_divergence", "so_lan_distill": 1}

lich_su = distiller.lay_lich_su()
```

---

## CatTiaMoHinh (Model Pruning)

Cắt tỉa trọng số không quan trọng để giảm kích thước và tăng tốc mô hình.

### Magnitude-based Pruning (mặc định)

Xóa trọng số có giá trị tuyệt đối nhỏ nhất:

```python
from vietnamese_ai import CatTiaMoHinh

pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
ket_qua = pruner.cat_tia(model, X_train, y_train)

print(f"Pruned: {ket_qua['ty_le_prune']*100:.1f}% weights")
print(f"Hiệu suất trước: {ket_qua['hieu_suat_truoc']}")
print(f"Hiệu suất sau: {ket_qua['hieu_suat_sau']}")
```

### Structured Pruning

Xóa nguyên neuron/layer dựa trên L2 norm:

```python
pruner = CatTiaMoHinh(che_do="structured", ty_le=0.3)
ket_qua = pruner.cat_tia(model, X_train, y_train)
```

### Iterative Pruning

Prune dần dần, mỗi lần prune một ít rồi retrain:

```python
pruner = CatTiaMoHinh(
    che_do="iterative",
    ty_le=0.5,
    so_vong_lap=5,
)
ket_qua = pruner.cat_tia(model, X_train, y_train)
```

### Random Pruning

Prune ngẫu nhiên (baseline để so sánh):

```python
pruner = CatTiaMoHinh(che_do="random", ty_le=0.5)
ket_qua = pruner.cat_tia(model)
```

### Chế độ pruning

| Chế độ | Mô tả |
|---|---|
| `"magnitude"` | Xóa trọng số nhỏ (|w| < threshold) |
| `"structured"` | Xóa nguyên neuron dựa trên L2 norm |
| `"iterative"` | Prune dần + retrain (Lottery Ticket) |
| `"random"` | Prune ngẫu nhiên (baseline) |

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `che_do` | `"magnitude"` | Chiến lược pruning |
| `ty_le` | `0.5` | Tỷ lệ trọng số cần prune (0-1) |
| `so_vong_lap` | `1` | Số vòng lặp (iterative) |
| `phuc_hoi` | `False` | Có phục hồi trọng số không |

### Lấy mask và thống kê

```python
mask = pruner.lay_mask()
print(f"Số trọng số giữ lại: {mask.sum()}/{len(mask)}")

lich_su = pruner.lay_lich_su()
print(pruner.thong_ke())
```

---

## So sánh Distillation vs Pruning

| Đặc điểm | Distillation | Pruning |
|---|---|---|
| Mục đích | Mô hình nhỏ hơn | Giảm trọng số không cần thiết |
| Đầu vào | Teacher model + Data | Model + Data (tùy chọn) |
| Đầu ra | Student model mới | Model đã prune |
| Tốc độ | Chậm (train lại) | Nhanh |
| Chất lượng | Thường tốt hơn | Có thể giảm nhẹ |

---

## Kết hợp Distillation + Pruning

```python
from vietnamese_ai import HocRutGon, CatTiaMoHinh

distiller = HocRutGon(teacher=teacher, nhiet_do=3.0)
student = PhanLoai(thuat_toan="logistic")
distiller.huan_luyen(student, X_train, y_train)

pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.3)
pruner.cat_tia(student, X_train, y_train)

print(f"Student size: {pruner._dem_tham_so(student)}")
```
