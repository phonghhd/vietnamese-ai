# Phân loại

## Các thuật toán hỗ trợ

| Thuật toán | Tham số | Mô tả |
|---|---|---|
| `logistic` | C, max_iter | Hồi quy Logistic |
| `knn` | n_neighbors | K-Nearest Neighbors |
| `svm` | C, kernel | Support Vector Machine |
| `cay_quyet_dinh` | max_depth | Cây quyết định |
| `rung_ngau_nhien` | n_estimators | Random Forest |
| `gradient_boosting` | n_estimators, learning_rate | Gradient Boosting |
| `naive_bayes` | - | Naive Bayes |

## Ví dụ cơ bản

```python
from vietnamese_ai import PhanLoai, XuLySo

# Chia dữ liệu
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

# Huấn luyện
pl = PhanLoai(thuat_toan="rung_ngau_nhien", n_estimators=100)
pl.huan_luyen(X_train, y_train)

# Đánh giá
print(pl.bao_cao(X_test, y_test))
# {'do_chinh_xac': 0.95, 'precision': 0.94, 'recall': 0.96, 'f1': 0.95}

# Dự đoán xác suất
xac_suat = pl.du_doan_xac_suat(X_test)
```

## Cross-Validation

```python
from vietnamese_ai import KiemDinhCheo, PhanLoai

kdc = KiemDinhCheo(so_fold=5)
ket_qua = kdc.chay(PhanLoai(thuat_toan="logistic"), X, y)
print(f"Accuracy: {ket_qua['diem_trung_binh']:.4f} (+/- {ket_qua['do_lech_chuan']:.4f})")
```

## Hyperparameter Tuning

```python
from vietnamese_ai import TimKiemThamSo, PhanLoai

ts = TimKiemThamSo(so_fold=5)
ket_qua = ts.tim_kiem_luoi(
    lop_mo_hinh=PhanLoai,
    luoi_tham_so={'thuat_toan': ['logistic', 'knn', 'rung_ngau_nhien']},
    X=X_train, y=y_train
)
print(f"Tham số tốt nhất: {ket_qua['tham_so_tot_nhat']}")
```
