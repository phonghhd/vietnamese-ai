# Hồi quy

## Các thuật toán hỗ trợ

| Thuật toán | Tham số | Mô tả |
|---|---|---|
| `tuyen_tinh` | - | Linear Regression |
| `ridge` | alpha | Ridge Regression |
| `lasso` | alpha | Lasso Regression |
| `elastic_net` | alpha, l1_ratio | Elastic Net |
| `svm` | C, kernel | SVR |
| `cay_quyet_dinh` | max_depth | Decision Tree |
| `rung_ngau_nhien` | n_estimators | Random Forest |
| `gradient_boosting` | n_estimators | Gradient Boosting |

## Ví dụ

```python
from vietnamese_ai import HoiQuy, XuLySo

X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

hq = HoiQuy(thuat_toan="gradient_boosting")
hq.huan_luyen(X_train, y_train)

bc = hq.bao_cao(X_test, y_test)
print(f"MSE: {bc['mse']:.2f}, R2: {bc['r2']:.4f}")
```

## So sánh thuật toán

```python
from vietnamese_ai import HoiQuy, XuLySo

for tt in ["tuyen_tinh", "ridge", "rung_ngau_nhien"]:
    hq = HoiQuy(thuat_toan=tt)
    hq.huan_luyen(X_train, y_train)
    bc = hq.bao_cao(X_test, y_test)
    print(f"{tt}: MSE={bc['mse']:.2f}, R2={bc['r2']:.4f}")
```
