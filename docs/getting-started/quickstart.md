# Sử dụng nhanh

## Phân loại

```python
from vietnamese_ai import PhanLoai, XuLySo
from vietnamese_ai.datasets.sample_data import DuLieuMau

X, y = DuLieuMau.phan_loai_don_gian(so_mau=400)
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)
print(pl.bao_cao(X_test, y_test))
```

## Hồi quy

```python
from vietnamese_ai import HoiQuy

hq = HoiQuy(thuat_toan="tuyen_tinh")
hq.huan_luyen(X_train, y_train)
print(hq.bao_cao(X_test, y_test))
```

## AutoML

```python
from vietnamese_ai import AutoML

auto = AutoML()
ket_qua = auto.fit(X_train, y_train)
print(auto.bao_cao())
du_doan = auto.predict(X_test)
```

## Xử lý văn bản

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()
xl.tach_tu("Trí tuệ nhân tạo rất hay")  # ['trí_tuệ_nhân_tạo', 'rất', 'hay']
xl.phan_tich_cam_xuc("Sản phẩm tốt")     # 'positive'
```

## Pipeline

```python
from vietnamese_ai import Pipeline, XuLySo, PhanLoai

pipe = Pipeline()
pipe.them_buoc("chuan_hoa", XuLySo())
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="gradient_boosting"))
pipe.fit(X_train, y_train)
pipe.luu("model.pkl")
```

## CLI

```bash
vai info
vai train --data data.csv --model logistic --output model.pkl
vai predict --model model.pkl --input new.csv
vai serve --model model.pkl --port 8080
```
