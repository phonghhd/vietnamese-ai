# Pipeline

Kết hợp tiền xử lý + mô hình trong một luồng thống nhất.

## Sử dụng

```python
from vietnamese_ai import Pipeline, XuLySo, PhanLoai

pipe = Pipeline()
pipe.them_buoc("chuan_hoa", XuLySo())
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="gradient_boosting"))

pipe.fit(X_train, y_train)
du_doan = pipe.predict(X_test)
```

## Save/Load

```python
# Lưu
pipe.luu("models/pipeline.pkl")

# Tải lại
pipe2 = Pipeline.tai("models/pipeline.pkl")
du_doan = pipe2.predict(X_test)
```

## Kết hợp với NLP

```python
from vietnamese_ai import Pipeline, XuLyVanBan, PhanLoai

# Pipeline cho text classification
pipe = Pipeline()
pipe.them_buoc("tfidf", tfidf_transformer)
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="logistic"))
```

## Kiểm tra

```python
print(pipe)  # Pipeline('Pipeline': chuan_hoa -> phan_loai [đã fit])
print(pipe.danh_sach_buoc())  # ['chuan_hoa', 'phan_loai']
```
