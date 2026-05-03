# Real-time ML Pipeline

## Pipeline thời gian thực

`PipelineThoiGianThuc` cung cấp message queue, feature store và real-time prediction.

## Đăng ký mô hình

```python
from vietnamese_ai import PipelineThoiGianThuc, PhanLoai

pl = PhanLoai(thuat_toan="logistic")
pl.huan_luyen(X_train, y_train)

pipeline = PipelineThoiGianThuc(kich_thuoc_buffer=1000)
pipeline.dang_ky_mo_hinh("sentiment", pl)
```

## Dự đoán real-time

```python
ket_qua = pipeline.du_doan("sentiment", X_test[0])
print(f"Kết quả: {ket_qua['ket_qua']}")
print(f"Latency: {ket_qua['latency_ms']:.2f}ms")
```

## Batch prediction

```python
batch = pipeline.du_doan_batch("sentiment", X_test[:100])
print(f"Throughput: {batch['throughput_mau_giay']:.0f} samples/s")
```

## Thống kê

```python
tk = pipeline.lay_thong_ke()
print(f"Latency p95: {tk['latency_p95_ms']:.2f}ms")
print(f"Tổng predictions: {tk['so_du_doan']}")
```

## Callback

```python
def on_prediction(msg):
    print(f"Prediction: {msg}")

pipeline.dang_ky_callback("sentiment", on_prediction)
```
