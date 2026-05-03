# Mobile/Edge Deployment

## Triển khai mô hình lên thiết bị di động

`TriKhaiDiDong` cho phép xuất mô hình sang các định dạng mobile: TFLite, CoreML, ONNX Mobile.

## Xuất mô hình

```python
from vietnamese_ai import TriKhaiDiDong, PhanLoai

# Huấn luyện mô hình
pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)

tkdd = TriKhaiDiDong()

# Xuất TFLite (Android, microcontrollers)
tkdd.xuat_tflite(pl, "model.tflite", kich_thuoc_dau_vao=(5,))

# Xuất CoreML (iOS, macOS)
tkdd.xuat_coreml(pl, "model.mlmodel", kich_thuoc_dau_vao=(5,))

# Xuất ONNX Mobile (cross-platform)
tkdd.xuat_onnx_mobile(pl, "model_mobile.onnx", kich_thuoc_dau_vao=(5,))
```

## Quantization

Giảm kích thước mô hình bằng INT8 quantization:

```python
tkdd.luong_hoa_int8("model.tflite", "model_int8.tflite")
```

## Benchmark

Đo hiệu suất mô hình trên edge device:

```python
ket_qua = tkdd.benchmark_edge(pl, kich_thuoc_dau_vao=(10, 5), so_lan=100)
print(f"Latency: {ket_qua['thoi_gian_trung_binh_ms']:.2f}ms")
print(f"Throughput: {ket_qua['throughput_mau_giay']:.0f} samples/s")
```

## Deployment Config

```python
tkdd.tao_config_deploy("my_model", dinh_dang="tflite", duong_dan="deploy/mobile")
```
