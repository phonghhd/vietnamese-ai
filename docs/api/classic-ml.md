# Classic ML API Reference (`vietnamese_ai.classic_ml`)

Gói mở rộng tùy chọn cho các mô hình học máy cổ điển (`pip install "vietnamese-ai[classic-ml]"`). Mọi biểu tượng vẫn được duy trì tương thích ngược tại `vietnamese_ai.models`.

> [!NOTE]
> Với các pipeline học máy bảng quy mô sản xuất, khuyến nghị sử dụng trực tiếp `scikit-learn` / `xgboost` / `lightgbm` hoặc cài đặt `vietnamese-ai[classic-ml]`. Khi chưa cài `scikit-learn`, các lớp tự động kích hoạt bộ ước lượng dự phòng thuần NumPy phục vụ học tập và thử nghiệm nhẹ.

## Exports

```python
from vietnamese_ai.classic_ml import (
    BaseModel,
    PhanLoai,
    HoiQuy,
    PhanCum,
    MoHinhTapHop,
    MangNron,
    MangSau,
    thong_tin_goi_classic_ml,
)
```

## `thong_tin_goi_classic_ml()`

```python
def thong_tin_goi_classic_ml() -> Dict[str, Any]
```

Trả về trạng thái cài đặt `scikit-learn` / `pandas`, tên extra `classic-ml` và khuyến nghị sử dụng. Chi tiết phương thức từng lớp mô hình xem tại [Models API Reference](models.md).
