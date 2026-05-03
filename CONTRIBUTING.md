# Hướng dẫn đóng góp

Cảm ơn bạn đã quan tâm đến việc đóng góp cho Vietnamese AI Framework!

## Bắt đầu nhanh

```bash
# 1. Fork repository trên GitHub

# 2. Clone repo đã fork
git clone https://github.com/yourusername/vietnamese-ai.git
cd vietnamese-ai

# 3. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Cài đặt dependencies
pip install -e ".[all]"

# 5. Tạo branch mới
git checkout -b feature/ten-tinh-nang

# 6. Chạy test để đảm bảo mọi thứ hoạt động
pytest tests/ -v
```

## Quy trình đóng góp

1. **Fork** repository
2. **Tạo branch** mới: `git checkout -b feature/ten-tinh-nang`
3. **Phát triển** tính năng mới
4. **Viết test** cho tính năng mới
5. **Chạy test**: `pytest tests/ -v`
6. **Chạy lint**: `ruff check vietnamese_ai/ tests/`
7. **Commit**: `git commit -m "feat: mo ta tinh nang"`
8. **Push**: `git push origin feature/ten-tinh-nang`
9. **Tạo Pull Request** trên GitHub

## Quy chuẩn code

### Đặt tên

Framework sử dụng tiếng Việt cho tất cả API công khai:

- **Biến, hàm**: tiếng Việt không dấu, snake_case → `du_lieu_dau_vao`, `tinh_diem_so`
- **Class**: PascalCase → `PhanLoai`, `XuLyVanBan`
- **Hằng số**: UPPER_SNAKE_CASE → `SO_LOP_TOI_DA`
- **File**: snake_case → `classifier.py`, `text_preprocessor.py`
- **Private**: prefix `_` → `_mo_hinh`, `_tai_du_lieu`

### Docstring

Mọi class và phương thức công khai phải có docstring tiếng Việt:

```python
class PhanLoai(BaseModel):
    """
    Bộ phân loại đa thuật toán.

    Hỗ trợ: logistic, knn, svm, ...

    Sử dụng:
        >>> pl = PhanLoai(thuat_toan="logistic")
        >>> pl.huan_luyen(X, y)
        >>> du_doan = pl.du_doan(X_test)
    """

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Huấn luyện mô hình phân loại.

        Args:
            X: Ma trận đặc trưng (n_samples, n_features)
            y: Vector nhãn (n_samples,)
        """
```

### Type hints

Sử dụng type hints cho tất cả tham số và giá trị trả về:

```python
def du_doan(self, X: np.ndarray) -> np.ndarray:
def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
def lay_thong_ke(self) -> Dict[str, Any]:
```

### Error messages

Thông báo lỗi bằng tiếng Việt:

```python
raise ValueError("thuat_toan phải là một trong: logistic, knn, svm")
raise RuntimeError("Mô hình chưa được huấn luyện. Gọi huan_luyen() trước.")
```

## Cấu trúc module mới

Khi tạo module mới, tuân theo cấu trúc:

```
vietnamese_ai/
└── ten_module/
    ├── __init__.py      # Export class chính
    └── ten_file.py      # Implementation
```

Ví dụ:

```python
# vietnamese_ai/ten_module/__init__.py
from vietnamese_ai.ten_module.ten_file import TenClass
__all__ = ["TenClass"]
```

## Kiểm thử

- Mọi tính năng mới phải có test tương ứng
- Test đặt trong `tests/test_phaseX.py` (X = phase number)
- Đặt tên test theo pattern: `test_<ten_tinh_nang>`
- Sử dụng `tmp_path` fixture cho test cần file I/O
- Chạy `pytest tests/ -v` trước khi commit
- Đảm bảo không có test nào bị skip hoặc xfail

### Ví dụ test

```python
class TestTenClass:
    def test_khoi_tao(self):
        from vietnamese_ai.ten_module.ten_file import TenClass
        tc = TenClass()
        assert tc is not None

    def test_chuc_nang(self):
        tc = TenClass()
        ket_qua = tc.chuc_nang_du_lieu(np.array([1, 2, 3]))
        assert len(ket_qua) == 3
```

## Báo cáo lỗi

Tạo Issue với nhãn `bug`:

- **Tiêu đề**: Mô tả ngắn gọn
- **Môi trường**: OS, Python version, package version (`pip show vietnamese-ai`)
- **Mô tả**: Các bước tái hiện lỗi
- **Kết quả mong đợi** vs **Kết quả thực tế**
- **Logs/Lỗi**: Copy toàn bộ traceback

## Đề xuất tính năng

Tạo Issue với nhãn `enhancement`:

- Mô tả vấn đề cần giải quyết
- Giải pháp đề xuất
- Ví dụ sử dụng (nếu có)
- Ảnh hưởng đến API hiện tại (breaking change?)

## Pull Request Checklist

Trước khi tạo PR, đảm bảo:

- [ ] Code tuân thủ quy chuẩn đặt tên (tiếng Việt)
- [ ] Có docstring cho class/method mới
- [ ] Có type hints
- [ ] Có test cho tính năng mới
- [ ] `pytest tests/ -v` pass tất cả
- [ ] `ruff check vietnamese_ai/ tests/` không có lỗi
- [ ] Cập nhật README.md nếu thêm tính năng mới
- [ ] Cập nhật CHANGELOG.md

## Liên hệ

- **Issues**: https://github.com/phonghhd/vietnamese-ai/issues
- **Email**: huynhduongphong9@gmail.com
