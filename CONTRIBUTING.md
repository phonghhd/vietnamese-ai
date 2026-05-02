# Hướng dẫn đóng góp

Cảm ơn bạn đã quan tâm đến việc đóng góp cho Vietnamese AI Framework!

## Quy trình đóng góp

1. **Fork** repository
2. **Clone** repo đã fork: `git clone https://github.com/yourusername/vietnamese-ai.git`
3. **Tạo branch** mới: `git checkout -b feature/ten-tinh-nang`
4. **Phát triển** tính năng mới
5. **Chạy test**: `pytest tests/ -v`
6. **Commit**: `git commit -m "feat: mo ta tinh nang"`
7. **Push**: `git push origin feature/ten-tinh-nang`
8. **Tạo Pull Request** trên GitHub

## Quy chuẩn code

### Đặt tên

- **Biến, hàm**: tiếng Việt không dấu, snake_case → `du_lieu_dau_vao`, `tinh_diem_so`
- **Class**: PascalCase → `PhanLoai`, `XuLyVanBan`
- **Hằng số**: UPPER_SNAKE_CASE → `SO_LOP_TOI_DA`
- **File**: snake_case → `classifier.py`, `text_preprocessor.py`

### Docstring

Mọi class và phương thức công khai phải có docstring tiếng Việt:

```python
class PhanLoai(BaseModel):
    """
    Bộ phân loại đa thuật toán.

    Sử dụng:
        >>> pl = PhanLoai(thuat_toan="logistic")
        >>> pl.huan_luyen(X, y)
    """

    def huan_luyen(self, X, y):
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
```

## Kiểm thử

- Mọi tính năng mới phải có test tương ứng
- Test đặt trong `tests/test_framework.py`
- Đặt tên test theo pattern: `test_<ten_tinh_nang>`
- Chạy `pytest tests/ -v` trước khi commit

## Báo cáo lỗi

Tạo Issue với:
- **Tiêu đề**: Mô tả ngắn gọn bằng tiếng Việt
- **Môi trường**: OS, Python version, package version
- **Mô tả**: Các bước tái hiện lỗi
- **Kết quả mong đợi** vs **Kết quả thực tế**

## Đề xuất tính năng

Tạo Issue với nhãn `enhancement`:
- Mô tả vấn đề cần giải quyết
- Giải pháp đề xuất
- Ví dụ sử dụng (nếu có)
