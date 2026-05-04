# Compression API Reference

## HocRutGon

```python
class HocRutGon:
    """Knowledge Distillation - huấn luyện mô hình nhỏ từ mô hình lớn."""

    def __init__(
        teacher,                       # Mô hình teacher (đã huấn luyện)
        nhiet_do=3.0,                  # Temperature scaling
        alpha=0.5,                     # Trọng số soft vs hard labels
        so_vong=10,                    # Số vòng huấn luyện
        ham_loss="kl_divergence",      # "kl_divergence", "mse", "cross_entropy"
    )

    def huan_luyen(student, X, y, X_val=None, y_val=None) -> dict
    def huan_luyen_ensemble(student, teachers, X, y) -> dict
    def lay_lich_su() -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `teacher` | Any | — | Mô hình teacher đã huấn luyện |
| `nhiet_do` | float | `3.0` | Nhiệt độ cho temperature scaling |
| `alpha` | float | `0.5` | Trọng số kết hợp soft/hard labels |
| `so_vong` | int | `10` | Số vòng lặp huấn luyện |
| `ham_loss` | str | `"kl_divergence"` | Hàm loss |

### `huan_luyen` Arguments

| Parameter | Type | Mô tả |
|---|---|---|
| `student` | Any | Mô hình student cần huấn luyện |
| `X` | np.ndarray | Dữ liệu huấn luyện |
| `y` | np.ndarray | Nhãn thực |
| `X_val` | np.ndarray | Dữ liệu validation (optional) |
| `y_val` | np.ndarray | Nhãn validation (optional) |

### `huan_luyen` Returns

```python
{
    "student": Any,                # Student model đã huấn luyện
    "teacher_acc": float,          # Độ chính xác teacher
    "student_acc": float,          # Độ chính xác student
    "teacher_val_acc": float,      # Độ chính xác teacher trên validation
    "student_val_acc": float,      # Độ chính xác student trên validation
    "ty_le_nen": float,            # Tỷ lệ nén (teacher/student params)
    "toc_do": str,                 # Thời gian huấn luyện
    "nhiet_do": float,             # Temperature đã dùng
    "alpha": float,                # Alpha đã dùng
}
```

### `huan_luyen_ensemble(student, teachers, X, y) -> dict`

Distillation từ nhiều teacher models. Soft labels được trung bình trước khi huấn luyện student.

**Returns:** `{"student": Any, "student_acc": float, "so_teachers": int}`

---

## CatTiaMoHinh

```python
class CatTiaMoHinh:
    """Cắt tỉa (pruning) mô hình để giảm kích thước và tăng tốc."""

    def __init__(
        che_do="magnitude",            # "magnitude", "structured", "iterative", "random"
        ty_le=0.5,                     # Tỷ lệ weights cần prune (0-1)
        so_vong_lap=1,                 # Số vòng lặp (cho iterative)
        phuc_hoi=False,                # Có phục hồi weights không
    )

    def cat_tia(model, X=None, y=None) -> dict
    def lay_mask() -> Optional[np.ndarray]
    def lay_lich_su() -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `che_do` | str | `"magnitude"` | Chiến lược pruning |
| `ty_le` | float | `0.5` | Tỷ lệ weights cần loại bỏ (0-1) |
| `so_vong_lap` | int | `1` | Số vòng lặp (chỉ dùng cho iterative) |
| `phuc_hoi` | bool | `False` | Có phục hồi weights sau prune |

### Pruning Modes

| Chế độ | Mô tả |
|---|---|
| `magnitude` | Xóa weights có giá trị tuyệt đối nhỏ nhất |
| `structured` | Xóa nguyên neuron dựa trên L2 norm |
| `iterative` | Prune dần nhiều vòng, retrain giữa các vòng |
| `random` | Xóa weights ngẫu nhiên |

### `cat_tia(model, X=None, y=None) -> dict`

```python
{
    "model": Any,                  # Model đã prune
    "mask": np.ndarray,            # Mask nhị phân (1=giữ, 0=prune)
    "hieu_suat_truoc": float,      # Độ chính xác trước prune
    "hieu_suat_sau": float,        # Độ chính xác sau prune
    "ty_le_prune": float,          # Tỷ lệ weights đã prune
    "so_tham_so_goc": int,         # Số tham số gốc
    "so_tham_so_sau": int,         # Số tham số sau prune
    "thoi_gian": str,              # Thời gian thực hiện
    "che_do": str,                 # Chế độ đã dùng
}
```

### Supported Models

- sklearn models (`coef_`, `feature_importances_`)
- Custom models với attributes: `_W`, `_weights`, `weights`, `_trong_so`
