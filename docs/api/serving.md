# Serving API Reference

## MayChuBatch

```python
class MayChuBatch:
    """Máy chủ dynamic batching cho model inference."""

    def __init__(
        mo_hinh=None,                  # Model object
        kich_thuoc_batch=32,           # Batch size tối đa
        timeout_batch=0.1,             # Timeout chờ batch đầy (giây)
        toi_da_cho=1000,               # Kích thước queue tối đa
        so_worker=1,                   # Số worker threads
        ham_du_doan=None,              # Custom predict function
    )

    def bat_dau() -> None
    def dung() -> None
    def du_doan(dau_vao) -> Any
    def du_doan_batch(dau_vao_list) -> list
    def lay_thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `mo_hinh` | Any | `None` | Model object (cần có `du_doan` hoặc `predict`) |
| `kich_thuoc_batch` | int | `32` | Số request gom thành một batch |
| `timeout_batch` | float | `0.1` | Thời gian chờ batch đầy (giây) |
| `toi_da_cho` | int | `1000` | Kích thước tối đa của queue |
| `so_worker` | int | `1` | Số worker threads xử lý batch |
| `ham_du_doan` | Callable | `None` | Hàm dự đoán tùy chỉnh |

### `du_doan(dau_vao) -> Any`

Gửi request dự đoán đơn lẻ (blocking). Tự động gom thành batch với các request khác.

### `du_doan_batch(dau_vao_list) -> list`

Gửi nhiều request cùng lúc, trả về danh sách kết quả theo thứ tự.

### `lay_thong_ke` Returns

```python
{
    "tong_request": int,         # Tổng số request
    "tong_batch": int,           # Tổng số batch đã xử lý
    "tong_thoi_gian": float,     # Tổng thời gian xử lý (giây)
    "batch_size_tb": float,      # Batch size trung bình
    "latency_p50": float,        # P50 latency (giây)
    "latency_p95": float,        # P95 latency
    "latency_p99": float,        # P99 latency
    "queue_size": int,           # Số request đang chờ
    "dang_chay": bool,           # Server đang chạy
}
```

---

## MayChuStream

```python
class MayChuStream:
    """Máy chủ streaming cho LLM responses."""

    def __init__(
        ham_sinh=None,                 # Callable - hàm sinh text
        toc_do_token=0.02,             # Thời gian giữa các token (giây)
        toi_da_client=100,             # Số client đồng thời tối đa
        timeout=60.0,                  # Timeout mỗi stream (giây)
    )

    def sinh_stream(prompt, client_id=None, **kwargs) -> Generator[str]
    def sinh_sse(prompt, client_id=None, **kwargs) -> Generator[str]
    def dang_ky_callback(client_id, callback) -> None
    def huy_client(client_id) -> bool
    def so_client() -> int
    def lay_thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `ham_sinh` | Callable | `None` | Hàm sinh text (trả về str, list, hoặc Generator) |
| `toc_do_token` | float | `0.02` | Delay giữa các token (giây) |
| `toi_da_client` | int | `100` | Số client streaming đồng thời tối đa |
| `timeout` | float | `60.0` | Timeout cho mỗi stream |

### `sinh_stream(prompt, client_id=None, **kwargs)`

Sinh nội dung theo dạng streaming, yield từng token.

```python
for token in server.sinh_stream("Xin chào"):
    print(token, end="")
```

### `sinh_sse(prompt, client_id=None, **kwargs)`

Sinh nội dung theo SSE (Server-Sent Events) format. Yield chuỗi `data: {...}\n\n`.

### `lay_thong_ke` Returns

```python
{
    "tong_stream": int,          # Tổng số stream đã tạo
    "tong_token": int,           # Tổng token đã sinh
    "client_hien_tai": int,      # Số client đang kết nối
    "thoi_gian_tb": float,       # Thời gian stream trung bình (giây)
}
```

---

## BoGioiHanTocDo

```python
class BoGioiHanTocDo:
    """Bộ giới hạn tốc độ (rate limiter) cho API."""

    def __init__(
        go_i_y=100,                    # Số request tối đa trong cửa sổ
        cua_so=60.0,                   # Kích thước cửa sổ thời gian (giây)
        kich_thuoc_burst=None,         # Burst size (mặc định = go_i_y)
        che_do="token_bucket",         # "token_bucket", "sliding_window"
    )

    def cho_phep(client_id="default") -> bool
    def lay_con_lai(client_id="default") -> int
    def reset(client_id=None) -> None
    def xoa_client_khong_hoat_dong(timeout=300.0) -> int
    def lay_thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `go_i_y` | int | `100` | Số request tối đa trong cửa sổ |
| `cua_so` | float | `60.0` | Kích thước cửa sổ thời gian (giây) |
| `kich_thuoc_burst` | int | `=go_i_y` | Số request burst tối đa |
| `che_do` | str | `"token_bucket"` | Thuật toán rate limiting |

### `cho_phep(client_id="default") -> bool`

Kiểm tra request có được phép hay không. Trả về `True` nếu được phép, `False` nếu bị giới hạn.

### `lay_con_lai(client_id="default") -> int`

Số request còn lại trong cửa sổ hiện tại.

### `reset(client_id=None) -> None`

Reset rate limit cho một client hoặc tất cả client (nếu `client_id=None`).

### `xoa_client_khong_hoat_dong(timeout=300.0) -> int`

Xóa client không hoạt động quá `timeout` giây. Trả về số client đã xóa.

### `lay_thong_ke` Returns

```python
{
    "tong_request": int,         # Tổng request
    "da_cho_phep": int,          # Số request được phép
    "da_tu_choi": int,           # Số request bị từ chối
    "client_hien_tai": int,      # Số client đang theo dõi
    "go_i_y": int,               # Giới hạn request
    "cua_so": float,             # Cửa sổ thời gian
    "che_do": str,               # Thuật toán đang dùng
    "ty_le_tu_choi": float,      # Tỷ lệ request bị từ chối
}
```
