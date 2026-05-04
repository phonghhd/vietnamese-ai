# Serving & Streaming

## Tổng quan

Bộ công cụ phục vụ model ML trong production:

- **MayChuBatch** - Dynamic batching tăng throughput
- **MayChuStream** - SSE streaming cho LLM response
- **BoGioiHanTocDo** - Rate limiting bảo vệ API

---

## MayChuBatch

Máy chủ dynamic batching - gom nhiều request thành batch để tăng throughput.

### Cơ bản

```python
from vietnamese_ai import MayChuBatch

server = MayChuBatch(
    mo_hinh=model,
    kich_thuoc_batch=32,
    timeout_batch=0.1,
    so_worker=2,
)

server.bat_dau()

ket_qua = server.du_doan(dau_vao)

ket_qua_batch = server.du_doan_batch([dau_vao_1, dau_vao_2, dau_vao_3])

server.dung()
```

### Custom predict function

```python
def my_predict(batch_inputs):
    return [model.predict(x) for x in batch_inputs]

server = MayChuBatch(
    mo_hinh=model,
    kich_thuoc_batch=16,
    ham_du_doan=my_predict,
)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `kich_thuoc_batch` | `32` | Kích thước batch tối đa |
| `timeout_batch` | `0.1` | Timeout gom batch (giây) |
| `toi_da_cho` | `1000` | Queue size tối đa |
| `so_worker` | `1` | Số worker threads |

### Theo dõi hiệu suất

```python
stats = server.lay_thong_ke()
print(f"Latency P50: {stats['latency_p50']:.3f}s")
print(f"Latency P95: {stats['latency_p95']:.3f}s")
print(f"Batch size TB: {stats['batch_size_tb']:.1f}")
print(f"Queue size: {stats['queue_size']}")
```

---

## MayChuStream

Máy chủ streaming cho LLM responses - SSE (Server-Sent Events).

### Cơ bản

```python
from vietnamese_ai import MayChuStream

server = MayChuStream(
    ham_sinh=my_generate_fn,
    toc_do_token=0.02,
    toi_da_client=100,
)

for token in server.sinh_stream("Viết bài luận về AI"):
    print(token, end="")
```

### SSE format (cho HTTP API)

```python
for sse_event in server.sinh_sse("Câu hỏi từ client"):
    # Format: "data: {"token": "..."}\n\n"
    # Hoặc: "data: [DONE]\n\n"
    print(sse_event)
```

### Callback-based streaming

```python
def on_token(token):
    print(token, end="")

server.dang_ky_callback("client_1", on_token)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `toc_do_token` | `0.02` | Delay giữa mỗi token (giây) |
| `toi_da_client` | `100` | Số client đồng thời tối đa |
| `timeout` | `60.0` | Timeout mỗi stream (giây) |

### Thống kê

```python
stats = server.lay_thong_ke()
print(f"Tổng stream: {stats['tong_stream']}")
print(f"Client hiện tại: {stats['client_hien_tai']}")
print(f"Thời gian TB: {stats['thoi_gian_tb']:.2f}s")
```

---

## BoGioiHanTocDo

Bộ giới hạn tốc độ (rate limiter) cho API.

### Token Bucket (mặc định)

```python
from vietnamese_ai import BoGioiHanTocDo

limiter = BoGioiHanTocDo(
    go_i_y=100,
    cua_so=60.0,
    kich_thuoc_burst=150,
    che_do="token_bucket",
)

if limiter.cho_phep("client_1"):
    # Xử lý request
    pass
else:
    # Bị giới hạn - trả về 429
    pass
```

### Sliding Window

```python
limiter = BoGioiHanTocDo(
    go_i_y=100,
    cua_so=60.0,
    che_do="sliding_window",
)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `go_i_y` | `100` | Số request tối đa mỗi cửa sổ |
| `cua_so` | `60.0` | Kích thước cửa sổ (giây) |
| `kich_thuoc_burst` | `= go_i_y` | Số token tối đa (burst) |
| `che_do` | `"token_bucket"` | Thuật toán: `token_bucket`, `sliding_window` |

### Quản lý client

```python
con_lai = limiter.lay_con_lai("client_1")

limiter.reset("client_1")
limiter.xoa_client_khong_hoat_dong(timeout=300.0)

stats = limiter.lay_thong_ke()
print(f"Tỷ lệ từ chối: {stats['ty_le_tu_choi']:.2%}")
```

---

## Production Deployment Example

Kết hợp cả ba thành phần cho production:

```python
from vietnamese_ai import MayChuBatch, MayChuStream, BoGioiHanTocDo

limiter = BoGioiHanTocDo(go_i_y=1000, cua_so=60.0)

batch_server = MayChuBatch(
    mo_hinh=model,
    kich_thuoc_batch=32,
    timeout_batch=0.05,
    so_worker=4,
)
batch_server.bat_dau()

stream_server = MayChuStream(
    ham_sinh=generate_fn,
    toc_do_token=0.01,
    toi_da_client=50,
)

def handle_request(client_id, prompt, stream=False):
    if not limiter.cho_phep(client_id):
        return {"error": "Rate limit exceeded", "retry_after": 60}

    if stream:
        return stream_server.sinh_sse(prompt, client_id=client_id)
    else:
        return batch_server.du_doan(prompt)

stats = {
    "batch": batch_server.lay_thong_ke(),
    "stream": stream_server.lay_thong_ke(),
    "rate_limit": limiter.lay_thong_ke(),
}
```

---

## Performance Tips

| Mẹo | Mô tả |
|---|---|
| Tăng `kich_thuoc_batch` | Tăng throughput nhưng tăng latency |
| Giảm `timeout_batch` | Giảm latency nhưng giảm batching efficiency |
| Tăng `so_worker` | Tăng concurrency nhưng tốn tài nguyên |
| Dùng `token_bucket` | Cho phép burst traffic |
| Giám sát `queue_size` | Phát hiện bottleneck |
