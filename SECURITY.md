# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 9.0.x   | :white_check_mark: |
| 8.0.x   | :white_check_mark: |
| < 8.0   | :x:                |

## Reporting a Vulnerability

Nếu bạn phát hiện lỗ hổng bảo mật, vui lòng **KHÔNG** tạo Issue công khai.

### Quy trình báo cáo

1. **Email**: Gửi mô tả chi tiết đến `huynhduongphong9@gmail.com`
2. **Mô tả**: Bao gồm các bước tái hiện, ảnh hưởng tiềm tàng
3. **Phản hồi**: Chúng tôi sẽ phản hồi trong vòng 48 giờ
4. **Xử lý**: Fix sẽ được phát hành trong bản cập nhật tiếp theo

### Scope

Các lỗ hổng bảo mật được quan tâm:

- **Remote Code Execution (RCE)**: Thực thi mã từ xa qua model loading
- **Path Traversal**: Truy cập file ngoài phạm vi cho phép
- **Deserialization**: Unsafe pickle/JSON deserialization
- **Authentication Bypass**: Bỏ qua xác thực API keys/tokens
- **Data Leakage**: Rò rỉ dữ liệu người dùng
- **Denial of Service**: Tấn công từ chối dịch vụ qua API

### Out of Scope

- Lỗ hổng trong dependencies bên thứ 3 (scikit-learn, numpy, etc.)
- Social engineering attacks
- Lỗ hổng yêu cầu quyền truy cập vật lý

## Security Best Practices

Khi sử dụng Vietnamese AI Framework:

### Model Loading

```python
# Nên: Sử dụng API chính thức
mo_hinh = PhanLoai.tai("model.pkl")

# Không nên: Load pickle từ nguồn không tin cậy
import pickle
with open("untrusted.pkl", "rb") as f:
    model = pickle.load(f)  # NGUY HIỂM
```

### API Keys

```python
# Nên: Lưu API key trong environment variable
import os
api_key = os.environ.get("VAI_API_KEY")

# Không nên: Hardcode API key trong source code
api_key = "vai_abc123..."  # NGUY HIỂM
```

### Input Validation

```python
# Nên: Validate dữ liệu đầu vào
from vietnamese_ai.utils.validators import Validator
hop_le, thong_bao = Validator.kiem_tra_du_lieu_hop_le(X, y)
if not hop_le:
    raise ValueError(thong_bao)

# Không nên: Trust user input trực tiếp
mo_hinh.du_doan(user_data)  # Có thể crash hoặc exploit
```

### File Operations

```python
# Nên: Sử dụng pathlib và kiểm tra đường dẫn
from pathlib import Path
duong_dan = Path(duong_dan).resolve()
if not str(duong_dan).startswith(str(expected_dir)):
    raise PermissionError("Đường dẫn không hợp lệ")

# Không nên: Sử dụng string concatenation cho đường dẫn
path = base_dir + "/" + user_input  # Path traversal risk
```

## Security Features

Framework đã tích hợp sẵn:

- **Input validation**: `Validator` class kiểm tra dữ liệu đầu vào
- **RBAC**: `HeThongXacThuc` phân quyền admin/developer/viewer
- **Audit logging**: `NhatKyHoatDong` ghi lại mọi hoạt động
- **API key auth**: Xác thực API key cho SaaS platform
- **Quota enforcement**: Giới hạn tài nguyên theo gói dịch vụ
- **No unsafe deserialization**: Modules mới dùng JSON thay vì pickle

## Acknowledgments

Chúng tôi cảm ơn các nhà nghiên cứu bảo mật đã giúp cải thiện framework.
