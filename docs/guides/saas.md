# Cloud Platform (SaaS)

## Nền tảng dịch vụ đám mây

`NenTangDichVu` cung cấp multi-tenant workspace, API keys, model deploy.

## Tạo workspace

```python
from vietnamese_ai import NenTangDichVu

ntdv = NenTangDichVu()
ws = ntdv.tao_workspace("AI Startup", "founder", goi_dich_vu="starter")
print(ws['ma'])  # workspace ID
```

## API Keys

```python
api_key = ntdv.tao_api_key(ws['ma'])
ws_info = ntdv.xac_thuc_api_key(api_key)
```

## Đăng ký và deploy model

```python
# Đăng ký model
model_info = ntdv.dang_ky_model(ws['ma'], "sentiment", mo_hinh)

# Deploy
deploy = ntdv.deploy_model(ws['ma'], model_info['ma'])

# Dự đoán qua deployment
ket_qua = ntdv.du_doan(ws['ma'], deploy['ma'], X_test[:5])
```

## Gói dịch vụ

| Gói | Models | Deployments | API Keys | Predictions/ngày |
|---|---|---|---|---|
| free | 3 | 1 | 1 | 100 |
| starter | 10 | 5 | 3 | 5,000 |
| pro | 50 | 20 | 10 | 100,000 |
| enterprise | unlimited | unlimited | unlimited | unlimited |

## Thống kê

```python
tk = ntdv.thong_ke_usage(ws['ma'])
print(f"Predictions: {tk['usage']['predictions']}")
```
