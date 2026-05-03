# Cloud Platform (SaaS) API Reference

## NenTangDichVu

```python
class NenTangDichVu:
    """Nền tảng dịch vụ AI đám mây."""

    def __init__(duong_dan="saas_data")
    def tao_workspace(ten, chu_so_huu, goi_dich_vu="free") -> dict
    def lay_workspace(ma) -> dict
    def danh_sach_workspaces() -> list
    def cap_nhat_goi(ma, goi_moi) -> dict
    def tao_api_key(ma_workspace) -> str
    def xac_thuc_api_key(api_key) -> dict
    def dang_ky_model(ma_workspace, ten_model, mo_hinh, mo_ta="") -> dict
    def lay_model(ma_workspace, ma_model) -> Any
    def deploy_model(ma_workspace, ma_model, ten_deployment="") -> dict
    def du_doan(ma_workspace, ma_deployment, du_lieu) -> dict
    def thong_ke_usage(ma_workspace) -> dict
    def xoa_workspace(ma) -> None
    def lay_goi_dich_vu() -> dict
```

### Gói dịch vụ

| Gói | max_models | max_deployments | max_api_keys | max_predictions/day |
|---|---|---|---|---|
| free | 3 | 1 | 1 | 100 |
| starter | 10 | 5 | 3 | 5,000 |
| pro | 50 | 20 | 10 | 100,000 |
| enterprise | -1 (unlimited) | -1 | -1 | -1 |
