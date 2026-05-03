# NAS API Reference

## TimKiemKienTruc

```python
class TimKiemKienTruc:
    """Neural Architecture Search."""

    def __init__(so_fold=3, seed=42, diem_toi_thieu=0.5, toi_da_hoa_do_phuc_tap=False)
    def tim_kiem_ngau_nhien(X, y, pham_vi=None, so_lan=10) -> dict
    def tim_kiem_luoi(X, y, luoi_tham_so=None) -> dict
    def so_sanh_voi_ml_truyen_thong(X, y) -> dict
    def bao_cao() -> str
```

### Phạm vi mặc định

```python
PHAM_VI_MAC_DINH = {
    "so_lop_an": [1, 2, 3],
    "so_neron_lop": [8, 16, 32, 64],
    "ham_kich_hoat": ["relu", "sigmoid", "tanh"],
    "so_vong": [50, 100, 200],
    "toc_do_hoc": [0.001, 0.01, 0.1],
}
```

### Returns

```python
{
    "kien_truc_tot_nhat": dict,  # {"lop_an": [...], "ham_kich_hoat": str, ...}
    "diem_tot_nhat": float,
    "so_lan_thu": int,
    "thanh_cong": int,
    "that_bai": int,
    "tong_thoi_gian": float,
    "tat_ca_ket_qua": list
}
```
