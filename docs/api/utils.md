# Utils API Reference

## Logger

```python
class Logger:
    def __init__(ten="VietnameseAI", level="INFO")
    def info(tin_nhan) -> None
    def error(tin_nhan) -> None
    def warning(tin_nhan) -> None
    def debug(tin_nhan) -> None
```

## Metrics

```python
class Metrics:
    @staticmethod
    def do_chinh_xac(y_thuc, y_du_doan) -> float
    @staticmethod
    def mse(y_thuc, y_du_doan) -> float
    @staticmethod
    def rmse(y_thuc, y_du_doan) -> float
    @staticmethod
    def mae(y_thuc, y_du_doan) -> float
    @staticmethod
    def r2_score(y_thuc, y_du_doan) -> float
    @staticmethod
    def precision_recall_f1(y_thuc, y_du_doan, lop_pos=1) -> dict
    @staticmethod
    def bao_cao_phan_loai(y_thuc, y_du_doan) -> dict
    @staticmethod
    def bao_cao_hoi_quy(y_thuc, y_du_doan) -> dict
```

## Validator

```python
class Validator:
    @staticmethod
    def kiem_tra_kich_thuoc(data, kich_thuoc_mong_duoi) -> bool
    @staticmethod
    def kiem_tra_gia_tri_thieu(data) -> bool
    @staticmethod
    def kiem_tra_du_lieu_hop_le(X, y=None, cho_phep_nan=False) -> tuple
    @staticmethod
    def kiem_tra_nhiem_vu(y) -> str  # "phan_loai" hoặc "hoi_quy"
```

## LuuTai

```python
class LuuTai:
    @staticmethod
    def luu_mo_hinh(mo_hinh, duong_dan) -> None
    @staticmethod
    def tai_mo_hinh(duong_dan) -> object
    @staticmethod
    def luu_numpy(duong_dan, **matran) -> None
    @staticmethod
    def tai_numpy(duong_dan) -> dict
    @staticmethod
    def luu_json(data, duong_dan) -> None
    @staticmethod
    def tai_json(duong_dan) -> dict
```
