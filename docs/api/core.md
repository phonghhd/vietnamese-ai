# Core API Reference

## Engine

```python
class Engine:
    def __init__(self, ten="VietnameseAI", log_level="INFO")
    def huan_luyen(mo_hinh, X, y, ten_mo_hinh=None) -> dict
    def du_doan(mo_hinh, X) -> np.ndarray
    def danh_gia(mo_hinh, X, y) -> dict
    def lay_lich_su() -> list
    def lay_mo_hinh(ten) -> object
    def danh_sach_mo_hinh() -> list
```

## Pipeline

```python
class Pipeline:
    def __init__(self, ten="Pipeline")
    def them_buoc(ten, bo_xu_ly) -> Pipeline
    def fit(X, y=None) -> Pipeline
    def predict(X) -> np.ndarray
    def danh_gia(X, y) -> float
    def luu(duong_dan) -> None
    @classmethod
    def tai(duong_dan) -> Pipeline
    def danh_sach_buoc() -> list
    def lay_buoc(ten) -> object
```

## KiemDinhCheo

```python
class KiemDinhCheo:
    def __init__(self, so_fold=5, lap_lai=1, seed=42)
    def chay(mo_hinh, X, y, chi_so="do_chinh_xac", stratified=True) -> dict
```

## TimKiemThamSo

```python
class TimKiemThamSo:
    def __init__(self, so_fold=5, seed=42)
    def tim_kiem_luoi(lop_mo_hinh, luoi_tham_so, X, y, chi_so, tot_nhat_cao) -> dict
    def tim_kiem_ngau_nhien(lop_mo_hinh, pham_vi_tham_so, X, y, so_lan, chi_so) -> dict
```

## AutoML

```python
class AutoML:
    def __init__(self, so_fold=5, chuan_hoa=True)
    def fit(X, y) -> dict
    def predict(X) -> np.ndarray
    def danh_gia(X, y) -> float
    def bao_cao() -> str
```
