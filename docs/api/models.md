# Models API Reference

## PhanLoai

```python
class PhanLoai(BaseModel):
    def __init__(self, thuat_toan="logistic", ten=None, **kwargs)
    def huan_luyen(X, y) -> None
    def du_doan(X) -> np.ndarray
    def du_doan_xac_suat(X) -> np.ndarray
    def danh_gia(X, y) -> float
    def bao_cao(X, y) -> dict
    def luu(duong_dan) -> None
    def tai(duong_dan) -> PhanLoai
```

## HoiQuy

```python
class HoiQuy(BaseModel):
    def __init__(self, thuat_toan="tuyen_tinh", ten=None, **kwargs)
    def huan_luyen(X, y) -> None
    def du_doan(X) -> np.ndarray
    def danh_gia(X, y) -> float  # MSE
    def bao_cao(X, y) -> dict    # mse, rmse, mae, r2
```

## PhanCum

```python
class PhanCum(BaseModel):
    def __init__(self, so_cum=3, thuat_toan="kmeans", ten=None, **kwargs)
    def huan_luyen(X, y=None) -> None
    def du_doan(X) -> np.ndarray
    def danh_gia(X) -> float      # Silhouette Score
    def lay_tam_cum() -> np.ndarray
```

## MangNron

```python
class MangNron(BaseModel):
    def __init__(self, lop_an=[64,32], ham_kich_hoat="relu", toc_do_hoc=0.01, so_vong=100)
    def huan_luyen(X, y) -> None
    def du_doan(X) -> np.ndarray
    def danh_gia(X, y) -> float
    def lay_lich_su_loss() -> list
```

## MoHinhTapHop

```python
class MoHinhTapHop(BaseModel):
    def __init__(self, loai="voting", cac_mo_hinh=None, nhiem_vu="phan_loai")
    def huan_luyen(X, y) -> None
    def du_doan(X) -> np.ndarray
    def danh_gia(X, y) -> float
```
