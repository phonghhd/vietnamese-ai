# Preprocessing API Reference

## XuLyVanBan

```python
class XuLyVanBan:
    def __init__(self, tu_dung=None, su_dung_underthesea=True)
    def chuan_hoa(text) -> str
    def tach_tu(text) -> list
    def tach_tu_chuoi(text) -> str
    def loai_bo_tu_dung(text) -> str
    def xu_ly_day_du(text) -> str
    def tao_tu_dien(cac_van_ban) -> dict
    def ma_hoa_bo_dem(text) -> np.ndarray
    def ma_hoa_tfidf(cac_van_ban) -> np.ndarray
    def trich_xuat_tu_khoa(text, top_n=5) -> list
    def gan_nhan_tu_loai(text) -> list  # underthesea
    def phan_tich_cam_xuc(text) -> str  # underthesea
```

## XuLySo

```python
class XuLySo:
    def __init__(self, phuong_phap="minmax")
    def fit(data) -> XuLySo
    def transform(data) -> np.ndarray
    def fit_transform(data) -> np.ndarray
    def chuan_hoa_minmax(data, fit=True) -> np.ndarray
    def chuan_hoa_zscore(data, fit=True) -> np.ndarray
    @staticmethod
    def xu_ly_gia_tri_thieu(data, phuong_phap="trung_vi") -> np.ndarray
    @staticmethod
    def ma_hoa_nhan(nhan) -> tuple
    @staticmethod
    def ma_hoa_onehot(nhan, so_lop=None) -> np.ndarray
    @staticmethod
    def chia_du_lieu(X, y, ty_le_test=0.2, ngau_nhien=42) -> tuple
```

## TaoDacTrung

```python
class TaoDacTrung:
    @staticmethod
    def dac_trung_da_thuc(X, bac=2) -> np.ndarray
    @staticmethod
    def dac_trung_tuong_tac(X) -> np.ndarray
    @staticmethod
    def giam_chieu_pca(X, so_chieu=2) -> np.ndarray
    @staticmethod
    def chon_dac_trung_phuong_sai(X, nguong=0.0) -> tuple
```
