# Vietnamese NLP Core API Reference (`vietnamese_ai.nlp`)

Gói lõi xử lý ngôn ngữ tự nhiên tiếng Việt (tải lười PEP 562). Phụ thuộc nâng cao tùy chọn: `pip install "vietnamese-ai[vietnamese-nlp]"`.

## `BoTachTuTiengVietSieuToc`

```python
class BoTachTuTiengVietSieuToc:
    def __init__(
        tu_dien_bo_sung: Optional[Set[str]] = None,
        su_dung_underthesea: bool = False,
    ) -> None
    @property
    def co_underthesea() -> bool
    def them_tu_dien(ds_tu: List[str]) -> None
    def chuan_hoa_unicode(van_ban: str) -> str
    def chuan_hoa_dau_thanh(van_ban: str) -> str
    def xoa_dau_tieng_viet(van_ban: str) -> str
    def tach_tu(
        van_ban: str,
        noi_tu_ghep: bool = False,
        ky_tu_noi: str = "_",
        chuan_hoa_dau: bool = True,
    ) -> List[str]
    def dem_so_tu(van_ban: str, noi_tu_ghep: bool = True, su_dung_rust: bool = True) -> int
    def tim_tu_khoa_nhanh(van_ban: str, tu_khoa: str) -> List[int]
    def ma_hoa(
        van_ban: str,
        them_cls_sep: bool = False,
        do_dai_toi_da: Optional[int] = None,
    ) -> List[int]
    def giai_ma(ids: List[int], bo_special: bool = True) -> str
    def pad(ids: List[int], do_dai: int, gia_tri_pad: int = 0) -> List[int]
```

Aliases: `FastVietnameseTokenizer`, `NativeVietnameseTokenizer`.

## `BoNhungVanBanSieuToc`

```python
class BoNhungVanBanSieuToc:
    def __init__(
        so_chieu: int = 256,
        tokenizer: Optional[BoTachTuTiengVietSieuToc] = None,
    ) -> None
    def nhung(van_ban: str) -> np.ndarray
    def nhung_lo(ds_van_ban: List[str]) -> np.ndarray
    def do_tuong_dong(vb1: str, vb2: str) -> float
    def tim_kiem_tuong_dong_hang_loat(
        query_vec: Union[np.ndarray, List[float]],
        ma_tran_vectors: Union[np.ndarray, List[List[float]]],
    ) -> np.ndarray
```

Alias: `FastVietnameseEmbedder`.

## `GanNhanTuLoai` (VLSP POS Tagger)

```python
class GanNhanTuLoai:
    def __init__(
        su_dung_underthesea: bool = True,
        tokenizer: Optional[BoTachTuTiengVietSieuToc] = None,
        tu_dien_bo_sung: Optional[Dict[str, str]] = None,
    ) -> None
    @property
    def co_underthesea() -> bool
    def gan_nhan(van_ban: str) -> List[Tuple[str, str]]
    def gan_nhan_chi_tiet(van_ban: str) -> List[Dict[str, Any]]
    def gan_nhan_lo(ds_van_ban: List[str]) -> List[List[Tuple[str, str]]]
    def thong_ke() -> Dict[str, Any]
```

Alias: `VietnamesePOSTagger`. Hỗ trợ bộ nhãn VLSP: `N`, `Np`, `Nc`, `Nu`, `V`, `A`, `R`, `P`, `L`, `M`, `E`, `C`, `T`, `CH`.

## `NhanDienThucThe` (Vietnamese NER)

```python
class NhanDienThucThe:
    def __init__(
        su_dung_underthesea: bool = True,
        mau_tuy_chinh: Optional[Dict[str, List[str]]] = None,
        tu_dien_tuy_chinh: Optional[Dict[str, Set[str]]] = None,
    ) -> None
    def nhan_dien(van_ban: str, loai_loc: Optional[List[str]] = None) -> List[Dict[str, Any]]
    def them_dia_danh(*ten: str) -> None
    def them_chuc_danh(*ten: str) -> None
    def them_to_chuc(*ten: str) -> None
    def them_mau(loai: str, mau: str) -> None
    def thong_ke() -> Dict[str, Any]
```

Các loại thực thể tích hợp sẵn: `NGAY_THANG`, `SO_DIEN_THOAI`, `EMAIL`, `URL`, `TIEN_TE`, `DIA_CHI`, `CCCD`, `MA_SO_THUE`, `VAN_BAN_PHAP_LUAT`, `TO_CHUC`, `NGUOI`, `DIA_DANH`, `CHUC_DANH`, `PERSON` (khi cài `underthesea`).

## `PhoBERTWrapper`

```python
class PhoBERTWrapper:
    def __init__(
        model_name: str = "vinai/phobert-base",
        che_do_mo_phong: bool = False,
        kich_thuoc_an: int = 768,
        tu_dong_tach_tu: bool = True,
        phuong_phap_gop: str = "mean",  # "mean" | "cls" | "max"
        thiet_bi: str = "auto",         # "auto" | "cpu" | "cuda"
        do_dai_toi_da: int = 256,
    ) -> None
    def tien_xu_ly_tieng_viet(van_ban: str) -> str
    def tai_mo_hinh() -> None
    def ma_hoa(cac_van_ban: List[str], batch_size: int = 32) -> np.ndarray
    def ma_hoa_van_ban(text: str) -> np.ndarray
    def do_tuong_dong(text1: str, text2: str) -> float
    def tim_tu_giong(text: str, cac_van_ban: List[str], top_n: int = 5) -> List[Tuple[str, float]]
    def huan_luyen_dau_phan_loai(
        ds_van_ban: List[str],
        ds_nhan: List[str],
        he_so_chinh_quy: float = 1.0,
    ) -> Dict[str, Any]
    def du_doan_phan_loai(ds_van_ban: List[str]) -> List[Dict[str, Any]]
```

## Tiện ích quản lý phụ thuộc NLP

```python
from vietnamese_ai.nlp import XuLyVanBan, TangCuongVanBan, kiem_tra_phu_thuoc_nlp

thong_tin = kiem_tra_phu_thuoc_nlp()
```
