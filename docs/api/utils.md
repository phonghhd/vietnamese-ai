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

## QuanLyPhuThuoc (`DependencyManager`)

```python
class QuanLyPhuThuoc:
    @classmethod
    def bat_tu_dong_cai(kich_hoat: bool = True) -> bool
    @classmethod
    def dang_bat_tu_dong_cai() -> bool
    @classmethod
    def phat_hien_phan_cung_cai_dat() -> Dict[str, Any]
    @classmethod
    def chon_goi_theo_phan_cung(nhom: str) -> Tuple[List[str], List[str]]
    @classmethod
    def tu_dong_bo_sung_tang_thap(chay_that: bool = False) -> Dict[str, Any]
    @classmethod
    def thiet_lap_tu_dong(profile: str = "core", chay_that: bool = False, bien_dich_rust: bool = True) -> Dict[str, Any]
    @classmethod
    def da_cai(ten_goi_hoac_module: str) -> bool
    @classmethod
    def kiem_tra_xung_dot_phu_thuoc() -> Dict[str, Any]
    @classmethod
    def yeu_cau(ten_module: str, nhom_extra: str = "ml", tinh_nang: str = "", tu_dong_cai: Optional[bool] = None) -> Any
    @classmethod
    def nhap_tuy_chon(ten_module: str, nhom_extra: str = "ml", tinh_nang: str = "") -> Optional[Any]
    @classmethod
    def cai_dat_nhom(nhom: str, chay_that: bool = False, dung_cache_neu_can: bool = False) -> Dict[str, Any]
    @classmethod
    def chan_doan_he_thong() -> Dict[str, Any]
```

Hỗ trợ cơ chế tự động cấp phát 3 tầng Opt-in (`vai setup --profile <...> --apply` hoặc biến môi trường `VAI_AUTO_INSTALL=1`), ưu tiên `uv`, tự động nhận diện PEP 668 (`~/.cache/vietnamese-ai/runtime/`) và tự biên dịch/nạp Rust Core (`abi3-py39`).

## Feature Status & Modular Registry (`vietnamese_ai.utils.feature_status`)

```python
class TrangThaiTinhNang:
    CORE = "core"
    EXTENSION = "extension"
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    SIMULATED = "simulated"
    EXTERNAL_DEPENDENCY = "external_dependency"

def gan_nhan_tinh_nang(trang_thai, mo_ta="", phu_thuoc_ngoai=None, nhom_extra=None, la_mo_phong=False)
def lay_thong_tin_tinh_nang(doi_tuong) -> Optional[Dict[str, Any]]
def liet_ke_tinh_nang(trang_thai=None) -> List[Dict[str, Any]]
```

## Semantic Versioning & Sub-Package Release (`vietnamese_ai.utils.versioning`)

```python
class PhienBanSemVer:
    @classmethod
    def phan_tich(chuoi_phien_ban: str) -> PhienBanSemVer
    @classmethod
    def hop_le(chuoi_phien_ban: str) -> bool
    def tang_patch() -> PhienBanSemVer
    def tang_minor() -> PhienBanSemVer
    def tang_major() -> PhienBanSemVer
    def tuong_thich_api(phien_ban_khac: PhienBanSemVer) -> bool

class QuanLyPhatHanhGoi:
    @staticmethod
    def danh_sach_goi() -> Dict[str, Dict[str, Any]]
    @staticmethod
    def tao_noi_dung_pyproject(ten_goi: str, phien_ban: Optional[str] = None) -> str
    @classmethod
    def xuat_pyproject_goi_con(ten_goi: str, thu_muc_goc: Union[str, Path], phien_ban: Optional[str] = None) -> Path
    @classmethod
    def kiem_tra_tuan_thu_semver(thu_muc_du_an: Optional[Union[str, Path]] = None) -> Dict[str, Any]
    @classmethod
    def kiem_tra_san_sang_phat_hanh(thu_muc_du_an: Optional[Union[str, Path]] = None) -> Dict[str, Any]
```

## Trình Chạy Demo 5 Phút & Kiểm Định Phát Hành CLI (`vai demo` / `vai release-check`)

```bash
vai demo                # Chạy đủ 4 kịch bản: Legal RAG 100 điều khoản, Edge CPU, Bảo mật & GPU Flywheel
vai demo legal-rag      # Riêng RAG pháp lý + kiểm toán hiệu lực văn bản hết hạn
vai release-check       # Kiểm định SemVer, xung đột phụ thuộc, .pyi type stubs, Rust abi3-py39
vai release-check --json
```
