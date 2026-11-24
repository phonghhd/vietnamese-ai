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

## QuyTrinhDauCuoi (`HeSinhThaiAI` / `HeThongAI`) & Production API

```python
class QuyTrinhDauCuoi:
    def mo_kho_tri_thuc_vaidb(duong_dan_vaidb: Union[str, Path], dung_onnx_semantic: bool = True, che_do_hanh_chinh: bool = True) -> RAGPipeline
    def nap_tep_phap_ly(duong_dan: Union[str, Path], che_do_hanh_chinh: bool = True, toi_uu_ivf: bool = False) -> Dict[str, Any]
    def ra_soat_hop_dong(hop_dong_doi_tac: Union[str, Path], hop_dong_chuan: Optional[Union[str, Path]] = None, ngay_tra_cuu: Optional[str] = None, xuat_docx: Optional[Union[str, Path]] = None) -> Dict[str, Any]
    def kiem_dinh_mo_hinh(ung_vien: Optional[Any] = None, ban_goc: Optional[Any] = None, gioi_han_mau: int = 8, nguong_dat: float = 0.55) -> Dict[str, Any]
    def tao_mcp_server(ten: str = "vietnamese-ai-mcp", nap_kho_phap_ly_mo: bool = True, quan_ly_plugin: Optional[Any] = None) -> MayChuMCPThucTe
    def tao_webhook_kenh(kenh: str = "both", host: str = "0.0.0.0", port: int = 8095, bo_xu_ly: Optional[Any] = None, quan_ly_workspace: Optional[Any] = None) -> MayChuWebhookKenh
    def tai_mo_hinh_edge(ten_mo_hinh: str, muc_luong_tu: str = "auto", chay_that: bool = False) -> Dict[str, Any]
    def tinh_chinh_gpu_xa(nguon_du_lieu: Any, nha_cung_cap: str = "modal", chay_that: bool = False, **kwargs: Any) -> Dict[str, Any]
    def khoi_tao_edge(duong_dan_mo_hinh: Optional[Union[str, Path]] = None, so_luong_thread: int = 4, do_dai_toi_da: int = 128) -> DongCoSuyLuanEdge
    def sinh_stream(prompt: str, do_dai_toi_da: int = 128, nhiet_do: float = 0.7) -> Iterator[str]
    def tao_api_server(ten: Optional[str] = None, quan_ly_plugin: Optional[Any] = None, **kwargs: Any) -> FastAPIServer
```

`FastAPIServer` hỗ trợ chuẩn OpenAI (nhận trực tiếp `TacTu`, `RAGPipeline`, `TroLyPhapLy`, `TongDaiCSKH`, `DongCoSuyLuanEdge` và `QuyTrinhDauCuoi`):
- `POST /v1/chat/completions`: SSE streaming thời gian thực từng token từ `sinh_stream()`, OpenAI Tool Calling JSON Schema tự khám phá từ `TacTu`/`QuanLyPlugin`.
- `GET /v1/tools`: Liệt kê công cụ đã đăng ký.
- `GET /metrics`: Xuất số liệu Prometheus Text Exposition Format qua `BoThuThapTelemetryAPI` & OpenTelemetry tracing.

## Sub-modules Lõi Mới (Modular Architecture)

- `vietnamese_ai.core.preprocessing`: `XuLySo`, `TaoDacTrung` (thuần NumPy, không kéo `underthesea`).
- `vietnamese_ai.core.edge`: `DongCoSuyLuanEdge`, `BoGiaiLuongTuQ4` / `BoGiaiLuongTuGGUF`, `GGUFMemoryMappedStreamer`, `CauNoiCtypesNative`, `KiemTraMoiTruongNative`, `thong_tin_kien_truc_cpu`.
- `vietnamese_ai.extensions`: Namespace tải lười cho 11 phân hệ mở rộng (`vision`, `timeseries`, `federated`, `fine_tuning`, `transformer`, `llm`, `nlp`, `mobile`, `web3`, `robotics`, `extreme`).
