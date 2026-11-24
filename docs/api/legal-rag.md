# Legal & Administrative RAG API Reference (`vietnamese_ai.rag.legal` / `vietnamese_ai.legal.rag`)

Gói RAG chuyên sâu cho văn bản quy phạm pháp luật, văn bản hành chính nhà nước và hợp đồng tiếng Việt (tải lười PEP 562). Extra tùy chọn cho PDF/OCR: `pip install "vietnamese-ai[legal-rag]"`.

## `BoToiUuNguCanhHanhChinh`

```python
class BoToiUuNguCanhHanhChinh:
    def __init__(
        embedder: Optional[BoNhungVanBanSieuToc] = None,
        bat_tuong_lua: bool = True,
    ) -> None
    def phan_loai_van_ban(van_ban: str) -> str
    def trich_xuat_can_cu_va_so_hieu(van_ban: str) -> Dict[str, List[str]]
    def trich_xuat_metadata_hanh_chinh(van_ban: str) -> Dict[str, Any]
    def phan_tich_cau_truc(van_ban: str) -> List[Dict[str, Any]]
    def phan_doan_giu_ngu_canh(
        van_ban: str,
        ten_tai_lieu: str = "Tài liệu",
        max_tokens: int = 250,
    ) -> List[Dict[str, Any]]
    def nap_tu_tep(
        duong_dan: Union[str, Path],
        ten_tai_lieu: Optional[str] = None,
        max_tokens: int = 250,
    ) -> Dict[str, Any]
    def truy_hoi_phap_ly(
        cau_hoi: str,
        cac_chunks: List[Dict[str, Any]],
        top_k: int = 3,
        trong_so_hybrid: float = 0.5,
    ) -> List[Dict[str, Any]]
    def tao_prompt_phap_ly(
        cau_hoi: str,
        ket_qua_truy_hoi: List[Dict[str, Any]],
        ten_mau: str = "tra_cuu_dieu_khoan",
        **kwargs: Any,
    ) -> str
```

Alias: `VietnameseAdministrativeRAG`.

## `BoDocTaiLieuPhapLy`

```python
class BoDocTaiLieuPhapLy:
    def __init__(bat_bao_ve_ocr: bool = True) -> None
    def lam_sach_van_ban_phap_ly(van_ban: str) -> str
    def doc_tep(duong_dan: Union[str, Path]) -> Dict[str, Any]
```

Hỗ trợ `.txt`, `.md`, `.docx` (ưu tiên `python-docx`, dự phòng OOXML chuẩn Python kèm bảng biểu Markdown), `.pdf` (ưu tiên `pdfplumber` trích xuất bảng biểu, dự phòng `pypdf`), ảnh scan `.png`/`.jpg`/`.jpeg`/`.bmp`/`.tif`/`.tiff` (qua `TrichXuatVanBanAnh`) và `.json`/`.jsonl`. Alias: `LegalDocumentParser`.

## Tích hợp `RAGPipeline` & Chỉ mục Vector IVF-Flat

```python
class RAGPipeline:
    def them_tep(
        duong_dan: Union[str, Path],
        ma: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> int
    def them_thu_muc(
        thu_muc: Union[str, Path],
        de_quy: bool = False,
        duoi_tep: Optional[List[str]] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> Dict[str, Any]
    def toi_uu_chi_muc_vector(
        so_cum: int = 16,
        so_vong_lap: int = 8,
        nprobe: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]
```

## Kho Dữ Liệu Pháp Lý Mở Việt Nam (`vietnamese_ai.rag.legal.open_corpus`)

```python
def tai_bo_phap_ly_mo() -> List[Dict[str, Any]]
def tai_bo_hoi_dap_phap_ly_mo() -> List[Dict[str, Any]]
```

Bao gồm **100 điều khoản pháp luật Việt Nam thực tế** (Bộ luật Lao động 2019, Bộ luật Dân sự 2015, Luật Doanh nghiệp 2020, Luật Đất đai 2024, Luật Giao dịch điện tử 2023, Nghị định 13/2023/NĐ-CP, Nghị định 30/2020/NĐ-CP, Luật BHXH 2024, Luật Thuế GTGT 2024, Luật Quản lý thuế 2019, Luật SHTT, Luật Đấu thầu 2023, Luật Nhà ở 2023, Luật KDBĐS 2023, Nghị định 70/2025/NĐ-CP...) cùng **35 cặp Hỏi–Đáp (QA) chuẩn đánh giá** và tích hợp nạp một dòng qua `TroLyPhapLy(nap_kho_mo=True)`.

## Theo Dõi Hiệu Lực Văn Bản Pháp Luật (`BoTheoDoiHieuLucPhapLy`)

```python
@dataclass
class HoSoHieuLucVanBan:
    so_hieu: str
    ten_van_ban: str
    linh_vuc: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    ngay_het_hieu_luc: Optional[str] = None
    thay_the_cho: List[str] = ...
    bi_thay_the_boi: List[str] = ...
    sua_doi_bo_sung_boi: List[str] = ...
    ghi_chu: str = ""

class BoTheoDoiHieuLucPhapLy:
    def __init__(du_lieu_bo_sung: Optional[List[HoSoHieuLucVanBan]] = None) -> None
    def dang_ky_van_ban(ho_so: HoSoHieuLucVanBan) -> None
    def danh_sach_van_ban(linh_vuc: Optional[str] = None) -> List[Dict[str, Any]]
    def kiem_tra_hieu_luc(
        so_hieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]
    def trich_xuat_so_hieu(van_ban: str) -> List[str]
    def kiem_tra_hieu_luc_trong_van_ban(
        van_ban: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]
    def bo_sung_hieu_luc_cho_rag(
        ket_qua_rag: List[Dict[str, Any]],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        loai_bo_het_hieu_luc: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]
```

Tự động kích hoạt trong `RAGPipeline(che_do_hanh_chinh=True).hoi()` (gắn `hieu_luc_van_ban` & `canh_bao_hieu_luc` vào `trich_dan_phap_ly`), trong `TroLyPhapLy.kiem_tra_hieu_luc()` và qua Web Studio endpoints `POST /api/legal/check-validity` & `POST /api/rag/upload-file` (kéo-thả `.docx`/`.pdf`/ảnh OCR). Alias: `LegalValidityTracker`, `LegalValidityRecord`.

## `BoMauPromptPhapLy`

```python
class BoMauPromptPhapLy:
    def danh_sach_mau() -> List[Dict[str, Any]]
    def lay_mau(ten_mau: str) -> MauPrompt
    def them_mau(ten_mau: str, mau_prompt: MauPrompt) -> None
    def tao_prompt(ten_mau: str, **kwargs: Any) -> str
    def tao_prompt_tu_truy_hoi(
        cau_hoi: str,
        ket_qua_truy_hoi: List[Dict[str, Any]],
        ten_mau: str = "tra_cuu_dieu_khoan",
        **kwargs: Any,
    ) -> str
```

Các mẫu tích hợp: `tra_cuu_dieu_khoan`, `ra_soat_hop_dong`, `tom_tat_van_ban_phap_ly`, `so_sanh_quy_dinh`, `soan_thao_cong_van`, `kiem_tra_tuan_thu_du_lieu`. Alias: `LegalPromptLibrary`.

## Đồ Thị Tham Chiếu Pháp Lý (`DoThiThamChieuPhapLy`)

```python
@dataclass(frozen=True)
class QuanHePhapLy:
    nguon: str
    dich: str
    loai_quan_he: str
    dieu_lien_quan: str = ""
    mo_ta: str = ""

class DoThiThamChieuPhapLy:
    def __init__(bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None) -> None
    def them_van_ban(ho_so: HoSoHieuLucVanBan) -> None
    def them_quan_he(quan_he: QuanHePhapLy) -> bool
    def trich_xuat_va_them_tu_van_ban(so_hieu_nguon: str, noi_dung: str) -> List[QuanHePhapLy]
    def truy_van_van_ban_lien_quan(
        so_hieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        chi_lay_con_hieu_luc: bool = False,
        do_sau: int = 2,
    ) -> Dict[str, Any]
    def mo_rong_ngu_canh_rag(
        so_hieu_list: List[str],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]
    def xuat_do_thi_mermaid(
        so_hieu_trung_tam: Optional[str] = None,
        do_sau: int = 2,
    ) -> str
    def thong_ke() -> Dict[str, Any]
```

Liên kết tự động `Luật ↔ Nghị định hướng dẫn ↔ Thông tư ↔ Sửa đổi/Thay thế`, tích hợp trong `RAGPipeline(che_do_hanh_chinh=True).hoi()`, `TroLyPhapLy.tra_cuu_van_ban_lien_quan()`, CLI `vai legal-graph <so_hieu>` và Web Studio `POST /api/legal/citation-graph`. Alias: `LegalCitationGraph`, `LegalRelationEdge`.

## Rà Soát & So Sánh Hợp Đồng (`BoSoSanhRaSoatHopDong`)

```python
class BoSoSanhRaSoatHopDong:
    def __init__(
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
        do_thi_tham_chieu: Optional[DoThiThamChieuPhapLy] = None,
    ) -> None
    def quet_rui_ro_don_le(van_ban_hop_dong: str, ngay_tra_cuu: Optional[Union[str, date]] = None) -> Dict[str, Any]
    def so_sanh_va_ra_soat(
        hop_dong_doi_tac: Union[str, Path],
        hop_dong_chuan: Optional[Union[str, Path]] = None,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]
    def xuat_bao_cao_markdown(ket_qua: Dict[str, Any]) -> str
    def xuat_bao_cao_docx(ket_qua: Dict[str, Any], duong_dan_xuat: Union[str, Path]) -> Path
```

Hỗ trợ đọc `.docx`/`.pdf`/`.txt`, sinh báo cáo Redline Markdown & `.docx` (OOXML), tích hợp qua CLI `vai contract-review` và Web Studio `POST /api/legal/contract-review`. Alias: `ContractReviewEngine`.
