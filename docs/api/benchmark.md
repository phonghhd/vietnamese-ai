# Benchmark API Reference (`vietnamese_ai.benchmark`)

Hệ thống đo lường định lượng tái lập cho hiệu năng CPU, suy luận LLM đồng thời, độ ổn định tài nguyên dài hạn, khả năng mở rộng cụm Swarm TCP LAN và tập dữ liệu tiếng Việt 5 miền.

## `BoDoLuongDoiDau` / `ComparativeBenchmarkSuite`

```python
class BoDoLuongDoiDau:
    @staticmethod
    def thu_thap_thong_tin_moi_truong(seed: int = 42) -> Dict[str, Any]
    def do_toc_do_tach_tu(so_vong_lap: int = 50) -> float
    def do_ram_tieu_thu() -> float
    def do_thoi_gian_khoi_dong() -> float
    def do_do_chinh_xac_rag_phap_ly() -> Dict[str, float]
    def do_hieu_nang_suy_luan_llm(
        so_yeu_cau: int = 24,
        so_luong_dong_thoi: int = 4,
        kich_thuoc_mo_hinh: Sequence[int] = (256, 512, 1024),
        seed: int = 42,
    ) -> Dict[str, Any]
    def do_do_on_dinh_tai_nguyen(
        so_chu_ky: int = 20,
        khoang_lay_mau: int = 5,
    ) -> Dict[str, Any]
    def do_kha_nang_mo_rong_swarm(
        danh_sach_so_node: Sequence[int] = (1, 2, 4),
        tong_so_layer: int = 8,
        bat_lan_thuc_te: bool = True,
        seed: int = 42,
    ) -> Dict[str, Any]
    def do_da_dang_mien_tieng_viet() -> Dict[str, Any]
    def chay_benchmark_toan_dien() -> Dict[str, Any]
    def chay_benchmark_mo_rong(
        nhanh: bool = False,
        seed: int = 42,
        bat_lan_thuc_te: bool = True,
    ) -> Dict[str, Any]
    def xuat_bang_so_sanh_markdown(du_lieu: Optional[Dict[str, Any]] = None) -> str
    def xuat_bao_cao_mo_rong_markdown(du_lieu: Optional[Dict[str, Any]] = None) -> str
    def xuat_bang_so_sanh_latex(du_lieu: Optional[Dict[str, Any]] = None) -> str
    def xuat_du_lieu_json(duong_dan: Optional[Union[str, Path]] = None) -> Dict[str, Any]
```

## Tập dữ liệu tiếng Việt đa miền (`vietnamese_ai.benchmark.datasets_vn`)

- `TAP_DU_LIEU_DA_MIEN_VN`: 5 lĩnh vực (`phap_ly_hanh_chinh`, `tai_chinh_ngan_hang`, `y_te_suc_khoe`, `cong_nghe_ai`, `thuong_mai_dien_tu`).
- `BO_TRUY_HOI_DA_MIEN_VN`: Cặp câu hỏi – tài liệu đích cho đánh giá Multi-Domain RAG.
- `BO_KIEM_THU_NER_VN`: Tập câu gán nhãn thực thể Việt Nam (CCCD, MST, Tổ chức, Văn bản pháp luật, Người, Địa danh).
- `BO_KIEM_THU_CAM_XUC_VN`: Tập câu đánh giá cảm xúc tiếng Việt.

## Bộ Chuẩn Học Thuật `ViLegal-TempBench` (`BoDanhGiaLegalTemporalRAG`)

```python
def tao_ham_sinh_llm(
    nha_cung_cap: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    nhiet_do: float = 0.0,
    timeout: float = 45.0,
    do_tre_giua_goi_giay: float = 0.0,
    so_lan_thu_lai: int = 2,
) -> Callable[[str], str]

class BoDanhGiaLegalTemporalRAG:
    def chay_ablation(
        cac_che_do: Optional[Sequence[str]] = None,
        nhanh: bool = False,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]
    def xuat_bao_cao_markdown(ket_qua: Dict[str, Any]) -> str
    def xuat_bang_latex(ket_qua: Dict[str, Any]) -> str
```

Alias: `ViLegalTemporalBenchmark`. Hỗ trợ 5 cấu hình Ablation (`bm25`, `hybrid`, `temporal`, `temporal_graph`, `full`), kiểm định thống kê `kiem_dinh_bootstrap_cap` và xuất bảng LaTeX `booktabs` qua `vai benchmark --suite legal-temporal --latex`.

## Bộ Chuẩn Hệ Thống Zero-GPU `ViEdge-SysBench` (`BoDanhGiaHeThongEdge`)

```python
class BoDanhGiaHeThongEdge:
    def do_tim_kiem_vector_luong_tu(
        n_vectors: int = 20000,
        dim: int = 384,
        n_queries: int = 80,
        top_k: int = 10,
        seed: int = 42,
    ) -> Dict[str, Any]
    def do_bitnet_158_dong_goi_2bit(
        cac_kich_thuoc: Sequence[Tuple[int, int]] = ((512, 512), (1024, 1024), (2048, 2048)),
        n_rows: int = 16,
        so_vong_lap: int = 8,
        seed: int = 42,
    ) -> Dict[str, Any]
    def do_chuoi_zero_gpu_dau_cuoi(
        so_tai_lieu: int = 200,
        dim: int = 128,
        seed: int = 42,
    ) -> Dict[str, Any]
    def chay_toan_dien(
        nhanh: bool = False,
        n_vectors: int = 20000,
        dim: int = 384,
        seed: int = 42,
    ) -> Dict[str, Any]
    def xuat_bao_cao_markdown(ket_qua: Dict[str, Any]) -> str
    def xuat_bang_latex(ket_qua: Dict[str, Any]) -> str
```

Alias: `ViEdgeSystemsBenchmark`. Tự động đối đầu thêm `faiss` và `hnswlib` nếu máy có cài, xuất Markdown & LaTeX `booktabs` qua `vai benchmark --suite edge-systems [--scale N] [--dim D] [--latex]`.

## Bộ Chuẩn Continual Alignment `ViAlign-SafeBench` (`BoDanhGiaCanChinhAnToan`)

```python
class BoDanhGiaCanChinhAnToan:
    def chay_ablation_da_vong(
        so_vong: int = 5,
        nhanh: bool = False,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]
    def xuat_bao_cao_markdown(ket_qua: Dict[str, Any]) -> str
    def xuat_bang_latex(ket_qua: Dict[str, Any]) -> str
```

Alias: `ViAlignSafeBenchmark`. Đối chứng 4 chiến lược (`naive_dpo`, `filtered_dpo`, `filtered_replay_cdpo`, `vialign_safeloop`) kèm Paired Bootstrap 95% CI và xuất LaTeX `booktabs` qua `vai benchmark --suite align-safety [--rounds 5] [--latex]`.
