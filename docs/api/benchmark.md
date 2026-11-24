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
