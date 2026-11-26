"""Gói Đo Lường và So Sánh Benchmark Tái Lập - Vietnamese AI Framework (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

__all__ = [
    "BoDoLuongDoiDau",
    "ComparativeBenchmarkSuite",
    "BoDanhGiaHeThongEdge",
    "ViEdgeSystemsBenchmark",
    "BoDanhGiaCanChinhAnToan",
    "ViAlignSafeBenchmark",
    "tao_bo_du_lieu_can_chinh_da_vong",
    "BoDanhGiaLegalTemporalRAG",
    "ViLegalTemporalBenchmark",
    "kiem_dinh_bootstrap_cap",
    "tao_ham_sinh_llm",
    "tinh_ndcg_at_k",
    "TAP_DU_LIEU_DA_MIEN_VN",
    "BO_TRUY_HOI_DA_MIEN_VN",
    "BO_KIEM_THU_NER_VN",
    "BO_KIEM_THU_CAM_XUC_VN",
    "tai_bo_phap_ly_mo",
    "tai_bo_hoi_dap_phap_ly_mo",
    "tai_bo_phap_ly_thoi_gian",
    "tai_canh_do_thi_phap_ly_thoi_gian",
    "tai_bo_hoi_dap_phap_ly_thoi_gian",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "BoDoLuongDoiDau": ("vietnamese_ai.benchmark.comparative_benchmark", "BoDoLuongDoiDau"),
        "ComparativeBenchmarkSuite": (
            "vietnamese_ai.benchmark.comparative_benchmark",
            "ComparativeBenchmarkSuite",
        ),
        "BoDanhGiaHeThongEdge": (
            "vietnamese_ai.benchmark.edge_systems_benchmark",
            "BoDanhGiaHeThongEdge",
        ),
        "ViEdgeSystemsBenchmark": (
            "vietnamese_ai.benchmark.edge_systems_benchmark",
            "ViEdgeSystemsBenchmark",
        ),
        "BoDanhGiaCanChinhAnToan": (
            "vietnamese_ai.benchmark.continual_alignment_benchmark",
            "BoDanhGiaCanChinhAnToan",
        ),
        "ViAlignSafeBenchmark": (
            "vietnamese_ai.benchmark.continual_alignment_benchmark",
            "ViAlignSafeBenchmark",
        ),
        "tao_bo_du_lieu_can_chinh_da_vong": (
            "vietnamese_ai.benchmark.continual_alignment_benchmark",
            "tao_bo_du_lieu_can_chinh_da_vong",
        ),
        "BoDanhGiaLegalTemporalRAG": (
            "vietnamese_ai.benchmark.legal_temporal_benchmark",
            "BoDanhGiaLegalTemporalRAG",
        ),
        "ViLegalTemporalBenchmark": (
            "vietnamese_ai.benchmark.legal_temporal_benchmark",
            "ViLegalTemporalBenchmark",
        ),
        "kiem_dinh_bootstrap_cap": (
            "vietnamese_ai.benchmark.legal_temporal_benchmark",
            "kiem_dinh_bootstrap_cap",
        ),
        "tao_ham_sinh_llm": (
            "vietnamese_ai.benchmark.legal_temporal_benchmark",
            "tao_ham_sinh_llm",
        ),
        "tinh_ndcg_at_k": (
            "vietnamese_ai.benchmark.legal_temporal_benchmark",
            "tinh_ndcg_at_k",
        ),
        "TAP_DU_LIEU_DA_MIEN_VN": (
            "vietnamese_ai.benchmark.datasets_vn",
            "TAP_DU_LIEU_DA_MIEN_VN",
        ),
        "BO_TRUY_HOI_DA_MIEN_VN": (
            "vietnamese_ai.benchmark.datasets_vn",
            "BO_TRUY_HOI_DA_MIEN_VN",
        ),
        "BO_KIEM_THU_NER_VN": (
            "vietnamese_ai.benchmark.datasets_vn",
            "BO_KIEM_THU_NER_VN",
        ),
        "BO_KIEM_THU_CAM_XUC_VN": (
            "vietnamese_ai.benchmark.datasets_vn",
            "BO_KIEM_THU_CAM_XUC_VN",
        ),
        "tai_bo_phap_ly_mo": (
            "vietnamese_ai.benchmark.datasets_vn",
            "tai_bo_phap_ly_mo",
        ),
        "tai_bo_hoi_dap_phap_ly_mo": (
            "vietnamese_ai.benchmark.datasets_vn",
            "tai_bo_hoi_dap_phap_ly_mo",
        ),
        "tai_bo_phap_ly_thoi_gian": (
            "vietnamese_ai.benchmark.datasets_vn",
            "tai_bo_phap_ly_thoi_gian",
        ),
        "tai_canh_do_thi_phap_ly_thoi_gian": (
            "vietnamese_ai.benchmark.datasets_vn",
            "tai_canh_do_thi_phap_ly_thoi_gian",
        ),
        "tai_bo_hoi_dap_phap_ly_thoi_gian": (
            "vietnamese_ai.benchmark.datasets_vn",
            "tai_bo_hoi_dap_phap_ly_thoi_gian",
        ),
    },
    globals(),
)
