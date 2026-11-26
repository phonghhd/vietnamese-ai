"""Kịch Bản Chạy Benchmark Tái Lập Chuẩn Công Nghiệp Cho Vietnamese AI Framework.

Sử dụng:
    python benchmarks/run_benchmarks.py --suite extended --output results/benchmark.json --markdown results/benchmark.md
    python benchmarks/run_benchmarks.py --suite extended --quick --profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vietnamese_ai.benchmark import (
    BoDanhGiaCanChinhAnToan,
    BoDanhGiaHeThongEdge,
    BoDanhGiaLegalTemporalRAG,
    BoDoLuongDoiDau,
    tao_ham_sinh_llm,
)


def chay_tu_dong(args: argparse.Namespace) -> Dict[str, Any]:
    """Thực thi benchmark theo tham số CLI, tùy chọn gắn profiler cProfile và xuất JSON/Markdown/LaTeX."""
    suite = BoDoLuongDoiDau()
    profiler = cProfile.Profile() if args.profile else None

    if profiler is not None:
        profiler.enable()

    if args.suite == "core":
        ket_qua: Dict[str, Any] = suite.chay_benchmark_toan_dien()
        bao_cao_md = suite.xuat_bang_so_sanh_latex(ket_qua) if getattr(args, "latex", False) else suite.xuat_bang_so_sanh_markdown(ket_qua)
    elif args.suite == "deep":
        ket_qua = suite.do_chieu_sau_loi_he_thong(nhanh=args.quick, seed=args.seed)
        bao_cao_md = suite.xuat_bao_cao_chieu_sau_markdown(ket_qua)
    elif args.suite == "legal-temporal":
        ham_llm = None
        ten_llm = "extractive-grounded"
        llm_provider = getattr(args, "llm_provider", "none") or "none"
        if llm_provider != "none":
            llm_model = getattr(args, "llm_model", None) or None
            ham_llm = tao_ham_sinh_llm(
                nha_cung_cap=llm_provider,
                model=llm_model,
                base_url=getattr(args, "base_url", None) or None,
                do_tre_giua_goi_giay=float(getattr(args, "llm_delay", 0.0) or 0.0),
            )
            ten_llm = f"{llm_provider}:{llm_model or 'default'}"
        legal_bench = BoDanhGiaLegalTemporalRAG(
            ham_sinh_llm=ham_llm,
            ten_mo_hinh_llm=ten_llm,
        )
        ket_qua = legal_bench.chay_ablation(nhanh=args.quick, seed=args.seed)
        bao_cao_md = legal_bench.xuat_bang_latex(ket_qua) if getattr(args, "latex", False) else legal_bench.xuat_bao_cao_markdown(ket_qua)
    elif args.suite == "edge-systems":
        edge_bench = BoDanhGiaHeThongEdge()
        ket_qua = edge_bench.chay_toan_dien(
            nhanh=args.quick,
            n_vectors=int(getattr(args, "scale", 20000) or 20000),
            dim=int(getattr(args, "dim", 384) or 384),
            seed=args.seed,
        )
        bao_cao_md = edge_bench.xuat_bang_latex(ket_qua) if getattr(args, "latex", False) else edge_bench.xuat_bao_cao_markdown(ket_qua)
    elif args.suite == "align-safety":
        align_bench = BoDanhGiaCanChinhAnToan()
        ket_qua = align_bench.chay_ablation_da_vong(
            so_vong=int(getattr(args, "rounds", 5) or 5),
            nhanh=args.quick,
            seed=args.seed,
        )
        bao_cao_md = align_bench.xuat_bang_latex(ket_qua) if getattr(args, "latex", False) else align_bench.xuat_bao_cao_markdown(ket_qua)
    else:
        ket_qua = suite.chay_benchmark_mo_rong(
            nhanh=args.quick,
            seed=args.seed,
            bat_lan_thuc_te=not args.no_lan,
            doi_dau_truc_tiep=getattr(args, "compare_live", False),
        )
        bao_cao_md = suite.xuat_bao_cao_mo_rong_markdown(ket_qua)

    if profiler is not None:
        profiler.disable()
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        ket_qua["ho_so_cprofile_top15"] = buf.getvalue()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ket_qua, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(bao_cao_md, encoding="utf-8")

    if not args.quiet:
        print(bao_cao_md)

    return ket_qua


def tao_parser() -> argparse.ArgumentParser:
    """Khởi tạo bộ phân tích tham số dòng lệnh cho benchmark tái lập."""
    parser = argparse.ArgumentParser(
        description="Bộ chạy Benchmark Tái Lập cho Vietnamese AI Framework (v31.1.0)"
    )
    parser.add_argument(
        "--suite",
        choices=("core", "extended", "deep", "legal-temporal", "edge-systems", "align-safety"),
        default="extended",
        help="Chọn bộ đo 'core', 'extended', 'deep', 'legal-temporal', 'edge-systems' hoặc 'align-safety' (ViAlign-SafeBench)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Số vòng lặp Continual Alignment cho suite 'align-safety' (mặc định: 5)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=20000,
        help="Số lượng vector cho suite 'edge-systems' (mặc định: 20000, có thể đặt 100000+)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Số chiều vector cho suite 'edge-systems' (mặc định: 384)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Xuất bảng kết quả chuẩn LaTeX (booktabs) phục vụ bài báo khoa học",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("none", "gemini", "openai", "ollama"),
        default="none",
        help="Tùy chọn chạy đánh giá sinh văn bản bằng LLM thật (gemini/openai/ollama) cho suite legal-temporal",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Tên mô hình LLM (mặc định: gemini-2.5-flash / gpt-4o-mini / qwen2.5:3b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="URL API tùy chỉnh (dùng cho Ollama hoặc endpoint tương thích OpenAI)",
    )
    parser.add_argument(
        "--llm-delay",
        type=float,
        default=0.0,
        help="Độ trễ giữa các lần gọi API LLM (giây, hữu ích cho Gemini Free Tier)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Chạy chế độ rút gọn phù hợp kiểm thử CI nhanh",
    )
    parser.add_argument(
        "--no-lan",
        action="store_true",
        help="Tắt bài đo Swarm TCP Socket LAN cục bộ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên cố định để tái lập kết quả (mặc định: 42)",
    )
    parser.add_argument(
        "--compare-live",
        action="store_true",
        help="Đo đối đầu trực tiếp trong subprocess nếu máy có cài langchain/llama_index",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Thu thập top 15 hàm tốn thời gian nhất bằng cProfile chuẩn Python",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Đường dẫn tệp JSON lưu kết quả đo lường",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default="",
        help="Đường dẫn tệp Markdown lưu báo cáo",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Không in báo cáo Markdown ra stdout",
    )
    return parser


def main() -> int:
    """Điểm vào chính cho công cụ benchmark tái lập."""
    parser = tao_parser()
    args = parser.parse_args()
    chay_tu_dong(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
