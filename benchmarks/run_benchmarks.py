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

from vietnamese_ai.benchmark import BoDoLuongDoiDau


def chay_tu_dong(args: argparse.Namespace) -> Dict[str, Any]:
    """Thực thi benchmark theo tham số CLI, tùy chọn gắn profiler cProfile và xuất JSON/Markdown."""
    suite = BoDoLuongDoiDau()
    profiler = cProfile.Profile() if args.profile else None

    if profiler is not None:
        profiler.enable()

    if args.suite == "core":
        ket_qua: Dict[str, Any] = suite.chay_benchmark_toan_dien()
        bao_cao_md = suite.xuat_bang_so_sanh_markdown(ket_qua)
    elif args.suite == "deep":
        ket_qua = suite.do_chieu_sau_loi_he_thong(nhanh=args.quick, seed=args.seed)
        bao_cao_md = suite.xuat_bao_cao_chieu_sau_markdown(ket_qua)
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
        description="Bộ chạy Benchmark Tái Lập cho Vietnamese AI Framework (v31.0.4)"
    )
    parser.add_argument(
        "--suite",
        choices=("core", "extended", "deep"),
        default="extended",
        help="Chọn bộ đo 'core', 'extended' hoặc 'deep' (6 trụ cột kiến trúc chiều sâu, mặc định: extended)",
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
