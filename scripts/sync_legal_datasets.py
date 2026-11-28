"""Đồng bộ & kiểm tra mã băm SHA-256 giữa nguồn dữ liệu pháp lý chuẩn `vietnamese_ai/rag/legal/data/` và `benchmarks/datasets/`."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CAC_TEP_DU_LIEU_PHAP_LY: Tuple[str, ...] = (
    "legal_vn_corpus.jsonl",
    "legal_vn_qa.jsonl",
    "vilegal_tempbench_corpus.jsonl",
    "vilegal_tempbench_edges.jsonl",
    "vilegal_tempbench_qa.jsonl",
)


def tinh_sha256(duong_dan: Path) -> str:
    """Tính mã băm SHA-256 hex cho một tệp."""
    return hashlib.sha256(duong_dan.read_bytes()).hexdigest()


def kiem_tra_dong_bo(thu_muc_du_an: Path) -> Dict[str, Any]:
    """Kiểm tra đối khớp SHA-256 giữa bản chuẩn trong gói và bản sao trong `benchmarks/datasets/`."""
    thu_muc_chuan = thu_muc_du_an / "vietnamese_ai" / "rag" / "legal" / "data"
    thu_muc_bench = thu_muc_du_an / "benchmarks" / "datasets"
    sai_lech: List[str] = []
    chi_tiet: Dict[str, Dict[str, str]] = {}

    for ten in CAC_TEP_DU_LIEU_PHAP_LY:
        p_chuan = thu_muc_chuan / ten
        p_bench = thu_muc_bench / ten
        if not p_chuan.is_file():
            sai_lech.append(f"Thiếu tệp nguồn chuẩn: {p_chuan}")
            continue
        sha_chuan = tinh_sha256(p_chuan)
        if not p_bench.is_file():
            sai_lech.append(f"Thiếu bản sao benchmark: {p_bench}")
            chi_tiet[ten] = {"nguon_chuan": sha_chuan, "benchmarks": "THIEU"}
            continue
        sha_bench = tinh_sha256(p_bench)
        chi_tiet[ten] = {"nguon_chuan": sha_chuan, "benchmarks": sha_bench}
        if sha_chuan != sha_bench:
            sai_lech.append(f"Lệch SHA-256 tại {ten}: {sha_chuan[:12]} != {sha_bench[:12]}")

    return {
        "dong_bo": len(sai_lech) == 0,
        "so_tep": len(CAC_TEP_DU_LIEU_PHAP_LY),
        "sai_lech": sai_lech,
        "chi_tiet": chi_tiet,
    }


def dong_bo_du_lieu(thu_muc_du_an: Path) -> List[str]:
    """Sao chép 1 chiều từ `vietnamese_ai/rag/legal/data/` sang `benchmarks/datasets/`."""
    thu_muc_chuan = thu_muc_du_an / "vietnamese_ai" / "rag" / "legal" / "data"
    thu_muc_bench = thu_muc_du_an / "benchmarks" / "datasets"
    thu_muc_bench.mkdir(parents=True, exist_ok=True)
    da_cap_nhat: List[str] = []

    for ten in CAC_TEP_DU_LIEU_PHAP_LY:
        p_chuan = thu_muc_chuan / ten
        p_bench = thu_muc_bench / ten
        if not p_chuan.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu pháp lý chuẩn: {p_chuan}")
        if not p_bench.is_file() or tinh_sha256(p_chuan) != tinh_sha256(p_bench):
            shutil.copy2(p_chuan, p_bench)
            da_cap_nhat.append(ten)
    return da_cap_nhat


def main() -> int:
    """Điểm vào CLI hỗ trợ `--sync` và `--check`."""
    parser = argparse.ArgumentParser(
        description="Đồng bộ & kiểm tra SHA-256 giữa vietnamese_ai/rag/legal/data/ và benchmarks/datasets/."
    )
    parser.add_argument("--sync", action="store_true", help="Sao chép bản chuẩn từ gói sang benchmarks/datasets/")
    parser.add_argument("--check", action="store_true", help="Kiểm tra khớp SHA-256 (trả mã thoát != 0 nếu lệch)")
    parser.add_argument("--root", type=str, default=None, help="Đường dẫn thư mục gốc dự án")
    args = parser.parse_args()

    goc = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if args.sync:
        cap_nhat = dong_bo_du_lieu(goc)
        if cap_nhat:
            print(f"Đã đồng bộ {len(cap_nhat)} tệp dữ liệu pháp lý: {', '.join(cap_nhat)}")
        else:
            print("Toàn bộ 5 tệp dữ liệu pháp lý đã khớp SHA-256, không cần sao chép.")

    if args.check or not args.sync:
        kq = kiem_tra_dong_bo(goc)
        if kq["dong_bo"]:
            print(f"OK: Toàn bộ {kq['so_tep']} tệp dữ liệu pháp lý khớp SHA-256 100%.")
            return 0
        for msg in kq["sai_lech"]:
            print(f"LỖI: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
