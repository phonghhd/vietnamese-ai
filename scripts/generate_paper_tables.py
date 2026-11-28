"""Kịch bản tái lập 1 lệnh sinh toàn bộ bảng số liệu LaTeX/Markdown/JSON thực nghiệm cho bài báo JOSS & ViLegal-TempBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def sinh_toan_bo_bang_bai_bao(
    thu_muc_xuat: Path = Path("paper/artifacts"),
    seed: int = 42,
    nhanh: bool = False,
) -> Dict[str, Any]:
    """Chạy đồng bộ 3 bộ benchmark lõi (`ViLegal-TempBench v2`, `ViAlign-SafeBench`, `ViEdge-SysBench`) và xuất tệp LaTeX/JSON."""
    from vietnamese_ai.benchmark.continual_alignment_benchmark import BoDanhGiaCanChinhAnToan
    from vietnamese_ai.benchmark.edge_systems_benchmark import BoDanhGiaHeThongEdge
    from vietnamese_ai.benchmark.legal_temporal_benchmark import BoDanhGiaLegalTemporalRAG

    thu_muc_xuat.mkdir(parents=True, exist_ok=True)

    legal_bench = BoDanhGiaLegalTemporalRAG()
    kq_legal = legal_bench.chay_ablation(nhanh=nhanh, seed=seed)
    (thu_muc_xuat / "vilegal_tempbench_v2_table.tex").write_text(
        legal_bench.xuat_bang_latex(kq_legal), encoding="utf-8"
    )
    (thu_muc_xuat / "vilegal_tempbench_v2_report.md").write_text(
        legal_bench.xuat_bao_cao_markdown(kq_legal), encoding="utf-8"
    )

    align_bench = BoDanhGiaCanChinhAnToan()
    kq_align = align_bench.chay_ablation_da_vong(so_vong=5, seed=seed, nhanh=nhanh)
    (thu_muc_xuat / "vialign_safebench_table.tex").write_text(
        align_bench.xuat_bang_latex(kq_align), encoding="utf-8"
    )
    (thu_muc_xuat / "vialign_safebench_report.md").write_text(
        align_bench.xuat_bao_cao_markdown(kq_align), encoding="utf-8"
    )

    edge_bench = BoDanhGiaHeThongEdge()
    kq_edge = edge_bench.chay_toan_dien(
        nhanh=True if nhanh else False,
        n_vectors=5000 if nhanh else 10000,
        dim=384,
        seed=seed,
    )
    (thu_muc_xuat / "viedge_sysbench_table.tex").write_text(
        edge_bench.xuat_bang_latex(kq_edge), encoding="utf-8"
    )
    (thu_muc_xuat / "viedge_sysbench_report.md").write_text(
        edge_bench.xuat_bao_cao_markdown(kq_edge), encoding="utf-8"
    )

    tong_hop: Dict[str, Any] = {
        "seed": seed,
        "nhanh": nhanh,
        "vilegal_tempbench_v2": kq_legal,
        "vialign_safebench": kq_align,
        "viedge_sysbench": kq_edge,
    }
    (thu_muc_xuat / "summary_metrics.json").write_text(
        json.dumps(tong_hop, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tong_hop


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh toàn bộ bảng LaTeX/Markdown/JSON thực nghiệm cho bài báo.")
    parser.add_argument("--out-dir", default="paper/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    kq = sinh_toan_bo_bang_bai_bao(thu_muc_xuat=Path(args.out_dir), seed=args.seed, nhanh=args.quick)
    legal_full = kq["vilegal_tempbench_v2"]["ket_qua"]["full"]
    align_safe = kq["vialign_safebench"]["ket_qua"]["vialign_safeloop"]
    print(
        f"ViLegal-TempBench v2 (full): R@3={legal_full['recall_at_3']}, MRR@10={legal_full['mrr_at_10']}, "
        f"TVA={legal_full['tva']}, T5-R@3={legal_full['colloquial_recall_at_3']}, AbstF1={legal_full['abstention_f1']}"
    )
    print(
        f"ViAlign-SafeBench (safeloop): Final={align_safe['diem_cuoi']['diem_tong_hop']}, "
        f"SafetyFloor={align_safe['san_an_toan_thap_nhat']}, Forgetting={align_safe['do_quen_kien_thuc']}, "
        f"FilterF1={align_safe['loc_nhieu_f1']}, Rollbacks={align_safe['so_vong_rollback']}"
    )


if __name__ == "__main__":
    main()
