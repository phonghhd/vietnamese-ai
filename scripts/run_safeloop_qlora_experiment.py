"""Kịch bản tái lập thực nghiệm ViAlign-SafeLoop đa vòng (CPU Proxy 4 chiến lược & QLoRA Modal/Local GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def chay_thuc_nghiem_cpu_proxy(
    so_vong: int = 3,
    seed: int = 42,
    nhanh: bool = False,
    thu_muc_xuat: Path | None = None,
) -> Dict[str, Any]:
    """Chạy đối chứng 4 chiến lược Continual Alignment trên `ViAlign-SafeBench` (tái lập 100% không cần GPU)."""
    from vietnamese_ai.benchmark.continual_alignment_benchmark import BoDanhGiaCanChinhAnToan

    bench = BoDanhGiaCanChinhAnToan()
    kq = bench.chay_ablation_da_vong(so_vong=so_vong, seed=seed, nhanh=nhanh)
    md = bench.xuat_bao_cao_markdown(kq)
    tex = bench.xuat_bang_latex(kq)
    if thu_muc_xuat is not None:
        thu_muc_xuat.mkdir(parents=True, exist_ok=True)
        (thu_muc_xuat / "vialign_safebench_cpu.json").write_text(
            json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (thu_muc_xuat / "vialign_safebench_cpu.md").write_text(md, encoding="utf-8")
        (thu_muc_xuat / "vialign_safebench_table.tex").write_text(tex, encoding="utf-8")
    return {"ket_qua": kq, "markdown": md, "latex": tex}


def lap_ke_hoach_qlora_da_chien_luoc(
    mo_hinh: str = "Qwen/Qwen2.5-1.5B-Instruct",
    nha_cung_cap: str = "modal",
    loai_gpu: str = "L4",
    so_vong_du_lieu: int = 3,
    so_epoch: int = 2,
    seed: int = 42,
    thu_muc_xuat: Path = Path("paper/artifacts/qlora_safeloop"),
    chay_that: bool = False,
    cho_phep_gui_cloud: bool = False,
) -> Dict[str, Any]:
    """Sinh kế hoạch thực nghiệm 4 chiến lược QLoRA (`naive_dpo`, `filtered_dpo`, `simpo_filtered`, `vialign_safeloop`) có ước tính chi phí và script tái lập."""
    from vietnamese_ai.benchmark.continual_alignment_benchmark import (
        tao_bo_du_lieu_can_chinh_da_vong,
    )
    from vietnamese_ai.fine_tuning.remote_gpu import BoDieuPhoiTinhChinhXa, CauHinhTinhChinhXa

    thu_muc_xuat.mkdir(parents=True, exist_ok=True)
    _, _, _, _, cac_lo = tao_bo_du_lieu_can_chinh_da_vong(so_vong=so_vong_du_lieu, seed=seed)
    mau_gop: List[Dict[str, Any]] = [
        {"prompt": m["prompt"], "chosen": m["chosen"], "rejected": m["rejected"]}
        for lo in cac_lo
        for m in lo
    ]

    cac_chien_luoc: List[Dict[str, Any]] = [
        {
            "ten": "naive_qlora_dpo",
            "nhiem_vu": "dpo",
            "che_do_dpo": "dpo",
            "label_smoothing": 0.0,
            "he_so_neo_kl": 0.0,
            "loai_phat_kl": "l2_tham_so",
            "loc_nhieu_safeloop": False,
            "loc_dau_doc_du_lieu": False,
        },
        {
            "ten": "filtered_qlora_dpo",
            "nhiem_vu": "dpo",
            "che_do_dpo": "dpo",
            "label_smoothing": 0.0,
            "he_so_neo_kl": 0.0,
            "loai_phat_kl": "l2_tham_so",
            "loc_nhieu_safeloop": True,
            "loc_dau_doc_du_lieu": True,
        },
        {
            "ten": "filtered_qlora_simpo",
            "nhiem_vu": "simpo",
            "che_do_dpo": "simpo",
            "label_smoothing": 0.0,
            "he_so_neo_kl": 0.0,
            "loai_phat_kl": "token_kl",
            "loc_nhieu_safeloop": True,
            "loc_dau_doc_du_lieu": True,
        },
        {
            "ten": "vialign_safeloop_qlora",
            "nhiem_vu": "safeloop",
            "che_do_dpo": "cdpo",
            "label_smoothing": 0.05,
            "he_so_neo_kl": 0.05,
            "loai_phat_kl": "token_kl",
            "loc_nhieu_safeloop": True,
            "loc_dau_doc_du_lieu": True,
        },
    ]

    ket_qua_cac_chien_luoc: Dict[str, Any] = {}
    for st in cac_chien_luoc:
        ten_st = str(st["ten"])
        dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec=str(thu_muc_xuat / ten_st))
        cfg = CauHinhTinhChinhXa(
            mo_hinh=mo_hinh,
            nhiem_vu=str(st["nhiem_vu"]),
            loai_gpu=loai_gpu,
            so_vong=so_epoch,
            rank_lora=16,
            toc_do_hoc=5e-5,
            dinh_dang_xuat="lora",
            ngan_sach_toi_da_usd=15.0,
            an_pii_truoc_khi_gui=True,
            loc_dau_doc_du_lieu=bool(st["loc_dau_doc_du_lieu"]),
            cho_phep_gui_cloud=cho_phep_gui_cloud or (nha_cung_cap == "local"),
            che_do_offline_worker=(nha_cung_cap == "local"),
            che_do_dpo=str(st["che_do_dpo"]),
            beta_dpo=0.1,
            label_smoothing=float(st["label_smoothing"]),
            he_so_neo_kl=float(st["he_so_neo_kl"]),
            loai_phat_kl=str(st["loai_phat_kl"]),
            dung_4bit_qlora=True,
            loc_nhieu_safeloop=bool(st["loc_nhieu_safeloop"]),
        )
        kq = dieu_phoi.chay_vong_lap_flywheel(
            nguon_du_lieu=mau_gop,
            ten_nha_cung_cap=nha_cung_cap,
            cau_hinh=cfg,
            chay_that=chay_that,
        )
        ket_qua_cac_chien_luoc[ten_st] = kq.to_dict()

    tong_hop = {
        "mo_hinh": mo_hinh,
        "nha_cung_cap": nha_cung_cap,
        "loai_gpu": loai_gpu,
        "so_mau_dau_vao": len(mau_gop),
        "chay_that": chay_that,
        "cac_chien_luoc": ket_qua_cac_chien_luoc,
    }
    (thu_muc_xuat / "qlora_experiment_plan.json").write_text(
        json.dumps(tong_hop, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tong_hop


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy hoặc dựng kế hoạch thực nghiệm ViAlign-SafeLoop (CPU Proxy / QLoRA GPU).")
    parser.add_argument("--mode", choices=("cpu-proxy", "qlora-modal", "qlora-local"), default="cpu-proxy")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--out-dir", default="paper/artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.mode == "cpu-proxy":
        kq = chay_thuc_nghiem_cpu_proxy(
            so_vong=args.rounds,
            seed=args.seed,
            nhanh=args.quick,
            thu_muc_xuat=out_dir,
        )
        print(kq["markdown"])
        return

    provider = "modal" if args.mode == "qlora-modal" else "local"
    plan = lap_ke_hoach_qlora_da_chien_luoc(
        mo_hinh=args.model,
        nha_cung_cap=provider,
        loai_gpu=args.gpu,
        so_vong_du_lieu=args.rounds,
        so_epoch=args.epochs,
        seed=args.seed,
        thu_muc_xuat=out_dir / "qlora_safeloop",
        chay_that=bool(args.apply),
        cho_phep_gui_cloud=bool(args.allow_cloud),
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
