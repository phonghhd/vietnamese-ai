from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest


def _load_script_module(script_name: str) -> Any:
    """Nạp động một script trong thư mục `scripts/` bằng `importlib` để kiểm thử trực tiếp."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"vai_script_{script_path.stem}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_m3_dpo_trainer_token_kl_and_validation() -> None:
    """Kiểm tra `loai_phat_kl='token_kl'` trên `DPOTrainer`, xác thực tham số và theo dõi `kl_divergence`."""
    from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

    with pytest.raises(ValueError, match="loai_phat_kl"):
        DPOTrainer(loai_phat_kl="khong_hop_le")

    samples = [
        {
            "prompt": "Thời giờ làm việc bình thường tối đa theo BLLĐ 2019?",
            "chosen": "Không quá 08 giờ trong 01 ngày và 48 giờ trong 01 tuần theo Điều 105 BLLĐ 45/2019/QH14.",
            "rejected": "Không có giới hạn giờ làm việc.",
            "label_smoothing": 0.05,
        },
        {
            "prompt": "Mức phạt vi phạm hợp đồng thương mại tối đa?",
            "chosen": "Không quá 8% giá trị phần nghĩa vụ bị vi phạm theo Điều 301 Luật Thương mại 36/2005/QH11.",
            "rejected": "Có thể phạt tùy ý đến 50%.",
            "label_smoothing": 0.05,
        },
    ]

    trainer = DPOTrainer(
        so_vong=2,
        toc_do_hoc=0.15,
        beta=0.2,
        label_smoothing=0.05,
        he_so_neo_kl=0.05,
        loai_phat_kl="token_kl",
    )
    kq = trainer.huan_luyen(model=None, ref_model=None, preference_data=samples)
    assert kq["loai_phat_kl"] == "token_kl"
    assert kq["final_kl_divergence"] >= 0.0
    assert len(kq["history"]["kl_divergence"]) == 2
    assert trainer.thong_ke()["loai_phat_kl"] == "token_kl"

    try:
        import torch
        import torch.nn as nn

        class _TinyLM(nn.Module):
            def __init__(self, vocab: int = 128, dim: int = 16) -> None:
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.head = nn.Linear(dim, vocab)

            def forward(self, x: Any) -> Any:
                return self.head(self.emb(x % 128))

        torch.manual_seed(42)
        m_pol = _TinyLM()
        m_ref = _TinyLM()
        m_ref.load_state_dict(m_pol.state_dict())

        pt_trainer = DPOTrainer(
            so_vong=1,
            toc_do_hoc=1e-2,
            beta=0.1,
            label_smoothing=0.05,
            he_so_neo_kl=0.1,
            loai_phat_kl="token_kl",
            gradient_accumulation=1,
        )
        kq_pt = pt_trainer.huan_luyen(m_pol, m_ref, samples)
        assert kq_pt["loai_phat_kl"] == "token_kl"
        assert kq_pt["final_kl_divergence"] >= 0.0
        assert len(kq_pt["history"]["kl_divergence"]) == 1
    except ImportError:
        pass


def test_m3_auto_finetuner_safeloop_and_remote_gpu_qlora_scripts(tmp_path: Path) -> None:
    """Kiểm tra `BoTinhChinhMoHinh` chế độ `safeloop`/`simpo` và cú pháp Python hợp lệ của script worker QLoRA & Modal."""
    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh
    from vietnamese_ai.fine_tuning.remote_gpu import (
        CauHinhTinhChinhXa,
        NhaCungCapModal,
        _sinh_noi_dung_worker_python,
    )

    pref_samples = [
        {
            "prompt": "Thời hạn thông báo vi phạm bảo vệ dữ liệu cá nhân theo NĐ 13/2023/NĐ-CP?",
            "chosen": "Chậm nhất 72 giờ kể từ khi xảy ra vi phạm theo Điều 23 Nghị định 13/2023/NĐ-CP.",
            "rejected": "Không cần thông báo cho Bộ Công an.",
        },
        {
            "prompt": "Thời giờ làm việc bình thường tối đa?",
            "chosen": "Không quá 08 giờ/ngày và 48 giờ/tuần theo Điều 105 Bộ luật Lao động 45/2019/QH14.",
            "rejected": "Làm việc 16 giờ mỗi ngày.",
        },
    ]

    tuner_sl = BoTinhChinhMoHinh(mo_hinh="vgpt2", nhiem_vu="safeloop", che_do_offline=True)
    kq_sl = tuner_sl.huan_luyen(pref_samples, so_vong=1, toc_do_hoc=0.1)
    assert "bao_cao_loc_nhieu" in kq_sl
    assert kq_sl["loai_phat_kl"] == "token_kl"

    tuner_sp = BoTinhChinhMoHinh(mo_hinh="vgpt2", nhiem_vu="simpo", che_do_offline=True)
    kq_sp = tuner_sp.huan_luyen(pref_samples, so_vong=1, toc_do_hoc=0.1)
    assert kq_sp["che_do"] == "simpo"

    worker_src = _sinh_noi_dung_worker_python()
    ast.parse(worker_src)
    assert "loai_phat_kl" in worker_src
    assert "loc_nhieu_safeloop" in worker_src

    cfg = CauHinhTinhChinhXa(
        mo_hinh="Qwen/Qwen2.5-1.5B-Instruct",
        nhiem_vu="safeloop",
        loai_gpu="L4",
        che_do_dpo="cdpo",
        loai_phat_kl="token_kl",
        dung_4bit_qlora=True,
        loc_nhieu_safeloop=True,
    )
    modal_provider = NhaCungCapModal()
    modal_script = modal_provider.tao_modal_app_script(cfg, tmp_path / "modal_check", "job_m3")
    modal_src = modal_script.read_text(encoding="utf-8")
    ast.parse(modal_src)
    assert "bitsandbytes>=0.43.0" in modal_src
    assert "loai_phat_kl" in modal_src
    assert "loc_nhieu_safeloop" in modal_src


def test_m3_repro_experiment_scripts(tmp_path: Path) -> None:
    """Kiểm tra kịch bản `run_safeloop_qlora_experiment.py` cho cả chế độ CPU Proxy nhanh và kế hoạch 4 chiến lược QLoRA Modal."""
    mod_exp = _load_script_module("run_safeloop_qlora_experiment.py")

    kq_cpu = mod_exp.chay_thuc_nghiem_cpu_proxy(
        so_vong=3,
        seed=42,
        nhanh=True,
        thu_muc_xuat=tmp_path / "cpu_out",
    )
    assert (tmp_path / "cpu_out" / "vialign_safebench_cpu.json").is_file()
    assert (tmp_path / "cpu_out" / "vialign_safebench_table.tex").is_file()
    d_safe = kq_cpu["ket_qua"]["ket_qua"]["vialign_safeloop"]["diem_cuoi"]["diem_tong_hop"]
    d_naive = kq_cpu["ket_qua"]["ket_qua"]["naive_dpo"]["diem_cuoi"]["diem_tong_hop"]
    assert d_safe >= d_naive

    plan = mod_exp.lap_ke_hoach_qlora_da_chien_luoc(
        mo_hinh="Qwen/Qwen2.5-1.5B-Instruct",
        nha_cung_cap="modal",
        loai_gpu="L4",
        so_vong_du_lieu=3,
        so_epoch=1,
        seed=42,
        thu_muc_xuat=tmp_path / "qlora_out",
        chay_that=False,
        cho_phep_gui_cloud=True,
    )
    assert set(plan["cac_chien_luoc"].keys()) == {
        "naive_qlora_dpo",
        "filtered_qlora_dpo",
        "filtered_qlora_simpo",
        "vialign_safeloop_qlora",
    }
    for st_info in plan["cac_chien_luoc"].values():
        script_p = Path(st_info["duong_dan_script"])
        assert script_p.is_file()
        ast.parse(script_p.read_text(encoding="utf-8"))
        assert st_info["bao_cao_chi_phi"]["vuot_ngan_sach"] is False

    mod_tables = _load_script_module("generate_paper_tables.py")
    assert callable(mod_tables.sinh_toan_bo_bang_bai_bao)


def test_m3_dual_paper_artifacts_and_bibtex_consistency() -> None:
    """Xác minh đồng bộ BibTeX 100%, số liệu thực nghiệm nhất quán và đủ bộ tệp JOSS + ACL + Datasheet."""
    root = Path(__file__).resolve().parents[1]
    paper_md = (root / "paper" / "paper.md").read_text(encoding="utf-8")
    paper_bib = (root / "paper" / "paper.bib").read_text(encoding="utf-8")
    acl_tex = (root / "docs" / "research" / "vilegal_tempbench_acl.tex").read_text(encoding="utf-8")
    datasheet = (root / "docs" / "research" / "DATASHEET_VILEGAL_TEMPBENCH_V2.md").read_text(encoding="utf-8")
    checklist = (root / "paper" / "JOSS_CHECKLIST.md").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    for required_section in (
        "# Summary",
        "# Statement of need",
        "# State of the field",
        "# Software design",
        "# Research impact statement",
        "# AI usage disclosure",
        "# References",
    ):
        assert required_section in paper_md

    assert (root / "README_EN.md").is_file()
    assert (root / "CONTRIBUTING_EN.md").is_file()
    assert (root / "CODE_OF_CONDUCT.md").is_file()
    assert (root / ".github" / "workflows" / "draft-pdf.yml").is_file()

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", paper_bib))
    cited_keys = set(re.findall(r"@([a-z]+[0-9]{4}[a-z]+)", paper_md))
    assert len(cited_keys) >= 10
    missing_keys = cited_keys - bib_keys
    assert not missing_keys, f"Khóa trích dẫn thiếu trong paper.bib: {missing_keys}"

    for metric in ("0.936", "0.907", "0.987", "0.923", "0.688", "0.5969", "1.0000", "0.9041"):
        assert metric in paper_md
        assert metric in acl_tex

    assert "84 QA" in citation
    assert "68 QA" not in citation
    assert "T5" in datasheet
    assert "Gebru" in datasheet
    assert "JOSS" in checklist
