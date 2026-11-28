# Vietnamese AI Framework (English Documentation & Reviewer Guide)

> 🇻🇳 Tài liệu chính bằng tiếng Việt: [README.md](README.md) • Contributing (EN): [CONTRIBUTING_EN.md](CONTRIBUTING_EN.md) • Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) • JOSS Paper: [paper/paper.md](paper/paper.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20357082.svg)](https://doi.org/10.5281/zenodo.20357082)
![version](https://img.shields.io/badge/version-31.3.0-blue.svg)
![python](https://img.shields.io/badge/python-3.9%20--%203.14-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)
![tests](https://img.shields.io/badge/tests-1450%2B%20passed-brightgreen.svg)
![mypy](https://img.shields.io/badge/mypy-0%20errors-brightgreen.svg)
![ruff](https://img.shields.io/badge/ruff-0%20errors-brightgreen.svg)

**Vietnamese AI Framework** (`vietnamese-ai`) is a local-first, lightweight Python framework (requiring only `numpy` at core install, with an optional Rust `abi3-py39` extension) designed for:

1. **Temporal, Clause-Level Graph Legal RAG (`ViLegal-TemporalRAG`)**: Point-in-time retrieval over Vietnamese statutory hierarchies (Laws, Decrees, Circulars) with a bundled 38,372-word MIT Vietnamese lexicon, portable single-file `.vaidb` knowledge stores (`SQLite/FTS5 + HNSW`), and a deterministic 3-tier citation, temporal, and numeric verifier (`LegalCitationVerifier` / `BoKiemChungTrichDanPhapLy`).
2. **Vietnamese Contract Review Pro (`ContractReviewEngine`)**: Automated risk auditing across five Vietnamese contract classes (`lao_dong`, `dich_vu`, `mua_ban`, `thue_nha`, `nda`), OOXML `.docx` merged-cell parsing, pure-NumPy red-stamp removal for scanned PDF OCR, citation-verified hybrid LLM auditing, and Word Track Changes export.
3. **Zero-GPU Edge Systems (`ViEdge-RustEngine`)**: Native Rust SIMD `SQ8`/`Binary`/`HNSW` vector retrieval, 2-bit packed BitNet 1.58-bit T-MAC LUT inference, Swarm CPU pipeline parallelism, and RFC 5114 Schnorr/Fiat-Shamir + RFC 6962 Merkle Tree verification.
4. **Eval-Gated Continual Alignment (`ViAlign-SafeLoop`)**: 6-signal noisy preference filtering, stratified anchor replay, weighted cDPO/SimPO with parameter L2 anchoring or token-level reverse KL regularization, 4-axis evaluation gate with atomic rollback, and 4-bit QLoRA remote GPU orchestration.

---

## Installation (Under 60 Seconds)

### Core Install (NumPy-only)

```bash
pip install vietnamese-ai
```

### Development & Reproducibility Install (Recommended for Reviewers)

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Optional extras: `.[legal-rag]`, `.[edge]`, `.[onnx]`, `.[cloud-gpu]`, `.[all]`.

---

## Quickstart Examples (Runnable Offline on CPU)

### 1. Temporal Legal RAG & 3-Tier Citation Verification

```python
from vietnamese_ai.rag.legal.temporal_rag import tao_dong_co_phap_ly_thoi_gian_mac_dinh

engine = tao_dong_co_phap_ly_thoi_gian_mac_dinh()

res_before = engine.tra_loi_phap_ly(
    "Vào tháng 3/2025, người bán có được phép hủy hóa đơn điện tử đã cấp mã bị sai sót không?"
)
res_after = engine.tra_loi_phap_ly(
    "Vào tháng 8/2025, người bán có được phép hủy hóa đơn điện tử đã cấp mã bị sai sót không?"
)

print("Before 2025-06-01:", res_before["ngay_tra_cuu"], "| Valid:", res_before["kiem_chung_trich_dan"]["hop_le"])
print("After  2025-06-01:", res_after["ngay_tra_cuu"], "| Valid:", res_after["kiem_chung_trich_dan"]["hop_le"])
```

### 2. Build a Portable `.vaidb` Legal Knowledge Pack & Generate MCP Config

```bash
vai pull-kb --list
vai pull-kb vn-business-core-2025 --output ./vn-business-core-2025.vaidb
vai mcp-config --client cursor --kb ./vn-business-core-2025.vaidb
```

### 3. Vietnamese Contract Review Pro

```python
from vietnamese_ai.rag.legal import BoSoSanhRaSoatHopDong

auditor = BoSoSanhRaSoatHopDong()
report = auditor.quet_rui_ro_don_le(
    "HỢP ĐỒNG LAO ĐỘNG\n"
    "Điều 1. Thử việc: Thời gian thử việc là 90 ngày, lương thử việc hưởng 70% lương chính thức.",
    ngay_tra_cuu="2026-01-15",
)
print("Detected type:", report["loai_hop_dong"], "| Alerts:", report["so_canh_bao"])
```

### 4. Launch Web Studio v2 (SSE Streaming + Split-View Citations)

```bash
vai studio --host 127.0.0.1 --port 8080
```

---

## Reproducing All Benchmarks & Running Tests

### Run Automated Test Suite & Static Checks

```bash
pytest tests/ -q --tb=short
ruff check vietnamese_ai/ tests/ benchmarks/ scripts/
mypy vietnamese_ai/ --ignore-missing-imports
vai release-check
python scripts/sync_legal_datasets.py --check
mkdocs build --strict
```

### Reproduce Paper Benchmark Tables (Zero-GPU CPU, `seed=42`)

```bash
vai benchmark --suite legal-temporal --latex
vai benchmark --suite align-safety --latex
vai benchmark --suite edge-systems --quick --latex
python scripts/generate_paper_tables.py --seed 42
python scripts/run_safeloop_qlora_experiment.py --mode cpu-proxy --rounds 5 --seed 42
```

### Module Status Transparency (`Core Production` vs `Optional` vs `Experimental`)

Run `vai status-map` to inspect the exact production readiness tier of every subsystem. Experimental/simulated legacy modules (`wasm_node`, `webxr_node`, `tee_zkp_node`, `mobile_tools`, `salm`, `federated`) emit explicit `DeprecationWarning` notices and are scheduled for separation in `v32.0.0`.

---

## Research Artifacts & Citation

- **JOSS Software Paper**: [paper/paper.md](paper/paper.md) & [paper/paper.bib](paper/paper.bib)
- **Dataset Datasheet (`ViLegal-TempBench v2`, Gebru et al., 2021)**: [docs/research/DATASHEET_VILEGAL_TEMPBENCH_V2.md](docs/research/DATASHEET_VILEGAL_TEMPBENCH_V2.md)
- **Citation Metadata**: [CITATION.cff](CITATION.cff) (Zenodo DOI: `10.5281/zenodo.20357082`)
