---
title: 'Vietnamese AI: A Local-First, Lightweight Framework for Vietnamese Legal RAG, Edge Inference, and Verifiable Alignment'
tags:
  - Python
  - Rust
  - Vietnamese NLP
  - Legal RAG
  - Temporal Knowledge Graphs
  - Continual Alignment
  - Edge Inference
  - Verifiable AI
authors:
  - name: Duong Phong Huynh
    orcid: 0009-0002-1458-3406
    affiliation: 1
affiliations:
  - name: Independent Researcher, Ho Chi Minh City, Vietnam
    index: 1
date: 28 November 2026
bibliography: paper.bib
---

# Summary

Modern Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) stacks predominantly target English corpora and high-memory GPU clusters. Deploying general-purpose frameworks such as LangChain [@chase2022langchain], LlamaIndex [@liu2022llamaindex], and PyTorch [@paszke2019pytorch] on low-power office hardware for Vietnamese statutory law introduces three recurring failures: (i) multi-second startup latency and >140 MB baseline RAM overhead; (ii) temporal hallucinations when laws are repealed or amended across multi-document Law–Decree–Circular hierarchies; and (iii) safety regressions when continual preference alignment updates ingest noisy or poisoned human feedback.

`Vietnamese AI` is an MIT-licensed, local-first Python framework (requiring only `numpy` at core install, with an optional Rust `abi3-py39` native extension) designed for reproducible Vietnamese Legal RAG, contract auditing, zero-GPU edge inference, and eval-gated continual alignment. It unites four cohesive pillars:

1. **Vietnamese Linguistic Core & Portable `.vaidb` Legal Knowledge Packs**: A bundled 38,372-word MIT Vietnamese lexicon with Rust/Python-synchronized Trie/Viterbi DAG word segmentation, SQLite/FTS5 + HNSW [@malkov2018efficient] `.vaidb` knowledge stores (`vai pull-kb`), and optional quantized ONNX INT8 semantic retrieval.
2. **Temporal, Clause-Level Graph Legal RAG & Deterministic 3-Tier Verification (`ViLegal-TemporalRAG`)**: Point-in-time legal retrieval combining BM25 [@robertson2009probabilistic], signed-hash/ONNX vectors via Reciprocal Rank Fusion [@cormack2009reciprocal], clause-level temporal validity tracking, multi-hop Law–Decree–Circular graph traversal, and deterministic citation, temporal, and numeric verification.
3. **Vietnamese Contract Review Pro & Local-First Surfaces**: Five domain-specific Vietnamese contract rulesets (Labor, Services, Sales, Leasing, NDA), OOXML `.docx` merged-cell parsing, pure-NumPy red-stamp removal for scanned PDF OCR, Hybrid LLM risk review gated by citation verification, Word Track Changes export, Web Studio v2 (SSE streaming + split-view citations), and 1-click MCP integration (`vai mcp-config`) for Cursor, Claude Desktop, VS Code, and Antigravity.
4. **Eval-Gated Continual Alignment & Verifiable Edge Systems (`ViAlign-SafeLoop` & `ViEdge-RustEngine`)**: Multi-round noisy preference filtering, anchor replay, weighted cDPO/SimPO [@rafailov2023direct; @meng2024simpo; @mitchell2023note] with token-level reverse KL regularization and QLoRA remote GPU orchestration [@hu2022lora], alongside 1.58-bit BitNet T-MAC LUT kernels [@ma2024era], Swarm CPU pipeline parallelism [@huang2019gpipe], and RFC 5114 Schnorr/Fiat-Shamir + RFC 6962 Merkle Tree verification [@schnorr1991efficient; @fiat1986how; @merkle1987digital].

# Statement of Need

Vietnamese legal and enterprise teams frequently handle sensitive contracts and statutory queries on standard office laptops (2–8 GB RAM, no discrete GPU) where sending confidential documents to external cloud APIs is restricted. Existing Vietnamese NLP libraries such as Underthesea [@vu2019underthesea] address linguistic preprocessing but do not provide point-in-time statutory validity graphs, deterministic numeric/citation verifiers, portable single-file `.vaidb` vector/FTS5 stores, or continual alignment safeguards against poisoned feedback. Conversely, general LLM orchestration libraries impose heavy dependency graphs and lack built-in Vietnamese legal temporal modeling. `Vietnamese AI` fills this gap by providing a lightweight, self-contained, and rigorously tested software stack that serves both immediate local-first enterprise deployment and reproducible academic research.

# Core Architecture

`Vietnamese AI` separates zero-dependency production modules (`Core Production`), optional extras (`Optional Extensions`), and clearly flagged experimental/simulated modules (`vai status-map`):

```text
vietnamese_ai/
├── nlp/           # 38,372-word lexicon, FST tone normalization, Trie/Viterbi DAG (Rust parity)
├── rag/           # RAGPipeline, KhoTriThucVAIDB (.vaidb + HNSW), onnx_semantic
│   └── legal/     # TemporalRAG, ClauseGraph, CitationVerifier, ContractReviewPro, KBHub
├── fine_tuning/   # DPO/SimPO/cDPO (token KL), ViAlign-SafeLoop, EvalGate, Remote GPU QLoRA
├── edge/          # Rust SIMD SQ8/Binary/HNSW, BitNet 1.58-bit LUT, Swarm CPU
├── security/      # LLM Firewall, Image CDR, RFC 5114 Schnorr & RFC 6962 Merkle PoI
├── agents/        # StateGraph, MayChuMCPThucTe (stdio/SSE + IDE config generator)
├── ui/            # StudioDashboard v2 (SSE streaming, split-view citations, SQLite DPO)
└── benchmark/     # ViLegal-TempBench v2, ViAlign-SafeBench, ViEdge-SysBench (LaTeX/JSON)
```

## Temporal Legal Retrieval & Deterministic Citation Verification

Let $C_t$ denote statutory clauses active at reference date $t$ where each clause version carries a half-open validity interval $[t_{\text{eff}}(c), t_{\text{exp}}(c))$, linked by a directed clause-level temporal graph $G_t = (V, E_t)$ capturing *replaces*, *amends*, and *implements* relations. Given a Vietnamese query $q$, `BoPhanTichThoiGianPhapLy` extracts the target legal date $t$ (disambiguating event origin dates from evaluation dates), `BoTruyHoiPhapLyThoiGian` fuses lexical/dense rankings with temporal validity filtering and multi-hop graph expansion, and `BoKiemChungTrichDanPhapLy` verifies generated answers across three deterministic tiers: (1) source-grounded citation matching, (2) temporal validity consistency, and (3) numeric/deadline fidelity.

## Eval-Gated Continual Alignment (`ViAlign-SafeLoop`)

To prevent preference-poisoning regressions during iterative feedback collection from Studio workspaces, `VongLapCanChinhAnToan` chains a 6-signal preference filter (`BoLocNhieuCapUuTien`), stratified anchor replay (`BoDemOnTapNeo`), weighted cDPO/SimPO training (`DPOTrainer` supporting both parameter L2 anchoring and token-level reverse KL penalty $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$), and a 4-axis evaluation gate (`CongKiemDinhMoHinh`) with atomic rollback. On GPU backends (`NhaCungCapModal`, `NhaCungCapSSH`), `BoDieuPhoiTinhChinhXa` executes 4-bit QLoRA alignment with PII scrubbing, budget caps, and SHA-256/HMAC artifact verification.

# Empirical Evaluation & Benchmarks

All local benchmarks were measured without GPU acceleration on an **Intel Celeron N3050 @ 1.60 GHz (2 cores, 6W TDP, SSE4.2, no AVX2, 3.2 GiB RAM)** running Linux `x86_64`, Python `3.14.4`, and the compiled `vietnamese_ai_rust` extension. Process startup and `VmRSS` are measured in isolated Python subprocesses. Dataset provenance and limitations are documented in a formal datasheet [@gebru2021datasheets].

## Core Efficiency & `ViLegal-TempBench v2` (84 Queries, 5 Splits, `seed=42`)

Core `import vietnamese_ai` cold-start latency is **24.68 ms** at **12.82 MB `VmRSS`** (**37.39 MB** after indexing/querying the 100-clause open corpus at `100.0%` Hit@1 / MRR `1.000`). On `ViLegal-TempBench v2` (`160` statutory clauses, `38` graph edges, `84` queries spanning `T1` single-hop, `T2` temporal contrast pairs, `T3` multi-hop chains, `T4` expired-statute traps and OOD abstention, and `T5` colloquial queries without statute IDs), temporal date extraction achieves **TQPA = 100.0% (84/84)**:

| Configuration | R@1 | R@3 | MRR@10 | TVA | TCA (T2) | Multi-Hop (T3) | Trap (T4) | T5-R@3 | Abst. F1 | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25` (VN Tokenizer) | 0.575 | 0.880 | 0.810 | 0.731 | 0.083 (1/12) | 0.500 | 0.125 | 0.625 | 0.000 | 7.71 |
| `hybrid` (BM25 + Dense RRF) | 0.575 | 0.900 | 0.804 | 0.705 | 0.167 (2/12) | 0.571 | 0.125 | 0.688 | 0.000 | 13.86 |
| `naive_date_filter` (Start-Date Only) | 0.639 | 0.900 | 0.836 | 0.727 | 0.583 (7/12) | 0.571 | 0.125 | 0.688 | 0.000 | 22.20 |
| `temporal` (+ Point-in-Time) | 0.793 | 0.932 | 0.917 | 1.000 | 1.000 (12/12) | 0.786 | 1.000 | 0.750 | 0.000 | 23.02 |
| `temporal_graph` (+ Clause Graph) | 0.793 | 0.936 | 0.910 | 1.000 | 1.000 (12/12) | 1.000 | 1.000 | 0.688 | 0.000 | 23.82 |
| **`full` (ViLegal-TemporalRAG)** | **0.793** | **0.936** | **0.907** | **0.987** | **1.000 (12/12)** | **1.000** | **1.000** | **0.688** | **0.923** | **101.80** |

Paired bootstrap tests (`n=1000`, `seed=42`) confirm statistically significant gains of `full` over `naive_date_filter` on MRR@10 (`+0.0709`, $p<0.0001$), TVA (`+0.2607`, $p<0.0001$), and Multi-Hop@3 (`+0.4286`, $p=0.0020$). Error analysis on `T5` transparently exposes 4 lexical-gap misses (`t5_01`, `t5_03`, `t5_06`, `t5_12`) and 1 conservative abstention (`t5_13`), motivating optional ONNX INT8 dense embeddings (`--onnx-embedder`, `--onnx-reranker`).

## Continual Alignment Ablation on `ViAlign-SafeBench` (5 Rounds, 70 Pairs, 55.7% Noise/Poisoning)

Evaluated on the deterministic 256-dim CPU ranking proxy (`seed=42`; GPU QLoRA scripts for `Qwen2.5-1.5B/3B-Instruct` are shipped in `scripts/run_safeloop_qlora_experiment.py`):

| Strategy | Final Score | Citation | Validity | Safety | Safety Floor | Forgetting $\downarrow$ | Filter F1 | Rollbacks | Silent Reg. $\downarrow$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive Continual DPO | 0.5969 | 0.5625 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 1 |
| Filtered DPO (6-signal filter) | 0.9812 | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9041 | 0 | 0 |
| Filtered + Replay + cDPO (+ L2 Anchor) | 0.9906 | 0.9792 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9041 | 0 | 0 |
| **`ViAlign-SafeLoop` (Full + Rollback)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **0.9041** | **1** | **0** |

`ViAlign-SafeLoop` prevents the complete Round-3 safety collapse of Naive DPO (`Safety: 0.0000`) and automatically rolls back the Round-5 subtle numeric-fabrication stress attack (`+0.2500` mean delta vs. Naive DPO, 95% CI `[+0.1260, +0.3740]`, $p<0.0001$).

# Software Quality & Reproducibility

- **Automated Testing & CI**: Over 1,450 unit, security, and integration tests pass across Python 3.9–3.14 and Rust `abi3-py39` wheels (Linux, macOS, Windows), with 0 Ruff errors, 0 Mypy errors across 343 source files (`check_untyped_defs = true` on core packages), and 349 PEP 561 stub exports.
- **One-Command Reproducibility**:
  ```bash
  pip install vietnamese-ai
  vai pull-kb vn-business-core-2025
  vai mcp-config --client cursor --kb vn-business-core-2025.vaidb
  python scripts/generate_paper_tables.py --seed 42
  python scripts/run_safeloop_qlora_experiment.py --mode cpu-proxy --rounds 5 --seed 42
  ```

# AI Use Disclosure

During development and manuscript preparation, generative AI coding assistants (including Google Antigravity / Gemini models) assisted with refactoring, test scaffolding, PEP 561 type stub maintenance, and English academic editing. The human author conceived and directed all core architectures, benchmarks, legal dataset curation, and experimental protocols, personally reviewed and verified all executed outputs and tests, and assumes full scientific, legal, and ethical responsibility for the submission.

# Acknowledgements

The author thanks the Vietnamese open-source AI community for feedback and testing support on local-first, low-resource deployments.

# References
