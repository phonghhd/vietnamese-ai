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

Modern Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) stacks predominantly target English corpora and high-memory GPU clusters. Deploying general-purpose frameworks on commodity office hardware for Vietnamese statutory law introduces three recurring failures: (i) multi-second startup latency and high baseline memory overhead; (ii) temporal hallucinations when statutes are repealed or amended across multi-document Law--Decree--Circular hierarchies; and (iii) safety regressions when continual preference alignment ingests noisy or poisoned human feedback.

`Vietnamese AI` (`v31.3.0`) is an MIT-licensed, local-first Python framework requiring only `numpy` at core install (with an optional Rust `abi3-py39` native extension) for reproducible Vietnamese legal RAG, deterministic citation auditing, zero-GPU edge vector/quantized inference, and eval-gated continual alignment. It integrates four pillars:

1. **Vietnamese Linguistic Core & Portable `.vaidb` Stores**: A bundled 38,372-word MIT Vietnamese lexicon with Rust/Python-synchronized Trie/Viterbi DAG word segmentation, SQLite/FTS5 + HNSW [@malkov2018efficient] `.vaidb` knowledge packs (`vai pull-kb`), and optional INT8 ONNX semantic retrieval.
2. **Temporal, Clause-Level Graph Legal RAG (`ViLegal-TemporalRAG`)**: Point-in-time retrieval combining BM25 [@robertson2009probabilistic] and dense vectors via Reciprocal Rank Fusion [@cormack2009reciprocal], clause-level temporal validity tracking, multi-hop graph traversal, and deterministic 3-tier citation, temporal, and numeric verification.
3. **Vietnamese Contract Review Pro & Local-First Interfaces**: Five Vietnamese contract rulesets (Labor, Services, Sales, Leasing, NDA), OOXML `.docx` merged-cell parsing, pure-NumPy red-stamp filtering for scanned PDF OCR, citation-verified hybrid LLM review, Word Track Changes export, Web Studio v2 (SSE streaming), and 1-click MCP integration (`vai mcp-config`).
4. **Eval-Gated Continual Alignment & Verifiable Edge Systems (`ViAlign-SafeLoop` & `ViEdge-RustEngine`)**: 6-signal preference noise filtering, anchor replay, weighted cDPO/SimPO [@rafailov2023direct; @meng2024simpo; @mitchell2023note] with parameter L2 anchoring or token-level reverse KL regularization and 4-bit QLoRA orchestration [@hu2022lora], alongside 2-bit packed BitNet 1.58-bit T-MAC LUT kernels [@ma2024era], Swarm CPU pipeline parallelism [@huang2019gpipe], and RFC 5114 Schnorr/Fiat-Shamir + RFC 6962 Merkle Tree verification [@schnorr1991efficient; @fiat1986how; @merkle1987digital].

# Statement of need

Researchers and practitioners studying Vietnamese legal NLP, low-resource RAG, and continual safety alignment face a fragmented software ecosystem. Legal documents in Vietnam undergo frequent legislative transitions (e.g., the 2024 Land Law, 2024 Social Insurance Law, 2024 VAT Law, and 2025 e-invoicing Decree 70/2025/ND-CP) where operational thresholds span separate Laws, Decrees, and Circulars. Furthermore, empirical studies of continual preference learning under noisy feedback require reproducible multi-round evaluation harnesses that run deterministically even on CPU-only laboratory machines. `Vietnamese AI` addresses this need by uniting temporal statutory retrieval, deterministic 3-tier verification, reproducible zero-GPU benchmarks, and eval-gated continual alignment in a single, self-contained package with minimal dependency overhead.

# State of the field

Existing open-source tools solve isolated parts of this stack but do not integrate them cleanly for low-resource Vietnamese legal research:

- **Vietnamese NLP toolkits** such as Underthesea [@vu2019underthesea] provide morphological processing and classification, but lack point-in-time statutory validity graphs, single-file vector/FTS5 stores, and citation verifiers.
- **General RAG orchestration frameworks** such as LangChain [@chase2022langchain] and LlamaIndex [@liu2022llamaindex] rely on deep dependency trees, incur substantial cold-start latency and RAM overhead on commodity hardware, and provide neither Vietnamese statutory hierarchy parsers nor deterministic temporal/numeric citation auditing.
- **Alignment and inference stacks** such as PyTorch [@paszke2019pytorch], TRL [@vonwerra2022trl], and `llama.cpp` [@gerganov2023llamacpp] excel at GPU fine-tuning or C++ LLM inference, yet do not couple noisy preference quarantine, stratified anchor replay, and 4-axis atomic rollback gates with legal validity checks.

Rather than duplicating upstream C++ or PyTorch engines, `Vietnamese AI` was built as a lightweight NumPy-only core that wraps optional upstream backends (`llama-cpp-python`, `onnxruntime`, `peft`/`trl`) while contributing domain-specific Vietnamese legal temporal graphs, deterministic verifiers, portable `.vaidb` packaging, and eval-gated continual alignment loops that do not fit naturally into any single upstream library.

# Software design

`Vietnamese AI` enforces a strict three-tier boundary inspectable at runtime via `vai status-map`: (i) **Core Production** modules depending solely on Python and `numpy` (exposed via PEP 562 lazy loading across 349 PEP 561 typed symbols); (ii) **Optional Extensions** (`onnx`, `legal-rag`, `edge`, `cloud-gpu`) loaded lazily with explicit validation (e.g., ONNX encoders refuse execution if `tokenizer.json` is missing); and (iii) **Experimental/Simulated** legacy modules marked with `DeprecationWarning`.

Key design trade-offs include:

- **Point-in-time legal graph vs. flat vector RAG**: Let $C_t$ denote statutory clauses active at reference date $t$ over half-open validity intervals $[t_{\text{eff}}(c), t_{\text{exp}}(c))$ linked by directed edges $G_t = (V, E_t)$ (*replaces*, *amends*, *implements*). `BoPhanTichThoiGianPhapLy` disambiguates event origin years from operative evaluation years, `BoTruyHoiPhapLyThoiGian` fuses lexical/dense rankings with temporal redirection and 2-hop graph expansion, and `BoKiemChungTrichDanPhapLy` deterministically checks citation grounding, temporal validity, and numeric/deadline consistency.
- **Zero-GPU portability vs. neural recall**: Default retrieval uses signed feature hashing and Rust SIMD/HNSW inside a single SQLite `.vaidb` file (`24.68 ms` cold-start, `12.82 MB` `VmRSS`), trading some paraphrase recall on colloquial queries for zero-GPU reproducibility while allowing drop-in INT8 ONNX embedders/rerankers (`--onnx-embedder`, `--onnx-reranker`).
- **Eval-gated continual alignment (`ViAlign-SafeLoop`)**: `VongLapCanChinhAnToan` chains a 6-signal preference filter (`BoLocNhieuCapUuTien`), stratified anchor replay (`BoDemOnTapNeo`), weighted cDPO/SimPO (`DPOTrainer`, supporting parameter L2 anchoring and token-level reverse KL $D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$), and a 4-axis evaluation gate (`CongKiemDinhMoHinh`) with atomic rollback. On GPU backends (`NhaCungCapModal`, `NhaCungCapSSH`), `BoDieuPhoiTinhChinhXa` automates PII scrubbing, budget caps, 4-bit QLoRA training, and SHA-256/HMAC artifact verification.

# Research impact statement

`Vietnamese AI` ships three reproducible, seed-controlled benchmark suites and open datasets (documented via a formal datasheet [@gebru2021datasheets] and parity-checked by SHA-256) that enable immediate quantitative research on commodity hardware. All measurements below were executed without a GPU on an **Intel Celeron N3050 @ 1.60 GHz (2 cores, 6W TDP, 3.2 GiB RAM)** running Linux `x86_64`, Python `3.14.4`, and the compiled `vietnamese_ai_rust` extension (`seed=42`). Core import latency is **24.68 ms** (`12.82 MB` `VmRSS`; `37.39 MB` after indexing the 100-clause open corpus at `100.0%` Hit@1 / MRR `1.000`).

On **`ViLegal-TempBench v2`** (`160` statutory clauses, `38` graph edges, `84` queries across 5 splits `T1`--`T5`), temporal date parsing achieves **TQPA = 100.0% (84/84)**:

| Configuration | R@3 | MRR@10 | TVA | TCA (T2) | MH@3 (T3) | T5-R@3 | Abst. F1 |
|:------------------------|--------:|-----------:|--------:|---------:|----------:|----------:|------------:|
| BM25 | 0.880 | 0.810 | 0.731 | 0.083 | 0.500 | 0.625 | 0.000 |
| Hybrid | 0.900 | 0.804 | 0.705 | 0.167 | 0.571 | 0.688 | 0.000 |
| Naive date filter | 0.900 | 0.836 | 0.727 | 0.583 | 0.571 | 0.688 | 0.000 |
| Temporal | 0.932 | 0.917 | 1.000 | 1.000 | 0.786 | 0.750 | 0.000 |
| Temporal + graph | 0.936 | 0.910 | 1.000 | 1.000 | 1.000 | 0.688 | 0.000 |
| **Full** | **0.936** | **0.907** | **0.987** | **1.000** | **1.000** | **0.688** | **0.923** |

Paired bootstrap tests (`n=1000`, `seed=42`) confirm significant improvements of `full` over `naive_date_filter` on MRR@10 (`+0.0709`, $p<0.0001$), TVA (`+0.2607`, $p<0.0001$), and Multi-Hop@3 (`+0.4286`, $p=0.0020$), with `1.000` expired-statute trap handling (`T4`) at `101.80 ms`/query (`7.71 ms` for `bm25`). Automated error analysis on colloquial split `T5` surfaces 4 lexical-gap misses (`t5_01`, `t5_03`, `t5_06`, `t5_12`) and 1 conservative abstention (`t5_13`), motivating optional INT8 ONNX dense models.

On **`ViAlign-SafeBench`** (5 rounds, 70 pairs, 55.7% poisoning on the deterministic 256-dim CPU ranking proxy; QLoRA GPU scripts for `Qwen2.5-1.5B/3B-Instruct` ship in `scripts/run_safeloop_qlora_experiment.py`):

| Strategy | Final | Citation | Validity | Safety | Floor | Filter F1 | Rollbacks |
|:-----------------------|---------:|----------:|----------:|---------:|---------:|-----------:|-----------:|
| Naive DPO | 0.5969 | 0.5625 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| Filtered DPO | 0.9812 | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 0.9041 | 0 |
| Filtered + Replay + cDPO | 0.9906 | 0.9792 | 1.0000 | 1.0000 | 1.0000 | 0.9041 | 0 |
| **ViAlign-SafeLoop** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9041** | **1** |

`ViAlign-SafeLoop` prevents Round-3 safety collapse (`0.0000` under Naive DPO) and automatically rolls back the Round-5 numeric-fabrication stress attack with zero forgetting (`+0.2500` mean delta vs. Naive DPO, 95% CI `[+0.1260, +0.3740]`, $p<0.0001$). All software artifacts are publicly archived on PyPI, GitHub, and Zenodo [@huynh2026vietnamese], verified by 1,450+ automated tests across Python 3.9--3.14 and Rust `abi3-py39` wheels, 0 Ruff errors, and 0 Mypy errors across 343 source files, with one-command reproducibility (`scripts/generate_paper_tables.py --seed 42`).

# AI usage disclosure

During development and manuscript preparation, generative AI coding assistants (Google Antigravity / Gemini models) were used to assist with code refactoring, unit/integration test scaffolding, PEP 561 type stub maintenance, and English academic copy-editing of this manuscript. The human author conceived and directed all software architectures, legal dataset curation, experimental protocols, and benchmark analyses; executed and verified all automated test suites, static checks, and empirical benchmark runs directly; and assumes full scientific, legal, and ethical responsibility for the entire submission.

# Acknowledgements

The author thanks the Vietnamese open-source AI community for feedback and testing support on local-first, low-resource deployments.

# References
