---
title: 'Vietnamese AI: A Lightweight, Zero-Dependency, and Verifiable Framework for Vietnamese-Centric AI and Distributed Swarm CPU Inference'
tags:
  - Python
  - Vietnamese NLP
  - Zero-Dependency
  - Distributed Inference
  - Swarm CPU
  - Pipeline Parallelism
  - Verifiable AI
  - Non-Interactive Zero-Knowledge Proofs
authors:
  - name: Duong Phong Huynh
    orcid: 0009-0002-1458-3406
    affiliation: 1
affiliations:
  - name: Independent Researcher, Ho Chi Minh City, Vietnam
    index: 1
date: 08 November 2026
bibliography: paper.bib
---

# Summary

The rapid expansion of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems has predominantly targeted high-resource languages and high-end GPU infrastructures. Applying conventional LLM frameworks (e.g., LangChain and LlamaIndex) to low-resource or complex morphosyntactic languages like Vietnamese often introduces severe performance bottlenecks: high cold-start latencies, excessive memory footprints, and token fragmentation caused by compound words and diacritical marks. Furthermore, decentralized edge environments frequently face adversarial threats such as Byzantine nodes falsifying tensor activations during collaborative inference.

`Vietnamese AI` is a production-grade, zero-dependency Python framework designed to resolve these challenges. The framework delivers three core technological pillars:
1. **Vietnamese-Centric Linguistic Engine**: A native compound-word tokenizer, syllable-level vector embedder, and hierarchy-preserving administrative RAG tailored for Vietnamese statutory legal corpora.
2. **Distributed Swarm CPU Inference**: A model-sharding pipeline parallelism engine (`NutSwarmCPU`, `DieuPhoiSwarmCPU`) that connects commodity office CPUs and Mini PCs across physical local area networks (LAN) to run large models collaboratively without GPU acceleration [@huang2019gpipe].
3. **Cryptographically Verifiable AI Inference (PoI)**: A zero-knowledge verifiable inference layer combining Fiat-Shamir Schnorr Non-Interactive Zero-Knowledge (NIZK) commitments with $O(\log N)$ Merkle audit paths to enforce 100% Byzantine fault tolerance against activation tampering [@schnorr1991efficient; @fiat1986how; @merkle1987digital].

# Statement of Need

Contemporary agentic and RAG workflows rely heavily on large runtime stacks such as PyTorch [@paszke2019pytorch], LangChain [@chase2022langchain], and LlamaIndex [@liu2022llamaindex]. On commodity edge hardware (e.g., office PCs with 2–4 GB RAM and no discrete GPUs), importing these frameworks incurs prohibitive initialization delays (often exceeding 2,000 ms) and consumes over 140 MB of RAM before processing a single token. Moreover, standard subword tokenizers frequently decompose Vietnamese multi-syllabic terms (e.g., *Ủy ban nhân dân* or *Nghị định*) into disjoint phonetic sub-units, corrupting administrative context in specialized domains such as law, governance, and public administration.

While existing Vietnamese NLP toolkits like Underthesea [@vu2019underthesea] offer morphological processing, they lack native integration with distributed pipeline parallelism, agentic state graphs, and cryptographic verification for distributed edge environments. `Vietnamese AI` bridges this gap by providing an end-to-end, zero-dependency architecture that maintains high computational throughput, strict memory parsimony, and provable cryptographic authenticity directly on standard CPU infrastructure.

# Core Architecture & System Innovations

`Vietnamese AI` is structured into modular, decoupled components conforming to strict zero-dependency and defensive programming guidelines:

```
vietnamese_ai/
├── edge/               # Swarm CPU pipeline parallelism (NutSwarmCPU, DieuPhoiSwarmCPU)
├── security/           # Verifiable inference (CayMerkleSuyLuan, XacThucSuyLuanMaoHiem)
├── nlp/                # Native fast tokenizer & vector embedder
├── rag/                # Administrative context chunker & statutory legal retriever
├── agents/             # Agentic StateGraph, DeepResearchAgent, CodeHealingAgent
└── benchmark/          # Comparative benchmark suite (Markdown & LaTeX exporters)
```

## 1. Vietnamese-Centric Linguistic Engine & Administrative RAG

Vietnamese is an isolating language where words are composed of individual syllables separated by whitespace. The native tokenizer (`BoTachTuTiengVietSieuToc`) implements a maximum-matching compound dictionary lookup, Unicode NFC normalization, modern/traditional diacritic harmonization (`hòa/hoà`, `thúy/thuý`), and diacritic stripping with processing speeds exceeding 20,000 words/second. 

The accompanying vectorizer (`BoNhungVanBanSieuToc`) maps tokens into an $L_2$-normalized 256-dimensional subword hashing space, yielding fast cosine similarity lookups ($< 0.05$ ms) without external neural dependencies. For legal corpora, `BoToiUuNguCanhHanhChinh` parses statutory hierarchies (Constitution, Laws, Decrees, Chapters, Articles, and Clauses) to preserve parental context during chunking and guarantee exact statutory citation.

## 2. Distributed Swarm CPU Pipeline Parallelism

To enable model execution across multiple resource-constrained office computers, `DieuPhoiSwarmCPU` shards linear transformation blocks and feed-forward activation layers across discrete worker nodes (`NutSwarmCPU`). Model inference operates via continuous pipeline parallelism:

$$\mathbf{x}_{l+1} = \sigma\left(\mathbf{x}_l \mathbf{W}_l + \mathbf{b}_l\right) + \alpha \mathbf{x}_l$$

where hidden states $\mathbf{x}_l$ are passed sequentially through worker nodes over local Gigabit LAN. The orchestrator explicitly supports both local in-process simulation (`khoi_tao_cum_gia_lap`) for unit testing and distributed physical network transport (`them_node_mang_vat_ly`) over HTTP/RPC socket endpoints.

## 3. Cryptographic Verifiable AI Inference (Proof of Inference)

To safeguard decentralized swarm inference against Byzantine workers that return falsified tensors or poisoned weights, `XacThucSuyLuanMaoHiem` implements a Schnorr-based Fiat-Shamir NIZK identification protocol over the RFC 5114 2048-bit MODP Group with a 256-bit prime-order subgroup ($p$: 2048-bit prime, $q$: 256-bit prime, $g$ of order $q$):

$$e = H(\text{domain} \parallel g \parallel y \parallel a \parallel \text{node\_id} \parallel \text{layer\_id} \parallel H(\mathbf{x}_{in}) \parallel H(\mathbf{x}_{out})) \pmod q$$
$$z = (r + e \cdot s) \pmod q$$
$$\text{Verification: } g^z \equiv a \cdot y^e \pmod p$$

Concurrently, `CayMerkleSuyLuan` constructs an RFC 6962 domain-separated cryptographic Merkle Tree (using `0x00` leaf and `0x01` internal node prefixes without odd-leaf duplication) over hidden-state activation slices, outputting compact $O(\log N)$ audit proofs. For single-layer linear projections $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$, `KiemChungFreivalds` optionally performs randomized $O(k \cdot n^2)$ matrix multiplication verification.

# Empirical Evaluation & Benchmarks

## Experimental Setup & Methodology

All benchmarks were executed directly without GPU acceleration on a commodity low-power office machine: **Intel Celeron N3050 @ 1.60 GHz (2 physical cores, 6W TDP, SSE4.2/POPCNT, no AVX2, no discrete GPU)** with **3.2 GiB RAM**, running Linux `x86_64` and Python `3.14.4` with the compiled `vietnamese_ai_rust` extension (`abi3-py39`).

Two open, version-controlled Vietnamese legal datasets bundled directly in the package (`vietnamese_ai/rag/legal/data/`, verified via SHA-256 parity with `benchmarks/datasets/`) are used for reproducible evaluation:
1. **Open Vietnamese Legal Corpus (`legal_vn_corpus.jsonl` & `legal_vn_qa.jsonl`)**: `100` statutory clauses spanning key Vietnamese laws paired with `35` ground-truth question-clause queries evaluated via `Hit@1`, `Hit@3`, and `MRR`.
2. **`ViLegal-TempBench` (`vilegal_tempbench_*.jsonl`)**: `160` indexed statutory clauses (`60` multi-version temporal clauses + `100` open background clauses), `38` clause-level temporal graph edges, and `68` queries across 4 splits (`T1`: 16 single-hop, `T2`: 24 temporal contrast pairs, `T3`: 14 multi-hop Law–Decree–Circular chains, `T4`: 8 expired-statute traps + 6 out-of-scope abstention queries).

Process startup and memory metrics are measured in isolated Python subprocesses using `time.perf_counter()` and Linux `/proc/self/status` (`VmRSS`) to eliminate parent-process peak RSS contamination. External frameworks (e.g., LangChain or LlamaIndex) are only benchmarked head-to-head when explicitly installed in the same runtime environment; no static external baseline tables are fabricated.

## Direct Quantitative Results (`v31.1.0`)

| Evaluation Metric | **Measured Value (`v31.1.0`)** | Measurement Protocol |
|:---|:---:|:---|
| **Cold-Start Import Latency** | **24.68 ms** | Isolated subprocess `import vietnamese_ai` |
| **Core Import Resident RAM (`VmRSS`)** | **12.82 MB** | `/proc/self/status` `VmRSS` after core import |
| **Legal RAG Resident RAM (`VmRSS`)** | **37.39 MB** | `/proc/self/status` `VmRSS` indexing + querying 100 clauses |
| **Tokenization Throughput (Rust Trie)** | **354,047.7 words/s** | `BoTachTuTiengVietSieuToc` on 100-clause legal corpus |
| **Open Legal RAG Hit@1 / Hit@3 / MRR** | **100.0% / 100.0% / 1.000** | `RAGPipeline` (`onnx_viet`) on 100 clauses / 35 QA |
| **`ViLegal-TempBench` Temporal Parsing (TQPA)** | **100.0% (68/68)** | `BoPhanTichThoiGianPhapLy` zero-leak evaluation |
| **`ViLegal-TempBench` Full Ablation (`full`)** | **R@3 = 1.000, MRR@10 = 0.968, TVA = 1.000** | TCA = `12/12`, Multi-Hop@3 = `1.000`, Trap = `1.000`, Abstention F1 = `1.000` (`47.02 ms`/query) |

## Ablation Study on `ViLegal-TempBench` (68 Queries, `seed=42`)

| Configuration | R@1 | R@3 | MRR@10 | TVA | TCA (T2) | Multi-Hop (T3) | Trap (T4) | Abst. F1 | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25` (VN Tokenizer) | 0.562 | 0.973 | 0.839 | 0.720 | 0.000 (0/12) | 0.786 | 0.125 | 0.000 | 6.64 |
| `hybrid` (BM25 + Dense RRF) | 0.562 | 0.949 | 0.827 | 0.694 | 0.167 (2/12) | 0.714 | 0.125 | 0.000 | 15.13 |
| `temporal` (+ Point-in-Time) | 0.836 | 0.981 | 0.968 | 1.000 | 1.000 (12/12) | 0.857 | 1.000 | 0.000 | 20.10 |
| `temporal_graph` (+ Clause Graph) | 0.836 | 1.000 | 0.968 | 1.000 | 1.000 (12/12) | 1.000 | 1.000 | 0.000 | 22.67 |
| **`full` (ViLegal-TemporalRAG)** | **0.836** | **1.000** | **0.968** | **1.000** | **1.000 (12/12)** | **1.000** | **1.000** | **1.000** | **47.02** |

Paired bootstrap testing (`n=1000`, `seed=42`) confirms statistically significant improvements of `full` over `bm25` on MRR@10 (`+0.1290`, $p=0.0010$) and TVA (`+0.2796`, $p<0.0001$), and over `hybrid` across Recall@3 (`+0.0511`, $p=0.0290$), MRR@10 (`+0.1411`, $p<0.0001$), TVA (`+0.3065`, $p<0.0001$), and Multi-Hop@3 (`+0.2857`, $p=0.0370$).

## Edge Systems & Computational Transparency

In addition to legal retrieval, the framework provides reproducible edge systems benchmarks (`vai benchmark --suite edge-systems`) measuring Rust SIMD SQ8/Binary/HNSW vector search and 2-bit packed BitNet 1.58-bit T-MAC LUT inference directly on the same Celeron N3050 host, as well as physical vs. in-process latency breakdown (`phan_ra_do_tre`) for Swarm CPU pipeline parallelism.

# Software Quality & Reproducibility

The framework adheres to high standards of software quality:
- **Comprehensive Automated Testing**: Over 1,076 automated unit and integration tests covering core modules, edge sharding, Byzantine detection, and API endpoints (`pytest tests/ -v`).
- **Code Hygiene**: 100% compliance with Ruff linting rules and PEP 561 static type stub definitions (`vietnamese_ai/py.typed`, `vietnamese_ai/__init__.pyi`).
- **Reproducibility CLI**: All benchmark tables and swarm verification runs are reproducible with standalone CLI commands:
  ```bash
  pip install -e .
  vai benchmark --compare
  vai swarm --prompt "Chuyển đổi số quốc gia" --nodes 2 --bench
  vai verify
  ```

# AI Use Disclosure

During the preparation of this project and manuscript, the author utilized generative AI coding assistants (including Google Antigravity / Gemini models) to assist with code refactoring, test suite scaffolding, PEP 561 static type stub generation, and English grammar/academic drafting of the manuscript. 

The human author confirms that all core system architectures, algorithm designs (including the zero-dependency pipeline parallelism, Vietnamese administrative RAG hierarchy, and Fiat-Shamir Schnorr cryptographic commitments), and experimental methodologies were conceived, directed, and evaluated by the author. The author thoroughly reviewed, debugged, tested, and validated all software code and text, and assumes full scientific, legal, and ethical responsibility for the entirety of the submitted work.

# Acknowledgements

The author thanks the Vietnamese open-source AI community for valuable insights, testing feedback, and continuous support in advancing native AI technologies for low-resource environments.

# References
