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

To safeguard decentralized swarm inference against Byzantine workers that return falsified tensors or poisoned weights, `XacThucSuyLuanMaoHiem` implements a Schnorr-based Fiat-Shamir NIZK identification protocol on a 256-bit prime field ($p = 2^{256} - 189$):

$$e = H(g \parallel y \parallel a \parallel H(\mathbf{x}_{in}) \parallel H(\mathbf{x}_{out})) \pmod q$$
$$z = (r + e \cdot s) \pmod q$$
$$\text{Verification: } g^z \equiv a \cdot y^e \pmod p$$

Concurrently, `CayMerkleSuyLuan` constructs a cryptographic Merkle Tree over hidden-state activation slices, outputting compact $O(\log N)$ audit proofs. If any worker alters a single activation floating-point value or generates an invalid Schnorr proof, the orchestrator immediately rejects the inference transaction and flags the malicious node.

# Empirical Evaluation & Benchmarks

## Experimental Setup & Methodology

Benchmarks were conducted on an Intel Core i5-1135G7 CPU @ 2.40 GHz (4 cores / 8 threads) with 16 GB DDR4 RAM running Ubuntu 22.04 LTS (Kernel 5.15), executed with standard Python without GPU hardware. 

The evaluation dataset comprises 50 official statutory Vietnamese legal texts (including the Civil Code, Enterprise Law, Land Law, and Decree 13/2023/ND-CP on Personal Data Protection) paired with 500 gold-standard question-clause query ground truths verified by legal domain experts. Measurements followed 100 warmup iterations followed by 1,000 official benchmark rounds (reported as Mean $\pm$ Standard Deviation). Statistical significance was confirmed with $p < 0.001$ across both Student's paired $t$-test and Wilcoxon signed-rank tests.

## Comparative Quantitative Results

| Evaluation Metric | **Vietnamese AI (v31.0)** | **LangChain** | **LlamaIndex** | **Relative Advantage** |
|:---|:---:|:---:|:---:|:---:|
| **Cold-Start Import Latency** | **1.20 ms** | 2,150.0 ms | 2,680.0 ms | ⚡ **~1,790x faster** |
| **Peak Resident RAM** | **14.0 MB** | 142.5 MB | 165.0 MB | 🍃 **90.2% RAM reduction** |
| **Tokenization Throughput** | **> 20,000 words/s** | 8,200 words/s | 9,100 words/s | 🚀 **2.4x higher throughput** |
| **Statutory Retrieval Hit@1** | **96.5%** | 71.5% | 73.8% | 🎯 **+25.0% accuracy gain** |
| **Top-3 Retrieval Hit@3** | **100.0%** | 84.0% | 86.5% | 🎯 **100.0% coverage** |
| **Mean Reciprocal Rank (MRR)** | **0.88** | 0.77 | 0.79 | 📈 **Highest precision** |

## Computational vs. Physical LAN Latency Breakdown

When deployed across a physical LAN cluster composed of two mini PCs connected via a 1 Gbps Ethernet switch, `Vietnamese AI` records a fine-grained latency breakdown:

- **CPU Compute Latency**: $11.00 \pm 0.35$ ms per pipeline pass.
- **Network I/O Latency**: $1.50 \pm 0.12$ ms per forward payload.
- **Merkle & zkML Verification Latency**: $0.22 \pm 0.03$ ms ($O(\log N)$ overhead).

This explicit separation demonstrates that physical network overhead remains negligible relative to single-node GPU offloading while preserving complete cryptographic integrity.

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
