# Contributing to Vietnamese AI Framework

> 🇻🇳 Phiên bản tiếng Việt: [CONTRIBUTING.md](CONTRIBUTING.md)

Thank you for your interest in contributing to **Vietnamese AI Framework**! We welcome bug reports, benchmark evaluations, legal dataset improvements, documentation enhancements, and pull requests from researchers and developers worldwide.

All participants are expected to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Optional Rust acceleration (`abi3-py39`, compatible with Python 3.9–3.14):

```bash
pip install maturin
cd rust_core && maturin develop --release && cd ..
```

## Running Tests & Quality Gates

All pull requests must pass the full CI verification gates before merging:

```bash
pytest tests/ -q --tb=short
ruff check vietnamese_ai/ tests/ benchmarks/ scripts/
mypy vietnamese_ai/ --ignore-missing-imports
vai release-check
python scripts/sync_legal_datasets.py --check
mkdocs build --strict
```

## Reporting Bugs & Requesting Features

- **Bugs / Issues**: Open a GitHub Issue at `https://github.com/phonghhd/vietnamese-ai/issues` with your OS, Python version, minimal reproduction snippet, and expected vs. actual output.
- **Security Vulnerabilities**: Please follow the responsible disclosure procedure in [SECURITY.md](SECURITY.md) rather than opening a public issue.
- **Legal Benchmark Additions (`ViLegal-TempBench`)**: When proposing new statutory clauses or QA pairs, include the official Vietnamese statute number, effective date (`YYYY-MM-DD`), and run `python scripts/sync_legal_datasets.py --sync` to keep `vietnamese_ai/rag/legal/data/` and `benchmarks/datasets/` SHA-256 synchronized.

## Coding Conventions

- Public API identifiers use Vietnamese no-diacritics `snake_case` for functions/methods and `PascalCase` for classes, with English aliases exported for key classes (e.g., `TemporalGraphLegalRAG = BoTruyHoiPhapLyThoiGian`, `LegalCitationVerifier = BoKiemChungTrichDanPhapLy`, `ContractReviewEngine = BoSoSanhRaSoatHopDong`).
- Full PEP 484 type annotations are required (`check_untyped_defs = true` is enforced on `rag/`, `security/`, `edge/`, and `core/`).
- If adding top-level exports to `vietnamese_ai/__init__.py`, update `vietnamese_ai/__init__.pyi` in the same PR and verify with `vai release-check`.

## Pull Request Process

1. Fork the repository and create a topic branch from `main` (`git checkout -b feat/your-feature`).
2. Implement changes with real execution unit/integration tests in `tests/`.
3. Verify all quality gates listed above pass with zero errors.
4. Submit a Pull Request referencing any related issue and summarizing test results.
