# AGENTS.md - Vietnamese AI Framework

## Project Info
- **Name**: Vietnamese AI Framework (`vietnamese-ai`)
- **Version**: 30.2.0
- **Language**: Python 3.8+
- **Package dir**: `vietnamese_ai/`

## Development Commands

### Install
```bash
pip install -e ".[all]"
```

### Run Tests
```bash
pytest tests/ -v --tb=short
```

### Run Tests with Coverage
```bash
pytest tests/ -v --tb=short --cov=vietnamese_ai --cov-report=html
```

### Lint
```bash
ruff check vietnamese_ai/ tests/
```

### Format
```bash
black vietnamese_ai/ tests/ examples/
ruff check vietnamese_ai/ tests/ --fix
```

### Type Check
```bash
mypy vietnamese_ai/ --ignore-missing-imports
```

### Build Docs
```bash
mkdocs build
```

### Build Package
```bash
python -m build
```

## Code Conventions

- **API names**: Vietnamese, no-diacritics, snake_case (`phan_loai`, `huan_luyen`)
- **Classes**: PascalCase (`PhanLoai`, `XuLyVanBan`)
- **Constants**: UPPER_SNAKE_CASE
- **Private**: prefix `_` (`_mo_hinh`)
- **Docstrings**: Vietnamese, for all public classes/methods
- **Type hints**: Required for all parameters and return values
- **Error messages**: Vietnamese

## Module Structure

```
vietnamese_ai/
├── rag/                 # RAG Pipeline (text + graph + multimodal)
├── serving/             # Batch, Streaming, Rate Limiter, Speculative
├── prompts/             # Prompt Templates, Chains, Guardrails
├── compression/         # Distillation, Pruning, BitNet
├── production/          # Health, Circuit Breaker, Logging, Metrics
├── nlp/                 # NER, QA, Summarization, Translation, Spelling
├── models/              # PhanLoai, HoiQuy, PhanCum, MangNron
├── core/                # Engine, Pipeline, CrossValidation
├── preprocessing/       # XuLyVanBan, XuLySo, TaoDacTrung
├── fine_tuning/         # SFT, DPO, RLHF, LoRA, PEFT
├── transformer/         # GPT, Tokenizer, Attention
├── llm/                 # VietnameseLLM (n-gram), Eval, Benchmark
├── automl/              # AutoML, NAS
├── embeddings/          # Word2Vec, FastText
├── federated/           # Federated Learning
├── mobile/              # Mobile Deployment
├── enterprise/          # Auth, Audit, SaaS platform (NenTangDichVu)
├── studio/              # No-code Studio
├── agents/              # Multi-agent, Swarm, MoA, MCTS
├── edge/                # Edge routing (một số tính năng mô phỏng)
├── orchestrator/        # Cluster: NutChinh / DieuPhoiCum
├── extreme/             # JIT / kernel fusion (thử nghiệm)
└── ...
```

## Important Notes

- All tests must pass before commit: `pytest tests/ -v`
- Lint must pass: `ruff check vietnamese_ai/ tests/`
- No comments in code unless asked
- Follow existing code patterns (see models/base.py for base class pattern)
- New modules need: `__init__.py` with exports, implementation file, tests
- Một số module Edge/DePIN/multimodal/WASM là **mô phỏng** — xem docstring từng class
- Version canonical: **30.2.0** (`__init__.py` == `pyproject.toml` == `setup.py`)
- Plugin: `QuanLyPlugin` = hot-swap; `PluginManager` = legacy hooks (cả hai đều hợp lệ)
- Baseline test (v30): ~959 passed, ~31 skipped (torch/optional deps)
