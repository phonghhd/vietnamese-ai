# AGENTS.md - Vietnamese AI Framework

## Project Info
- **Name**: Vietnamese AI Framework (`vietnamese-ai`)
- **Version**: 11.0.0
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
├── rag/              # RAG Pipeline (v10)
├── serving/          # Batch Server, Streaming, Rate Limiter (v10)
├── prompts/          # Prompt Templates, Chains, Guardrails (v10)
├── compression/      # Knowledge Distillation, Pruning (v10)
├── production/       # Health Check, Circuit Breaker, Logging, Metrics (v10)
├── nlp/              # NER, QA, Summarization, Translation, Spelling (v10)
├── models/           # PhanLoai, HoiQuy, PhanCum, MangNron
├── core/             # Engine, Pipeline, CrossValidation
├── preprocessing/    # XuLyVanBan, XuLySo, TaoDacTrung
├── fine_tuning/      # SFT, DPO, RLHF, LoRA, PEFT
├── transformer/      # GPT, Tokenizer, Attention
├── llm/              # VietnameseLLLM, Eval, Benchmark
├── automl/           # AutoML, NAS
├── embeddings/       # Word2Vec, FastText
├── federated/        # Federated Learning
├── mobile/           # Mobile Deployment
├── saas/             # SaaS Platform
├── studio/           # No-code Studio
└── ...               # Other modules
```

## Important Notes

- All tests must pass before commit: `pytest tests/ -v`
- Lint must pass: `ruff check vietnamese_ai/ tests/`
- No comments in code unless asked
- Follow existing code patterns (see models/base.py for base class pattern)
- New modules need: `__init__.py` with exports, implementation file, tests
