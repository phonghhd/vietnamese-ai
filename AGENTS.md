# AGENTS.md - Vietnamese AI Framework

## Project Info
- **Name**: Vietnamese AI Framework (`vietnamese-ai`)
- **Version**: 31.3.0
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

### Build Docs & Release Check
```bash
mkdocs build --strict
vai release-check
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
├── rag/                 # RAGPipeline, KhoTriThucVAIDB (.vaidb), onnx_semantic, legal/ (100 điều luật + hiệu lực + đồ thị + hợp đồng)
├── connectors/          # KetNoiZaloOA, KetNoiTelegram, MayChuWebhookKenh (lưu SQLite Workspace)
├── solutions/           # TroLyPhapLy, TongDaiCSKH, TrichXuatThongMinh
├── ui/                  # StudioDashboard (6 tabs), QuanLyPhienChatWorkspace (SQLite + DPO), V-UI
├── serving/             # Batch, Streaming, Rate Limiter, Speculative
├── prompts/             # Prompt Templates, Chains, Guardrails
├── compression/         # Distillation, Pruning, BitNet
├── production/          # Health, Circuit Breaker, Logging, Metrics
├── nlp/                 # Native Tokenizer (38k+ từ nén + Rust parity), NER, QA, Summarization, Translation, Spelling (38k+ từ)
├── models/              # PhanLoai, HoiQuy, PhanCum, MangNron
├── core/                # Engine, Pipeline, CrossValidation, QuyTrinhDauCuoi
├── preprocessing/       # XuLyVanBan, XuLySo, TaoDacTrung
├── fine_tuning/         # SFT, DPO, RLHF, LoRA, PEFT, remote_gpu (Modal/SSH), eval_gate (4 trục)
├── transformer/         # GPT, Tokenizer, Attention
├── llm/                 # VietnameseLLM, Eval, Benchmark
├── automl/              # AutoML, NAS
├── embeddings/          # Word2Vec, FastText
├── federated/           # Federated Learning
├── mobile/              # Mobile Deployment
├── enterprise/          # Auth (RBAC), Audit, SaaS platform (NenTangDichVu)
├── studio/              # No-code Studio
├── agents/              # StateGraph, DeepResearch, CodeHealing, MayChuMCPThucTe (stdio/SSE)
├── edge/                # QuanLyMoHinhEdge, DongCoSuyLuanEdge, Swarm CPU (một số tính năng DePIN mô phỏng)
├── orchestrator/        # Cluster: NutChinh / DieuPhoiCum
├── extreme/             # JIT / kernel fusion (thử nghiệm)
└── ...
```

## Important Notes

- All tests must pass before commit: `pytest tests/ -q --tb=short`
- Lint, type check & release gate must pass: `ruff check vietnamese_ai/ tests/ benchmarks/`, `mypy vietnamese_ai/ --ignore-missing-imports`, `vai release-check`, `mkdocs build --strict`
- No comments in code unless asked
- Follow existing code patterns (see models/base.py for base class pattern)
- New modules need: `__init__.py` with exports, implementation file, tests (and update `__init__.pyi` if adding top-level exports)
- Một số module Edge/DePIN/multimodal/WASM là **mô phỏng** — kiểm tra trực tiếp qua `vai status-map` hoặc docstring từng class
- Version canonical: **31.3.0** (`__init__.py` == `pyproject.toml` == `setup.py` == `packages/*/pyproject.toml` == `rust_core/Cargo.toml`)
- Plugin: `QuanLyPlugin` = hot-swap (tự gắn vào FastAPI & MCP); `PluginManager` = legacy hooks (cả hai đều hợp lệ); tài chính tách tại `vietnamese_ai/plugins/finance/`
- Baseline test (v31.3.0): 1,450+ passed, 0 mypy errors (`check_untyped_defs = true` cho `rag/`, `security/`, `edge/`, `core/`), 349 public exports (đã hoàn tất trọn bộ P0 → P4 và M1 → M4)
