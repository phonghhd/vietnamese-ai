# JOSS (Journal of Open Source Software) Pre-Submission Checklist

Bản kiểm tra trước khi nộp bài báo phần mềm [paper/paper.md](paper.md) cho **JOSS** (`https://joss.theoj.org/papers/new`).

## 1. Yêu cầu kho mã nguồn & giấy phép
- [x] **Open-source license (OSI-approved)**: `LICENSE` (MIT License) tại thư mục gốc repository.
- [x] **Public Git repository & Release tag**: `https://github.com/phonghhd/vietnamese-ai` kèm PyPI package (`vietnamese-ai`) và Zenodo DOI (`10.5281/zenodo.20357082`).
- [x] **Citation metadata**: `CITATION.cff` chuẩn CFF 1.2.0 khớp phiên bản, ORCID (`0009-0002-1458-3406`) và phạm vi dataset (`84 QA`, 5 splits).
- [x] **Community Guidelines**: `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md` đầy đủ hướng dẫn báo lỗi, gửi PR và kiểm thử.

## 2. Tài liệu & trải nghiệm 5 phút
- [x] **Installation & Quickstart**: `README.md`, `docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`, `docs/guides/mcp-cursor-claude.md`.
- [x] **API Documentation**: MkDocs Material build sạch với `mkdocs build --strict` (0 lỗi liên kết).
- [x] **Reproducibility CLI**: `vai pull-kb`, `vai mcp-config`, `vai benchmark --suite legal-temporal --latex`, `scripts/generate_paper_tables.py`, `scripts/run_safeloop_qlora_experiment.py`.

## 3. Chất lượng phần mềm & kiểm thử tự động
- [x] **Automated Tests**: > 1.450 unit/security/integration tests xanh trên GitHub Actions CI (Python 3.9–3.13 + local Python 3.14 + Rust `abi3-py39` trên Linux/macOS/Windows).
- [x] **Static Quality Gates**: `ruff check` (0 lỗi), `mypy` (0 lỗi trên 343 tệp nguồn, `check_untyped_defs = true` cho nhóm lõi), `vai release-check` ĐẠT, `scripts/sync_legal_datasets.py --check` 100% SHA-256.
- [x] **Transparency & Scope Honesty**: Tách bạch rõ `Core Production` vs `Optional Extensions` vs `Experimental/Simulated` qua `vai status-map`; công bố rõ giới hạn lexical CPU proxy vs optional ONNX INT8 / GPU QLoRA.

## 4. Kiểm tra bản thảo `paper/paper.md` & `paper/paper.bib`
- [x] **YAML Frontmatter hợp lệ**: `title`, `tags`, `authors` (có ORCID), `affiliations`, `date`, `bibliography: paper.bib`.
- [x] **Đủ các mục bắt buộc của JOSS**: `Summary`, `Statement of Need`, `Core Architecture`, `Empirical Evaluation & Benchmarks`, `Software Quality & Reproducibility`, `AI Use Disclosure`, `Acknowledgements`, `References`.
- [x] **Trích dẫn BibTeX khớp 100%**: Mọi khóa `@...` trong `paper/paper.md` đều có mục tương ứng trong `paper/paper.bib`.
- [x] **Độ dài chuẩn JOSS**: ~1.100–1.300 từ (trong vùng khuyến nghị của JOSS ~1.000–1.500 từ).

## 5. Lệnh biên dịch PDF thử nghiệm

```bash
docker run --rm --volume "$PWD/paper:/data" --user $(id -u):$(id -g) \
  --env JOURNAL=joss openjournals/inara
```
