# JOSS (Journal of Open Source Software) Pre-Submission Checklist

Bản kiểm tra trước khi nộp bài báo phần mềm [paper/paper.md](paper.md) cho **JOSS** (`https://joss.theoj.org/papers/new`).

## 1. Yêu cầu kho mã nguồn, giấy phép & cộng đồng
- [x] **Open-source license (OSI-approved)**: `LICENSE` (MIT License) tại thư mục gốc repository.
- [x] **Public Git repository & Release tag**: `https://github.com/phonghhd/vietnamese-ai` kèm PyPI package (`vietnamese-ai` `v31.3.0`) và Zenodo DOI (`10.5281/zenodo.20357082`).
- [x] **Citation metadata**: `CITATION.cff` chuẩn CFF 1.2.0 khớp phiên bản `31.3.0`, ORCID (`0009-0002-1458-3406`) và phạm vi dataset (`84 QA`, 5 splits).
- [x] **Community Guidelines & English Reviewer Docs**: [README_EN.md](../README_EN.md), [CONTRIBUTING_EN.md](../CONTRIBUTING_EN.md), [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md), [SECURITY.md](../SECURITY.md), [CHANGELOG.md](../CHANGELOG.md).

## 2. Tài liệu & trải nghiệm 5 phút
- [x] **Installation & Quickstart (VI & EN)**: `README.md`, `README_EN.md`, `docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`, `docs/guides/mcp-cursor-claude.md`, `docs/guides/enterprise-pilot-kit.md`.
- [x] **API Documentation**: MkDocs Material build sạch với `mkdocs build --strict` (0 lỗi liên kết).
- [x] **Reproducibility CLI**: `vai pull-kb`, `vai mcp-config`, `vai benchmark --suite legal-temporal --latex`, `scripts/generate_paper_tables.py`, `scripts/run_safeloop_qlora_experiment.py`.

## 3. Chất lượng phần mềm & kiểm thử tự động
- [x] **Automated Tests**: > 1.450 unit/security/integration tests xanh trên GitHub Actions CI (Python 3.9–3.13 + local Python 3.14 + Rust `abi3-py39` trên Linux/macOS/Windows).
- [x] **Static Quality Gates**: `ruff check` (0 lỗi), `mypy` (0 lỗi trên 343 tệp nguồn, `check_untyped_defs = true` cho nhóm lõi), `vai release-check` ĐẠT, `scripts/sync_legal_datasets.py --check` 100% SHA-256.
- [x] **Transparency & Scope Honesty**: Tách bạch rõ `Core Production` vs `Optional Extensions` vs `Experimental/Simulated` qua `vai status-map`; công bố rõ giới hạn lexical CPU proxy vs optional ONNX INT8 / GPU QLoRA.

## 4. Kiểm tra bản thảo `paper/paper.md` & `paper/paper.bib`
- [x] **Thư mục `paper/` gọn sạch chuẩn JOSS**: chỉ chứa `paper.md`, `paper.bib`, `JOSS_CHECKLIST.md` (bản thảo ACL & Datasheet đã tách sang `docs/research/`).
- [x] **Đủ 6 mục bắt buộc theo chuẩn JOSS mới nhất**: `Summary`, `Statement of need`, `State of the field` (có lập luận *build vs. contribute*), `Software design`, `Research impact statement`, `AI usage disclosure`, kèm `Acknowledgements` và `References`.
- [x] **Bảng gọn chống tràn lề PDF**: cả 2 bảng thực nghiệm rút gọn còn 8 cột chuẩn trang PDF JOSS, không chứa ký tự box-drawing gây lỗi font XeLaTeX.
- [x] **Trích dẫn BibTeX khớp 100%**: 20 mục trong `paper/paper.bib`, mọi khóa `@...` trong `paper/paper.md` đều tồn tại hợp lệ.
- [x] **GitHub Action biên dịch PDF tự động**: [`.github/workflows/draft-pdf.yml`](../.github/workflows/draft-pdf.yml) chạy `openjournals/inara:latest` sinh artifact `paper.pdf` + `paper.jats`.

## 5. Lệnh biên dịch PDF thử nghiệm cục bộ (Docker)

```bash
docker run --rm --volume "$PWD/paper:/data" --user $(id -u):$(id -g) \
  --env JOURNAL=joss openjournals/inara:latest -o pdf,jats paper.md
```
