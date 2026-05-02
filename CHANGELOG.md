# CHANGELOG

## v1.0.0 (2026-05-03)

### Production Release

- Phiên bản chính thức đầu tiên
- 57 test cases PASSED
- Hỗ trợ Python 3.8 - 3.13
- CI/CD với GitHub Actions
- PyPI publishing (`pip install vietnamese-ai`)
- MkDocs documentation site
- pyproject.toml (PEP 621)
- Makefile cho development workflow
- MANIFEST.in cho source distribution

## v0.3.0 (2026-05-03)

### Thêm mới
- **Word2Vec tiếng Việt** (`Word2VecTiengViet`): Skip-gram + Negative Sampling, tự cài đặt
- **FastText tiếng Việt** (`FastTextTiengViet`): Character n-gram, xử lý từ mới (OOV)
- **Sentiment Analysis** (`PhanTichCamXuc`): 3 chế độ (underthesea, từ điển, tự huấn luyện)
- **Model Interpretability** (`GiaiThichMoHinh`): Feature Importance, Permutation Importance, LIME cơ bản
- **Experiment Tracking** (`TheoDoiThiNghiem`): Theo dõi thí nghiệm, tương thích MLflow API
- **AutoML** (`AutoML`): Tự động phát hiện nhiệm vụ, thử tất cả thuật toán, chọn mô hình tốt nhất
- **Data Augmentation** (`TangCuongVanBan`): 5 kỹ thuật tăng cường văn bản tiếng Việt
- Thêm 12 test cases mới (tổng: 57 tests)

### Cải thiện
- Cập nhật README với badges, tables, roadmap chuẩn quốc tế
- Thêm CHANGELOG.md

## v0.2.0 (2026-05-02)

### Thêm mới
- **underthesea integration**: Tách từ tiếng Việt chuẩn, POS tagging, sentiment
- **Cross-validation** (`KiemDinhCheo`): K-Fold, Stratified K-Fold
- **Hyperparameter Tuning** (`TimKiemThamSo`): GridSearch, RandomSearch
- **Save/Load Pipeline**: `pipe.luu()` / `Pipeline.tai()`
- **CLI tool**: `vai info/train/predict/evaluate/serve`
- **Docker**: Dockerfile + docker-compose.yml
- **.gitignore**

### Cải thiện
- Pipeline hỗ trợ `fit_transform`, `transform` từ `XuLySo`
- `XuLySo` thêm `phuong_phap` parameter (minmax/zscore)

## v0.1.0 (2026-05-02)

### Thêm mới
- **Models**: PhanLoai (7 thuật toán), HoiQuy (8), PhanCum (3), MangNron (custom MLP), MoHinhTapHop (3)
- **Preprocessing**: XuLyVanBan, XuLySo, TaoDacTrung
- **Core**: Engine, Pipeline
- **Utils**: Logger, Metrics, Validator, LuuTai
- **Visualization**: BieuDo
- **Datasets**: DuLieuMau
- **API**: ServerDonGian
- 45 test cases
