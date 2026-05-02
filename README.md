<p align="center">
  <h1 align="center">Vietnamese AI Framework</h1>
  <p align="center">
    <strong>Framework AI thuần tiếng Việt cho Python</strong>
  </p>
  <p align="center">
    <em>Học máy đơn giản. API tiếng Việt. Production-ready.</em>
  </p>
</p>

<p align="center">
  <a href="#cài-đặt">Cài đặt</a> •
  <a href="#tính-năng">Tính năng</a> •
  <a href="#sử-dụng-nhanh">Sử dụng nhanh</a> •
  <a href="#cli">CLI</a> •
  <a href="#docker">Docker</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#đóng-góp">Đóng góp</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-57%2F57-brightgreen.svg" alt="tests">
  <img src="https://img.shields.io/badge/language-Vietnamese-red.svg" alt="language">
</p>

---

## Giới thiệu

**Vietnamese AI Framework** là framework học máy mã nguồn mở, được thiết kế đặc biệt cho cộng đồng phát triển AI tại Việt Nam. Toàn bộ API, thông báo lỗi, tài liệu và ví dụ đều sử dụng tiếng Việt.

### Tại sao lại là Vietnamese AI?

| Vấn đề hiện tại | Giải pháp của chúng tôi |
|---|---|
| Framework AI lớn quá phức tạp cho người mới | API đơn giản, học trong 5 phút |
| Tài liệu toàn bằng tiếng Anh | Tài liệu và ví dụ 100% tiếng Việt |
| Thiếu toolkit xử lý văn bản tiếng Việt | Tích hợp sẵn underthesea + TF-IDF + stopwords |
| Không có framework "all-in-one" cho người Việt | Models + Preprocessing + CV + Tuning + API + CLI + Docker |

---

## Cài đặt

### Cơ bản

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Với NLP tiếng Việt (underthesea)

```bash
pip install -e ".[nlp]"
```

### Tất cả tính năng

```bash
pip install -e ".[all]"
```

---

## Tính năng

### Models

| Mô hình | Thuật toán |
|---|---|
| `PhanLoai` | logistic, knn, svm, cay_quyet_dinh, rung_ngau_nhien, gradient_boosting, naive_bayes |
| `HoiQuy` | tuyen_tinh, ridge, lasso, elastic_net, svm, cay_quyet_dinh, rung_ngau_nhien, gradient_boosting |
| `PhanCum` | kmeans, dbscan, hierarchical |
| `MangNron` | Custom MLP (ReLU, Sigmoid, Tanh) - không cần TensorFlow/PyTorch |
| `MoHinhTapHop` | voting, bagging, boosting |

### Core

| Module | Chức năng |
|---|---|
| `Engine` | Điều phối huấn luyện, đánh giá, quản lý lịch sử |
| `Pipeline` | Chuỗi tiền xử lý + mô hình, hỗ trợ save/load |
| `KiemDinhCheo` | K-Fold, Stratified K-Fold, Repeated K-Fold |
| `TimKiemThamSo` | GridSearch, RandomSearch |

### NLP & Embeddings

| Module | Chức năng |
|---|---|
| `Word2VecTiengViet` | Word2Vec Skip-gram/CBOW tự cài đặt |
| `FastTextTiengViet` | Character n-gram embeddings (xử lý OOV) |
| `PhanTichCamXuc` | Sentiment: underthesea, từ điển, tự huấn luyện |

### AutoML & Experiment Tracking

| Module | Chức năng |
|---|---|
| `AutoML` | Tự động chọn mô hình + thuật toán tốt nhất |
| `TheoDoiThiNghiem` | Theo dõi thí nghiệm (tương thích MLflow) |

### Interpretability & Augmentation

| Module | Chức năng |
|---|---|
| `GiaiThichMoHinh` | Feature Importance, Permutation Importance, LIME |
| `TangCuongVanBan` | Tăng cường dữ liệu văn bản (đồng nghĩa, xóa từ, hoán vị, thiếu dấu) |

### Preprocessing

| Module | Chức năng |
|---|---|
| `XuLyVanBan` | Tách từ (underthesea), TF-IDF, stopwords, sentiment, POS tagging |
| `XuLySo` | Min-Max, Z-Score, missing values, one-hot, chia dữ liệu |
| `TaoDacTrung` | Polynomial, interaction, PCA, variance selection |

### CLI

```
vai info                              Thông tin framework
vai train --data FILE --model M       Huấn luyện mô hình
vai predict --model M --input FILE    Dự đoán
vai evaluate --model M --data FILE    Đánh giá
vai serve --model M --port 8080       API server
```

---

## Sử dụng nhanh

### Phân loại

```python
from vietnamese_ai import PhanLoai, XuLySo, DuLieuMau

X, y = DuLieuMau.phan_loai_don_gian(so_mau=400)
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)
print(pl.bao_cao(X_test, y_test))
```

### Cross-Validation

```python
from vietnamese_ai import KiemDinhCheo, PhanLoai

kdc = KiemDinhCheo(so_fold=5)
ket_qua = kdc.chay(PhanLoai(thuat_toan="logistic"), X, y)
print(f"Điểm CV: {ket_qua['diem_trung_binh']:.4f} (+/- {ket_qua['do_lech_chuan']:.4f})")
```

### Hyperparameter Tuning

```python
from vietnamese_ai import TimKiemThamSo, PhanLoai

ts = TimKiemThamSo(so_fold=5)
ket_qua = ts.tim_kiem_luoi(
    lop_mo_hinh=PhanLoai,
    luoi_tham_so={
        'thuat_toan': ['logistic', 'knn', 'rung_ngau_nhien'],
    },
    X=X_train, y=y_train
)
print(f"Tham số tốt nhất: {ket_qua['tham_so_tot_nhat']}")
```

### Pipeline + Save/Load

```python
from vietnamese_ai import Pipeline, XuLySo, PhanLoai

pipe = Pipeline()
pipe.them_buoc("chuan_hoa", XuLySo())
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="gradient_boosting"))
pipe.fit(X_train, y_train)

# Lưu
pipe.luu("models/pipe.pkl")

# Tải lại
pipe2 = Pipeline.tai("models/pipe.pkl")
du_doan = pipe2.predict(X_test)
```

### Xử lý văn bản tiếng Việt (underthesea)

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()

# Tách từ chuẩn (nhận diện từ ghép)
xl.tach_tu("Trí tuệ nhân tạo rất hay")
# → ['trí_tuệ_nhân_tạo', 'rất', 'hay']

# Phân tích cảm xúc
xl.phan_tich_cam_xuc("Sản phẩm rất tốt, tôi rất hài lòng")
# → 'positive'

# Gán nhãn từ loại
xl.gan_nhan_tu_loai("Học máy rất thú vị")
# → [('Học', 'V'), ('máy', 'N'), ('rất', 'R'), ('thú vị', 'A')]

# TF-IDF
tfidf = xl.ma_hoa_tfidf(["văn bản 1", "văn bản 2"])
```

### Mạng nơ-ron

```python
from vietnamese_ai import MangNron

mang = MangNron(lop_an=[64, 32], ham_kich_hoat="relu", so_vong=100)
mang.huan_luyen(X_train, y_train)
print(f"Độ chính xác: {mang.danh_gia(X_test, y_test):.4f}")
```

---

## CLI

Sau khi cài đặt, sử dụng lệnh `vai`:

```bash
# Thông tin framework
vai info

# Huấn luyện từ file CSV
vai train --data data.csv --model logistic --output model.pkl --test-size 0.2

# Dự đoán
vai predict --model model.pkl --input new_data.csv --output results.csv

# Đánh giá
vai evaluate --model model.pkl --data test.csv

# Khởi động API server
vai serve --model model.pkl --port 8080
```

---

## Docker

### Build và chạy

```bash
# Build image
docker build -t vietnamese-ai .

# Khởi động API server
docker run -p 8080:8080 -v ./models:/app/models vietnamese-ai serve --model /app/models/model.pkl --port 8080
```

### Docker Compose

```bash
# Tạo thư mục
mkdir -p data models

# Copy dữ liệu vào data/
# Huấn luyện
docker-compose run vai-train

# Khởi động server
docker-compose up vai-serve
```

---

## API Server

```bash
# Khởi động
vai serve --model model.pkl --port 8080

# Gửi request
curl -X POST http://localhost:8080/du_doan \
  -H "Content-Type: application/json" \
  -d '{"du_lieu": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

# Kiểm tra sức khỏe
curl http://localhost:8080/suc_khoe
```

---

## API Reference

### PhanLoai(thuat_toan, ten, **kwargs)

| Phương thức | Mô tả |
|---|---|
| `huan_luyen(X, y)` | Huấn luyện mô hình |
| `du_doan(X)` | Dự đoán nhãn |
| `du_doan_xac_suat(X)` | Dự đoán xác suất |
| `danh_gia(X, y)` | Độ chính xác (0-1) |
| `bao_cao(X, y)` | accuracy, precision, recall, f1 |
| `luu(duong_dan)` | Lưu mô hình |
| `BaseModel.tai(duong_dan)` | Tải mô hình |

### KiemDinhCheo(so_fold, lap_lai, seed)

| Phương thức | Mô tả |
|---|---|
| `chay(mo_hinh, X, y, chi_so)` | Chạy K-Fold CV |

### TimKiemThamSo(so_fold, seed)

| Phương thức | Mô tả |
|---|---|
| `tim_kiem_luoi(lop_mo_hinh, luoi_tham_so, X, y)` | Grid Search |
| `tim_kiem_ngau_nhien(lop_mo_hinh, pham_vi_tham_so, X, y, so_lan)` | Random Search |

### Pipeline(ten)

| Phương thức | Mô tả |
|---|---|
| `them_buoc(ten, bo_xu_ly)` | Thêm bước |
| `fit(X, y)` | Huấn luyện pipeline |
| `predict(X)` | Dự đoán |
| `luu(duong_dan)` | Lưu pipeline |
| `Pipeline.tai(duong_dan)` | Tải pipeline |

### XuLyVanBan(tu_dung, su_dung_underthesea)

| Phương thức | Mô tả |
|---|---|
| `tach_tu(text)` | Tách từ tiếng Việt (underthesea) |
| `phan_tich_cam_xuc(text)` | Sentiment: positive/negative/neutral |
| `gan_nhan_tu_loai(text)` | POS tagging |
| `ma_hoa_tfidf(cac_van_ban)` | TF-IDF matrix |
| `trich_xuat_tu_khoa(text, top_n)` | Keyword extraction |

---

## Cấu trúc thư mục

```
vietnamese-ai/
├── vietnamese_ai/
│   ├── core/              # Engine, Pipeline, KiemDinhCheo, TimKiemThamSo
│   ├── models/            # PhanLoai, HoiQuy, PhanCum, MangNron, MoHinhTapHop
│   ├── preprocessing/     # XuLyVanBan, XuLySo, TaoDacTrung
│   ├── utils/             # Logger, Metrics, Validator, LuuTai
│   ├── visualization/     # BieuDo
│   ├── datasets/          # DuLieuMau
│   ├── api/               # ServerDonGian
│   └── cli/               # CLI (vai command)
├── tests/                 # 45 tests
├── examples/              # Ví dụ sử dụng
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── requirements.txt
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## Chạy test

```bash
pytest tests/ -v
# ========================= 45 passed =========================
```

---

## Roadmap

### v1.0.0 - Production Release

- [x] 6 mô hình học máy (PhanLoai, HoiQuy, PhanCum, MangNron, MoHinhTapHop)
- [x] NLP tiếng Việt (Word2Vec, FastText, Sentiment, underthesea)
- [x] AutoML (tự động chọn mô hình tốt nhất)
- [x] Cross-validation + Hyperparameter Tuning
- [x] Model Interpretability (Feature Importance, Permutation, LIME)
- [x] Experiment Tracking (MLflow compatible)
- [x] Data Augmentation cho văn bản tiếng Việt
- [x] Pipeline (save/load)
- [x] CLI tool (`vai` command)
- [x] Docker + docker-compose
- [x] API Server
- [x] CI/CD (GitHub Actions)
- [x] PyPI publishing (`pip install vietnamese-ai`)
- [x] MkDocs documentation
- [x] 57 test cases

### v1.1.0 - Kế hoạch

- [ ] Deep Learning GPU (PyTorch backend)
- [ ] Pre-trained models (PhoBERT, ViBERT)
- [ ] Image classification (CNN)
- [ ] Time series forecasting
- [ ] No-code/low-code interface

---

## Giấy phép

MIT License - xem [LICENSE](LICENSE)

---

<p align="center">
  <strong>Vietnamese AI Framework</strong> - Được xây dựng với ❤️ cho cộng đồng AI Việt Nam
</p>
