# Kiến Trúc Chiều Sâu & Tích Hợp 5 Bề Mặt Giao Tiếp

Tài liệu này hướng dẫn khai thác đồng bộ toàn bộ 6 năng lực chiều sâu của **Vietnamese AI Framework (`v31.0.4`)** trên cả 5 bề mặt người dùng: **Python API (`TroLyPhapLy`)**, **CLI (`vai`)**, **FastAPI REST Server**, **MCP Server cho IDE** và **Web Studio Dashboard**.

---

## 1. Tổng Quan 6 Năng Lực Chiều Sâu

| Trụ cột | Mô-đun cốt lõi | Giá trị thực tiễn |
|---|---|---|
| **NLP Tiếng Việt FST & Viterbi DAG** | `BoTachTuTiengVietSieuToc`, `vietnamese_ai_rust` | Chuẩn hóa dấu thanh âm vị học (`qu`/`gi`/âm tiết mở & khép) và tách từ ghép tối ưu toàn cục Viterbi DAG khớp tuyệt đối giữa Rust và Python. |
| **Pháp lý Point-in-Time** | `BoTheoDoiHieuLucPhapLy`, `PhienBanDieuKhoan` | Tra cứu chính xác nội dung một **Điều/Khoản** có hiệu lực tại mốc ngày bất kỳ (ví dụ `Điều 19 Nghị định 123/2020/NĐ-CP` trước và sau `01/06/2025` khi `70/2025/NĐ-CP` có hiệu lực). |
| **Hợp đồng Tách/Gộp & Word Track Changes** | `BoSoSanhRaSoatHopDong` | Phát hiện đối tác tách 1 điều khoản thành nhiều điều (hoặc gộp lại), đồng thời xuất tệp `.docx` chuẩn OOXML chứa `<w:del>`, `<w:ins>` và nhận xét lề `word/comments.xml`. |
| **Kho `.vaidb` Persistent HNSW** | `KhoTriThucVAIDB` | Dựng đồ thị đa tầng HNSW, lưu bền vững xuống 2 bảng SQLite (`vai_hnsw_meta`, `vai_hnsw_nodes`), tự nạp lại khi mở tệp và hỗ trợ lọc metadata trước khi trả kết quả. |
| **Rust Zero-Copy Kernels & Freivalds** | `vietnamese_ai_rust`, `KiemChungFreivalds` | Tăng tốc giải lượng tử `Q5_K`/`Q6_K`, `RMSNorm`, `SwiGLU` qua FFI `PyBytes` không cấp phát `PyFloat`, kết hợp xác thực nhân ma trận $O(k \cdot n^2)$ bắt 100% gian lận Byzantine. |
| **SimPO & Eval Gate Bootstrap 95% CI** | `DPOTrainer`, `CongKiemDinhMoHinh`, `ThucThiDocLap` | Tinh chỉnh SimPO không cần `ref_model` (tiết kiệm 50% VRAM), kiểm định hồi quy có khoảng tin cậy Paired Bootstrap 95% CI, tường lửa chống Prompt Injection không dấu/Zero-Width và Sandbox giới hạn RAM cấp HĐH (`RLIMIT_AS`). |

---

## 2. Sử Dụng Qua Python API (`TroLyPhapLy`)

```python
from vietnamese_ai import TroLyPhapLy

tro_ly = TroLyPhapLy(nap_kho_mo=True, duong_dan_vaidb="tri_thuc_luat.vaidb")

pb_2024 = tro_ly.tra_cuu_dieu_khoan_theo_ngay(
    so_hieu="123/2020/NĐ-CP",
    dieu="Điều 19",
    ngay_tra_cuu="2024-12-01",
)
pb_2025 = tro_ly.tra_cuu_dieu_khoan_theo_ngay(
    so_hieu="123/2020/NĐ-CP",
    dieu="Điều 19",
    ngay_tra_cuu="2025-07-01",
)

kq = tro_ly.tra_cuu(
    "Điều 19 Nghị định 123/2020/NĐ-CP quy định xử lý hóa đơn sai sót thế nào?",
    ngay_tra_cuu="2025-07-01",
)
print(kq["tra_loi"])

hnsw_info = tro_ly.xay_dung_chi_muc_hnsw(m=12, ef_construction=64, ef_search=48)
print("Đã dựng HNSW:", hnsw_info)
tro_ly.rag.dong()
```

---

## 3. Sử Dụng Qua CLI (`vai`)

```bash
# 1. Tách từ Viterbi DAG & chuẩn hóa dấu thanh âm vị học FST
vai tokenize "Hoà Bình toà án nhân dân và cơ quan nhà nước pháp quyền" --method viterbi --normalize-tone

# 2. Dựng & lưu bền vững đồ thị HNSW trong tệp .vaidb
vai rag --docs thu_muc_luat/ --db kho_luat.vaidb --admin --build-hnsw

# 3. Tra cứu phiên bản Điều/Khoản theo mốc ngày (Point-in-Time)
vai legal-graph 123/2020/NĐ-CP --dieu "Điều 19" --date 2025-07-01

# 4. Rà soát hợp đồng tách/gộp điều khoản & xuất Word Track Changes + Comments (.docx)
vai contract-review hop_dong_doi_tac.docx \
  --template hop_dong_chuan.docx \
  --track-changes-docx hop_dong_track_changes.docx

# 5. Khởi động MCP Server với đầy đủ 9 công cụ (6 mặc định + 3 công cụ mở rộng)
vai mcp-server --transport stdio --vaidb kho_luat.vaidb --extended-tools

# 6. Chạy Eval Gate so sánh bản gốc kèm Paired Bootstrap 95% CI & bộ tấn công mở rộng
vai eval-gate --limit 8 --compare-baseline --extended-attacks --json

# 7. Nạp tự động thư mục văn bản pháp quy / văn bản hợp nhất -> 1 tệp .vaidb HNSW + Point-in-Time + Đồ thị
vai legal-ingest thu_muc_vbhn/ --output kho_phap_quy.vaidb --json
```

### Cross-Encoder Reranker ONNX Tiếng Việt (BGE-Reranker-v2-m3 / PhoRanker)

```python
from vietnamese_ai import RAGPipeline

rag = RAGPipeline(
    backend="vaidb",
    duong_dan_vaidb="kho_phap_quy.vaidb",
    che_do_nhung="onnx_viet",
    che_do_rerank="bge_reranker",
    duong_dan_onnx_rerank="models/bge_reranker_v2_m3_int8.onnx",
    che_do_hanh_chinh=True,
)
```

---

## 4. Sử Dụng Qua FastAPI REST Server

```python
from vietnamese_ai import FastAPIServer, TroLyPhapLy

server = FastAPIServer(bo_xu_ly=TroLyPhapLy(nap_kho_mo=True, duong_dan_vaidb="kho_luat.vaidb"))
server.chay(host="127.0.0.1", port=8000)
```

Các endpoint REST mới:

| Endpoint | Phương thức | Chức năng |
|---|---|---|
| `/v1/legal/clause-version` | `POST` | Tra cứu phiên bản Điều/Khoản Point-in-Time (`so_hieu`, `dieu`, `khoan`, `ngay_tra_cuu`) |
| `/v1/legal/contract-review` | `POST` | So sánh hợp đồng tách/gộp + quét rủi ro + tùy chọn `xuat_track_changes_docx: true` trả về `track_changes_docx_base64` |
| `/v1/nlp/tokenize` | `POST` | Chuẩn hóa dấu thanh FST & tách từ (`phuong_phap`: `"viterbi"` hoặc `"max_match"`) |
| `/v1/vaidb/hnsw` | `POST` | Xem thống kê hoặc dựng chỉ mục HNSW bền vững trên `.vaidb` gắn với bộ xử lý |

---

## 5. Sử Dụng Qua MCP IDE Server & Web Studio Dashboard

### MCP Server (Cursor / VS Code / Claude Desktop)
- **Mặc định (6 công cụ)**: `tra_cuu_phap_ly` hỗ trợ thêm `ngay_tra_cuu`; `kiem_tra_hieu_luc` hỗ trợ thêm `dieu`, `khoan`; `ra_soat_hop_dong` hỗ trợ thêm `ngay_tra_cuu` và `xuat_track_changes_docx`.
- **Chế độ mở rộng (`--extended-tools` / `bat_cong_cu_mo_rong=True` — 9 công cụ)**: bổ sung thêm `tra_cuu_dieu_khoan_theo_ngay`, `tach_tu_tieng_viet` và `xay_dung_chi_muc_hnsw_vaidb`.

### Web Studio Dashboard (`vai studio --vaidb kho_luat.vaidb --auth`)
- **Tab 3 (RAG & Tri thức)**: nút dựng chỉ mục HNSW (`POST /api/rag/vaidb/build-hnsw`, bảo vệ RBAC quyền `"manage"`), biểu mẫu tra cứu Điều/Khoản theo ngày (`POST /api/legal/clause-version`), và khung thử nghiệm NLP FST/Viterbi (`POST /api/nlp/tokenize`).
- **Tab 4 (Huấn luyện & Flywheel)**: chọn tác vụ `SimPO` không cần `ref_model` (`POST /api/training/start`) và nút chạy Eval Gate + Paired Bootstrap 95% CI (`POST /api/flywheel/eval-gate`).
- **Tab 5 (Giải pháp Pháp lý)**: trường chọn ngày hiệu lực `ngay_tra_cuu`.
- **Tab 6 (Hợp đồng & Chat Workspace)**: trường ngày kiểm toán hiệu lực và nút tải trực tiếp tệp Word Track Changes + Comments (`.docx`).
