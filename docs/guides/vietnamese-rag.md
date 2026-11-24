# RAG Pháp Lý & Hành Chính Việt Nam (100 Điều Khoản, Theo Dõi Hiệu Lực & Studio Kéo-Thả)

Khi áp dụng RAG thông thường (Naive RAG) cho văn bản quy phạm pháp luật Việt Nam, hệ thống thường gặp 4 lỗi nghiêm trọng:
1. **Xé rời cấu trúc thứ bậc**: Làm mất liên kết `Phần > Chương > Mục > Điều > Khoản > Điểm`.
2. **Đánh mất điều kiện ngoại lệ**: Cắt rời Khoản 1 (quy định chung) khỏi Khoản 2 (ngoại lệ loại trừ).
3. **Viện dẫn nhầm luật hết hiệu lực**: Trích dẫn Luật Đất đai `45/2013/QH13` hay Luật BHXH `58/2014/QH13` dù các luật này đã bị thay thế bởi `31/2024/QH15` và `41/2024/QH15`.
4. **Khó nạp hồ sơ thực tế**: Tài liệu văn phòng nằm rải rác ở `.docx`, `.pdf`, ảnh chụp scan OCR và bảng biểu.

Vietnamese AI Framework giải quyết trọn vẹn cả 4 bài toán qua cụm mô-đun `vietnamese_ai.rag.legal` (`BoToiUuNguCanhHanhChinh`, `BoDocTaiLieuPhapLy`, `BoTheoDoiHieuLucPhapLy`, `tai_bo_phap_ly_mo`) tích hợp trực tiếp vào `RAGPipeline`, `TroLyPhapLy`, CLI và Web Studio.

---

## 1. Kiến trúc Đường ống Legal RAG Đầy đủ

```text
[Tệp .docx / .pdf / Ảnh OCR / .txt / Thư mục luật]
                       │
                       ▼
          [BoDocTaiLieuPhapLy]
           ├─ Đọc DOCX (giữ bảng biểu Markdown, có fallback OOXML thuần Python)
           ├─ Đọc PDF (pdfplumber / pypdf) & Ảnh scan OCR (khử độc CDR)
                       │
                       ▼
        [BoToiUuNguCanhHanhChinh]
           ├─ Nhận diện Chương > Mục > Điều > Khoản > Điểm (Nghị định 30/2020/NĐ-CP)
           ├─ Trích xuất số hiệu văn bản & căn cứ pháp lý
           └─ Phân đoạn bảo toàn ngữ cảnh cha-con (Hierarchy-Preserving Chunking)
                       │
                       ▼
       [RAGPipeline + Chỉ mục IVF-Flat]
           ├─ Tìm kiếm lai BM25 tiếng Việt + Vector Blake2b / Embedding
           └─ Tự động kiểm toán qua BoTheoDoiHieuLucPhapLy:
               • Gắn trạng thái hiệu lực theo ngày tra cứu
               • Giảm điểm (0.65x) hoặc lọc bỏ văn bản hết hiệu lực
               • Xuất danh sách cảnh báo sửa đổi / thay thế
```

---

## 2. Nạp Tệp Đa Định Dạng (`.docx`, `.pdf`, Ảnh OCR, Thư mục) & Hỏi Đáp

```python
from vietnamese_ai import RAGPipeline

rag = RAGPipeline(che_do_hanh_chinh=True, top_k=3)

# Nạp một tệp .docx / .pdf / .txt / ảnh scan / .jsonl cụ thể
rag.them_tep("hop_dong_chuyen_nhuong.docx", che_do_hanh_chinh=True)

# Hoặc nạp nguyên một thư mục hồ sơ pháp lý
rag.them_thu_muc("kho_van_ban_phap_ly/", de_quy=True, che_do_hanh_chinh=True)

# (Tùy chọn) Tạo chỉ mục vector IVF-Flat cho kho tài liệu lớn
rag.toi_uu_chi_muc_vector(so_cum=16, nprobe=3)

ket_qua = rag.hoi("Quyền chuyển nhượng quyền sử dụng đất theo Luật Đất đai được quy định thế nào?")
print("Trả lời:\n", ket_qua["tra_loi"])

trich_dan = ket_qua.get("trich_dan_phap_ly", {})
print("Điều khoản căn cứ:", trich_dan.get("dieu_khoan"))
print("Số hiệu văn bản:", trich_dan.get("so_hieu_van_ban"))
print("Cảnh báo hiệu lực:", trich_dan.get("canh_bao_hieu_luc"))
```

### Sử dụng nhanh từ CLI (`vai rag --admin`)

```bash
vai rag \
  --docs hop_dong_chuyen_nhuong.docx \
  --admin \
  --query "Hợp đồng viện dẫn những căn cứ pháp lý nào và có văn bản nào hết hiệu lực không?"
```

---

## 3. Kho Pháp Lý Mở 100 Điều Khoản & `TroLyPhapLy`

Framework đóng gói sẵn **100 điều khoản pháp luật Việt Nam thực tế** (Bộ luật Lao động 2019, Bộ luật Dân sự 2015, Luật Doanh nghiệp 2020, Luật Đất đai 2024, Luật Giao dịch điện tử 2023, Nghị định `13/2023/NĐ-CP`, Nghị định `30/2020/NĐ-CP`, Luật BHXH 2024 `41/2024/QH15`, Luật Thuế GTGT 2024 `48/2024/QH15`, Luật Quản lý thuế 2019, Luật SHTT, Luật Đấu thầu 2023, Luật Nhà ở 2023, Luật KDBĐS 2023, Nghị định `70/2025/NĐ-CP`, Luật XLVPHC...) cùng **35 cặp Hỏi–Đáp (QA) chuẩn kiểm thử**.

```python
from vietnamese_ai import TroLyPhapLy, tai_bo_hoi_dap_phap_ly_mo, tai_bo_phap_ly_mo

print("Tổng điều khoản mở:", len(tai_bo_phap_ly_mo()))        # 100
print("Tổng cặp QA chuẩn:", len(tai_bo_hoi_dap_phap_ly_mo()))  # 35

# Nạp trọn bộ 100 điều khoản chỉ bằng một tham số `nap_kho_mo=True`
tro_ly = TroLyPhapLy(nap_kho_mo=True)

kq = tro_ly.tra_cuu("Theo Luật Bảo hiểm xã hội 2024 cần đóng tối thiểu bao nhiêu năm để hưởng lương hưu?")
print(kq["tra_loi"])
print("Điểm bám sát nguồn:", kq["diem_bam_sat_nguon"])
```

---

## 4. Theo Dõi Hiệu Lực Văn Bản Pháp Luật (`BoTheoDoiHieuLucPhapLy`)

### Kiểm tra theo mốc thời gian (Point-in-Time Legal Audit)

```python
from vietnamese_ai import BoTheoDoiHieuLucPhapLy, HoSoHieuLucVanBan

tracker = BoTheoDoiHieuLucPhapLy()

# Kiểm tra lịch sử vs hiện tại
print(tracker.kiem_tra_hieu_luc("45/2013/QH13", ngay_tra_cuu="2022-01-01")["trang_thai"])
# -> "con_hieu_luc"

kq_het = tracker.kiem_tra_hieu_luc("45/2013/QH13", ngay_tra_cuu="2026-01-01")
print(kq_het["trang_thai"], "->", kq_het["bi_thay_the_boi"])
# -> "het_hieu_luc" -> ['31/2024/QH15']

kq_sua = tracker.kiem_tra_hieu_luc("123/2020/NĐ-CP", ngay_tra_cuu="2026-01-01")
print(kq_sua["trang_thai"], "->", kq_sua["sua_doi_bo_sung_boi"])
# -> "con_hieu_luc_da_sua_doi" -> ['70/2025/NĐ-CP']
```

### Quét toàn bộ trích dẫn trong hợp đồng / công văn

```python
hop_dong = "Hợp đồng căn cứ Luật Đất đai 45/2013/QH13 và Nghị định 123/2020/NĐ-CP về hóa đơn."
bao_cao = tracker.kiem_tra_hieu_luc_trong_van_ban(hop_dong, ngay_tra_cuu="2026-01-15")

print("Hợp lệ hiệu lực:", bao_cao["hop_le_hieu_luc"])
for canh_bao in bao_cao["canh_bao"]:
    print(" -", canh_bao)
```

### Đăng ký bổ sung văn bản mới của tổ chức / bộ ngành

```python
tracker.dang_ky_van_ban(
    HoSoHieuLucVanBan(
        so_hieu="99/2026/NĐ-CP",
        ten_van_ban="Nghị định mới thay thế quy định cũ",
        linh_vuc="Hành chính",
        ngay_ban_hanh="2026-02-01",
        ngay_hieu_luc="2026-03-01",
        thay_the_cho=["30/2020/NĐ-CP"],
    )
)
```

---

## 5. Kéo-Thả Tài liệu & Kiểm toán Hiệu lực trên Web Studio

Khởi chạy giao diện không cần viết mã (hỗ trợ gắn sẵn `.vaidb`, thư mục Workspace SQLite đa phòng ban và xác thực RBAC):

```bash
vai studio --vaidb kho_luat_cong_ty.vaidb --workspace-dir ./vai_studio_workspaces --port 8080
```

Mở trình duyệt tại `http://127.0.0.1:8080`, chuyển sang **Tab RAG Tri Thức**:
1. Kéo-thả trực tiếp tệp `.docx`, `.pdf`, `.txt`, `.json` hoặc ảnh chụp văn bản (`.png`/`.jpg`) vào khung tải lên (giới hạn an toàn 20 MB); có thể chỉ định `ma_workspace` (ví dụ `phap_che`, `nhan_su`) để lưu thẳng vào tệp `.vaidb` tách biệt của phòng ban đó.
2. Hệ thống tự tách cấu trúc Điều/Khoản qua `BoDocTaiLieuPhapLy`, nạp vào RAG (ngữ nghĩa `onnx_viet`) và trả ngay **báo cáo kiểm toán hiệu lực** cùng **đồ thị tham chiếu văn bản thay thế/hướng dẫn (`do_thi_tham_chieu`)**.
3. Có thể gọi trực tiếp qua HTTP API cho ứng dụng nội bộ:
   - `POST /api/rag/upload-file`: Nhận `{filename, content_base64, che_do_hanh_chinh, ma_workspace}`.
   - `POST /api/legal/check-validity`: Nhận `{so_hieu, ngay_tra_cuu}` hoặc `{van_ban, ngay_tra_cuu}`.
   - `POST /api/legal/citation-graph`: Nhận `{so_hieu, ngay_tra_cuu, chi_lay_con_hieu_luc}` trả kèm sơ đồ Mermaid.

---

## 6. Kho Tri Thức Bền Vững `.vaidb`, Ngữ Nghĩa Đồng Nghĩa ONNX & Đồ Thị Tham Chiếu Luật

### Lưu trữ bền vững 1 tệp `.vaidb` + Đồng bộ tăng dần theo SHA-256

Không cần dựng máy chủ Chroma/Qdrant rời: toàn bộ văn bản, chỉ mục toàn văn SQLite **FTS5 BM25** (hỗ trợ cả tiếng Việt có dấu lẫn không dấu) và ma trận vector nằm gọn trong một tệp `.vaidb` duy nhất:

```python
from vietnamese_ai import RAGPipeline

rag = RAGPipeline(
    backend="vaidb",
    duong_dan_vaidb="kho_luat_doanh_nghiep.vaidb",
    che_do_nhung="onnx_viet",
    che_do_rerank="onnx_viet",
    che_do_hanh_chinh=True,
)

# Lần đầu lập chỉ mục cả thư mục; các lần sau tự bỏ qua tệp chưa thay đổi SHA-256
bc = rag.dong_bo_thu_muc("thu_muc_hop_dong/", de_quy=True, xoa_tep_da_mat=True)
print("Cập nhật:", bc["so_tep_cap_nhat"], "| Bỏ qua:", bc["so_tep_bo_qua_khong_doi"])

# Hiểu cả câu hỏi đồng nghĩa đời thường ("sa thải", "sổ đỏ", "lộ dữ liệu", "hóa đơn đỏ")
kq = rag.hoi("Công ty sa thải nhân viên trước hạn phải báo trước bao lâu?")
print(kq["tra_loi"])

# Hoặc đóng gói / sao lưu trực tiếp bất kỳ RAGPipeline nào sang 1 tệp `.vaidb` và nạp lại bằng `tai()`
rag.luu("ban_sao_kho_luat.vaidb")
rag.dong()

rag_moi = RAGPipeline(che_do_nhung="onnx_viet", che_do_rerank="onnx_viet", che_do_hanh_chinh=True)
rag_moi.tai("ban_sao_kho_luat.vaidb")
rag_moi.dong()
```

### Đồ thị Tham chiếu Pháp lý (`DoThiThamChieuPhapLy` & `vai legal-graph`)

Tự động kéo kèm **Nghị định hướng dẫn thi hành**, **Thông tư mới nhất** và **Luật thay thế hiện hành** ngay cả khi người dùng chỉ biết số hiệu luật cũ đã hết hiệu lực:

```python
from vietnamese_ai import DoThiThamChieuPhapLy

do_thi = DoThiThamChieuPhapLy()

# Hỏi Luật Đất đai cũ 45/2013/QH13 -> tự mở rộng sang Luật 31/2024/QH15 + NĐ 102/2024/NĐ-CP & 101/2024/NĐ-CP
kq = do_thi.truy_van_van_ban_lien_quan("45/2013/QH13", ngay_tra_cuu="2026-01-15", chi_lay_con_hieu_luc=True)
print("Thay thế hiện hành:", [x["so_hieu"] for x in kq["van_ban_thay_the_hien_hanh"]])
print("Nghị định hướng dẫn:", [x["so_hieu"] for x in kq["van_ban_huong_dan"]])
print(do_thi.xuat_do_thi_mermaid("45/2013/QH13"))
```

Hoặc tra cứu tức thì trên dòng lệnh CLI:

```bash
vai legal-graph 45/2013/QH13 --active-only --mermaid
vai rag --db kho_luat.vaidb --docs thu_muc_luat/ --admin --semantic-onnx --query "lộ dữ liệu cá nhân báo ai?"
# Lần truy vấn sau mở thẳng .vaidb không cần nạp lại --docs:
vai rag --db kho_luat.vaidb --admin --semantic-onnx --query "lộ dữ liệu cá nhân báo ai?"
```

---

## 7. Cài đặt Nhóm Mở rộng PDF/OCR & ONNX Runtime (Tùy chọn)

Lõi đọc `.txt`, `.md`, `.json`, `.jsonl`, `.docx` (OOXML), kho `.vaidb` (SQLite FTS5) và chế độ ngữ nghĩa đồng nghĩa tiếng Việt chạy hoàn toàn không cần thư viện ngoài. Để mở khóa đọc bảng `.pdf`, OCR ảnh scan và chạy trực tiếp tệp mô hình ONNX INT8 trên CPU:

```bash
pip install "vietnamese-ai[legal-rag,export]"
# Hoặc dùng trình tự động 3 tầng:
vai setup --profile legal-rag --apply
```

Xem đầy đủ API chi tiết tại [Legal & Administrative RAG API Reference](../api/legal-rag.md) và [RAG API Reference](../api/rag.md).
