# Datasheet for `ViLegal-TempBench v2`

*Theo khung chuẩn Datasheets for Datasets (Gebru et al., 2021).*

## 1. Động cơ xây dựng (Motivation)
- **Mục đích**: Cung cấp bộ chuẩn mở, có thể tái lập 100% trên CPU để đánh giá hệ thống RAG pháp lý tiếng Việt về: (1) truy hồi đúng phiên bản điều luật theo mốc thời gian hiệu lực (*Point-in-Time Validity*), (2) suy luận đa bước trên đồ thị Luật → Nghị định → Thông tư, (3) khả năng phát hiện bẫy văn bản hết hiệu lực và từ chối trả lời ngoài phạm vi (*Abstention*), và (4) khả năng hiểu câu hỏi đời thường không chứa số hiệu luật (`T5`).
- **Tác giả & đơn vị duy trì**: Duong Phong Huynh (`phonghhd`), dự án mã nguồn mở `Vietnamese AI Framework` (MIT License).

## 2. Thành phần dữ liệu (Composition)
Bộ dữ liệu gồm 3 tệp JSONL đồng bộ 100% SHA-256 tại `vietnamese_ai/rag/legal/data/` và `benchmarks/datasets/` (kiểm tra tự động qua `scripts/sync_legal_datasets.py --check`):
1. **`vilegal_tempbench_corpus.jsonl`** (`60` điều khoản đa phiên bản theo thời gian) + **`legal_vn_corpus.jsonl`** (`100` điều khoản nền hiện hành) = **`160` điều khoản pháp lý** phủ 10 lĩnh vực cốt lõi doanh nghiệp (Lao động, BHXH, Thuế GTGT & Hóa đơn điện tử, Đất đai, Nhà ở, Kinh doanh BĐS, Bảo vệ dữ liệu cá nhân, Doanh nghiệp, Dân sự, Thương mại).
2. **`vilegal_tempbench_graph.jsonl`**: **`38` cạnh đồ thị có hướng cấp điều khoản** (`thay_the`, `sua_doi`, `huong_dan`, `tham_chieu`) kèm mốc ngày hiệu lực.
3. **`vilegal_tempbench_qa.jsonl`**: **`84` câu hỏi kiểm chuẩn** chia 5 split tường minh:
   - `T1` (`16` câu): Tra cứu 1 bước văn bản hiện hành.
   - `T2` (`24` câu = `12` cặp tương phản): Cùng câu hỏi ở 2 mốc thời gian trước/sau hiệu lực chuyển tiếp.
   - `T3` (`14` câu): Suy luận đa bước 2–3 hop qua chuỗi Luật ↔ Nghị định ↔ Thông tư.
   - `T4` (`14` câu): `8` bẫy viện dẫn văn bản hết hiệu lực + `6` câu ngoài phạm vi (OOD) đo Abstention F1.
   - `T5` (`16` câu): Câu hỏi ngôn ngữ đời thường/paraphrase của HR, kế toán, chủ doanh nghiệp (không chứa bất kỳ số hiệu văn bản nào).

Mỗi bản ghi QA có trường: `id`, `split`, `cau_hoi`, `ngay_tra_cuu_chuan`, `id_dieu_khoan_dung`, `id_dieu_khoan_het_hieu_luc`, `so_hieu_mong_doi`, `can_tu_choi`, `cap_tuong_phan` (nếu thuộc `T2`), và `tu_khoa_so_lieu`.

## 3. Quy trình thu thập & làm sạch (Collection & Preprocessing)
- **Nguồn văn bản**: Tổng hợp từ các văn bản quy phạm pháp luật công khai của Quốc hội và Chính phủ nước CHXHCN Việt Nam (thuộc phạm vi không bảo hộ quyền tác giả theo Điều 15 Luật Sở hữu trí tuệ Việt Nam).
- **Chuẩn hóa**: Áp dụng chuẩn hóa Unicode NFC, đồng nhất vị trí dấu thanh tiếng Việt hiện đại/cổ điển và gắn nhãn khoảng hiệu lực nửa mở $[t_{\text{eff}}, t_{\text{exp}})$.
- **Quyền riêng tư**: Dữ liệu không chứa bất kỳ thông tin định danh cá nhân (PII) nào của cá nhân thực tế.

## 4. Mục đích sử dụng & giới hạn (Uses & Limitations)
- **Phù hợp cho**: Đánh giá đối chứng retriever lexical/dense/ONNX, temporal graph RAG, deterministic citation verifiers, và end-to-end LLM hallucination benchmarks.
- **Không phù hợp cho**: Thay thế tư vấn pháp lý chính thức của luật sư hành nghề; suy luận ngoài các văn bản/thời điểm đã lập chỉ mục.
- **Giới hạn hiện tại**: Bản đóng gói nhẹ trong wheel gồm `160` điều khoản và `84` câu QA để chạy tái lập nhanh trên CPU; quy mô toàn văn 1.500+ điều khoản được phân phối tách rời dưới dạng `.vaidb` knowledge pack qua lệnh `vai pull-kb`.

## 5. Phân phối & bảo trì (Distribution & Maintenance)
- **Giấy phép**: MIT License (kho mã nguồn: `https://github.com/phonghhd/vietnamese-ai`, Zenodo DOI: `10.5281/zenodo.20357082`).
- **Xác minh toàn vẹn**: `python scripts/sync_legal_datasets.py --check`.
- **Cập nhật phiên bản**: Mọi sửa đổi luật mới tại Việt Nam được gắn thẻ phiên bản theo SemVer và ghi rõ trong `CHANGELOG.md`.
