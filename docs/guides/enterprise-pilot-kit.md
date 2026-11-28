# Bộ Triển Khai Thử Nghiệm Doanh Nghiệp & Cộng Đồng (Enterprise Pilot Kit — `v31.3.0`)

Hướng dẫn 15 phút giúp **văn phòng luật, bộ phận pháp chế/nhân sự SME** và **đội phát triển AI tại Việt Nam** chạy thử nghiệm thực tế Vietnamese AI Framework `v31.3.0` 100% cục bộ (không gửi hợp đồng ra ngoài máy văn phòng).

## Kịch bản 1: Pháp chế / Văn phòng luật (100% Offline trên Laptop Văn Phòng)

### 1. Cài đặt & dựng kho tri thức pháp lý `.vaidb`

```bash
pip install "vietnamese-ai[legal-rag]"
vai pull-kb --list
vai pull-kb vn-business-core-2025 --output ./data/vaidb/vn-business-core-2025.vaidb
```

### 2. Rà soát hợp đồng chuyên sâu theo 5 nhóm luật Việt Nam

Hỗ trợ tự động nhận diện hoặc chỉ định rõ `--type lao_dong|dich_vu|mua_ban|thue_nha|nda` kèm xuất tệp Word Track Changes + Comments (`word/comments.xml`):

```bash
vai contract-review hop_dong_doi_tac.docx \
  --template hop_dong_mau.docx \
  --type lao_dong \
  --date 2026-01-15 \
  --track-changes-docx bao_cao_track_changes.docx
```

### 3. Mở Web Studio v2 (SSE Streaming + Split-View Căn cứ Pháp lý)

```bash
vai studio --host 127.0.0.1 --port 8080 \
  --vaidb ./data/vaidb/vn-business-core-2025.vaidb \
  --workspace-dir ./data/workspace \
  --auth
```

Trong Studio:
- **Tab RAG → Mục 6**: nhấn **Stream Trực tiếp (SSE) + Split-View Căn cứ** để xem dòng trả lời trực tuyến và nhấn trực tiếp vào từng thẻ điều luật (nhãn màu Xanh: Còn hiệu lực / Vàng: Đã sửa đổi / Đỏ: Hết hiệu lực).
- **Tab Hợp đồng & Chat Workspace**: chọn 1 trong 5 loại hợp đồng, xem Diff tô màu `+`/`-` ngay trên web và tải Word Track Changes.

---

## Kịch bản 2: Đội Dev AI Cắm MCP IDE & Triển Khai Docker trong 60 Giây

```bash
vai mcp-config --client cursor --kb ./data/vaidb/vn-business-core-2025.vaidb --write .cursor/mcp.json
vai mcp-config --client vscode --kb ./data/vaidb/vn-business-core-2025.vaidb --write .vscode/mcp.json
vai mcp-config --client claude --kb ./data/vaidb/vn-business-core-2025.vaidb
```

Hoặc triển khai trọn bộ qua Docker Compose (mặc định chỉ lắng nghe loopback `127.0.0.1` an toàn):

```bash
docker compose up -d
docker compose --profile mcp up -d
docker compose --profile connectors up -d
```

---

## Kịch bản 3: Vòng Lặp Phản Hồi Thực Tế → `ViAlign-SafeLoop` & Đóng Góp Dữ Liệu

1. Người dùng đánh giá **Thumbs Up / Thumbs Down** (có thể nhập `cau_tra_loi_sua`) trực tiếp trong Chat Workspace.
2. Xuất tập DPO sạch đã ẩn PII và lọc 6 tín hiệu nhiễu/đầu độc:
   ```bash
   vai export-dpo --workspace-dir ./data/workspace --filter-noise --output pilot_dpo.jsonl
   ```
3. Kiểm định hồi quy 4 trục hoặc dựng kế hoạch QLoRA `ViAlign-SafeLoop`:
   ```bash
   vai eval-gate --compare-baseline --extended-attacks
   python scripts/run_safeloop_qlora_experiment.py --mode qlora-modal --model Qwen/Qwen2.5-1.5B-Instruct --gpu L4
   ```

## Mẫu Phiếu Ghi Nhận Phản Hồi Pilot (Nạp ngược vào `ViLegal-TempBench`)

Khi nhóm thử nghiệm phát hiện câu hỏi đời thường hoặc tình huống hợp đồng còn sót, gửi Issue theo mẫu tối giản:

- **Lĩnh vực**: Lao động / Thuế-Hóa đơn / Đất đai-Nhà ở / Hợp đồng thương mại / NDA
- **Câu hỏi đời thường (không nêu số hiệu luật)**: `...`
- **Mốc ngày áp dụng**: `YYYY-MM-DD`
- **Điều khoản đúng kỳ vọng**: `Điều ... Luật/Nghị định ...`
- **Phản hồi thực tế của hệ thống & ghi chú sai sót**: `...`
