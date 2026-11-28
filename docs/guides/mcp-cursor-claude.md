# Cắm Vietnamese AI MCP vào Cursor, Claude Desktop, VS Code & Antigravity trong 60 Giây

Vietnamese AI Framework cung cấp máy chủ MCP (Model Context Protocol) thuần Python (`stdio` và `HTTP/SSE`) tích hợp trực tiếp **ViLegal-TemporalRAG**, **Kiểm chứng trích dẫn 3 tầng**, **Contract Review Pro (5 loại hợp đồng Việt Nam)** và **Tường lửa AI**.

## Bước 1: Tạo hoặc Tải Kho Tri Thức Pháp Lý `.vaidb`

Liệt kê 5 gói tri thức pháp lý theo ngành và dựng gói cục bộ có chỉ mục HNSW + chữ ký SHA-256:

```bash
pip install vietnamese-ai
vai pull-kb --list
vai pull-kb vn-business-core-2025 --output ./data/vaidb/vn-business-core-2025.vaidb
```

## Bước 2: Sinh Cấu Hình MCP 1-Click (`vai mcp-config`)

### Cursor (`.cursor/mcp.json`)

```bash
vai mcp-config --client cursor --kb ./data/vaidb/vn-business-core-2025.vaidb --write .cursor/mcp.json
```

### Claude Desktop (`claude_desktop_config.json`)

```bash
vai mcp-config --client claude --kb ./data/vaidb/vn-business-core-2025.vaidb --extended-tools
```

### VS Code (`.vscode/mcp.json`)

```bash
vai mcp-config --client vscode --kb ./data/vaidb/vn-business-core-2025.vaidb --write .vscode/mcp.json
```

### Antigravity (`mcp_config.json`)

```bash
vai mcp-config --client antigravity --kb ./data/vaidb/vn-business-core-2025.vaidb
```

### Chế độ HTTP/SSE (dùng chung trong mạng LAN hoặc Docker)

```bash
vai mcp-server --transport sse --host 127.0.0.1 --port 8090 --extended-tools --temporal-legal-tools
vai mcp-config --client cursor --transport sse --host 127.0.0.1 --port 8090
```

## Bước 3: Chạy Web Studio v2 & Docker Compose 1 Lệnh

Web Studio v2 hỗ trợ **SSE Streaming trực tuyến** (`/api/legal/temporal-ask/stream`, `/api/chat/stream`), **Split-View trích dẫn điều luật** (nhãn hiệu lực Xanh/Vàng/Đỏ) và **Contract Review Pro** cho 5 loại hợp đồng Việt Nam (`lao_dong`, `dich_vu`, `mua_ban`, `thue_nha`, `nda`):

```bash
vai studio --host 127.0.0.1 --port 8080 --workspace-dir ./data/workspace
```

Hoặc triển khai bằng Docker Compose (Studio + tùy chọn MCP SSE + Zalo/Telegram Webhook):

```bash
docker compose up -d
docker compose --profile mcp up -d
docker compose --profile connectors up -d
```

## Danh Sách Công Cụ MCP Nổi Bật

| Công cụ MCP | Mô tả |
| :--- | :--- |
| `tra_cuu_phap_ly_thoi_gian` | Hỏi đáp pháp lý theo mốc hiệu lực thời gian + đồ thị điều khoản đa bước (`ViLegal-TemporalRAG`) |
| `kiem_chung_trich_dan_phap_ly` | Kiểm chứng 3 tầng (Grounding, Temporal Consistency, Numeric/Deadline) chống ảo giác điều luật |
| `ra_soat_hop_dong` | Rà soát rủi ro pháp lý, so sánh hợp đồng mẫu và kiểm toán văn bản luật hết hiệu lực |
| `truy_van_do_thi_phap_ly` | Truy vết quan hệ Luật ↔ Nghị định ↔ Thông tư ↔ Sửa đổi/Thay thế |
| `quet_bao_mat_du_lieu` | Phát hiện Prompt Injection, Unicode Trojan Bidi và ẩn tự động dữ liệu PII |
| `tra_cuu_dieu_khoan_theo_ngay` | Tra cứu phiên bản Point-in-Time của một Điều/Khoản luật tại ngày bất kỳ (`--extended-tools`) |
