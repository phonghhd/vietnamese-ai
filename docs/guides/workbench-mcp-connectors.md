# Hướng Dẫn Bàn Làm Việc Hợp Đồng, Chat Workspace, MCP IDE Server & Zalo/Telegram Connectors

Hướng dẫn thực chiến triển khai **Bàn làm việc AI văn phòng** (so sánh & rà soát hợp đồng `.docx`, chat đa phiên theo phòng ban lưu SQLite) và **kết nối trực tiếp vào công cụ làm việc hằng ngày** (Cursor, VS Code, Claude Desktop qua `vai mcp-server`, cùng Zalo OA và Telegram Bot).

---

## 1. Rà Soát & So Sánh Hợp Đồng (`BoSoSanhRaSoatHopDong` / `vai contract-review`)

Tự động đọc `.docx`, `.pdf`, `.txt`, so sánh từng điều khoản giữa **Hợp đồng mẫu chuẩn** và **Dự thảo đối tác**, phát hiện ngay:
- Phạt vi phạm vượt trần **8%** (Điều 301 Luật Thương mại `36/2005/QH11`).
- Lãi suất vượt trần **20%/năm** (Điều 468 Bộ luật Dân sự `91/2015/QH13`).
- Miễn trừ trách nhiệm một chiều, bồi thường không giới hạn, chấm dứt không cần báo trước, tự động gia hạn ẩn.
- Thiếu điều khoản bắt buộc (đối tượng, thanh toán, bảo mật dữ liệu, bất khả kháng, giải quyết tranh chấp).
- Viện dẫn văn bản pháp luật **hết hiệu lực / đã sửa đổi** (`BoTheoDoiHieuLucPhapLy` + `DoThiThamChieuPhapLy`).
- **Phát hiện tách/gộp điều khoản** (1 điều mẫu bị đối tác tách thành 2 điều hoặc ngược lại) và xuất trực tiếp báo cáo Redline định dạng **Markdown (`.md`)**, **Word (`.docx`)** hoặc **Word Track Changes + Comments chuẩn OOXML** (`<w:del>`, `<w:ins>`, `word/comments.xml`).

### Chạy qua CLI

```bash
vai contract-review hop_dong_doi_tac.docx \
  --template hop_dong_chuan.docx \
  --output-md bao_cao_ra_soat.md \
  --output-docx bao_cao_redline.docx \
  --track-changes-docx hop_dong_track_changes.docx
```

### Chạy qua Python API

```python
from vietnamese_ai import BoSoSanhRaSoatHopDong

engine = BoSoSanhRaSoatHopDong()
kq = engine.so_sanh_va_ra_soat(
    hop_dong_doi_tac="hop_dong_doi_tac.docx",
    hop_dong_chuan="hop_dong_chuan.docx",
)
print("Điểm an toàn:", kq["diem_an_toan_phap_ly"], "| Rủi ro:", kq["xep_loai_rui_ro"])
engine.xuat_bao_cao_docx(kq, "bao_cao_redline.docx")
engine.xuat_bao_cao_track_changes_docx(kq, "hop_dong_track_changes.docx")
```

---

## 2. Chat Đa Phiên Theo Phòng Ban & Web Studio Workspace (`QuanLyPhienChatWorkspace`)

Mỗi phòng ban (`phap_che`, `nhan_su`, `ke_toan`) có tệp `.vaidb` riêng biệt, lịch sử chat lưu bền vững trên SQLite kèm **đoạn trích dẫn gốc**, phản hồi `thumbs_up`/`thumbs_down` phục vụ vòng lặp DPO, và tùy chọn bật RBAC (`HeThongXacThuc`) + Audit (`NhatKyHoatDong`):

```python
from vietnamese_ai import QuanLyPhienChatWorkspace

ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu="vai_studio_workspaces", bat_xac_thuc=False)
ql.tao_workspace(ten="Phòng Pháp chế", ma_workspace="phap_che")
phien = ql.tao_phien_chat(ma_workspace="phap_che", tieu_de="Rà soát hợp đồng Q4")

rag_pc = ql.lay_rag_workspace("phap_che")
rag_pc.them_tai_lieu("hd_01", "Điều 105 Bộ luật Lao động 2019: Thời giờ làm việc không quá 08 giờ/ngày.")

tra_loi = ql.gui_tin_nhan(phien["ma_phien"], "Thời giờ làm việc bình thường là bao nhiêu?")
print(tra_loi["tra_loi"])
print("Trích dẫn nổi bật:", tra_loi["trich_dan_noi_bat"])
ql.dong()
```

Khởi động Web Studio có sẵn Tab **📑 Hợp đồng & Chat Workspace** (hỗ trợ gắn `.vaidb`, thư mục SQLite Workspace và xác thực RBAC):

```bash
vai studio --port 8080 --vaidb tri_thuc_cong_ty.vaidb --workspace-dir ./vai_studio_workspaces --auth
```

---

## 3. Cắm Thẳng Vào Cursor, VS Code & Claude Desktop (`vai mcp-server`)

`MayChuMCPThucTe` xuất bản sẵn **6 công cụ Vietnamese AI** theo chuẩn Model Context Protocol (JSON-RPC 2.0) qua cả `stdio` lẫn `HTTP/SSE`, đồng thời hỗ trợ tự động phơi bày toàn bộ công cụ **V-Plugin 2.0** qua `gan_quan_ly_plugin(quan_ly_plugin)` hoặc cờ `--plugin-dir`:
1. `tra_cuu_phap_ly`
2. `kiem_tra_hieu_luc`
3. `do_thi_tham_chieu_luat`
4. `ra_soat_hop_dong`
5. `quet_bao_mat_du_lieu`
6. `hoi_edge_llm`

```bash
# In ngay cấu hình JSON để dán vào Cursor / Claude Desktop / VS Code
vai mcp-server --print-config cursor --vaidb tri_thuc_cong_ty.vaidb
vai mcp-server --print-config claude
vai mcp-server --print-config vscode

# Chạy giao vận stdio kèm tự nạp V-Plugin 2.0 và bật 3 công cụ mở rộng (Point-in-Time Điều luật, Viterbi NLP, HNSW .vaidb)
vai mcp-server --transport stdio --vaidb tri_thuc_cong_ty.vaidb --plugin-dir ./plugins --extended-tools

# Hoặc chạy giao vận HTTP + SSE (cho Open WebUI / mạng nội bộ)
vai mcp-server --transport sse --host 127.0.0.1 --port 8090 --plugin-dir ./plugins --extended-tools
```

---

## 4. Kết Nối Zalo OA & Telegram Bot (`vai connector`)

Bộ kết nối thuần Việt bảo vệ sẵn bằng `TuongLuaBaoVeFramework`, nạp mặc định kho **100 điều khoản pháp lý mở + `onnx_viet`**, xác thực chữ ký MAC SHA-256 của Zalo OA (`X-ZEvent-Signature`), xác thực `X-Telegram-Bot-Api-Secret-Token`, cắt đoạn tin nhắn Telegram an toàn dưới 4,096 ký tự, hỗ trợ lưu lịch sử bền vững vào **SQLite Workspace** (gom thẳng dữ liệu về vòng lặp DPO) và tích hợp sẵn 3 lệnh nhanh:
- `/hieuluc <so_hieu>`: Kiểm tra hiệu lực văn bản luật.
- `/luat <so_hieu>`: Tra cứu đồ thị văn bản thay thế & nghị định hướng dẫn.
- `/hopdong <nội dung điều khoản>`: Rà soát rủi ro pháp lý nhanh ngay trong khung chat.

### Chạy Webhook Server hợp nhất bằng CLI

```bash
# Cấu hình biến môi trường khi triển khai thật
export ZALO_APP_ID="..."
export ZALO_OA_SECRET="..."
export ZALO_OA_ACCESS_TOKEN="..."
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_WEBHOOK_SECRET="..."

# Khởi động máy chủ Webhook cho cả Zalo OA (/webhook/zalo) và Telegram (/webhook/telegram) kèm lưu lịch sử Workspace
vai connector both --solution phap-ly --port 8095 --vaidb tri_thuc_cong_ty.vaidb --workspace-dir ./vai_studio_workspaces
```

### Tích hợp trực tiếp trong Python & `QuyTrinhDauCuoi`

```python
from vietnamese_ai import (
    KetNoiTelegram,
    KetNoiZaloOA,
    MayChuWebhookKenh,
    QuanLyPhienChatWorkspace,
    QuyTrinhDauCuoi,
    TroLyPhapLy,
)

ql_ws = QuanLyPhienChatWorkspace(thu_muc_du_lieu="./vai_studio_workspaces", bat_xac_thuc=False)
tro_ly = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True)
zalo = KetNoiZaloOA(bo_xu_ly=tro_ly, app_id="APP_ID", oa_secret_key="SECRET", quan_ly_workspace=ql_ws)
telegram = KetNoiTelegram(bo_xu_ly=tro_ly, bot_token="BOT_TOKEN", webhook_secret="WEBHOOK_SECRET", quan_ly_workspace=ql_ws)

server = MayChuWebhookKenh(ket_noi_zalo=zalo, ket_noi_telegram=telegram, port=8095)
# server.chay()

# Hoặc dùng cầu nối 1 dòng trên QuyTrinhDauCuoi:
wf = QuyTrinhDauCuoi(ten_du_an="doanh_nghiep")
kq_hd = wf.ra_soat_hop_dong("hop_dong_doi_tac.docx", hop_dong_chuan="hop_dong_chuan.docx")
kq_eg = wf.kiem_dinh_mo_hinh(gioi_han_mau=8)
mcp_srv = wf.tao_mcp_server()
wh_srv = wf.tao_webhook_kenh(kenh="both", quan_ly_workspace=ql_ws)
```
