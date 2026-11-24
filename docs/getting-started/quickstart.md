# Sử dụng nhanh

## Phân loại

```python
from vietnamese_ai import PhanLoai, XuLySo
from vietnamese_ai.datalake.sample_data import DuLieuMau

X, y = DuLieuMau.phan_loai_don_gian(so_mau=400)
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)
print(pl.bao_cao(X_test, y_test))
```

## Hồi quy

```python
from vietnamese_ai import HoiQuy

hq = HoiQuy(thuat_toan="tuyen_tinh")
hq.huan_luyen(X_train, y_train)
print(hq.bao_cao(X_test, y_test))
```

## AutoML

```python
from vietnamese_ai import AutoML

auto = AutoML()
ket_qua = auto.fit(X_train, y_train)
print(auto.bao_cao())
du_doan = auto.predict(X_test)
```

## Xử lý văn bản

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()
xl.tach_tu("Trí tuệ nhân tạo rất hay")  # ['trí_tuệ_nhân_tạo', 'rất', 'hay']
xl.phan_tich_cam_xuc("Sản phẩm tốt")     # 'positive'
```

## Pipeline

```python
from vietnamese_ai import Pipeline, XuLySo, PhanLoai

pipe = Pipeline()
pipe.them_buoc("chuan_hoa", XuLySo())
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="gradient_boosting"))
pipe.fit(X_train, y_train)
pipe.luu("model.pkl")
```

## No-code Studio

```python
from vietnamese_ai import StudioKeoTha

studio = StudioKeoTha()
studio.tai_template("phan_loai_co_ban")
ket_qua = studio.chay()
print(ket_qua['trang_thai'])  # 'thanh_cong'
```

## Vietnamese LLM

```python
from vietnamese_ai import VietnameseLLM

llm = VietnameseLLM(bac=3)
llm.huan_luyen(cac_van_ban, so_vong=5)
van_ban = llm.sinh_van_ban("học máy là", do_dai=50)
goi_y = llm.lay_tu_ke_tiep("trí tuệ nhân", top_n=5)
```

## Federated Learning

```python
from vietnamese_ai import HocLienKet, PhanLoai

hl = HocLienKet(so_client=5, so_vong=10)
ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
print(f"Điểm global: {ket_qua['diem_toan_cuc']:.4f}")
```

## Trải Nghiệm 5 Phút Với Hệ Sinh Thái `v31.0.3`

```bash
# 1. Trải nghiệm 1-Click đủ 4 trụ cột offline (.vaidb, Đồ thị luật, Hợp đồng, Edge CPU, Bảo mật, GPU Flywheel)
vai demo
vai status-map

# 2. Lập chỉ mục bền vững 1 tệp .vaidb + ngữ nghĩa đồng nghĩa tiếng Việt ONNX + Đồ thị & Hợp đồng
vai rag --db cong_ty.vaidb --docs ./thu_muc_luat --semantic-onnx --admin --query "Thời hạn thông báo vi phạm dữ liệu cá nhân?"
vai legal-graph 45/2013/QH13 --active-only --mermaid
vai contract-review hop_dong_doi_tac.docx --template hop_dong_chuan.docx --output-docx redline.docx

# 3. Kéo & chạy mô hình LLM tiếng Việt Zero-GPU trên CPU gắn thẳng kho .vaidb
vai pull --list
vai pull qwen2.5-3b
vai run qwen2.5-3b --rag cong_ty.vaidb --admin --prompt "Rà soát văn bản hết hiệu lực"

# 4. Mở Web Studio (Workspace phòng ban + RBAC), cắm MCP IDE & Webhook Zalo/Telegram
vai studio --vaidb cong_ty.vaidb --workspace-dir ./studio_ws --port 8080
vai mcp-server --print-config cursor --vaidb cong_ty.vaidb
vai connector both --solution phap-ly --workspace-dir ./studio_ws --dry-run

# 5. Xuất DPO từ Workspace -> Tinh chỉnh GPU từ xa (Modal/SSH/Local) -> Kiểm định 4 trục & Phát hành PyPI
vai export-dpo --workspace-dir ./studio_ws --output studio_dpo.jsonl
vai train-cloud --from-workspace ./studio_ws --provider modal --gpu T4 --eval-4-axis
vai eval-gate --limit 8 --min-score 0.55
vai release-check
```

Xem hướng dẫn chi tiết từng trụ cột:

- [Hướng dẫn RAG Pháp lý 100 điều khoản, `.vaidb`, Ngữ nghĩa ONNX & Đồ thị Luật](../guides/vietnamese-rag.md)
- [Hướng dẫn Rà soát Hợp đồng Redline, Chat Workspace, MCP IDE & Zalo/Telegram](../guides/workbench-mcp-connectors.md)
- [Hướng dẫn LLM Tiếng Việt Zero-GPU (`vai pull` & `vai run`)](../guides/edge-cpu-llm-runtime.md)
- [Hướng dẫn Remote GPU Flywheel (Modal / SSH / Local) & Eval Gate 4 Trục](../guides/remote-gpu-flywheel.md)
- [Hướng dẫn Trải nghiệm 5 Phút, `vai status-map`, `vai release-check` & Benchmark Docker](../guides/five-minute-dx-and-release.md)

## CLI Tiêu Biểu

```bash
vai info
vai doctor
vai status-map
vai demo
vai studio --vaidb cong_ty.vaidb --workspace-dir ./studio_ws --port 8080
vai train --data data.csv --model logistic --output model.pkl
vai predict --model model.pkl --input new_data.csv
vai serve --model model.pkl --port 8080 --backend fastapi
vai release-check
```
