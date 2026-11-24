# Hướng Dẫn Động Cơ LLM Tiếng Việt Zero-GPU Trên CPU (`vai pull` & `vai run`)

Vietnamese AI Framework tích hợp trọn gói trải nghiệm kéo mô hình và chạy suy luận LLM tiếng Việt trực tiếp trên CPU máy văn phòng — tương tự phong cách `ollama pull / run` nhưng bổ sung **tự chọn mức lượng tử theo RAM thực**, **tải nối tiếp chống đứt mạng**, **xác minh toàn vẹn SHA-256 + chữ ký HMAC-SHA256**, **Tường lửa AI** và **RAG Pháp lý Việt Nam gắn sẵn**.

---

## 1. Kiến trúc Suy luận 3 Tầng Native & Tự chọn Lượng tử Theo RAM

```text
[vai pull qwen2.5-3b]
       │
       ├── 1. Đo RAM khả dụng thực (`do_ram_he_thong_gb`: Linux /proc/meminfo, macOS vm_stat, Windows API)
       ├── 2. Tự chọn mức lượng tử tối ưu (`Q8_0` / `Q5_K_M` / `Q4_K_M` / `Q3_K_M`) tránh tràn RAM
       ├── 3. Tải HTTP Range Resume (`206 Partial Content`) -> không mất công tải lại khi rớt mạng
       └── 4. Xác minh tiêu đề nhị phân `GGUF` + SHA-256 + Ký HMAC-SHA256 cục bộ
               │
               ▼
[vai run qwen2.5-3b --rag hop_dong.docx --admin]
       │
       ├── Tầng 1 (Nhanh nhất): `llama-cpp-python` (nếu đã cài `vietnamese-ai[local-llm]`)
       ├── Tầng 2 (C++ Native): `llama-cli` / `libllama.so` qua `CauNoiCtypesNative`
       └── Tầng 3 (Zero-Dependency): Bộ giải mã GGUF mmap + BPE thuần Python/NumPy (`BoGiaiMaTransformerEdge`)
```

---

## 2. Danh mục 8 Mô hình LLM Tiếng Việt Tích hợp Sẵn (`DANH_MUC_MO_HINH_VIET`)

| Mã mô hình | Tham số | Mẫu Chat | Biến thể lượng tử | Khuyến nghị RAM | Phù hợp cho |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `qwen2.5-1.5b` | 1.5B | `chatml` | `Q4_K_M`, `Q5_K_M`, `Q8_0` | 2.5 – 4.0 GB | Laptop văn phòng 4–8 GB RAM, tốc độ sinh rất nhanh |
| `gemma2-2b` | 2.6B | `gemma` | `Q4_K_M`, `Q8_0` | 3.8 – 5.5 GB | Hỏi đáp và tóm tắt tài liệu gọn nhẹ |
| `qwen2.5-3b` | 3.0B | `chatml` | `Q4_K_M`, `Q5_K_M`, `Q8_0` | 4.5 – 7.0 GB | **Cân bằng tốt nhất** cho máy văn phòng 8 GB RAM |
| `phogpt-4b` | 3.7B | `phogpt` | `Q4_K_M`, `Q8_0` | 5.0 – 7.5 GB | Văn phong thuần Việt chuyên sâu (VinAI) |
| `qwen3-4b` | 4.0B | `chatml` | `Q4_K_M`, `Q5_K_M`, `Q8_0` | 5.0 – 8.0 GB | Suy luận logic từng bước trên CPU 8–16 GB RAM |
| `vistral-7b` | 7.2B | `mistral` | `Q4_K_M`, `Q8_0` | 8.0 – 13.0 GB | RAG hành chính, pháp lý và viết lách tiếng Việt |
| `qwen2.5-7b` | 7.6B | `chatml` | `Q3_K_M`, `Q4_K_M`, `Q5_K_M`, `Q8_0` | 6.5 – 14.0 GB | Phân tích pháp lý & doanh nghiệp chuyên sâu (16 GB RAM) |
| `seallm-7b` | 7.6B | `chatml` | `Q4_K_M`, `Q8_0` | 8.0 – 14.0 GB | Ngôn ngữ Đông Nam Á & tiếng Việt tự nhiên, ít ảo giác |

> Mẹo: Bạn cũng có thể kéo bất kỳ tệp `.gguf` nào trên HuggingFace bằng cú pháp trực tiếp:
> `vai pull hf://ten-org/ten-repo/ten-file.gguf`

---

## 3. Sử dụng qua CLI (`vai pull` & `vai run`)

### Xem gợi ý mô hình khớp với RAM máy hiện tại

```bash
vai pull --list
```

### Xem trước URL & biến thể lượng tử trước khi tải (Dry-Run)

```bash
vai pull qwen2.5-3b --dry-run
```

### Tải mô hình thực tế (tự chọn lượng tử theo RAM hoặc ép mức lượng tử)

```bash
# Tự động chọn Q4_K_M / Q5_K_M / Q8_0 phù hợp RAM thực của máy
vai pull qwen2.5-3b

# Hoặc chỉ định rõ mức lượng tử và mã băm SHA-256 kiểm chứng
vai pull phogpt-4b --quant Q4_K_M --sha256 <ma_sha256_ky_vong>

# Liệt kê mô hình đã tải kèm trạng thái chữ ký HMAC
vai pull --installed

# Xóa mô hình khỏi bộ nhớ đệm ~/.cache/vietnamese-ai/models
vai pull --remove qwen2.5-3b-Q4_K_M
```

### Chạy suy luận Zero-GPU & Hỏi đáp trực tiếp trên Tài liệu Pháp lý (`--rag --admin`)

```bash
# 1. Hỏi đơn lẻ kèm bảng đo tốc độ Token/s, TTFT và RAM
vai run qwen2.5-3b --prompt "Tóm tắt quyền lợi người lao động về thời giờ làm việc" --bench

# 2. Chế độ hội thoại streaming trực tiếp trên CPU (ưu tiên tầng Native C++ nếu có)
vai run qwen2.5-3b --chat --threads 4 --native

# 3. Gắn trực tiếp kho bền vững `.vaidb`, tệp hợp đồng `.docx`/`.pdf` hoặc thư mục luật làm ngữ cảnh RAG pháp lý
# (Mặc định dùng ngữ nghĩa `onnx_viet`, tự in Cảnh báo hiệu lực & Đồ thị văn bản thay thế/hướng dẫn)
vai run qwen2.5-3b \
  --rag kho_luat_cong_ty.vaidb \
  --admin \
  --prompt "Hợp đồng này có điều khoản nào viện dẫn văn bản luật đã hết hiệu lực không?"
```

---

## 4. Sử dụng qua Python API

### Quản lý tải mô hình & định dạng prompt hội thoại

```python
from vietnamese_ai import (
    QuanLyMoHinhEdge,
    dinh_dang_prompt_chat,
    do_ram_he_thong_gb,
)

ram = do_ram_he_thong_gb()
print(f"RAM khả dụng: {ram['ram_kha_dung_gb']} / {ram['tong_ram_gb']} GB ({ram['so_nhan_cpu']} CPU)")

hub = QuanLyMoHinhEdge()
de_xuat = hub.chon_muc_luong_tu_theo_ram("qwen2.5-3b")
print("Lượng tử đề xuất:", de_xuat["muc_luong_tu"], "-", de_xuat["ly_do"])

# Xem trước kế hoạch tải không tốn băng thông
ke_hoach = hub.tai_mo_hinh("qwen2.5-3b", muc_luong_tu="auto", chay_that=False)
print("URL nguồn:", ke_hoach["url"])

# Định dạng prompt đúng chuẩn từng dòng mô hình (chatml / phogpt / mistral / gemma)
prompt = dinh_dang_prompt_chat(
    tin_nhan="Giải thích ngắn gọn Điều 105 Bộ luật Lao động 2019",
    he_thong="Bạn là trợ lý pháp lý tiếng Việt chính xác và súc tích.",
    chat_template="chatml",
)
```

### Chạy suy luận kết hợp RAG Pháp lý trong 5 dòng code

```python
from vietnamese_ai import DongCoSuyLuanEdge

engine = DongCoSuyLuanEdge(
    duong_dan_mo_hinh="qwen2.5-3b",
    so_luong_thread=4,
    uu_tien_native_c=True,
)
# Hỗ trợ truyền thẳng tệp `.vaidb`, tệp `.docx`/`.pdf`/`.txt` hoặc thư mục (mặc định bật `dung_onnx_semantic=True`)
engine.nap_tai_lieu_rag("kho_luat_cong_ty.vaidb", che_do_hanh_chinh=True)

ket_qua = engine.hoi_kem_rag("Thời hạn thông báo sự cố dữ liệu cá nhân là bao lâu?")
print(ket_qua["tra_loi"])
print("Căn cứ, hiệu lực & đồ thị pháp lý:", ket_qua.get("trich_dan_phap_ly"))
```

### Streaming trực tiếp với `LlamaCppWrapper` hoặc `QuyTrinhDauCuoi`

```python
from vietnamese_ai import LlamaCppWrapper, QuyTrinhDauCuoi

# Nối liền từ QuyTrinhDauCuoi
wf = QuyTrinhDauCuoi()
info = wf.tai_mo_hinh_edge("qwen2.5-1.5b", chay_that=False)

# Streaming token khi đã có tệp .gguf cục bộ + cài `vietnamese-ai[local-llm]`
wrapper = LlamaCppWrapper(duong_dan_mo_hinh="/path/to/model.gguf")
for token in wrapper.sinh_stream("Xin chào Việt Nam"):
    print(token, end="", flush=True)
```

---

## 5. Tăng tốc Tối đa Trên CPU (Tùy chọn)

Khi muốn đạt tốc độ sinh văn bản cao nhất trên CPU Intel AVX2 / AMD / Apple Silicon:

```bash
# Cài lớp tăng tốc llama-cpp-python (tầng 1 Native C++)
pip install "vietnamese-ai[local-llm]"

# Kiểm tra tập lệnh SIMD & Rust Core trên máy
vai doctor
```

Quản lý và tải mô hình trực quan từ trình duyệt cũng có sẵn qua Web Studio endpoints `GET /api/edge/models` và `POST /api/edge/pull`. Chi tiết chữ ký đầy đủ xem tại [Edge & Swarm CPU API Reference](../api/edge.md).
