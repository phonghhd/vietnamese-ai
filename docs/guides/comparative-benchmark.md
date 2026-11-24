# Bảng Đo Lường & Benchmark Tái Lập (Reproducible Quantitative Benchmarks)

Tài liệu này mô tả phương pháp đo lường minh bạch, dễ tái lập cho **Vietnamese AI Framework** (`v31.0.3`) bằng các công cụ chuẩn (`time.perf_counter`, `tracemalloc`, `resource`, `cProfile`, `ThreadPoolExecutor` và TCP Socket thực tế).

---

## 1. Cách Tái Lập Toàn Bộ Benchmark Trong 1 Lệnh

### Chạy cục bộ
```bash
python benchmarks/run_benchmarks.py \
  --suite extended \
  --seed 42 \
  --profile \
  --output results/benchmark.json \
  --markdown results/benchmark.md
```

### Chạy trong Docker cô lập
```bash
docker build -f benchmarks/Dockerfile.benchmark -t vietnamese-ai-bench:31.0.3 .
docker run --rm -v "$(pwd)/results:/results" vietnamese-ai-bench:31.0.3
```

---

## 2. Phạm Vi Đo Lường Chuẩn Sản Xuất (`BoDoLuongDoiDau`)

| Hạng mục | Phương thức | Chỉ số đo thực tế |
|---|---|---|
| **Core Micro-Architecture** | `chay_benchmark_toan_dien()` | Cold-start (ms), RAM tiêu thụ (MB), tốc độ tách từ (từ/s), Legal RAG Hit@1 / Hit@3 / MRR |
| **LLM Inference & Quantized CPU** | `do_hieu_nang_suy_luan_llm()` | Độ trễ `p50`/`p95`/`p99` (ms), thông lượng `req/s` & `tokens/s` dưới tải đa luồng, tốc độ `nhan_ma_tran_luong_tu` trên `Q4_0`, `Q4_K_M`, `Q8_0` theo kích thước tầng `256`/`512`/`1024` |
| **Độ ổn định dài hạn (Long-running)** | `do_do_on_dinh_tai_nguyen()` | Biến thiên Python Heap (`tracemalloc`), RSS tiến trình (`resource`), thời gian CPU, cảnh báo rò rỉ bộ nhớ |
| **Mở rộng cụm Swarm CPU & LAN** | `do_kha_nang_mo_rong_swarm()` | So sánh cụm giả lập (`khoi_tao_cum_gia_lap`) 1→2→4 node (có/không zkML RFC 5114) và cụm TCP Socket thực tế (`them_node_mang_vat_ly`, FP32 vs FP16) |
| **Tổng quát đa miền tiếng Việt** | `do_da_dang_mien_tieng_viet()` | 5 miền (Pháp lý, Tài chính, Y tế, Công nghệ AI, Thương mại điện tử): Tokenizer, POS VLSP, NER Recall, Sentiment Accuracy, Multi-Domain RAG |

---

## 3. Minh Bạch Về Dữ Liệu So Sánh (Baseline Disclosure)

- Mọi chỉ số của **Vietnamese AI Framework** trong báo cáo JSON/Markdown đều được **đo trực tiếp trong thời gian chạy** trên chính phần cứng người dùng và gắn kèm siêu dữ liệu môi trường (OS, Python, kiến trúc CPU, cờ SIMD, `seed`, thời điểm UTC).
- Các giá trị của `LangChain` và `LlamaIndex` trong `BoDoLuongDoiDau.BASELINE_LANGCHAIN` / `BASELINE_LLAMAINDEX` là **mốc tham chiếu tĩnh (reference baselines)** để tránh buộc người dùng cài thêm hai framework nặng này khi chạy lõi tối thiểu. Để đối chiếu trực tiếp 1-1 trên cùng phần cứng, hãy cài đặt song song trong container benchmark và đo trên cùng bộ dữ liệu đầu vào.
- Bảng phụ thuộc cốt lõi của `vietnamese-ai`: chỉ yêu cầu duy nhất `numpy>=1.20.0,<3.0.0` (các nhóm `[vietnamese-nlp]`, `[classic-ml]`, `[legal-rag]`, `[dl]`, `[rag]`, `[api]` là tùy chọn).
