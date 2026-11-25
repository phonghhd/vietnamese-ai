# Bộ Benchmark Tái Lập (Reproducible Benchmarks)

Thư mục này cung cấp kịch bản và Docker image chuẩn hóa để bất kỳ ai cũng có thể tái lập đầy đủ các chỉ số hiệu năng, độ ổn định tài nguyên và khả năng mở rộng của **Vietnamese AI Framework**.

## 1. Chạy trực tiếp trên máy cục bộ

```bash
# Cài lõi tối thiểu
pip install -e .

# Chạy toàn bộ suite mở rộng (Core + LLM Concurrency/Quantized MatMul + Stability + Swarm LAN + 5-Domain VN)
python benchmarks/run_benchmarks.py \
  --suite extended \
  --seed 42 \
  --profile \
  --output results/benchmark.json \
  --markdown results/benchmark.md

# Chạy nhanh cho CI/CD
python benchmarks/run_benchmarks.py --suite extended --quick --output results/quick.json

# Chạy bộ đo chiều sâu 6 trụ cột mới (FST/Viterbi, HNSW .vaidb, Rust Kernels, Freivalds, SimPO/EvalGate, Track Changes)
python benchmarks/run_benchmarks.py \
  --suite deep \
  --seed 42 \
  --output results/deep_benchmark.json \
  --markdown results/deep_benchmark.md
```

## 2. Tái lập trong Docker Container cô lập

```bash
docker build -f benchmarks/Dockerfile.benchmark -t vietnamese-ai-bench:31.0.4 .
docker run --rm -v "$(pwd)/results:/results" vietnamese-ai-bench:31.0.4
```

Báo cáo JSON và Markdown kèm đầy đủ thông tin phần cứng (`x86_64`/`arm64`, số lõi, SIMD, Python, `seed`) sẽ được ghi vào `results/benchmark.json` và `results/benchmark.md`.

## 3. Các hạng mục đo lường

1. **Core Comparative Suite**: Cold-start, RAM (`tracemalloc`), tốc độ tách từ tiếng Việt, Legal RAG Hit@1 / Hit@3 / MRR.
2. **LLM Inference & Quantized MatMul**: Độ trễ `p50`/`p95`/`p99`, thông lượng `req/s` và `tokens/s` dưới tải đa luồng đồng thời, cùng hiệu năng nhân ma trận lượng tử hóa `Q4_0` / `Q4_K_M` / `Q8_0` theo nhiều kích thước tầng (`256`, `512`, `1024`).
3. **Long-Running Resource Stability**: Giám sát RSS RAM (`resource`), Python Heap (`tracemalloc`), thời gian CPU (`time.process_time`) qua nhiều chu kỳ liên tục để phát hiện rò rỉ bộ nhớ.
4. **Swarm CPU Scalability & Real TCP LAN**: Đo mở rộng cụm giả lập (`khoi_tao_cum_gia_lap`) 1→2→4 node và truyền tải tensor qua TCP Socket thực tế (`them_node_mang_vat_ly`, FP32 vs FP16 + xác thực RFC 5114 zkML).
5. **5-Domain Vietnamese Generalization**: Đánh giá trên 5 lĩnh vực (Pháp lý, Tài chính, Y tế, Công nghệ AI, Thương mại điện tử) gồm Tokenizer, POS VLSP, NER Recall, Sentiment và Multi-Domain RAG.
