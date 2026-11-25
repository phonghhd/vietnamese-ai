"""CLI chính cho Vietnamese AI Framework — Siêu nhẹ, Bảo mật Ký số HMAC & Quản lý Gói Mở rộng."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _tao_parser():
    """Tạo argument parser cho CLI."""
    parser = argparse.ArgumentParser(
        prog="vai",
        description="Vietnamese AI Framework - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  vai doctor
  vai install ml --dry-run
  vai train --data data.csv --model logistic --output model.pkl
  vai predict --model model.pkl --input new_data.csv --output results.csv
  vai chat --model vgpt2 --prompt "Trí tuệ nhân tạo là gì?"
  vai benchmark
  vai serve --model model.pkl --port 8080
  vai info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")

    # === INFO ===
    subparsers.add_parser("info", help="Thông tin framework")

    # === DOCTOR ===
    subparsers.add_parser("doctor", help="Chẩn đoán sức khỏe hệ thống, phần cứng, tốc độ nạp & gói phụ thuộc")

    # === INSTALL ===
    inst_parser = subparsers.add_parser("install", help="Cài đặt nhóm gói mở rộng tùy chọn (ml, viz, dl, fast, rag, llm, api, full)")
    inst_parser.add_argument("group", help="Tên nhóm mở rộng (ml, tabular, viz, nlp, dl, fast, rag, legal-rag, audio, ocr-vn, llm, api, full)")
    inst_parser.add_argument("--dry-run", action="store_true", help="Chỉ hiển thị câu lệnh pip mà không chạy thật")

    # === SETUP ===
    setup_parser = subparsers.add_parser(
        "setup",
        help="Tự động quét phần cứng và thiết lập môi trường thông minh 3 tầng (gói mở rộng + Rust Core tầng thấp)",
    )
    setup_parser.add_argument(
        "--profile",
        default="core",
        choices=["core", "edge", "legal-rag", "nlp", "audio", "vision", "ocr-vn", "ml", "dl", "api", "cloud-gpu", "full"],
        help="Hồ sơ cài đặt phù hợp nhu cầu (mặc định: core)",
    )
    setup_parser.add_argument(
        "--apply",
        action="store_true",
        help="Thực thi cài đặt thực tế (mặc định chỉ chạy xem trước an toàn dry-run)",
    )
    setup_parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Bỏ qua bước kiểm tra/biên dịch tầng thấp Rust Core",
    )

    # === TRAIN ===
    train_parser = subparsers.add_parser("train", help="Huấn luyện mô hình (tự động ký số HMAC-SHA256)")
    train_parser.add_argument("--data", required=True, help="Đường dẫn file dữ liệu (CSV)")
    train_parser.add_argument(
        "--model",
        default="logistic",
        help="Thuật toán (logistic, knn, svm, rung_ngau_nhien, gradient_boosting, xgboost, lightgbm)",
    )
    train_parser.add_argument("--output", default="model.pkl", help="Đường dẫn lưu mô hình")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ dữ liệu test")
    train_parser.add_argument("--target", default=None, help="Tên cột nhãn (mặc định: cột cuối)")

    # === PREDICT ===
    predict_parser = subparsers.add_parser("predict", help="Dự đoán với mô hình đã huấn luyện (kiểm tra chữ ký HMAC)")
    predict_parser.add_argument("--model", required=True, help="Đường dẫn file mô hình (.pkl)")
    predict_parser.add_argument(
        "--input", required=True, help="Đường dẫn file dữ liệu đầu vào (CSV)"
    )
    predict_parser.add_argument("--output", default="results.csv", help="Đường dẫn lưu kết quả")

    # === SERVE ===
    serve_parser = subparsers.add_parser("serve", help="Khởi động API server (tích hợp Tường lửa AI & OpenAI API)")
    serve_parser.add_argument("--model", required=True, help="Đường dẫn file mô hình hoặc tên LLM (VD: vgpt2)")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port (mặc định: 8080)")
    serve_parser.add_argument(
        "--backend",
        default="simple",
        choices=["simple", "fastapi"],
        help="Backend API server (`simple` zero-dependency hoặc `fastapi` hỗ trợ SSE Streaming, Tool Calling, /metrics)",
    )

    # === EVALUATE ===
    eval_parser = subparsers.add_parser("evaluate", help="Đánh giá mô hình")
    eval_parser.add_argument("--model", required=True, help="Đường dẫn file mô hình (.pkl)")
    eval_parser.add_argument("--data", required=True, help="Đường dẫn file dữ liệu test (CSV)")
    eval_parser.add_argument("--target", default=None, help="Tên cột nhãn")

    # === WEB ===
    web_parser = subparsers.add_parser("web", help="Khởi động giao diện web no-code")
    web_parser.add_argument("--port", type=int, default=5000, help="Port (mặc định: 5000)")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")

    # === FINETUNE ===
    ft_parser = subparsers.add_parser("finetune", help="Tinh chỉnh (Fine-tune) LLM / Transformers với LoRA/DoRA")
    ft_parser.add_argument("--data", required=True, help="Đường dẫn file dữ liệu (.jsonl / .json)")
    ft_parser.add_argument("--model", default="vgpt2", help="Tên model (phobert, vit5-base, qwen3.8, gemma4, vgpt2)")
    ft_parser.add_argument("--task", default="sft", help="Nhiệm vụ (sft, dpo, grpo, rlhf, instruction, phan_loai, ner, qa)")
    ft_parser.add_argument("--epochs", type=int, default=3, help="Số vòng huấn luyện")
    ft_parser.add_argument("--rank", type=int, default=8, help="Rank của LoRA")
    ft_parser.add_argument("--output", default="outputs/finetuned", help="Thư mục lưu mô hình")

    # === TRAIN-CLOUD (MULTI-PROVIDER REMOTE GPU FLYWHEEL) ===
    tc_parser = subparsers.add_parser(
        "train-cloud",
        help="Tinh chỉnh GPU từ xa đa nhà cung cấp (Modal Serverless GPU, SSH GPU nội bộ, Local Subprocess) và kéo GGUF/LoRA về Edge",
    )
    tc_parser.add_argument("--data", default=None, help="Đường dẫn tệp dữ liệu (.jsonl / .json)")
    tc_parser.add_argument(
        "--from-workspace",
        default=None,
        help="Xuất thẳng phản hồi DPO đã ẩn PII từ thư mục Studio Workspace SQLite để đưa lên GPU tinh chỉnh",
    )
    tc_parser.add_argument("--workspace", default=None, help="Lọc mã Workspace khi dùng `--from-workspace`")
    tc_parser.add_argument(
        "--provider",
        default="modal",
        choices=["modal", "ssh", "local"],
        help="Nhà cung cấp GPU (modal, ssh, local — mặc định: modal)",
    )
    tc_parser.add_argument("--model", default="vgpt2", help="Tên mô hình gốc (VD: vgpt2, qwen2.5-1.5b, phobert)")
    tc_parser.add_argument("--task", default="sft", help="Nhiệm vụ tinh chỉnh (sft, dpo, grpo, instruction)")
    tc_parser.add_argument("--gpu", default="T4", help="Loại GPU (T4, L4, A10G, A100, H100, CPU)")
    tc_parser.add_argument("--epochs", type=int, default=3, help="Số vòng huấn luyện")
    tc_parser.add_argument("--rank", type=int, default=8, help="Rank của LoRA")
    tc_parser.add_argument("--quantization", default="q8_0", help="Mức lượng tử GGUF tải về (q4_0, q8_0, f16)")
    tc_parser.add_argument("--max-budget-usd", type=float, default=2.0, help="Trần ngân sách GPU tối đa (USD)")
    tc_parser.add_argument(
        "--eval-4-axis",
        action="store_true",
        help="Chấm hồi quy 4 trục (`CongKiemDinhMoHinh`) trước khi promote model sau khi tải artifact",
    )
    tc_parser.add_argument(
        "--allow-cloud",
        action="store_true",
        help="Xác nhận rõ ràng cho phép gửi dữ liệu đã ẩn PII lên GPU đám mây bên ngoài (Modal)",
    )
    tc_parser.add_argument(
        "--apply",
        action="store_true",
        help="Thực thi thật (mặc định chạy xem trước dry-run an toàn và xuất script)",
    )
    tc_parser.add_argument("--ssh-host", default="", help="Địa chỉ máy chủ GPU SSH (khi --provider ssh)")
    tc_parser.add_argument("--ssh-user", default=None, help="Tên người dùng SSH")
    tc_parser.add_argument("--ssh-port", type=int, default=22, help="Cổng SSH (mặc định: 22)")
    tc_parser.add_argument("--ssh-key", default=None, help="Đường dẫn khóa riêng SSH (.pem / id_ed25519)")
    tc_parser.add_argument("--output", default="vai_remote_jobs", help="Thư mục lưu tác vụ và artifact tải về")

    # === RAG ===
    rag_parser = subparsers.add_parser("rag", help="Truy vấn cơ sở tri thức RAG đa định dạng & kho bền vững .vaidb")
    rag_parser.add_argument("--docs", default=None, help="Đường dẫn tệp (.txt/.md/.docx/.pdf/.png/.json/.jsonl) hoặc thư mục tài liệu")
    rag_parser.add_argument("--query", default=None, help="Câu hỏi cần truy vấn")
    rag_parser.add_argument("--db", default=None, help="Đường dẫn tệp kho tri thức bền vững `.vaidb` (SQLite + FTS5 + Vector)")
    rag_parser.add_argument(
        "--build-hnsw",
        action="store_true",
        help="Xây dựng và lưu bền vững đồ thị chỉ mục HNSW bên trong tệp `.vaidb`",
    )
    rag_parser.add_argument(
        "--semantic-onnx",
        action="store_true",
        help="Dùng Bộ nhúng & Reranker ngữ nghĩa đồng nghĩa tiếng Việt / ONNX INT8 CPU",
    )
    rag_parser.add_argument(
        "--admin",
        "--hanh-chinh",
        action="store_true",
        help="Kích hoạt chế độ tối ưu hóa văn bản hành chính, hiệu lực & đồ thị tham chiếu pháp luật Việt Nam",
    )

    # === CHAT ===
    chat_parser = subparsers.add_parser("chat", help="Trò chuyện trực tiếp với mô hình kèm Tường lửa AI")
    chat_parser.add_argument("--model", default="vgpt2", help="Tên hoặc đường dẫn mô hình")
    chat_parser.add_argument("--prompt", default=None, help="Câu hỏi đơn lẻ (bỏ trống để vào chế độ hội thoại tương tác)")

    # === BENCHMARK ===
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Đo kiểm hiệu năng và so sánh đối đầu khoa học định lượng (vs LangChain / LlamaIndex)",
    )
    bench_parser.add_argument("--compare", action="store_true", help="Chạy bộ đo lường đối đầu định lượng vs LangChain/LlamaIndex")
    bench_parser.add_argument("--latex", action="store_true", help="Xuất bảng số liệu định dạng LaTeX cho bài báo khoa học")
    bench_parser.add_argument("--output", default=None, help="Đường dẫn tệp xuất báo cáo (.json hoặc .md)")

    # === TOKENIZE ===
    tok_parser = subparsers.add_parser(
        "tokenize",
        help="Tách từ và chuẩn hóa văn bản tiếng Việt siêu tốc (Native Fast Tokenizer)",
    )
    tok_parser.add_argument("text", nargs="?", default="Trí tuệ nhân tạo và chuyển đổi số tại Việt Nam", help="Văn bản tiếng Việt cần phân tích")
    tok_parser.add_argument("--no-compound", action="store_true", help="Không nối từ ghép bằng dấu gạch dưới")
    tok_parser.add_argument("--unaccent", action="store_true", help="Loại bỏ dấu tiếng Việt (unaccented / slugify)")
    tok_parser.add_argument("--encode", action="store_true", help="Mã hóa văn bản thành chuỗi Token IDs số nguyên")
    tok_parser.add_argument(
        "--method",
        default="max_match",
        choices=["max_match", "viterbi"],
        help="Chiến lược tách từ: max_match hoặc viterbi (Word Lattice DAG tối ưu toàn cục)",
    )
    tok_parser.add_argument(
        "--normalize-tone",
        action="store_true",
        help="Hiển thị văn bản sau khi chuẩn hóa dấu thanh âm vị học FST",
    )

    # === SCAN ===
    scan_parser = subparsers.add_parser(
        "scan",
        help="Quét bảo mật hình ảnh (Polyglot, Image Bomb, Stego) hoặc Dataset lớn (Sleeper Backdoor, Split-Payload)",
    )
    scan_parser.add_argument("target", help="Đường dẫn tệp ảnh (.png/.jpg/.svg) hoặc tập dữ liệu (.jsonl/.json/.csv)")
    scan_parser.add_argument("--cdr-out", default=None, help="Đường dẫn xuất tệp ảnh sạch sau khi khử độc CDR")

    # === EXPORT ===
    exp_parser = subparsers.add_parser("export", help="Xuất mô hình sang GGUF hoặc ONNX")
    exp_parser.add_argument("--model", default="vgpt2", help="Tên hoặc đường dẫn mô hình")
    exp_parser.add_argument("--format", default="gguf", choices=["gguf", "onnx"], help="Định dạng xuất")
    exp_parser.add_argument("--quantization", default="q8_0", help="Kiểu lượng tử hóa GGUF (q8_0, q4_k_m, f16)")
    exp_parser.add_argument("--output", required=True, help="Đường dẫn file xuất ra")

    # === STUDIO (WEB DASHBOARD) ===
    studio_parser = subparsers.add_parser("studio", help="Khởi động Vietnamese AI Web Studio Dashboard")
    studio_parser.add_argument("--port", type=int, default=8080, help="Cổng máy chủ (mặc định: 8080)")
    studio_parser.add_argument("--host", default="0.0.0.0", help="Địa chỉ host (mặc định: 0.0.0.0)")
    studio_parser.add_argument("--vaidb", default=None, help="Đường dẫn tệp kho tri thức bền vững `.vaidb` (tùy chọn)")
    studio_parser.add_argument("--workspace-dir", default=None, help="Thư mục lưu SQLite Chat Workspace & `.vaidb` theo phòng ban")
    studio_parser.add_argument("--auth", action="store_true", help="Bật xác thực RBAC & nhật ký kiểm toán trên Studio Workspace")

    # === SOLUTION (1-CLICK REAL-WORLD TEMPLATES) ===
    sol_parser = subparsers.add_parser("solution", help="Chạy các giải pháp thực chiến 1-Click đóng gói sẵn")
    sol_parser.add_argument(
        "name",
        choices=["phap-ly", "phap_ly", "trich-xuat", "trich_xuat", "cskh"],
        help="Tên giải pháp (phap-ly, trich-xuat, cskh)",
    )
    sol_parser.add_argument("--query", "-q", default=None, help="Câu hỏi hoặc yêu cầu xử lý")
    sol_parser.add_argument("--input", "-i", default=None, help="Đường dẫn tệp ảnh hoặc nội dung văn bản đầu vào")
    sol_parser.add_argument("--type", default="cccd", choices=["cccd", "hoa_don"], help="Loại giấy tờ cho trich-xuat")
    sol_parser.add_argument("--vaidb", default=None, help="Đường dẫn tệp kho tri thức `.vaidb` cho giải pháp `phap-ly`")
    sol_parser.add_argument("--no-open-corpus", action="store_true", help="Không nạp thêm 100 điều khoản pháp lý mở mặc định")
    sol_parser.add_argument("--voice", action="store_true", help="Khởi động máy chủ WebRTC Voicebot thời gian thực cho CSKH")

    # === AGENT (DEEP RESEARCH & CODE HEALING) ===
    agent_parser = subparsers.add_parser("agent", help="Tác tử tự hành cao cấp (Nghiên cứu sâu & Tự sửa mã nguồn)")
    agent_subs = agent_parser.add_subparsers(dest="subagent", help="Loại tác tử")

    res_parser = agent_subs.add_parser("research", help="Tác tử nghiên cứu tự hành đa bước (TacTuNghienCuuSau)")
    res_parser.add_argument("query", help="Chủ đề hoặc câu hỏi cần nghiên cứu sâu")
    res_parser.add_argument("--output", "-o", default=None, help="Đường dẫn file markdown xuất báo cáo")
    res_parser.add_argument("--steps", type=int, default=5, help="Số bước thu thập tối đa")

    code_parser = agent_subs.add_parser("code", help="Tác tử lập trình TDD tự chạy test & sửa lỗi (TacTuLapTrinhTuSua)")
    code_parser.add_argument("prompt", help="Yêu cầu lập trình")
    code_parser.add_argument("--output", "-o", default=None, help="Đường dẫn lưu mã nguồn đã kiểm thử")
    code_parser.add_argument("--max-retries", type=int, default=3, help="Số lần tự sửa tối đa")

    # === PULL (VIETNAMESE GGUF MODEL HUB & RAM-AWARE QUANTIZATION) ===
    pull_parser = subparsers.add_parser(
        "pull",
        help="Tải mô hình LLM GGUF tiếng Việt chuẩn (tự chọn Q4_K_M/Q8_0 theo RAM, hỗ trợ Range Resume & SHA-256)",
    )
    pull_parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Tên mô hình (VD: qwen2.5-3b-q4, phogpt-4b, vistral-7b, hf://org/repo/file.gguf)",
    )
    pull_parser.add_argument("--list", action="store_true", help="Liệt kê danh mục mô hình tiếng Việt và đề xuất theo RAM máy")
    pull_parser.add_argument("--installed", action="store_true", help="Liệt kê các mô hình GGUF đã tải về máy")
    pull_parser.add_argument("--remove", default=None, help="Xóa một mô hình GGUF đã tải khỏi bộ nhớ đệm")
    pull_parser.add_argument("--quant", default="auto", help="Mức lượng tử hóa (auto, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0)")
    pull_parser.add_argument("--dry-run", action="store_true", help="Xem trước URL, dung lượng và biến thể lượng tử mà không tải")
    pull_parser.add_argument("--force", action="store_true", help="Tải lại kể cả khi mô hình đã tồn tại")
    pull_parser.add_argument("--dir", default=None, help="Thư mục lưu mô hình tùy chỉnh (mặc định: ~/.cache/vietnamese-ai/models)")
    pull_parser.add_argument("--sha256", default=None, help="Mã băm SHA-256 kỳ vọng để xác minh toàn vẹn")
    pull_parser.add_argument("--hf-token", default=None, help="HuggingFace access token (tùy chọn cho repo riêng tư)")

    # === RUN (ZERO-GPU NATIVE INFERENCE ENGINE) ===
    run_parser = subparsers.add_parser(
        "run", help="Chạy suy luận CPU siêu tốc Zero-GPU (mmap GGUF, llama.cpp native, Tường lửa & RAG pháp lý)"
    )
    run_parser.add_argument("model", help="Đường dẫn file .gguf hoặc bí danh mô hình đã tải (VD: qwen2.5-3b-q4, phogpt-4b, vgpt2)")
    run_parser.add_argument("--prompt", "-p", default=None, help="Câu hỏi đơn lẻ cần suy luận")
    run_parser.add_argument("--chat", action="store_true", help="Vào chế độ đối thoại hội thoại tương tác trực tiếp")
    run_parser.add_argument("--threads", "-t", type=int, default=4, help="Số luồng CPU (mặc định: 4)")
    run_parser.add_argument("--bench", action="store_true", help="Hiển thị bảng thống kê hiệu năng chi tiết (T/s, TTFT, RAM)")
    run_parser.add_argument("--temp", type=float, default=0.7, help="Nhiệt độ ngẫu nhiên (mặc định: 0.7)")
    run_parser.add_argument("--max-tokens", type=int, default=128, help="Số token tối đa sinh ra")
    run_parser.add_argument("--rag", default=None, help="Đường dẫn tệp (`.vaidb`/`.txt`/`.docx`/`.pdf`/`.json`) hoặc thư mục tri thức gắn kèm RAG")
    run_parser.add_argument("--admin", "--hanh-chinh", action="store_true", help="Bật chế độ RAG pháp lý/hành chính Việt Nam khi dùng --rag")
    run_parser.add_argument("--native", action="store_true", help="Ưu tiên chạy trực tiếp qua llama-cpp-python / llama-cli nếu có")

    # === PACK (1-CLICK DESKTOP OFFLINE STANDALONE PACKAGER) ===
    pack_parser = subparsers.add_parser("pack", help="Tự động tạo gói Desktop 1-Click Offline (run_desktop.sh / .bat)")
    pack_parser.add_argument("--output", "-o", default="vietnamese_ai_desktop", help="Thư mục xuất gói ứng dụng")
    pack_parser.add_argument("--mode", default="studio", choices=["studio", "chat"], help="Chế độ khởi chạy mặc định")
    pack_parser.add_argument("--port", type=int, default=8080, help="Cổng chạy Web Studio")
    pack_parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cấu hình")

    # === SWARM (CPU SWARM INFERENCE & PIPELINE PARALLELISM) ===
    swarm_parser = subparsers.add_parser(
        "swarm", help="Suy luận bầy đàn phân tán trên CPU văn phòng (Pipeline Parallelism & Verifiable AI)"
    )
    swarm_parser.add_argument("--nodes", "-n", type=int, default=3, help="Số node CPU văn phòng tham gia (mặc định: 3)")
    swarm_parser.add_argument("--layers", "-l", type=int, default=6, help="Tổng số layer mô hình sharding (mặc định: 6)")
    swarm_parser.add_argument("--prompt", "-p", default="Trí tuệ nhân tạo Việt Nam phát triển bền vững.", help="Câu prompt đầu vào")
    swarm_parser.add_argument("--tokens", type=int, default=16, help="Số token tối đa cần sinh")
    swarm_parser.add_argument("--verify", action="store_true", default=True, help="Bật xác thực mật mã học Merkle Tree & zkML")
    swarm_parser.add_argument("--bench", action="store_true", help="Hiển thị bảng đo lường hiệu năng Swarm")
    swarm_parser.add_argument("--host", action="store_true", help="Khởi chạy máy chủ Master lắng nghe gia nhập cụm Swarm")
    swarm_parser.add_argument("--join", default=None, help="Gia nhập cụm máy chủ qua địa chỉ IP:PORT")
    swarm_parser.add_argument("--port", type=int, default=9999, help="Cổng TCP Socket lắng nghe (mặc định: 9999)")
    swarm_parser.add_argument("--pin", default=None, help="Mã PIN bảo mật xác thực cụm Zero-Trust")
    swarm_parser.add_argument("--tls-cert", default=None, help="Đường dẫn chứng chỉ TLS (.pem/.crt) cho Swarm TCP LAN")
    swarm_parser.add_argument("--tls-key", default=None, help="Đường dẫn khóa riêng TLS (.pem/.key) cho Swarm TCP LAN")
    swarm_parser.add_argument("--tls-ca", default=None, help="Đường dẫn chứng chỉ CA (.pem) cho TLS/mTLS Swarm")
    swarm_parser.add_argument("--mtls", action="store_true", help="Bắt buộc xác thực hai chiều mTLS (cần --tls-ca)")
    swarm_parser.add_argument("--fp16", action="store_true", help="Bật nén tensor Float16 giảm 50%% băng thông mạng LAN")
    swarm_parser.add_argument("--tailscale", action="store_true", help="Tự động phát hiện và liên kết qua mạng Tailscale")
    swarm_parser.add_argument(
        "--model",
        "--gguf",
        default=None,
        dest="gguf",
        help="Đường dẫn file GGUF để tự động phân mảnh trọng số vào các Node CPU",
    )
    swarm_parser.add_argument(
        "--orchestrator-port",
        type=int,
        default=None,
        help="Khởi động Cầu nối HTTP Orchestrator Master (NutCauNoiSwarm) trên cổng chỉ định (ví dụ: 8085)",
    )

    # === VOICE (REAL-TIME VOICE AI & WEBRTC WEBSOCKET) ===
    voice_parser = subparsers.add_parser(
        "voice",
        help="Khởi động máy chủ Voice AI thời gian thực (WebRTC / WebSocket & VAD thuần Việt)",
    )
    voice_parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ IP lắng nghe (mặc định: 127.0.0.1)")
    voice_parser.add_argument("--port", type=int, default=8765, help="Cổng mạng WebSocket (mặc định: 8765)")
    voice_parser.add_argument("--sample-rate", type=int, default=16000, help="Tần số lấy mẫu âm thanh Hz (mặc định: 16000)")
    voice_parser.add_argument("--llm", action="store_true", help="Kích hoạt mô hình ngôn ngữ VietnameseLLM để phản hồi thoại")
    voice_parser.add_argument("--model", default="vgpt2", help="Tên hoặc đường dẫn mô hình LLM (mặc định: vgpt2)")
    voice_parser.add_argument("--rag", default=None, help="Đường dẫn tệp tài liệu tri thức (văn bản/JSONL) cho RAG Voice")
    voice_parser.add_argument("--cskh", action="store_true", help="Kích hoạt chế độ Tổng đài CSKH thoại tự động (tra cứu đơn hàng, đổi trả, bảo hành)")
    voice_parser.add_argument("--open-browser", action="store_true", help="Tự động mở giao diện EvoNet OS Voice trong trình duyệt")

    # === VERIFY (VERIFIABLE INFERENCE AUDIT) ===
    verify_parser = subparsers.add_parser(
        "verify", help="Kiểm tra và xác thực tính trung thực của kết quả suy luận AI (Merkle Proof & zkML)"
    )
    verify_parser.add_argument("--proof", default=None, help="Chuỗi cam kết zkML (zkml1:...) hoặc mã hash cần xác thực")
    verify_parser.add_argument("--text", "-t", default=None, help="Văn bản cần kiểm tra tính toàn vẹn Merkle Tree")
    verify_parser.add_argument("--tamper-check", action="store_true", help="Chạy thử nghiệm phát hiện Byzantine can thiệp dữ liệu")

    # === PLUGIN (V-PLUGIN 2.0 DOMAIN EXTENSION ECOSYSTEM) ===
    plugin_parser = subparsers.add_parser(
        "plugin", help="Quản lý và thực thi hệ sinh thái Plugin chuyên ngành (V-Plugin 2.0)"
    )
    plugin_parser.add_argument("action", choices=["list", "verify", "info", "run"], default="list", nargs="?", help="Hành động: list, verify, info, run")
    plugin_parser.add_argument("target", default=None, nargs="?", help="Đường dẫn file .py hoặc tên Plugin")
    plugin_parser.add_argument("--tool", default=None, help="Tên công cụ muốn thực thi trong Plugin")
    plugin_parser.add_argument("--params", default="{}", help="Tham số thực thi dạng JSON (ví dụ: '{\"thu_nhap_thang\": 25000000}')")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Chạy trình diễn 1-Click thực chiến trong 5 phút (Legal RAG 100 điều khoản, Edge CPU, Bảo mật, GPU Flywheel)",
    )
    demo_parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "legal-rag", "edge-chat", "security", "flywheel"],
        help="Kịch bản demo cần chạy (mặc định: all)",
    )

    rel_parser = subparsers.add_parser(
        "release-check",
        help="Kiểm định toàn diện trước khi phát hành TestPyPI/PyPI (SemVer, phụ thuộc, type stubs, Rust abi3)",
    )
    rel_parser.add_argument(
        "--json",
        action="store_true",
        help="Xuất báo cáo kiểm định dưới dạng JSON thuần để tích hợp CI/CD",
    )

    lg_parser = subparsers.add_parser(
        "legal-graph",
        help="Truy vấn Đồ thị Tham chiếu Pháp lý Việt Nam (Luật ↔ Nghị định ↔ Thông tư ↔ Sửa đổi/Thay thế)",
    )
    lg_parser.add_argument("so_hieu", help="Số hiệu văn bản cần tra cứu (VD: 45/2013/QH13, 31/2024/QH15, 123/2020/NĐ-CP)")
    lg_parser.add_argument("--date", default=None, help="Ngày tra cứu hiệu lực dạng YYYY-MM-DD (mặc định: hôm nay)")
    lg_parser.add_argument("--dieu", default=None, help="Tra cứu phiên bản Point-in-Time của một Điều cụ thể (VD: 'Điều 19')")
    lg_parser.add_argument("--khoan", default=None, help="Lọc thêm theo Khoản cụ thể khi dùng `--dieu`")
    lg_parser.add_argument("--active-only", action="store_true", help="Chỉ hiển thị văn bản liên quan còn hiệu lực")
    lg_parser.add_argument("--mermaid", action="store_true", help="In thêm sơ đồ Mermaid flowchart LR")

    li_parser = subparsers.add_parser(
        "legal-ingest",
        help="Nạp tự động văn bản pháp quy / văn bản hợp nhất (.docx/.pdf/.txt/.md hoặc thư mục) thành 1 tệp .vaidb HNSW + Point-in-Time",
    )
    li_parser.add_argument("source", help="Đường dẫn tệp hoặc thư mục văn bản pháp quy cần nạp")
    li_parser.add_argument("--output", "-o", required=True, help="Đường dẫn tệp kho tri thức `.vaidb` đầu ra")
    li_parser.add_argument("--no-hnsw", action="store_true", help="Không dựng chỉ mục HNSW bền vững trong tệp `.vaidb`")
    li_parser.add_argument("--json", action="store_true", help="Xuất báo cáo nạp tự động dưới dạng JSON thuần")

    cr_parser = subparsers.add_parser(
        "contract-review",
        help="Rà soát rủi ro & so sánh hợp đồng (.docx/.pdf/.txt) kèm xuất báo cáo Redline Markdown & Word .docx",
    )
    cr_parser.add_argument("partner", help="Đường dẫn tệp hợp đồng đối tác gửi (.docx/.pdf/.txt)")
    cr_parser.add_argument("--template", default=None, help="Đường dẫn tệp hợp đồng mẫu chuẩn để đối chiếu từng điều khoản")
    cr_parser.add_argument("--date", default=None, help="Ngày kiểm toán hiệu lực luật dạng YYYY-MM-DD")
    cr_parser.add_argument("--output-md", default=None, help="Đường dẫn xuất báo cáo Redline định dạng Markdown (.md)")
    cr_parser.add_argument("--output-docx", default=None, help="Đường dẫn xuất báo cáo Redline định dạng Word (.docx)")
    cr_parser.add_argument(
        "--track-changes-docx",
        default=None,
        help="Đường dẫn xuất tệp Word (.docx) chứa Track Changes thực (<w:ins>/<w:del>) và chú thích lề comments.xml",
    )

    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Khởi chạy MCP Server chuẩn stdio/SSE cắm trực tiếp vào Cursor, VS Code & Claude Desktop",
    )
    mcp_parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="Giao vận MCP (stdio hoặc sse)")
    mcp_parser.add_argument("--host", default="127.0.0.1", help="Host cho chế độ SSE (mặc định: 127.0.0.1)")
    mcp_parser.add_argument("--port", type=int, default=8090, help="Cổng cho chế độ SSE (mặc định: 8090)")
    mcp_parser.add_argument("--vaidb", default=None, help="Đường dẫn tệp kho tri thức `.vaidb` gắn cho MCP tools")
    mcp_parser.add_argument("--plugin-dir", default=None, help="Thư mục V-Plugin 2.0 tự động nạp thêm vào MCP Server")
    mcp_parser.add_argument(
        "--extended-tools",
        action="store_true",
        help="Bật thêm các công cụ MCP chuyên sâu (Point-in-Time Điều luật, Viterbi NLP, HNSW .vaidb)",
    )
    mcp_parser.add_argument(
        "--print-config",
        default=None,
        choices=["cursor", "claude", "vscode"],
        help="In cấu hình JSON mẫu để dán trực tiếp vào Cursor / Claude Desktop / VS Code rồi thoát",
    )

    conn_parser = subparsers.add_parser(
        "connector",
        help="Khởi động Webhook Server kết nối Zalo OA & Telegram Bot với Trợ lý Pháp lý / CSKH",
    )
    conn_parser.add_argument("channel", choices=["zalo", "telegram", "both"], help="Kênh kết nối cần chạy")
    conn_parser.add_argument("--solution", default="phap-ly", choices=["phap-ly", "cskh"], help="Bộ xử lý AI gắn vào kênh")
    conn_parser.add_argument("--host", default="0.0.0.0", help="Host lắng nghe Webhook")
    conn_parser.add_argument("--port", type=int, default=8095, help="Cổng lắng nghe Webhook (mặc định: 8095)")
    conn_parser.add_argument("--vaidb", default=None, help="Đường dẫn tệp `.vaidb` (khi dùng giải pháp pháp lý)")
    conn_parser.add_argument("--workspace-dir", default=None, help="Thư mục SQLite Workspace để lưu bền vững lịch sử chat & gom DPO")
    conn_parser.add_argument("--dry-run", action="store_true", help="Chỉ in cấu hình endpoint và trạng thái mà không mở socket chặn")

    dpo_parser = subparsers.add_parser(
        "export-dpo",
        help="Xuất phản hồi Thumbs Up/Down từ Studio Workspace sang tập dữ liệu DPO JSONL đã ẩn PII",
    )
    dpo_parser.add_argument("--workspace-dir", required=True, help="Thư mục dữ liệu Workspace SQLite")
    dpo_parser.add_argument("--workspace", default=None, help="Lọc theo mã Workspace (tùy chọn)")
    dpo_parser.add_argument("--output", default="studio_dpo_feedback.jsonl", help="Đường dẫn tệp DPO JSONL xuất ra")

    eg_parser = subparsers.add_parser(
        "eval-gate",
        help="Chạy Cổng Kiểm Định hồi quy tự động (Citation Accuracy, Hiệu lực luật, Chống ảo giác & An toàn)",
    )
    eg_parser.add_argument("--limit", type=int, default=8, help="Số câu hỏi pháp lý chuẩn cần chấm (mặc định: 8)")
    eg_parser.add_argument("--min-score", type=float, default=0.55, help="Ngưỡng điểm tổng hợp tối thiểu để đạt chuẩn")
    eg_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="So sánh với mô hình nền và tính khoảng tin cậy Paired Bootstrap 95%% CI",
    )
    eg_parser.add_argument(
        "--extended-attacks",
        action="store_true",
        help="Bật bộ kiểm thử tấn công mở rộng gồm cả Prompt Injection tiếng Việt không dấu",
    )
    eg_parser.add_argument("--json", action="store_true", help="Xuất kết quả kiểm định dưới dạng JSON thuần")

    sm_parser = subparsers.add_parser(
        "status-map",
        help="Hiển thị bản đồ minh bạch 3 tầng Core Production vs Optional Extensions vs Experimental/Simulated",
    )
    sm_parser.add_argument("--json", action="store_true", help="Xuất bản đồ trạng thái dưới dạng JSON")

    return parser


def _doc_du_lieu_csv(duong_dan: str, target: Optional[str] = None):
    """Đọc dữ liệu từ file CSV (ưu tiên pandas nếu có, tự động fallback về `csv` chuẩn của Python)."""
    try:
        import pandas as pd

        df = pd.read_csv(duong_dan)
        if target:
            if target not in df.columns:
                print(f"Lỗi: Không tìm thấy cột '{target}' trong dữ liệu")
                print(f"Các cột có sẵn: {list(df.columns)}")
                sys.exit(1)
            y = df[target].values
            X = df.drop(columns=[target]).values
        else:
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1].values
        return X, y
    except ImportError:
        from vietnamese_ai.utils.lazy_loader import doc_csv_nhe

        try:
            return doc_csv_nhe(duong_dan, cot_nhan=target)
        except ValueError as e:
            print(f"Lỗi: {e}")
            sys.exit(1)


def _doc_ma_tran_csv(duong_dan: str) -> np.ndarray:
    """Đọc ma trận đặc trưng đầu vào cho `predict` không bắt buộc `pandas`."""
    try:
        import pandas as pd

        return pd.read_csv(duong_dan).values
    except ImportError:
        import csv

        with open(duong_dan, "r", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
        if not rows:
            return np.empty((0, 0))
        try:
            for c in rows[0]:
                float(c)
            start = 0
        except ValueError:
            start = 1
        return np.asarray([[float(x) for x in r] for r in rows[start:]], dtype=float)


def _luu_ket_qua_csv(duong_dan: str, ket_qua: Any):
    """Lưu kết quả dự đoán ra CSV (tự động fallback sang thư viện chuẩn `csv` khi không có `pandas`)."""
    try:
        import pandas as pd

        df = pd.DataFrame({"du_doan": ket_qua})
        df.to_csv(duong_dan, index=False)
    except ImportError:
        from vietnamese_ai.utils.lazy_loader import luu_csv_nhe

        luu_csv_nhe(duong_dan, ket_qua, ten_cot="du_doan")
    print(f"Đã lưu kết quả tại: {duong_dan}")


def _tai_mo_hinh_an_toan(duong_dan_hoac_ten: str) -> Any:
    """
    Nạp mô hình an toàn cho CLI:
    1. Kiểm tra chữ ký toàn vẹn HMAC-SHA256 (`LuuTai.xac_thuc_file`) nếu có tệp `.sig` đi kèm.
    2. Sử dụng `RestrictedUnpickler` (`tai_an_toan`) chống tấn công thực thi mã độc Pickle RCE.
    3. Hỗ trợ cả tệp `.pkl`, thư mục fine-tune, hoặc alias LLM (`vgpt2`, `qwen3.8`...).
    """
    from vietnamese_ai.utils.io_utils import LuuTai, tai_an_toan

    p = Path(duong_dan_hoac_ten)
    if p.is_file():
        sig_file = p.with_suffix(p.suffix + ".sig")
        if sig_file.exists() and not LuuTai.xac_thuc_file(str(p)):
            raise ValueError(
                f"Bảo mật: Chữ ký HMAC-SHA256 của mô hình '{p}' không hợp lệ! Tệp có thể đã bị chỉnh sửa trái phép."
            )
        return tai_an_toan(str(p), cho_phep_mo_hinh=True)

    if p.is_dir():
        from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

        tuner = BoTinhChinhMoHinh(che_do_offline=True)
        tuner.tai(str(p))
        return tuner

    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    return BoTinhChinhMoHinh(mo_hinh=duong_dan_hoac_ten, che_do_offline=True)


def cmd_info():
    """Hiển thị thông tin framework."""
    from vietnamese_ai import __all__ as public_exports
    from vietnamese_ai import __version__

    print(f"""
╔════════════════════════════════════════════╗
║       Vietnamese AI Framework              ║
║       Version: {__version__:<26s}║
╚════════════════════════════════════════════╝

Kiến trúc: Micro-Core Siêu Nhẹ (Lazy Import PEP 562, {len(public_exports)} biểu tượng) + Tường lửa AI Đa tầng
Phân hệ chính:
  - RAG & Pháp lý VN:     RAGPipeline (.vaidb + onnx_viet), BoTheoDoiHieuLucPhapLy, DoThiThamChieuPhapLy, BoSoSanhRaSoatHopDong
  - Edge & GPU Flywheel:  QuanLyMoHinhEdge, DongCoSuyLuanEdge, BoDieuPhoiTinhChinhXa (Modal/SSH/Local), CongKiemDinhMoHinh
  - Workbench & Kênh:     StudioDashboard, QuanLyPhienChatWorkspace (DPO), MayChuMCPThucTe, KetNoiZaloOA, KetNoiTelegram

Lệnh tiêu biểu:
  vai demo [all|legal-rag|edge-chat]       Trình diễn 1-Click trọn vẹn trong 5 phút
  vai doctor / vai status-map              Chẩn đoán phần cứng & bản đồ minh bạch 3 tầng
  vai pull qwen2.5-3b-q4 && vai run ...    Tải & chạy LLM GGUF tiếng Việt trên CPU (--rag .vaidb --admin)
  vai rag --db kb.vaidb --docs ./luat --semantic-onnx --admin --query "..."
  vai legal-graph 45/2013/QH13 --mermaid   Truy vấn Đồ thị Tham chiếu Pháp lý Việt Nam
  vai contract-review hd.docx --template mau.docx --output-docx redline.docx
  vai studio --vaidb kb.vaidb --workspace-dir ./ws --port 8080
  vai mcp-server --transport stdio|sse     Cắm trực tiếp vào Cursor, VS Code & Claude Desktop
  vai connector both --solution phap-ly --workspace-dir ./ws
  vai export-dpo --workspace-dir ./ws      Xuất phản hồi người dùng sang DPO JSONL đã ẩn PII
  vai train-cloud --from-workspace ./ws --provider modal --eval-4-axis
  vai eval-gate / vai release-check        Kiểm định hồi quy 4 trục & cổng phát hành PyPI
""")


def cmd_doctor():
    """Chẩn đoán toàn diện sức khỏe môi trường (`vai doctor`)."""
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    bc = QuanLyPhuThuoc.chan_doan_he_thong()
    cpu_info = bc.get("kien_truc_cpu", {})
    simd_str = ", ".join(cpu_info.get("tap_lenh_simd", [])) or "chuẩn cơ bản"
    print(f"=== CHẨN ĐOÁN HỆ THỐNG VIETNAMESE AI v{bc['phien_ban_framework']} ===")
    print(f"  Python:            {bc['python']} ({bc['he_dieu_hanh']})")
    print(f"  CPU / GPU:         {bc['so_nhan_cpu']} nhân CPU ({cpu_info.get('kien_truc', 'unknown')}) | CUDA GPU: {bc['ten_gpu'] or 'Không (chạy tối ưu CPU/NumPy)'}")
    print(f"  Tập lệnh SIMD:     {simd_str}")
    print(f"  Rust Core (abi3):  {'SẴN SÀNG (tăng tốc native)' if bc.get('co_rust_native') else 'Fallback Python/NumPy'}")
    print(f"  Trình cài nhanh:   {'uv + pip' if bc.get('co_uv') else 'pip'} | VAI_AUTO_INSTALL: {'BẬT' if bc.get('tu_dong_cai_bat') else 'TẮT (Opt-in)'}")
    print(f"  Cache PEP 668:     {bc.get('thu_muc_cache_runtime', '')}")
    print(f"  Tốc độ nạp Core:   {bc['thoi_gian_nap_core_ms']} ms")
    print("  Bảo mật mặc định:  Tường lửa AI Đa tầng + Ký số HMAC-SHA256 [BẬT]")
    print("\nTrạng thái các nhóm gói mở rộng:")
    for nhom, tt in bc["nhom_phu_thuoc"].items():
        icon = "[SẴN SÀNG]" if tt["san_sang"] else "[TÙY CHỌN]"
        print(f"  {icon:<12s} {nhom:<9s} (~{tt['kich_thuoc_uoc_tinh_mb']:>3d} MB) - {tt['mo_ta']}")
    return bc


def cmd_install(args):
    """Cài đặt nhóm phụ thuộc mở rộng (`vai install <group>`)."""
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    kq = QuanLyPhuThuoc.cai_dat_nhom(args.group, chay_that=not getattr(args, "dry_run", False))
    if getattr(args, "dry_run", False):
        print(f"[Dry-Run] Nhóm '{kq['nhom']}': {kq['lenh_goi_y']}")
        print(f"  Các gói sẽ cài: {', '.join(kq['danh_sach_goi'])}")
    elif kq["thanh_cong"]:
        print(f"Đã cài đặt thành công nhóm '{kq['nhom']}'!")
    else:
        print(f"Cài đặt nhóm '{kq['nhom']}' thất bại:\n{kq.get('dau_ra', '')}")
        sys.exit(1)
    return kq


def cmd_setup(args):
    """Thiết lập môi trường thông minh 3 tầng (`vai setup --profile <...> [--apply]`)."""
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    chay_that = bool(getattr(args, "apply", False))
    bien_dich_rust = not bool(getattr(args, "no_rust", False))
    profile = str(getattr(args, "profile", "core"))
    kq = QuanLyPhuThuoc.thiet_lap_tu_dong(
        profile=profile,
        chay_that=chay_that,
        bien_dich_rust=bien_dich_rust,
    )
    hw = kq["phan_cung"]
    print(f"=== THIẾT LẬP THÔNG MINH VIETNAMESE AI (Hồ sơ: {kq['ho_so']}) ===")
    print(
        f"  Phần cứng:  {hw['he_dieu_hanh']} ({hw['kien_truc']}, {hw['so_nhan_cpu']} CPU) | "
        f"CUDA: {'Có' if hw['co_cuda_gpu'] else 'Không'} | Rust Core: {'Sẵn sàng' if hw['co_rust_native'] else 'Fallback Python'}"
    )
    for item in kq["cac_nhom_goi"]:
        print(
            f"  Nhóm [{item['nhom']}] ({item['trinh_cai']}): {', '.join(item['danh_sach_goi'])}"
            f"{' [cache runtime]' if item.get('su_dung_cache_runtime') else ''}"
        )
    print(f"  Tầng thấp:  {kq['tang_thap_rust'].get('trang_thai')} - {kq['tang_thap_rust'].get('thong_bao', '')}")
    if not chay_that:
        print(f"  [Dry-Run] Chạy lại với cờ --apply để thực thi cài đặt: vai setup --profile {profile} --apply")
    return kq


def cmd_train(args):
    """Huấn luyện mô hình từ CLI và tự động tạo chữ ký toàn vẹn HMAC-SHA256."""
    from vietnamese_ai.models.classifier import PhanLoai
    from vietnamese_ai.models.regression import HoiQuy
    from vietnamese_ai.preprocessing.numerical import XuLySo
    from vietnamese_ai.utils.io_utils import LuuTai
    from vietnamese_ai.utils.logger import Logger
    from vietnamese_ai.utils.validators import Validator

    logger = Logger("CLI-Train")
    logger.info(f"Đọc dữ liệu từ: {args.data}")

    X, y = _doc_du_lieu_csv(args.data, args.target)
    logger.info(f"Dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")

    nhiem_vu = Validator.kiem_tra_nhiem_vu(y)
    logger.info(f"Nhiệm vụ: {nhiem_vu}")

    X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y, ty_le_test=args.test_size)

    if nhiem_vu == "phan_loai":
        mo_hinh = PhanLoai(thuat_toan=args.model)
    else:
        mo_hinh = HoiQuy(thuat_toan=args.model)

    logger.info(f"Huấn luyện mô hình: {args.model}")
    mo_hinh.huan_luyen(X_train, y_train)

    diem = mo_hinh.danh_gia(X_test, y_test)
    logger.info(f"Điểm đánh giá: {diem:.4f}")

    mo_hinh.luu(args.output)
    LuuTai.ky_file(args.output)
    logger.info(f"Đã lưu và ký số HMAC-SHA256 mô hình tại: {args.output}")


def cmd_predict(args):
    """Dự đoán từ CLI (có kiểm tra chữ ký HMAC-SHA256 và RestrictedUnpickler)."""
    from vietnamese_ai.utils.logger import Logger

    logger = Logger("CLI-Predict")
    logger.info(f"Tải mô hình an toàn từ: {args.model}")

    mo_hinh = _tai_mo_hinh_an_toan(args.model)
    logger.info(f"Loại mô hình: {type(mo_hinh).__name__}")

    X = _doc_ma_tran_csv(args.input)
    logger.info(f"Dự đoán {len(X)} mẫu")
    ket_qua = mo_hinh.du_doan(X)

    _luu_ket_qua_csv(args.output, ket_qua)
    logger.info("Dự đoán hoàn tất")


def cmd_serve(args):
    """Khởi động API server từ CLI (hỗ trợ cả `simple` zero-dependency và `fastapi` SSE/Tools/Metrics)."""
    from vietnamese_ai.utils.logger import Logger

    logger = Logger("CLI-Serve")
    logger.info(f"Tải mô hình an toàn từ: {args.model}")

    mo_hinh = _tai_mo_hinh_an_toan(args.model)
    logger.info(f"Loại mô hình: {type(mo_hinh).__name__}")

    backend = getattr(args, "backend", "simple")
    if backend == "fastapi":
        from vietnamese_ai.api.fastapi_server import FastAPIServer

        server_fa = FastAPIServer(bo_xu_ly=mo_hinh, ten="VAI-FastAPI-Server")
        server_fa.chay(host=args.host, port=args.port)
        return server_fa

    from vietnamese_ai.api.server import ServerDonGian

    server = ServerDonGian(mo_hinh=mo_hinh, ten="VAI-Server", bat_tuong_lua=True)
    server.chay(host=args.host, port=args.port)
    return server


def cmd_evaluate(args):
    """Đánh giá mô hình từ CLI."""
    from vietnamese_ai.utils.logger import Logger
    from vietnamese_ai.utils.metrics import Metrics
    from vietnamese_ai.utils.validators import Validator

    logger = Logger("CLI-Evaluate")
    logger.info(f"Tải mô hình an toàn từ: {args.model}")

    mo_hinh = _tai_mo_hinh_an_toan(args.model)
    X, y = _doc_du_lieu_csv(args.data, args.target)

    logger.info(f"Đánh giá trên {len(X)} mẫu")
    mo_hinh.danh_gia(X, y)

    nhiem_vu = Validator.kiem_tra_nhiem_vu(y)
    du_doan = mo_hinh.du_doan(X)

    if nhiem_vu == "phan_loai":
        bc = Metrics.bao_cao_phan_loai(y, du_doan)
    else:
        bc = Metrics.bao_cao_hoi_quy(y, du_doan)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    for chi_so, gia_tri in bc.items():
        print(f"  {chi_so}: {gia_tri:.4f}")
    print()


def cmd_web(args):
    """Khởi động giao diện web no-code."""
    from vietnamese_ai.ui.web_app import UngDungWeb

    app = UngDungWeb(port=args.port, host=args.host)
    app.chay()


def cmd_finetune(args):
    """Tinh chỉnh mô hình từ CLI."""
    import json

    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu '{args.data}'")
        sys.exit(1)

    lines = data_path.read_text(encoding="utf-8").strip().splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    tuner = BoTinhChinhMoHinh(
        mo_hinh=args.model,
        nhiem_vu=args.task,
        dung_lora=True,
        rank_lora=args.rank,
        che_do_offline=True,
    )
    kq = tuner.huan_luyen(samples, so_vong=args.epochs)
    tuner.luu(args.output)
    print(f"Đã fine-tune thành công ({args.epochs} epochs, loss_min={kq.get('train_loss_min', 0.0):.4f}) -> {args.output}")


def cmd_train_cloud(args):
    """Tinh chỉnh GPU từ xa đa nhà cung cấp (`vai train-cloud`), hỗ trợ lấy trực tiếp tập DPO từ `--from-workspace`."""
    from vietnamese_ai.fine_tuning.remote_gpu import (
        BoDieuPhoiTinhChinhXa,
        CauHinhTinhChinhXa,
        NhaCungCapSSH,
    )

    data_source = getattr(args, "data", None)
    from_ws = getattr(args, "from_workspace", None)
    if not data_source and not from_ws:
        print("Lỗi: Cần truyền `--data <file.jsonl>` hoặc `--from-workspace <thu_muc_workspace>`.")
        sys.exit(1)

    dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec=args.output)
    task_name = str(getattr(args, "task", "sft"))
    if from_ws:
        kq_dpo = dieu_phoi.chuan_bi_du_lieu_tu_workspace(
            thu_muc_workspace=from_ws,
            ma_workspace=getattr(args, "workspace", None),
            an_pii=True,
        )
        data_source = kq_dpo["duong_dan_jsonl"]
        if task_name == "sft":
            task_name = "dpo"
        print(f"[WORKSPACE -> DPO] Đã xuất {kq_dpo['so_mau_dpo']} mẫu DPO sạch tại: {data_source}")

    if args.provider == "ssh" and getattr(args, "ssh_host", ""):
        dieu_phoi.dang_ky_nha_cung_cap(
            "ssh",
            NhaCungCapSSH(
                host=args.ssh_host,
                user=getattr(args, "ssh_user", None),
                port=int(getattr(args, "ssh_port", 22)),
                khoa_ssh=getattr(args, "ssh_key", None),
            ),
        )

    cfg = CauHinhTinhChinhXa(
        mo_hinh=args.model,
        nhiem_vu=task_name,
        loai_gpu=args.gpu,
        so_vong=args.epochs,
        rank_lora=args.rank,
        quantization=args.quantization,
        ngan_sach_toi_da_usd=float(getattr(args, "max_budget_usd", 2.0)),
        cho_phep_gui_cloud=bool(getattr(args, "allow_cloud", False)),
    )
    kq = dieu_phoi.chay_vong_lap_flywheel(
        nguon_du_lieu=data_source,
        ten_nha_cung_cap=args.provider,
        cau_hinh=cfg,
        chay_that=bool(getattr(args, "apply", False)),
        su_dung_eval_gate_4_truc=bool(getattr(args, "eval_4_axis", False)),
    )
    print(f"=== REMOTE GPU FLYWHEEL [{kq.nha_cung_cap.upper()}] ({kq.trang_thai}) ===")
    print(f"  Mã tác vụ:          {kq.ma_tac_vu}")
    if kq.bao_cao_an_ninh_du_lieu:
        an = kq.bao_cao_an_ninh_du_lieu
        print(
            f"  Làm sạch dữ liệu:   {an.get('so_mau_sau_lam_sach')}/{an.get('tong_mau_ban_dau')} mẫu hợp lệ "
            f"(đã ẩn PII: {an.get('so_mau_da_an_pii', 0)})"
        )
    if kq.bao_cao_chi_phi:
        cp = kq.bao_cao_chi_phi
        print(
            f"  Chi phí ước tính:   {cp.get('chi_phi_uoc_tinh_usd', 0.0)} USD "
            f"(~{int(cp.get('chi_phi_uoc_tinh_vnd', 0))} VNĐ | giới hạn {cp.get('ngan_sach_toi_da_usd')} USD)"
        )
    if kq.duong_dan_gguf:
        print(f"  GGUF tải về:        {kq.duong_dan_gguf}")
    if kq.duong_dan_script:
        print(f"  Script tác vụ:      {kq.duong_dan_script}")
    if kq.chi_tiet_loi:
        print(f"  Cảnh báo / Lỗi:     {kq.chi_tiet_loi}")
    return kq.to_dict()


def cmd_rag(args):
    """Truy vấn RAG đa định dạng từ CLI (TXT/MD/DOCX/PDF/Ảnh OCR/JSON/JSONL, thư mục & kho bền vững `.vaidb`)."""
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    docs_arg = getattr(args, "docs", None)
    db_arg = getattr(args, "db", None)
    query_arg = getattr(args, "query", None)
    build_hnsw = bool(getattr(args, "build_hnsw", False))
    if not docs_arg and not db_arg:
        print("Lỗi: Cần truyền ít nhất `--docs <tệp/thư mục>` hoặc `--db <file.vaidb>`.")
        sys.exit(1)
    if not query_arg and not build_hnsw:
        print("Lỗi: Cần truyền `--query <câu hỏi>` hoặc `--build-hnsw`.")
        sys.exit(1)

    che_do_hc = bool(getattr(args, "admin", False))
    dung_onnx = bool(getattr(args, "semantic_onnx", False))
    rag = RAGPipeline(
        backend="vaidb" if db_arg else "memory",
        duong_dan_vaidb=db_arg,
        che_do_nhung="onnx_viet" if dung_onnx else "auto",
        che_do_rerank="onnx_viet" if dung_onnx else "mmr",
        che_do_hanh_chinh=che_do_hc,
    )

    try:
        if docs_arg:
            doc_path = Path(docs_arg)
            if not doc_path.exists():
                print(f"Lỗi: Không tìm thấy tài liệu '{docs_arg}'")
                sys.exit(1)
            if doc_path.is_dir():
                if db_arg:
                    sync_bc = rag.dong_bo_thu_muc(doc_path, che_do_hanh_chinh=che_do_hc)
                    print(
                        f"[VAIDB] Đồng bộ thư mục: cập nhật {sync_bc['so_tep_cap_nhat']}, "
                        f"bỏ qua không đổi {sync_bc['so_tep_bo_qua_khong_doi']}, xóa {sync_bc['so_tep_da_xoa']}."
                    )
                else:
                    rag.them_thu_muc(doc_path, che_do_hanh_chinh=che_do_hc)
            else:
                rag.them_tep(doc_path, ma=doc_path.name, che_do_hanh_chinh=che_do_hc)

        hnsw_info: Optional[Dict[str, Any]] = None
        if build_hnsw:
            if hasattr(rag.csdl_vector, "xay_dung_chi_muc_hnsw"):
                hnsw_info = rag.csdl_vector.xay_dung_chi_muc_hnsw()
                print(
                    f"[VAIDB HNSW] Đã xây dựng chỉ mục bền vững: "
                    f"{hnsw_info.get('so_nut', 0)} nút, {hnsw_info.get('so_tang', 0)} tầng."
                )
            else:
                print("[VAIDB HNSW] Cảnh báo: Cần truyền `--db <file.vaidb>` để lưu đồ thị HNSW bền vững.")

        if not query_arg:
            return {"hnsw": hnsw_info, "thong_ke": rag.thong_ke()}

        kq = rag.hoi(query_arg)
        if hnsw_info is not None:
            kq["hnsw"] = hnsw_info
        print(f"Trả lời: {kq.get('tra_loi', '')}")
        if che_do_hc and kq.get("trich_dan_phap_ly"):
            td = kq["trich_dan_phap_ly"]
            if td.get("dieu_khoan"):
                print(f"Điều khoản căn cứ: {', '.join(td['dieu_khoan'])}")
            if td.get("so_hieu_van_ban"):
                print(f"Số hiệu văn bản: {', '.join(td['so_hieu_van_ban'])}")
            for cb in td.get("canh_bao_hieu_luc") or []:
                print(f"Cảnh báo hiệu lực: {cb}")
            dt = td.get("do_thi_tham_chieu") or {}
            hd_list = [x["so_hieu"] for x in dt.get("van_ban_huong_dan_con_hieu_luc") or []]
            if hd_list:
                print(f"Văn bản hướng dẫn thi hành còn hiệu lực: {', '.join(hd_list)}")
        return kq
    finally:
        rag.dong()


def cmd_chat(args):
    """Trò chuyện trực tiếp qua CLI được bảo vệ bởi `TuongLuaBaoVeFramework`."""
    from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

    mo_hinh = _tai_mo_hinh_an_toan(getattr(args, "model", "vgpt2"))
    tuong_lua = TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000)

    def _tra_loi_mot_luot(cau_hoi: str) -> str:
        kq_fw = tuong_lua.kiem_tra_dau_vao(cau_hoi, client_id="cli_user")
        if not kq_fw["an_toan"]:
            return f"[Bị chặn bởi Tường lửa AI]: {kq_fw['ly_do']}"
        if hasattr(mo_hinh, "sinh_van_ban"):
            raw = str(mo_hinh.sinh_van_ban(cau_hoi, do_dai=32))
        elif hasattr(mo_hinh, "hoi"):
            r = mo_hinh.hoi(cau_hoi)
            raw = str(r.get("tra_loi", r) if isinstance(r, dict) else r)
        else:
            raw = str(mo_hinh)
        return tuong_lua.loc_dau_ra(raw)

    prompt_don = getattr(args, "prompt", None)
    if prompt_don is not None:
        tra_loi = _tra_loi_mot_luot(prompt_don)
        print(f"AI: {tra_loi}")
        return tra_loi

    print("Nhập câu hỏi (gõ 'thoat' hoặc 'exit' để kết thúc):")
    while True:
        try:
            cau = input("Bạn: ").strip()
        except EOFError:
            break
        if not cau or cau.lower() in ("thoat", "exit", "quit"):
            break
        print(f"AI: {_tra_loi_mot_luot(cau)}")
    return ""


def cmd_benchmark(args=None):
    """Đo kiểm tốc độ huấn luyện hoặc so sánh đối đầu khoa học vs LangChain/LlamaIndex."""
    is_compare = getattr(args, "compare", False) if args else False
    is_latex = getattr(args, "latex", False) if args else False
    out_file = getattr(args, "output", None) if args else None

    if is_compare:
        from vietnamese_ai.benchmark.comparative_benchmark import BoDoLuongDoiDau

        print("Đang thực hiện đo kiểm đối đầu định lượng (Vietnamese AI vs LangChain / LlamaIndex)...")
        suite = BoDoLuongDoiDau()
        kq = suite.chay_benchmark_toan_dien()

        if is_latex:
            bao_cao = suite.xuat_bang_so_sanh_latex(kq)
        else:
            bao_cao = suite.xuat_bang_so_sanh_markdown(kq)

        print(bao_cao)

        if out_file:
            p = Path(out_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix.lower() == ".json":
                suite.xuat_du_lieu_json(p)
            else:
                p.write_text(bao_cao, encoding="utf-8")
            print(f"-> Đã lưu báo cáo benchmark ra tệp: {out_file}")
        return kq

    from vietnamese_ai.fine_tuning.training_accelerator import BenchmarkTangToc

    kq_train = BenchmarkTangToc.do_toc_do_huan_luyen(so_mau=24, so_vong=2)
    kq_kv = BenchmarkTangToc.do_toc_do_suy_luan_kv_cache(do_dai_prompt=16, so_token_sinh=24)
    print("=== KẾT QUẢ BENCHMARK TĂNG TỐC THỰC TẾ ===")
    print(
        f"  [Huấn luyện] Gốc: {kq_train['tokens_moi_giay_goc']} tok/s -> "
        f"Tăng tốc: {kq_train['tokens_moi_giay_tang_toc']} tok/s "
        f"(Nhanh gấp {kq_train['he_so_tang_toc_train']}x)"
    )
    print(
        f"  [KV-Cache]   Không cache: {kq_kv['tokens_moi_giay_khong_cache']} tok/s -> "
        f"KV-Cache: {kq_kv['tokens_moi_giay_kv_cache']} tok/s "
        f"(Nhanh gấp {kq_kv['he_so_tang_toc_kv_cache']}x | Khớp tuyệt đối: {kq_kv['khop_dau_ra_tuyet_doi']})"
    )
    return {"train": kq_train, "kv_cache": kq_kv}


def cmd_tokenize(args):
    """Xử lý lệnh `vai tokenize` (Tách từ & chuẩn hóa tiếng Việt)."""
    from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc

    tok = BoTachTuTiengVietSieuToc()
    text = getattr(args, "text", "Trí tuệ nhân tạo và chuyển đổi số tại Việt Nam")
    noi_tu = not getattr(args, "no_compound", False)
    xoa_dau = getattr(args, "unaccent", False)
    phuong_phap = getattr(args, "method", "max_match")
    chuan_dau = bool(getattr(args, "normalize_tone", False))

    print("=== TÁCH TỪ & CHUẨN HÓA TIẾNG VIỆT (NATIVE FAST TOKENIZER) ===")
    print(f"Văn bản gốc   : {text}")
    print(f"Phương pháp   : {phuong_phap}")

    if chuan_dau:
        vb_dau = tok.chuan_hoa_dau_thanh(text)
        print(f"Chuẩn hóa dấu : {vb_dau}")

    if xoa_dau:
        khong_dau = tok.xoa_dau_tieng_viet(text)
        print(f"Không dấu      : {khong_dau}")

    tokens = tok.tach_tu(text, noi_tu_ghep=noi_tu, phuong_phap=phuong_phap)
    print(f"Số lượng tokens: {len(tokens)}")
    print(f"Tokens kết quả : {tokens}")

    if getattr(args, "encode", False):
        token_ids = tok.ma_hoa(text)
        print(f"Token IDs      : {token_ids}")
        return token_ids

    return tokens


def cmd_scan(args):
    """Quét kiểm toán an ninh & chất lượng cho tệp ảnh hoặc tập dữ liệu (`vai scan <target>`)."""
    import json

    from vietnamese_ai.security.data_defense import (
        CongChatLuongDauVao,
        KhienBaoVeHinhAnh,
        PhongThuDuLieu,
    )

    p = Path(args.target)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn cần quét: {args.target}")

    duoi = p.suffix.lower()
    if duoi in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"):
        khien = KhienBaoVeHinhAnh()
        cdr_out = getattr(args, "cdr_out", None)
        if cdr_out:
            _, bc = khien.khu_doc_anh_cdr(p, duong_dan_xuat=cdr_out)
            print(f"[CDR] Đã tái tạo ảnh sạch tại: {cdr_out}")
        else:
            bc = khien.quet_anh(p, chan_khi_nguy_hiem=False)
        trang_thai = "AN TOÀN" if bc["an_toan"] else "PHÁT HIỆN MỐI ĐE DỌA"
        print(f"=== KẾT QUẢ QUÉT AN NINH HÌNH ẢNH [{trang_thai}] ===")
        print(f"  Định dạng / Kích thước: {bc['dinh_dang']} | {bc['kich_thuoc'][0]}x{bc['kich_thuoc'][1]}")
        if bc["moi_de_doa"]:
            for m in bc["moi_de_doa"]:
                print(f"  [!] {m}")
        return bc

    if duoi == ".csv":
        X, y = _doc_du_lieu_csv(str(p), None)
        bc_in = CongChatLuongDauVao.kiem_tra_mang_so(X, y=y)
        _, bc_def = PhongThuDuLieu.quet_va_lam_sach_dataset(X, y=y, loai_bo_mau_doc=False)
        kq_csv = {**bc_in, "phong_thu": bc_def}
        print(
            f"=== QUÉT DATASET CSV === Điểm chất lượng: {bc_in['diem_chat_luong']}/100 | "
            f"Nghi ngờ SVD/Đảo nhãn: {len(bc_def['chi_so_bi_cach_ly'])} mẫu"
        )
        return kq_csv

    raw_txt = p.read_text(encoding="utf-8").strip()
    records = (
        json.loads(raw_txt)
        if duoi == ".json" and raw_txt.startswith("[")
        else [json.loads(line) for line in raw_txt.splitlines() if line.strip()]
    )
    _, bc_ds = PhongThuDuLieu.quet_va_lam_sach_dataset(records, loai_bo_mau_doc=False)
    print(
        f"=== QUÉT DATASET VĂN BẢN === Tổng: {bc_ds['tong_mau_kiem_tra']} mẫu | "
        f"Bom tích tụ: {bc_ds['phat_hien_bom_hen_gio']} | "
        f"Mẫu độc hại: {bc_ds['so_mau_dau_doc_bi_loai']}"
    )
    return bc_ds


def cmd_export(args):
    """Xuất mô hình sang GGUF hoặc ONNX từ CLI."""
    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    tuner = BoTinhChinhMoHinh(mo_hinh=args.model, che_do_offline=True)
    if args.format == "gguf":
        res = tuner.xuat_gguf(args.output, quantization=args.quantization)
    else:
        res = tuner.xuat_onnx(args.output)
    print(f"Đã xuất mô hình ({args.format}) tại: {res.get('duong_dan', args.output)}")


def cmd_studio(args):
    """Khởi động Vietnamese AI Web Studio Dashboard."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    app = StudioDashboard(
        host=args.host,
        port=args.port,
        duong_dan_vaidb=getattr(args, "vaidb", None),
        thu_muc_workspace=getattr(args, "workspace_dir", None),
        bat_xac_thuc=bool(getattr(args, "auth", False)),
    )
    app.chay()


def cmd_solution(args):
    """Chạy giải pháp thực chiến 1-Click từ CLI."""
    name = args.name.replace("-", "_").lower()
    if name == "phap_ly":
        from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

        tro_ly = TroLyPhapLy(
            nap_kho_mo=not bool(getattr(args, "no_open_corpus", False)),
            duong_dan_vaidb=getattr(args, "vaidb", None),
            dung_onnx_semantic=True,
        )
        try:
            q = args.query or args.input or "Thời giờ làm việc bình thường là bao nhiêu?"
            print(f"=== TRỢ LÝ PHÁP LÝ & DOANH NGHIỆP ===\nCâu hỏi: {q}\n")
            kq = tro_ly.tra_cuu(q)
            print(f"Trả lời:\n{kq['tra_loi']}\n")
            if kq["can_cu_phap_ly"]:
                print(f"Căn cứ pháp lý: {', '.join(kq['can_cu_phap_ly'])}")
            td_pl = kq.get("trich_dan_phap_ly") or {}
            for cb in td_pl.get("canh_bao_hieu_luc") or []:
                print(f"Cảnh báo hiệu lực: {cb}")
            return kq
        finally:
            tro_ly.rag.dong()

    if name == "trich_xuat":
        from vietnamese_ai.solutions.trich_xuat import TrichXuatThongMinh

        extractor = TrichXuatThongMinh()
        inp = args.input or args.query or ""
        loai = getattr(args, "type", "cccd")
        print(f"=== TRÍCH XUẤT THÔNG MINH [{loai.upper()}] ===")
        if loai == "hoa_don":
            kq = extractor.trich_xuat_hoa_don(inp)
        else:
            kq = extractor.trich_xuat_cccd(inp)
        for k, v in kq.items():
            if k != "van_ban_goc":
                print(f"  {k}: {v}")
        return kq

    if name == "cskh":
        from vietnamese_ai.solutions.cskh import TongDaiCSKH

        bot = TongDaiCSKH()
        if getattr(args, "voice", False):
            host = "127.0.0.1"
            port = 8765
            print("======================================================================")
            print("🎙️ TỔNG ĐÀI CSKH THOẠI THỜI GIAN THỰC (VOICEBOT WEBRTC) ĐANG CHẠY...")
            print("======================================================================")
            print(f"📍 WebSocket Voice Endpoint: ws://{host}:{port}")
            print("🎧 Đã nạp cơ sở dữ liệu đơn hàng & chính sách chăm sóc khách hàng")
            print("⚡ Hỗ trợ đàm thoại 2 chiều full-duplex, VAD ngắt câu & TTS tiếng Việt 6 thanh điệu")
            srv = bot.khoi_dong_may_chu_thoai(host=host, port=port, nen_chay_ngam=True)
            print(f"✅ Voicebot CSKH đang lắng nghe tại ws://{host}:{port}")
            return {"trang_thai": "dang_chay", "endpoint": f"ws://{host}:{port}", "server": srv}

        q = args.query or args.input or "Kiểm tra giúp tôi đơn hàng DH1001"
        print(f"=== TỔNG ĐÀI CSKH TỰ ĐỘNG ===\nKhách hàng: {q}\n")
        kq = bot.xu_ly_yeu_cau(q)
        print(f"Tổng đài: {kq['tra_loi']}")
        print(f"Ý định: {kq['y_dinh']}")
        return kq

    raise ValueError(f"Không hỗ trợ giải pháp: {args.name}")


def cmd_agent(args):
    """Xử lý lệnh `vai agent` (research | code)."""
    subagent = getattr(args, "subagent", None)
    if subagent == "research":
        from vietnamese_ai.agents.deep_research import TacTuNghienCuuSau

        print("=== TÁC TỬ NGHIÊN CỨU SÂU TIẾNG VIỆT ===")
        print(f"Đề tài: {args.query}\n")
        agent = TacTuNghienCuuSau(buoc_thu_thap_toi_da=args.steps)
        kq = agent.nghien_cuu(args.query, tep_xuat=args.output)
        print(f"Hoàn thành nghiên cứu! Điểm kiểm chứng: {kq.get('diem_kiem_chung', 1.0):.2f}")
        print(f"Số bằng chứng thu thập: {len(kq.get('bang_chung', []))}")
        if args.output:
            print(f"Báo cáo Markdown đã được lưu tại: {args.output}")
        else:
            print("\n--- BÁO CÁO TÓM TẮT ---")
            lines = kq.get("bao_cao_cuoi", "").split("\n")[:15]
            print("\n".join(lines) + ("\n..." if len(lines) >= 15 else ""))
        return kq

    if subagent == "code":
        from vietnamese_ai.agents.code_healing import TacTuLapTrinhTuSua

        print("=== TÁC TỬ LẬP TRÌNH TỰ SỬA MÃ (TDD SELF-HEALING) ===")
        print(f"Yêu cầu: {args.prompt}\n")
        agent = TacTuLapTrinhTuSua(so_lan_thu_lai_toi_da=args.max_retries)
        kq = agent.lap_trinh_va_tu_sua(args.prompt)
        print(f"Trạng thái: {'THÀNH CÔNG' if kq.get('thanh_cong') else 'CẦN KIỂM TRA THÊM'}")
        print(f"Số chu kỳ tự sửa: {kq.get('so_chu_ky', 1)}")
        if args.output and kq.get("ma_nguon"):
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(kq["ma_nguon"])
            print(f"Mã nguồn đã được lưu tại: {args.output}")
        else:
            print("\n--- MÃ NGUỒN HOÀN THIỆN ---")
            print(kq.get("ma_nguon", ""))
        return kq

    print("Vui lòng chọn loại tác tử: vai agent research ... hoặc vai agent code ...")
    return None


def cmd_pull(args):
    """Xử lý lệnh `vai pull` (Quản lý tải mô hình GGUF tiếng Việt tự chọn lượng tử theo RAM máy)."""
    from vietnamese_ai.edge.model_hub import QuanLyMoHinhEdge, do_ram_he_thong_gb

    hub = QuanLyMoHinhEdge(thu_muc_luu=getattr(args, "dir", None))
    ram_info = do_ram_he_thong_gb()

    if getattr(args, "remove", None):
        da_xoa = hub.xoa_mo_hinh(args.remove)
        if da_xoa:
            print(f"Đã xóa mô hình '{args.remove}' khỏi bộ nhớ đệm.")
        else:
            print(f"Không tìm thấy mô hình '{args.remove}' trong bộ nhớ đệm.")
        return {"da_xoa": da_xoa, "mo_hinh": args.remove}

    if getattr(args, "installed", False):
        ds_cai = hub.danh_sach_da_tai()
        print(f"=== MÔ HÌNH GGUF ĐÃ TẢI ({hub.thu_muc_luu}) ===")
        if not ds_cai:
            print("  (Chưa có mô hình nào. Dùng: vai pull qwen2.5-3b-q4)")
        for item in ds_cai:
            mb = round(item.get("kich_thuoc_bytes", 0) / (1024 * 1024), 2)
            print(f"  * {item['khoa_mo_hinh']:<22s} {mb:>8.2f} MB | HMAC: {'HỢP LỆ' if item.get('hmac_hop_le') else 'CẢNH BÁO'} | {item['duong_dan']}")
        return ds_cai

    if getattr(args, "list", False) or not getattr(args, "model", None):
        goi_y = hub.goi_y_mo_hinh_theo_may()
        print(
            f"=== DANH MỤC MÔ HÌNH LLM TIẾNG VIỆT GGUF "
            f"(RAM khả dụng: {ram_info['ram_kha_dung_gb']} / {ram_info['tong_ram_gb']} GB | {ram_info['so_nhan_cpu']} CPU) ==="
        )
        for m in goi_y:
            trang_thai = "[ĐÃ TẢI]" if m["da_tai_cuc_bo"] else ("[PHÙ HỢP]" if m["phu_hop_may"] else "[NẶNG RAM]")
            print(
                f"  {trang_thai:<10s} {m['ma_mo_hinh']:<15s} | Đề xuất: {m['muc_luong_tu_de_xuat']:<6s} "
                f"(~{m['dung_luong_gb']:>4.2f} GB) - {m['ten_hien_thi']}"
            )
        return goi_y

    kq = hub.tai_mo_hinh(
        ten_hoac_nguon=args.model,
        muc_luong_tu=getattr(args, "quant", "auto"),
        chay_that=not getattr(args, "dry_run", False),
        tai_lai=getattr(args, "force", False),
        sha256_mong_doi=getattr(args, "sha256", None),
        hf_token=getattr(args, "hf_token", None),
    )
    print(f"=== VAI PULL [{kq['khoa_mo_hinh']}] ({kq['trang_thai']}) ===")
    print(f"  Lý do chọn:   {kq.get('ly_do', '')}")
    print(f"  Nguồn URL:    {kq.get('url', '')}")
    print(f"  Lưu cục bộ:   {kq.get('duong_dan', '')}")
    if kq.get("sha256"):
        print(f"  SHA-256:      {kq['sha256']}")
    return kq


def cmd_run(args):
    """Xử lý lệnh `vai run` (Động cơ suy luận Edge Native Zero-GPU kèm Tường lửa & RAG pháp lý)."""
    from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge

    engine = DongCoSuyLuanEdge(
        duong_dan_mo_hinh=args.model,
        so_luong_thread=args.threads,
        nhiet_do=args.temp,
        do_dai_toi_da=args.max_tokens,
        uu_tien_native_c=bool(getattr(args, "native", False)),
    )

    rag_src = getattr(args, "rag", None)
    che_do_hc = bool(getattr(args, "admin", False))
    if rag_src:
        kq_nap = engine.nap_tai_lieu_rag(rag_src, che_do_hanh_chinh=che_do_hc)
        print(f"[RAG] Đã nạp {kq_nap['so_chunks']} phân đoạn từ '{rag_src}' (hành chính/pháp lý={che_do_hc}).")

    if args.chat:
        print("=== VIETNAMESE AI ZERO-GPU CHAT (CPU STREAMING) ===")
        print(f"Mô hình: {args.model} | Chế độ: {engine._che_do_suy_luan} | CPU Threads: {args.threads} | Gõ 'exit' để thoát.\n")
        while True:
            try:
                user_msg = input("Bạn: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ("exit", "quit", "thoat"):
                    print("Tạm biệt!")
                    break
                if rag_src:
                    kq_r = engine.hoi_kem_rag(user_msg, do_dai_toi_da=args.max_tokens, nhiet_do=args.temp)
                    print(f"\nAI: {kq_r['tra_loi']}\n")
                else:
                    print("\nAI: ", end="", flush=True)
                    for tok in engine.sinh_stream(user_msg):
                        print(tok, end="", flush=True)
                    print("\n")
            except (KeyboardInterrupt, EOFError):
                print("\nKết thúc phiên hội thoại.")
                break
        return None

    prompt = args.prompt or "Xin chào! Bạn có thể giới thiệu về khả năng của mình không?"
    print("=== ĐỘNG CƠ SUY LUẬN EDGE NATIVE (ZERO-GPU) ===")
    print(f"Mô hình: {args.model} | Chế độ: {engine._che_do_suy_luan} | Threads: {args.threads}\n")
    print(f"Prompt: {prompt}\n")

    if rag_src:
        kq_rag = engine.hoi_kem_rag(prompt, do_dai_toi_da=args.max_tokens, nhiet_do=args.temp)
        print("AI:\n" + kq_rag["tra_loi"])
        td = kq_rag.get("trich_dan_phap_ly")
        if td:
            if td.get("so_hieu_van_ban"):
                print(f"\nCăn cứ pháp lý: {', '.join(td['so_hieu_van_ban'])}")
            for cb in td.get("canh_bao_hieu_luc") or []:
                print(f"Cảnh báo hiệu lực: {cb}")
            dt = td.get("do_thi_tham_chieu") or {}
            hd_list = [x["so_hieu"] for x in dt.get("van_ban_huong_dan_con_hieu_luc") or []]
            if hd_list:
                print(f"Văn bản hướng dẫn thi hành còn hiệu lực: {', '.join(hd_list)}")
            tt_list = [x["so_hieu"] for x in dt.get("van_ban_thay_the_hien_hanh") or []]
            if tt_list:
                print(f"Văn bản thay thế hiện hành: {', '.join(tt_list)}")
        return kq_rag

    if args.bench:
        kq = engine.sinh_kem_thong_ke(prompt, do_dai_toi_da=args.max_tokens, nhiet_do=args.temp)
        print("AI:\n" + kq["van_ban"])
        print("\n" + "=" * 60)
        print("┌────────────────────────────────────────────────────────┐")
        print("│ BẢNG THỐNG KÊ HIỆU NĂNG EDGE NATIVE (CPU INTEL/AMD/ARM)│")
        print("├──────────────────────┬─────────────────────────────────┤")
        print(f"│ Tốc độ sinh văn bản  │ {kq['toc_do_token_giay']:>6.2f} tokens/giây             │")
        print(f"│ Độ trễ token đầu     │ {kq['ttft_ms']:>6.2f} ms (TTFT)                 │")
        print(f"│ Bộ nhớ RAM tiêu thụ  │ {kq['ram_su_dung_mb']:>6.2f} MB                        │")
        print(f"│ Tổng số tokens sinh  │ {kq['so_token']:>6d} tokens                     │")
        print(f"│ Thời gian thực thi   │ {kq['thoi_gian_giay']:>6.3f} giây                       │")
        print(f"│ Bảo vệ Tường lửa     │ {'AN TOÀN' if kq['an_toan'] else 'BỊ CHẶN':>12s}                    │")
        print("└──────────────────────┴─────────────────────────────────┘")
        return kq
    else:
        print("AI: ", end="", flush=True)
        full_tokens = []
        for tok in engine.sinh_stream(prompt, do_dai_toi_da=args.max_tokens, nhiet_do=args.temp):
            print(tok, end="", flush=True)
            full_tokens.append(tok)
        print("\n")
        return "".join(full_tokens)


def cmd_pack(args):
    """Xử lý lệnh `vai pack` (Đóng gói Desktop 1-Click)."""
    from vietnamese_ai.portable.desktop_packager import GoiDocLapDesktop

    packager = GoiDocLapDesktop()
    out_dir = getattr(args, "output", "vietnamese_ai_desktop")
    mode = getattr(args, "mode", "studio")
    port = getattr(args, "port", 8080)
    threads = getattr(args, "threads", 4)

    print("=== ĐÓNG GÓI 1-CLICK DESKTOP OFFLINE STANDALONE ===")
    print(f"Thư mục xuất : {out_dir}")
    print(f"Chế độ chạy  : {mode.upper()}")
    print(f"Cổng máy chủ : {port}")
    print(f"CPU Threads  : {threads}\n")

    res = packager.dong_goi_desktop(out_dir, che_do=mode, port=port, so_thread=threads)
    print("Đóng gói thành công!")
    print(f"  * Script Linux/macOS : {res['file_sh']}")
    print(f"  * Script Windows     : {res['file_bat']}")
    print(f"  * Tệp cấu hình       : {res['file_cfg']}")
    print(f"  * Hướng dẫn sử dụng  : {res['file_readme']}")
    print("\nNgười dùng chỉ cần nhấp đúp file run_desktop.sh (hoặc .bat) để khởi chạy!")
    return res


def cmd_voice(args):
    """Khởi động máy chủ Voice AI thời gian thực (WebRTC / WebSocket RFC 6455)."""
    import time
    import webbrowser

    from vietnamese_ai.audio.webrtc_bridge import MayChuWebRTCVoice

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8765)
    sample_rate = getattr(args, "sample_rate", 16000)
    open_browser = getattr(args, "open_browser", False)

    ham_tra_loi = None
    cskh_obj = None
    che_do_str = "Trợ lý Thoại Mặc định (Rule-based Assistant)"

    if getattr(args, "cskh", False):
        from vietnamese_ai.solutions.cskh import TongDaiCSKH

        print("🎧 Khởi tạo Tổng đài CSKH Đa phương thức...")
        cskh_obj = TongDaiCSKH()
        che_do_str = "Tổng đài CSKH Tự động (TongDaiCSKH Voicebot)"
    elif getattr(args, "rag", None):
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        print(f"📚 Nạp cơ sở tri thức RAG từ: {args.rag}")
        rag_p = Path(args.rag)
        if rag_p.suffix.lower() == ".vaidb":
            rag = RAGPipeline(
                backend="vaidb",
                duong_dan_vaidb=str(rag_p),
                che_do_nhung="onnx_viet",
                che_do_rerank="onnx_viet",
                che_do_hanh_chinh=True,
            )
        else:
            rag = RAGPipeline(che_do_nhung="onnx_viet", che_do_rerank="onnx_viet")
            if rag_p.is_dir():
                rag.them_thu_muc(rag_p)
            else:
                rag.them_tep(rag_p)

        def _voice_rag_reply(q: str) -> str:
            res = rag.hoi(q)
            return res.get("tra_loi", res.get("cau_tra_loi", "Tôi đã tra cứu nhưng không tìm thấy thông tin phù hợp."))

        ham_tra_loi = _voice_rag_reply
        che_do_str = f"RAG Tri Thức ({Path(args.rag).name})"
    elif getattr(args, "llm", False):
        from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

        model_name = getattr(args, "model", "vgpt2")
        print(f"🧠 Khởi tạo mô hình ngôn ngữ: {model_name}")
        llm = VietnameseLLM(model_name=model_name)

        def _voice_llm_reply(q: str) -> str:
            return llm.sinh_van_ban(q, do_dai_toi_da=64)

        ham_tra_loi = _voice_llm_reply
        che_do_str = f"Generative AI ({model_name})"

    print("======================================================================")
    print("🎙️ MÁY CHỦ VOICE AI THỜI GIAN THỰC (v31.0) ĐANG KHỞI ĐỘNG...")
    print("======================================================================")
    print(f"📍 WebSocket Endpoint: ws://{host}:{port}")
    print(f"🎧 Tần số âm thanh: {sample_rate} Hz (PCM 16-bit Mono)")
    print(f"🤖 Chế độ đối thoại: {che_do_str}")
    print("⚡ Bắt tay chuẩn WebSocket RFC 6455 & VAD 6 thanh điệu (Zero-Dependency)")

    server = MayChuWebRTCVoice(
        host=host,
        port=port,
        tan_so_lay_mau=sample_rate,
        ham_tra_loi=ham_tra_loi,
        cskh=cskh_obj,
    )
    server.khoi_dong_chay_ngam()
    print(f"✅ Máy chủ Voice AI đang lắng nghe tại ws://{host}:{port}")

    if open_browser:
        ui_path = Path(__file__).resolve().parent.parent.parent / "evonet_os" / "index.html"
        if ui_path.is_file():
            print(f"🌐 Đang mở giao diện EvoNet OS: {ui_path.as_uri()}")
            webbrowser.open(ui_path.as_uri())

    print("\nNhấn Ctrl+C để dừng máy chủ Voice AI.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nĐang dừng máy chủ Voice AI...")
        server.dung_lai()
        print("Đã dừng máy chủ Voice AI thành công.")


def cmd_swarm(args):
    """Xử lý lệnh `vai swarm` (Suy luận bầy đàn phân tán trên CPU văn phòng)."""
    import time

    import numpy as np

    # 1. Chế độ Máy Chủ Lắng Nghe (Host / Master Node)
    if getattr(args, "host", False):
        from vietnamese_ai.edge.swarm_auth import phat_hien_ip_tailscale, tao_ma_pin_ngau_nhien
        from vietnamese_ai.edge.swarm_inference import NutSwarmCPU, tao_ssl_context_server

        ip_ts = phat_hien_ip_tailscale() if getattr(args, "tailscale", False) else None
        host = "0.0.0.0"
        port = getattr(args, "port", 9999)
        pin = getattr(args, "pin", None) or tao_ma_pin_ngau_nhien(6)
        tls_cert = getattr(args, "tls_cert", None)
        tls_key = getattr(args, "tls_key", None)
        tls_ca = getattr(args, "tls_ca", None)
        use_mtls = bool(getattr(args, "mtls", False))
        ssl_ctx = (
            tao_ssl_context_server(
                certfile=tls_cert,
                keyfile=tls_key,
                cafile=tls_ca,
                yeu_cau_mtls=use_mtls,
            )
            if (tls_cert and tls_key)
            else None
        )

        print("======================================================================")
        print("🌐 MÁY CHỦ CỤM SWARM CPU (MASTER NODE) ĐANG KHỞI ĐỘNG...")
        print("======================================================================")
        if ip_ts:
            print(f"📍 IP Tailscale WireGuard: {ip_ts}:{port}")
        else:
            print(f"📍 Lắng nghe trên mọi mạng: 0.0.0.0:{port}")
        print(f"🔑 Mã PIN bảo mật Zero-Trust: {pin}")
        if ssl_ctx is not None:
            print(f"🔒 Mã hóa kênh truyền: {'mTLS 2 chiều (TLS 1.2+)' if use_mtls else 'TLS 1.2+'}")
        print("\n⚡ Đang chờ các máy worker gia nhập cụm...")
        join_ip = ip_ts if ip_ts else "127.0.0.1"
        print("👉 Trên máy worker, chỉ cần gõ:")
        print(f"   vietnamese-ai swarm --join {join_ip}:{port} --pin {pin}\n")

        server = NutSwarmCPU(node_id="master_server", dai_layer=(0, 4), ma_pin=pin, ssl_context=ssl_ctx)
        server.khoi_dong_server_lan(host=host, port=port, ma_pin=pin, ssl_context=ssl_ctx)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nĐang dừng máy chủ Swarm...")
            server.dung_server_lan()
        return

    # 2. Chế độ Máy Con Gia Nhập (Join / Worker Node)
    endpoint = getattr(args, "join", None)
    if endpoint:
        from vietnamese_ai.edge.swarm_inference import NutSwarmCPU, tao_ssl_context_client

        pin = getattr(args, "pin", None)
        if not pin:
            print("⚠️ Cảnh báo: Bạn chưa chỉ định --pin. Bắt tay xác thực có thể bị máy chủ từ chối.")
        tls_cert = getattr(args, "tls_cert", None)
        tls_key = getattr(args, "tls_key", None)
        tls_ca = getattr(args, "tls_ca", None)
        use_mtls = bool(getattr(args, "mtls", False))
        ssl_ctx_client = (
            tao_ssl_context_client(
                cafile=tls_ca,
                certfile=tls_cert,
                keyfile=tls_key,
                kiem_tra_hostname=False,
            )
            if (tls_ca or (tls_cert and tls_key) or use_mtls)
            else None
        )
        print(f"🔍 Đang kiểm tra kết nối tới cụm Swarm tại {endpoint}...")
        worker = NutSwarmCPU(
            node_id="worker_cli",
            dai_layer=(0, 4),
            endpoint_url=endpoint,
            che_do="mang_vat_ly_lan",
            ma_pin=pin,
            nen_fp16=getattr(args, "fp16", False),
            ssl_context=ssl_ctx_client,
        )
        x_dummy = np.ones((1, 128), dtype=np.float32)
        try:
            x_out, proof = worker.thuc_thi_layer(x_dummy, tao_proof=True)
            print("✅ Bắt tay xác thực và kết nối thành công tới cụm Swarm!")
            print("  * Trạng thái xác thực: ĐÃ ĐƯỢC CẤP PHÉP")
            if getattr(args, "fp16", False):
                print("  * Nén tensor FP16: BẬT (Tiết kiệm 50% băng thông)")
            if proof and "proof_zk" in proof:
                print(f"  * Bằng chứng Verifiable AI: {proof['proof_zk'][:32]}...")
            print(f"  * Độ trễ kết nối mạng: {worker.thoi_gian_mang_gan_nhat_ms:.2f}ms")
        except PermissionError as e:
            print(f"❌ Xác thực thất bại: {e}")
        except Exception as e:
            print(f"❌ Không thể kết nối tới {endpoint}: {e}")
        return

    # 3. Chế độ Giả Lập Cụm Tính Toán Mặc Định
    from vietnamese_ai.edge.swarm_inference import DieuPhoiSwarmCPU

    so_node = getattr(args, "nodes", 3)
    tong_so_layer = getattr(args, "layers", 6)
    prompt = getattr(args, "prompt", "Trí tuệ nhân tạo Việt Nam")
    so_token = getattr(args, "tokens", 16)
    xac_thuc = getattr(args, "verify", True)
    bench = getattr(args, "bench", False)
    nen_fp16 = getattr(args, "fp16", False)

    print("=== SUY LUẬN BẦY ĐÀN PHÂN TÁN TRÊN CPU (SWARM PIPELINE PARALLELISM) ===")
    print(f"  * Số node CPU tham gia: {so_node}")
    print(f"  * Tổng số layer:       {tong_so_layer}")
    print(f"  * Nén tensor FP16:     {'BẬT (Float16)' if nen_fp16 else 'TẮT (Float32)'}")
    print(f"  * Xác thực Verifiable: {'BẬT (Merkle Tree & zkML)' if xac_thuc else 'TẮT'}")
    print(f"  * Prompt:              \"{prompt}\"\n")

    orchestrator = DieuPhoiSwarmCPU(
        tong_so_layer=tong_so_layer,
        bat_xac_thuc_mat_ma=xac_thuc,
        nen_fp16=nen_fp16,
    )
    orchestrator.khoi_tao_cum_gia_lap(so_node=so_node, so_core_moi_node=4)

    gguf_path = getattr(args, "gguf", None)
    if gguf_path:
        print(f"📦 Đang nạp và phân mảnh mô hình GGUF từ '{gguf_path}'...")
        kq_pm = orchestrator.phan_manh_mo_hinh_gguf(gguf_path)
        print(f"✅ Đã phân mảnh thành công {kq_pm.get('tong_so_tensors', 0)} tensors vào {len(orchestrator.danh_sach_node)} nodes.")

    thong_tin_cum = orchestrator.lay_trang_thai_cum()
    print("Phân bổ Layer trên các Node CPU:")
    for n in thong_tin_cum["danh_sach_node"]:
        print(f"  Node [{n['node_id']}]: Layer {n['dai_layer'][0]} -> {n['dai_layer'][1]} ({n['so_core_cpu']} cores, {n['ram_mb']}MB RAM)")
    print()

    print("AI (Swarm Streaming): ", end="", flush=True)
    tokens = []
    t0 = time.perf_counter()
    for tok in orchestrator.sinh_stream(prompt, do_dai_toi_da=so_token):
        print(tok, end="", flush=True)
        tokens.append(tok)
    thoi_gian = time.perf_counter() - t0
    van_ban = "".join(tokens)
    print("\n")

    if bench:
        print("┌────────────────────────────────────────────────────────┐")
        print("│ BẢNG THỐNG KÊ SWARM CPU PIPELINE PARALLELISM           │")
        print("├──────────────────────┬─────────────────────────────────┤")
        print(f"│ Số node hoạt động    │ {so_node:>6d} nodes                     │")
        print(f"│ Tổng tokens sinh     │ {len(tokens):>6d} tokens                    │")
        print(f"│ Thời gian thực thi   │ {thoi_gian:>6.3f} giây                      │")
        toc_do = len(tokens) / thoi_gian if thoi_gian > 0 else 0
        print(f"│ Tốc độ trung bình    │ {toc_do:>6.2f} tokens/giây             │")
        print(f"│ Bảo vệ Byzantine     │ {'ĐÃ XÁC THỰC MẬT MÃ' if xac_thuc else 'CHƯA BẬT':>20s}      │")
        print("└──────────────────────┴─────────────────────────────────┘")

    orch_port = getattr(args, "orchestrator_port", None)
    if orch_port:
        from vietnamese_ai.orchestrator.swarm_bridge import NutCauNoiSwarm

        bridge = NutCauNoiSwarm(swarm=orchestrator, port=orch_port)
        print("======================================================================")
        print(f"🚀 CẦU NỐI HTTP ORCHESTRATOR ĐANG CHẠY TẠI http://127.0.0.1:{orch_port}")
        print(f"👉 Điểm cuối Health:  http://127.0.0.1:{orch_port}/health")
        print(f"👉 Điểm cuối Predict: http://127.0.0.1:{orch_port}/predict")
        print(f"👉 Kết nối vào NutChinh: nut_chinh.them_nut_swarm(port={orch_port})")
        print("======================================================================")
        try:
            bridge.chay()
        except KeyboardInterrupt:
            print("\nĐang tắt Cầu nối HTTP Orchestrator...")
            bridge.dung()
        return {"van_ban": van_ban, "so_token": len(tokens), "so_node": so_node, "bridge_port": orch_port}

    return {"van_ban": van_ban, "so_token": len(tokens), "so_node": so_node}


def cmd_verify(args):
    """Xử lý lệnh `vai verify` (Xác thực suy luận mật mã học Merkle Tree & zkML)."""
    from vietnamese_ai.security.verifiable_inference import CayMerkleSuyLuan, XacThucSuyLuanMaoHiem

    proof = getattr(args, "proof", None)
    text = getattr(args, "text", None)
    tamper_check = getattr(args, "tamper_check", False)

    print("=== XÁC THỰC MẬT MÃ HỌC SUY LUẬN AI (VERIFIABLE INFERENCE AUDIT) ===")
    verifier = XacThucSuyLuanMaoHiem()

    if proof:
        print(f"Kiểm tra cam kết zkML: {proof[:40]}...")
        hop_le = verifier.xac_nhan_cam_ket_chuoi(proof)
        status = "HỢP LỆ (BẰNG CHỨNG XÁC THỰC THÀNH CÔNG)" if hop_le else "KHÔNG HỢP LỆ (CÓ DẤU HIỆU GIAN LẬN/BYZANTINE)"
        print(f"  Kết quả kiểm tra: {status}")
        return {"hop_le": hop_le}
    elif text:
        tokens = text.split()
        merkle = CayMerkleSuyLuan(tokens)
        root = merkle.lay_root_hash()
        print(f"  * Số lá (Leaf tokens): {len(tokens)}")
        print(f"  * Merkle Root Hash:    {root}")
        ok = True
        if len(tokens) > 0:
            p = merkle.sinh_bang_chung(0)
            ok = merkle.kiem_tra_bang_chung(tokens[0], 0, p, root)
            print(f"  * Kiểm tra Proof lá #0: {'CHÍNH XÁC (O(log N))' if ok else 'THẤT BẠI'}")
        return {"root": root, "so_la": len(tokens), "hop_le": ok}
    else:
        print("Chạy kiểm định mật mã mẫu tự động:")
        arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        cam_ket = verifier.tao_cam_ket_zkml(arr)
        print(f"  * Tạo cam kết zkML: {cam_ket['chuoi_cam_ket'][:35]}...")
        ok = verifier.xac_thuc_cam_ket(cam_ket["y"], cam_ket["a"], cam_ket["e"], cam_ket["z"])
        print(f"  * Xác thực Schnorr/Fiat-Shamir: {'HỢP LỆ (TRUE)' if ok else 'THẤT BẠI'}")
        if tamper_check:
            arr_tampered = arr.copy()
            arr_tampered[0] += 0.5
            hash_tampered = verifier.bam_tensor(arr_tampered)
            gian_lan = hash_tampered != cam_ket["tensor_hash"]
            print(f"  * Phát hiện can thiệp tensor (Tamper Detection): {'ĐÃ CHẶN GIAN LẬN' if gian_lan else 'LỖI'}")
        return {"hop_le": ok}


def cmd_plugin(args):
    """Xử lý lệnh `vai plugin` (Quản trị và thực thi V-Plugin 2.0)."""
    import json
    import os
    from pathlib import Path

    from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin

    ql = QuanLyPlugin()
    action = getattr(args, "action", "list") or "list"
    target = getattr(args, "target", None)

    # 1. Action: list
    if action == "list":
        plugins_dir = Path("plugins")
        if plugins_dir.exists():
            for f in sorted(plugins_dir.glob("*.py")):
                if not f.name.startswith("__"):
                    try:
                        ql.nap_plugin(str(f))
                    except Exception:
                        pass

        print("=== DANH SÁCH V-PLUGIN 2.0 ĐANG HOẠT ĐỘNG ===")
        ds = ql.danh_sach_plugin()
        if not ds:
            print("  (Chưa có plugin nào được cài đặt hoặc đang hoạt động)")
            return {"so_luong": 0, "plugins": []}

        print(f"Tổng số: {len(ds)} plugin\n")
        for p in ds:
            print(f"  * [{p['id']}] {p['ten']} (v{p['phien_ban']}) - Loại: {p['loai'].upper()}")
            print(f"    Mô tả: {p['mo_ta']}")
            print(f"    Trạng thái: {p['trang_thai']}")
            print()
        return {"so_luong": len(ds), "plugins": ds}

    # 2. Action: verify
    elif action == "verify":
        if not target:
            print("❌ Vui lòng chỉ định đường dẫn tệp plugin cần kiểm tra (VD: vai plugin verify ./plugins/plugin_taichinh.py)")
            return {"hop_le": False, "ly_do": "Thiếu đường dẫn tệp"}

        print(f"🔍 Đang kiểm tra an ninh AST Sandbox cho: {target}...")
        hop_le, ly_do = ql.kiem_tra_an_ninh(target)
        if hop_le:
            print(f"✅ AN TOÀN: Tệp plugin '{target}' vượt qua kiểm duyệt AST Sandbox!")
        else:
            print(f"❌ NGUY HIỂM / KHÔNG HỢP LỆ: {ly_do}")
        return {"hop_le": hop_le, "ly_do": ly_do}

    # 3. Action: info
    elif action == "info":
        if not target:
            print("❌ Vui lòng chỉ định tên hoặc đường dẫn tệp plugin (VD: vai plugin info ./plugins/plugin_taichinh.py)")
            return {"tim_thay": False}

        p_obj = None
        if os.path.exists(target):
            p_obj = ql.nap_plugin(target)
        else:
            p_obj = ql.lay_plugin(target)

        if not p_obj:
            print(f"❌ Không tìm thấy hoặc không thể nạp plugin: {target}")
            return {"tim_thay": False}

        print("=== THÔNG TIN CHI TIẾT PLUGIN ===")
        print(f"  * Tên:       {p_obj.ten}")
        print(f"  * Phiên bản: {p_obj.phien_ban}")
        print(f"  * Loại:      {getattr(p_obj, 'loai', 'co_so').upper()}")
        print(f"  * Mô tả:     {p_obj.mo_ta}")

        if hasattr(p_obj, "dang_ky_cong_cu"):
            tools = p_obj.dang_ky_cong_cu()
            print(f"\n  🛠️ Danh sách Công cụ ({len(tools)} tools):")
            for t in tools:
                print(f"    - {t.get('ten')}: {t.get('mo_ta')}")

        if hasattr(p_obj, "nap_nguon_tri_thuc"):
            docs = p_obj.nap_nguon_tri_thuc()
            print(f"\n  📚 Nguồn Tri thức ({len(docs)} tài liệu):")
            for d in docs:
                print(f"    - [{d.get('tieu_de')}] (Nguồn: {d.get('nguon', 'N/A')})")

        return {"tim_thay": True, "ten": p_obj.ten, "loai": getattr(p_obj, "loai", "co_so")}

    # 4. Action: run
    elif action == "run":
        if not target:
            print("❌ Vui lòng chỉ định đường dẫn tệp hoặc tên plugin cần chạy.")
            return {"thanh_cong": False}

        p_obj = None
        if os.path.exists(target):
            p_obj = ql.nap_plugin(target)
        else:
            p_obj = ql.lay_plugin(target)

        if not p_obj:
            print(f"❌ Không tìm thấy plugin: {target}")
            return {"thanh_cong": False}

        tool_name = getattr(args, "tool", None)
        raw_params = getattr(args, "params", "{}") or "{}"
        try:
            params = json.loads(raw_params)
        except Exception:
            params = {}

        if tool_name and hasattr(p_obj, "thuc_thi_cong_cu"):
            print(f"⚡ Đang thực thi công cụ '{tool_name}' trên plugin '{p_obj.ten}'...")
            try:
                kq = p_obj.thuc_thi_cong_cu(tool_name, **params)
                print(f"✅ Kết quả: {json.dumps(kq, ensure_ascii=False, indent=2) if isinstance(kq, (dict, list)) else kq}")
                return {"thanh_cong": True, "ket_qua": kq}
            except Exception as e:
                print(f"❌ Lỗi thực thi công cụ: {e}")
                return {"thanh_cong": False, "loi": str(e)}
        else:
            print(f"⚡ Đang thực thi plugin '{p_obj.ten}'...")
            try:
                kq = p_obj.thuc_thi(**params)
                print(f"✅ Kết quả: {json.dumps(kq, ensure_ascii=False, indent=2) if isinstance(kq, (dict, list)) else kq}")
                return {"thanh_cong": True, "ket_qua": kq}
            except Exception as e:
                print(f"❌ Lỗi thực thi plugin: {e}")
                return {"thanh_cong": False, "loi": str(e)}


def cmd_demo(args: Any) -> Any:
    """Chạy trình diễn 1-Click thực chiến (`vai demo [all|legal-rag|edge-chat|security|flywheel]`)."""
    from vietnamese_ai.cli.demo import chay_demo_nhanh

    che_do = str(getattr(args, "mode", "all") or "all")
    return chay_demo_nhanh(che_do=che_do, in_ra=True)


def cmd_release_check(args: Any) -> Any:
    """Kiểm định toàn diện trước khi phát hành PyPI/TestPyPI (`vai release-check`)."""
    import json

    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    bc = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
    if getattr(args, "json", False):
        print(json.dumps(bc, ensure_ascii=False, indent=2))
        return bc

    trang_thai = "SẴN SÀNG PHÁT HÀNH" if bc["san_sang_phat_hanh"] else "CẦN BỔ SUNG TRƯỚC KHI PHÁT HÀNH"
    print(f"=== KIỂM ĐỊNH PHÁT HÀNH PYPI VIETNAMESE AI v{bc['phien_ban']} [{trang_thai}] ===")
    print(
        f"  SemVer & Đồng bộ:     {'ĐẠT' if bc['semver']['hop_le_semver'] and bc['semver']['dong_bo_phien_ban'] else 'CHƯA ĐẠT'} "
        f"({len(bc['semver']['cac_nguon'])} tệp phiên bản)"
    )
    print(
        f"  Xung đột phụ thuộc:   {'ĐẠT (0 xung đột)' if bc['phu_thuoc']['hop_le'] else 'CÓ XUNG ĐỘT'}"
    )
    print(
        f"  Type Stubs (.pyi):    {'ĐẠT' if not bc['thieu_type_stub'] else 'THIẾU ' + str(len(bc['thieu_type_stub']))} "
        f"({bc['so_bieu_tuong_cong_khai']} biểu tượng công khai)"
    )
    print(f"  Rust Core (abi3):     {'ĐẠT (abi3-py39)' if bc['rust_abi3_san_sang'] else 'CHƯA ĐẠT'}")
    print(
        f"  Tệp phát hành chuẩn:  {'ĐẠT' if not bc['thieu_tep_bat_buoc'] else 'THIẾU: ' + ', '.join(bc['thieu_tep_bat_buoc'])}"
    )
    print(f"  Lệnh đóng gói gợi ý:  {bc['lenh_dong_goi_goi_y']}")
    return bc


def cmd_legal_graph(args: Any) -> Any:
    """Truy vấn Đồ thị Tham chiếu Pháp lý Việt Nam (`vai legal-graph <so_hieu>`)."""
    from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy

    do_thi = DoThiThamChieuPhapLy()
    kq = do_thi.truy_van_van_ban_lien_quan(
        args.so_hieu,
        ngay_tra_cuu=getattr(args, "date", None),
        chi_lay_con_hieu_luc=bool(getattr(args, "active_only", False)),
    )
    hl = kq["hieu_luc_goc"]
    print(f"=== ĐỒ THỊ THAM CHIẾU PHÁP LÝ: {kq['so_hieu']} ===")
    print(f"  Tên văn bản:          {hl.get('ten_van_ban', kq['so_hieu'])}")
    print(f"  Trạng thái hiệu lực:  {hl['trang_thai']} (ngày tra cứu: {hl['ngay_tra_cuu']})")
    if kq["van_ban_thay_the_hien_hanh"]:
        ds_tt = ", ".join(x["so_hieu"] for x in kq["van_ban_thay_the_hien_hanh"])
        print(f"  Thay thế hiện hành:   {ds_tt}")
    if kq["van_ban_huong_dan"]:
        print("  Nghị định / Thông tư hướng dẫn:")
        for item in kq["van_ban_huong_dan"]:
            print(f"    - {item['so_hieu']} [{item['trang_thai']}]: {item['ten_van_ban']}")
    if kq["van_ban_sua_doi_bo_sung"]:
        print("  Văn bản sửa đổi, bổ sung:")
        for item in kq["van_ban_sua_doi_bo_sung"]:
            print(f"    - {item['so_hieu']} [{item['trang_thai']}]: {item['ten_van_ban']}")
    if getattr(args, "dieu", None):
        pb = do_thi.bo_theo_doi.tra_cuu_dieu_khoan_theo_ngay(
            so_hieu=args.so_hieu,
            dieu=str(args.dieu),
            ngay_tra_cuu=getattr(args, "date", None),
            khoan=getattr(args, "khoan", None),
        )
        kq["phien_ban_dieu_khoan"] = pb
        print(f"\n--- PHIÊN BẢN ĐIỀU KHOẢN POINT-IN-TIME ({pb['dieu']} @ {pb['ngay_tra_cuu']}) ---")
        print(f"  Trạng thái: {pb['trang_thai']}")
        if pb.get("phien_ban_hieu_luc"):
            print(f"  Nội dung:   {pb['phien_ban_hieu_luc']['noi_dung']}")
    if getattr(args, "mermaid", False):
        mmd = do_thi.xuat_do_thi_mermaid(so_hieu_trung_tam=args.so_hieu)
        kq["mermaid"] = mmd
        print("\n--- SƠ ĐỒ MERMAID ---")
        print(mmd)
    return kq


def cmd_legal_ingest(args: Any) -> Any:
    """Nạp tự động văn bản pháp quy / văn bản hợp nhất thành 1 tệp `.vaidb` HNSW + Point-in-Time (`vai legal-ingest`)."""
    import json

    from vietnamese_ai.rag.legal.auto_ingest import BoNapVanBanPhapQuyTuDong

    pipeline = BoNapVanBanPhapQuyTuDong(che_do_nhung="onnx_viet")
    kq = pipeline.dong_goi_kho_vaidb(
        nguon_dau_vao=args.source,
        duong_dan_vaidb=args.output,
        xay_dung_hnsw=not bool(getattr(args, "no_hnsw", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(kq, ensure_ascii=False, indent=2))
        return kq
    print(f"=== ĐÃ ĐÓNG GÓI KHO PHÁP QUY .VAIDB: {kq['duong_dan_vaidb']} ===")
    print(
        f"  Tệp đã nạp: {kq['so_tep_da_nap']} | Điều khoản: {kq['tong_dieu_khoan']} | "
        f"Phiên bản Point-in-Time: {kq['so_phien_ban_point_in_time']} | Quan hệ đồ thị: {kq['so_quan_he_do_thi']}"
    )
    if kq.get("hnsw"):
        print(
            f"  HNSW bền vững: {kq['hnsw'].get('so_nut', 0)} nút, {kq['hnsw'].get('so_tang', 0)} tầng"
        )
    return kq


def cmd_contract_review(args: Any) -> Any:
    """Rà soát rủi ro & so sánh hợp đồng (`vai contract-review <partner> [--template ...] [--output-docx ...] [--track-changes-docx ...]`)."""
    from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong

    engine = BoSoSanhRaSoatHopDong()
    kq = engine.so_sanh_va_ra_soat(
        hop_dong_doi_tac=Path(args.partner),
        hop_dong_chuan=Path(args.template) if getattr(args, "template", None) else None,
        ngay_tra_cuu=getattr(args, "date", None),
    )
    md = engine.xuat_bao_cao_markdown(kq)
    print(md)
    if kq.get("cau_truc_tach_gop"):
        tk = kq.get("thong_ke_so_sanh", {})
        print(
            f"Phát hiện cấu trúc tách/gộp điều khoản: "
            f"tách={tk.get('tach_dieu_khoan', 0)}, gộp={tk.get('gop_dieu_khoan', 0)}"
        )
    if getattr(args, "output_md", None):
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        print(f"Đã xuất báo cáo Markdown tại: {out_md}")
    if getattr(args, "output_docx", None):
        out_docx = engine.xuat_bao_cao_docx(kq, args.output_docx)
        print(f"Đã xuất báo cáo Word (.docx) tại: {out_docx}")
    if getattr(args, "track_changes_docx", None):
        out_tc = engine.xuat_bao_cao_track_changes_docx(kq, args.track_changes_docx)
        kq["track_changes_docx_path"] = str(out_tc)
        print(f"Đã xuất Word Track Changes + Comments (.docx) tại: {out_tc}")
    return kq


def cmd_mcp_server(args: Any) -> Any:
    """Khởi chạy máy chủ MCP chuẩn stdio/SSE hoặc in cấu hình IDE (`vai mcp-server`)."""
    import json

    from vietnamese_ai.agents.mcp_transport import MayChuMCPThucTe

    print_cfg = getattr(args, "print_config", None)
    vaidb = getattr(args, "vaidb", None)
    if print_cfg:
        cfg = MayChuMCPThucTe.xuat_cau_hinh_ide(ten_ide=print_cfg, duong_dan_vaidb=vaidb)
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return cfg

    quan_ly_plugin = None
    plugin_dir = getattr(args, "plugin_dir", None)
    if plugin_dir:
        from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin

        quan_ly_plugin = QuanLyPlugin()
        p_dir = Path(plugin_dir)
        if p_dir.is_dir():
            for py_f in sorted(p_dir.glob("*.py")):
                if not py_f.name.startswith("__"):
                    try:
                        quan_ly_plugin.nap_plugin(str(py_f))
                    except Exception:
                        pass

    server = MayChuMCPThucTe(
        duong_dan_vaidb=vaidb,
        quan_ly_plugin=quan_ly_plugin,
        bat_cong_cu_mo_rong=bool(getattr(args, "extended_tools", False)),
    )
    try:
        if getattr(args, "transport", "stdio") == "sse":
            server.chay_sse(host=getattr(args, "host", "127.0.0.1"), port=int(getattr(args, "port", 8090)))
        else:
            server.chay_stdio()
    finally:
        server.dong()
    return {"trang_thai": "da_dung"}


def cmd_connector(args: Any) -> Any:
    """Khởi động Webhook Server cho Zalo OA / Telegram Bot (`vai connector <zalo|telegram|both>`)."""
    import os

    from vietnamese_ai.connectors.telegram_bot import KetNoiTelegram
    from vietnamese_ai.connectors.webhook_server import MayChuWebhookKenh
    from vietnamese_ai.connectors.zalo_oa import KetNoiZaloOA

    sol = getattr(args, "solution", "phap-ly")
    if sol == "cskh":
        from vietnamese_ai.solutions.cskh import TongDaiCSKH

        handler: Any = TongDaiCSKH()
    else:
        from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

        handler = TroLyPhapLy(
            nap_kho_mo=True,
            duong_dan_vaidb=getattr(args, "vaidb", None),
            dung_onnx_semantic=True,
        )

    ws_dir = getattr(args, "workspace_dir", None)
    ql_ws = None
    if ws_dir:
        from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

        ql_ws = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=False)

    channel = getattr(args, "channel", "both")
    zalo = (
        KetNoiZaloOA(
            bo_xu_ly=handler,
            app_id=os.environ.get("ZALO_APP_ID", ""),
            oa_secret_key=os.environ.get("ZALO_OA_SECRET", ""),
            access_token=os.environ.get("ZALO_OA_ACCESS_TOKEN", ""),
            che_do_gui_that=bool(os.environ.get("ZALO_OA_ACCESS_TOKEN")),
            quan_ly_workspace=ql_ws,
        )
        if channel in ("zalo", "both")
        else None
    )
    telegram = (
        KetNoiTelegram(
            bo_xu_ly=handler,
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            webhook_secret=os.environ.get("TELEGRAM_WEBHOOK_SECRET", ""),
            che_do_gui_that=bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
            quan_ly_workspace=ql_ws,
        )
        if channel in ("telegram", "both")
        else None
    )

    host = getattr(args, "host", "0.0.0.0")
    port = int(getattr(args, "port", 8095))
    endpoints: List[str] = [
        *(["/webhook/zalo"] if zalo is not None else []),
        *(["/webhook/telegram"] if telegram is not None else []),
        "/health",
    ]
    info: Dict[str, Any] = {
        "kenh": channel,
        "giai_phap": sol,
        "host": host,
        "port": port,
        "endpoints": endpoints,
        "workspace_dir": ws_dir,
    }
    if getattr(args, "dry_run", False):
        print(f"=== [DRY-RUN] VIETNAMESE AI CONNECTOR ({channel.upper()}) ===")
        print(f"  Giải pháp gắn kết: {sol} | Endpoints: {', '.join(endpoints)}")
        if ql_ws is not None:
            ql_ws.dong()
        return info

    srv = MayChuWebhookKenh(ket_noi_zalo=zalo, ket_noi_telegram=telegram, host=host, port=port)
    try:
        srv.chay()
    finally:
        if ql_ws is not None:
            ql_ws.dong()
    return info


def cmd_export_dpo(args: Any) -> Any:
    """Xuất phản hồi người dùng từ Workspace sang tập huấn luyện DPO (`vai export-dpo`)."""
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

    ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=args.workspace_dir, bat_xac_thuc=False)
    try:
        kq = ql.xuat_tap_du_lieu_dpo(
            duong_dan_jsonl=args.output,
            ma_workspace=getattr(args, "workspace", None),
            an_pii=True,
        )
        print(f"Đã xuất {kq['so_mau_dpo']} mẫu DPO sạch (đã ẩn PII) tại: {kq['duong_dan_jsonl']}")
        return kq
    finally:
        ql.dong()


def cmd_eval_gate(args: Any) -> Any:
    """Chạy Cổng Kiểm Định tự động (`vai eval-gate`)."""
    import json

    from vietnamese_ai.fine_tuning.eval_gate import BO_TAN_CONG_MO_RONG, CongKiemDinhMoHinh
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    tro_ly = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True)
    ban_goc = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=False) if getattr(args, "compare_baseline", False) else None
    bo_atk = BO_TAN_CONG_MO_RONG if getattr(args, "extended_attacks", False) else None
    gate = CongKiemDinhMoHinh(bo_tan_cong=bo_atk)
    kq = gate.so_sanh_va_quyet_dinh(
        ung_vien=tro_ly,
        ban_goc=ban_goc,
        gioi_han_mau=int(getattr(args, "limit", 8)),
        nguong_dat=float(getattr(args, "min_score", 0.55)),
    )
    out = kq.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    print(f"=== CỔNG KIỂM ĐỊNH EVAL GATE [{out['quyet_dinh']}] ===")
    for k, v in out["diem_ung_vien"].items():
        print(f"  {k:<32s}: {v:.4f}")
    ci = (out.get("chi_tiet_thong_ke") or {}).get("bootstrap_ci_95")
    if ci:
        print(
            f"  Paired Bootstrap 95% CI         : "
            f"[{ci['ci_duoi']:+.4f}, {ci['ci_tren']:+.4f}] "
            f"(Δ={ci['chenh_lech_trung_binh']:+.4f}, ý nghĩa={ci['co_y_nghia_thong_ke']})"
        )
    print(f"  Kết luận: {out['ly_do']}")
    return out


def cmd_status_map(args: Any) -> Any:
    """Hiển thị bản đồ phân tách 3 tầng Core Production / Optional Extensions / Experimental-Simulated (`vai status-map`)."""
    import json

    from vietnamese_ai.utils.feature_status import bao_cao_ranh_gioi_he_thong

    bc = bao_cao_ranh_gioi_he_thong()
    if getattr(args, "json", False):
        print(json.dumps(bc, ensure_ascii=False, indent=2))
        return bc
    tq = bc["tong_quan"]
    print("=== BẢN ĐỒ MINH BẠCH TRẠNG THÁI HỆ THỐNG VIETNAMESE AI ===")
    print(f"  1. Core Production (0 phụ thuộc ngoài): {tq['so_phan_he_core_production']} phân hệ")
    for item in bc["core_production"]:
        print(f"     [CORE] {item['ten']} ({item['module']})")
    print(f"  2. Optional Extensions (cài thêm extra): {tq['so_phan_he_optional_extensions']} phân hệ")
    for item in bc["optional_extensions"]:
        print(f"     [EXT ] {item['nhom']:<12s} extra=[{item['extra']}] ({item['module']})")
    print(f"  3. Experimental / Simulated / External:  {tq['so_phan_he_experimental_simulated']} phân hệ")
    for item in bc["experimental_or_simulated"]:
        print(f"     [EXP ] {item['nhom']:<12s} trạng thái={item['trang_thai']} ({item['module']})")
    return bc


def main():
    """Entry point cho CLI."""
    parser = _tao_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    bang_lenh = {
        "info": lambda: cmd_info(),
        "doctor": lambda: cmd_doctor(),
        "install": lambda: cmd_install(args),
        "setup": lambda: cmd_setup(args),
        "train": lambda: cmd_train(args),
        "predict": lambda: cmd_predict(args),
        "serve": lambda: cmd_serve(args),
        "evaluate": lambda: cmd_evaluate(args),
        "web": lambda: cmd_web(args),
        "finetune": lambda: cmd_finetune(args),
        "train-cloud": lambda: cmd_train_cloud(args),
        "rag": lambda: cmd_rag(args),
        "chat": lambda: cmd_chat(args),
        "benchmark": lambda: cmd_benchmark(args),
        "tokenize": lambda: cmd_tokenize(args),
        "scan": lambda: cmd_scan(args),
        "export": lambda: cmd_export(args),
        "studio": lambda: cmd_studio(args),
        "solution": lambda: cmd_solution(args),
        "agent": lambda: cmd_agent(args),
        "pull": lambda: cmd_pull(args),
        "run": lambda: cmd_run(args),
        "pack": lambda: cmd_pack(args),
        "swarm": lambda: cmd_swarm(args),
        "voice": lambda: cmd_voice(args),
        "verify": lambda: cmd_verify(args),
        "plugin": lambda: cmd_plugin(args),
        "demo": lambda: cmd_demo(args),
        "release-check": lambda: cmd_release_check(args),
        "legal-graph": lambda: cmd_legal_graph(args),
        "legal-ingest": lambda: cmd_legal_ingest(args),
        "contract-review": lambda: cmd_contract_review(args),
        "mcp-server": lambda: cmd_mcp_server(args),
        "connector": lambda: cmd_connector(args),
        "export-dpo": lambda: cmd_export_dpo(args),
        "eval-gate": lambda: cmd_eval_gate(args),
        "status-map": lambda: cmd_status_map(args),
    }
    ham = bang_lenh.get(args.command)
    if ham is not None:
        ham()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
