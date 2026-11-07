"""CLI chính cho Vietnamese AI Framework — Siêu nhẹ, Bảo mật Ký số HMAC & Quản lý Gói Mở rộng."""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

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
    inst_parser.add_argument("group", help="Tên nhóm mở rộng (ml, tabular, viz, nlp, dl, fast, rag, llm, api, full)")
    inst_parser.add_argument("--dry-run", action="store_true", help="Chỉ hiển thị câu lệnh pip mà không chạy thật")

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

    # === RAG ===
    rag_parser = subparsers.add_parser("rag", help="Truy vấn cơ sở tri thức RAG (có khử độc tài liệu & Tường lửa)")
    rag_parser.add_argument("--docs", required=True, help="Đường dẫn file văn bản hoặc JSONL")
    rag_parser.add_argument("--query", required=True, help="Câu hỏi cần truy vấn")

    # === CHAT ===
    chat_parser = subparsers.add_parser("chat", help="Trò chuyện trực tiếp với mô hình kèm Tường lửa AI")
    chat_parser.add_argument("--model", default="vgpt2", help="Tên hoặc đường dẫn mô hình")
    chat_parser.add_argument("--prompt", default=None, help="Câu hỏi đơn lẻ (bỏ trống để vào chế độ hội thoại tương tác)")

    # === BENCHMARK ===
    subparsers.add_parser("benchmark", help="Đo kiểm tốc độ huấn luyện Sequence Packing & suy luận KV-Cache thực tế")

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

    # === RUN (ZERO-GPU NATIVE INFERENCE ENGINE) ===
    run_parser = subparsers.add_parser(
        "run", help="Chạy suy luận CPU siêu tốc Zero-GPU (mmap GGUF, Streaming Token 60 FPS)"
    )
    run_parser.add_argument("model", help="Đường dẫn file mô hình .gguf hoặc tên mô hình (vgpt2)")
    run_parser.add_argument("--prompt", "-p", default=None, help="Câu hỏi đơn lẻ cần suy luận")
    run_parser.add_argument("--chat", action="store_true", help="Vào chế độ đối thoại hội thoại tương tác trực tiếp")
    run_parser.add_argument("--threads", "-t", type=int, default=4, help="Số luồng CPU (mặc định: 4)")
    run_parser.add_argument("--bench", action="store_true", help="Hiển thị bảng thống kê hiệu năng chi tiết (T/s, TTFT, RAM)")
    run_parser.add_argument("--temp", type=float, default=0.7, help="Nhiệt độ ngẫu nhiên (mặc định: 0.7)")
    run_parser.add_argument("--max-tokens", type=int, default=128, help="Số token tối đa sinh ra")

    # === PACK (1-CLICK DESKTOP OFFLINE STANDALONE PACKAGER) ===
    pack_parser = subparsers.add_parser("pack", help="Tự động tạo gói Desktop 1-Click Offline (run_desktop.sh / .bat)")
    pack_parser.add_argument("--output", "-o", default="vietnamese_ai_desktop", help="Thư mục xuất gói ứng dụng")
    pack_parser.add_argument("--mode", default="studio", choices=["studio", "chat"], help="Chế độ khởi chạy mặc định")
    pack_parser.add_argument("--port", type=int, default=8080, help="Cổng chạy Web Studio")
    pack_parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cấu hình")

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

        return BoTinhChinhMoHinh.tai(str(p))

    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

    return BoTinhChinhMoHinh(mo_hinh=duong_dan_hoac_ten, che_do_offline=True)


def cmd_info():
    """Hiển thị thông tin framework."""
    from vietnamese_ai import __version__

    print(f"""
╔══════════════════════════════════════════╗
║       Vietnamese AI Framework            ║
║       Version: {__version__:<24s}║
╚══════════════════════════════════════════╝

Kiến trúc: Micro-Core Siêu Nhẹ (< 5ms Lazy Import) + Tường lửa AI Đa tầng
Modules:
  - models & fine_tuning: PhanLoai, HoiQuy, GPTModel, BoTinhChinhMoHinh, BoTangTocHuanLuyen
  - rag & agents:         RAGPipeline, TacTu, MCPServer, HethongNhoMemGPT
  - security & serving:   TuongLuaBaoVeFramework, ContinuousBatcher, ServerDonGian

Lệnh:
  vai doctor                        Chẩn đoán hệ thống, phần cứng & gói phụ thuộc
  vai install <nhom>                Cài gói mở rộng (ml, viz, dl, fast, rag, llm, api, full)
  vai train --data FILE --model M   Huấn luyện & ký số HMAC-SHA256
  vai predict --model M --input I   Dự đoán an toàn (RestrictedUnpickler)
  vai chat --model vgpt2            Trò chuyện trực tiếp kèm Tường lửa AI
  vai benchmark                     Đo tốc độ Sequence Packing & KV-Cache
  vai serve --model M --port P      Khởi động OpenAI-compatible API server
""")


def cmd_doctor():
    """Chẩn đoán toàn diện sức khỏe môi trường (`vai doctor`)."""
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    bc = QuanLyPhuThuoc.chan_doan_he_thong()
    print(f"=== CHẨN ĐOÁN HỆ THỐNG VIETNAMESE AI v{bc['phien_ban_framework']} ===")
    print(f"  Python:            {bc['python']} ({bc['he_dieu_hanh']})")
    print(f"  CPU / GPU:         {bc['so_nhan_cpu']} nhân CPU | CUDA GPU: {bc['ten_gpu'] or 'Không (chạy tối ưu CPU/NumPy)'}")
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
    """Khởi động API server từ CLI."""
    from vietnamese_ai.api.server import ServerDonGian
    from vietnamese_ai.utils.logger import Logger

    logger = Logger("CLI-Serve")
    logger.info(f"Tải mô hình an toàn từ: {args.model}")

    mo_hinh = _tai_mo_hinh_an_toan(args.model)
    logger.info(f"Loại mô hình: {type(mo_hinh).__name__}")

    server = ServerDonGian(mo_hinh=mo_hinh, ten="VAI-Server", bat_tuong_lua=True)
    server.chay(host=args.host, port=args.port)


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


def cmd_rag(args):
    """Truy vấn RAG từ CLI."""
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    doc_path = Path(args.docs)
    if not doc_path.exists():
        print(f"Lỗi: Không tìm thấy tài liệu '{args.docs}'")
        sys.exit(1)

    rag = RAGPipeline()
    rag.them_tai_lieu(doc_path.name, doc_path.read_text(encoding="utf-8"))
    kq = rag.hoi(args.query)
    print(f"Trả lời: {kq.get('tra_loi', '')}")


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


def cmd_benchmark():
    """Đo kiểm tốc độ huấn luyện và KV-Cache thực tế ngay từ dòng lệnh."""
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

    app = StudioDashboard(host=args.host, port=args.port)
    app.chay()


def cmd_solution(args):
    """Chạy giải pháp thực chiến 1-Click từ CLI."""
    name = args.name.replace("-", "_").lower()
    if name == "phap_ly":
        from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

        tro_ly = TroLyPhapLy()
        q = args.query or args.input or "Thời giờ làm việc bình thường là bao nhiêu?"
        print(f"=== TRỢ LÝ PHÁP LÝ & DOANH NGHIỆP ===\nCâu hỏi: {q}\n")
        kq = tro_ly.tra_cuu(q)
        print(f"Trả lời:\n{kq['tra_loi']}\n")
        if kq["can_cu_phap_ly"]:
            print(f"Căn cứ pháp lý: {', '.join(kq['can_cu_phap_ly'])}")
        return kq

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


def cmd_run(args):
    """Xử lý lệnh `vai run` (Động cơ suy luận Edge Native Zero-GPU)."""
    from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge

    engine = DongCoSuyLuanEdge(
        duong_dan_mo_hinh=args.model,
        so_luong_thread=args.threads,
        nhiet_do=args.temp,
        do_dai_toi_da=args.max_tokens,
    )

    if args.chat:
        print("=== VIETNAMESE AI ZERO-GPU CHAT (CPU STREAMING) ===")
        print(f"Mô hình: {args.model} | CPU Threads: {args.threads} | Gõ 'exit' hoặc 'quit' để thoát.\n")
        while True:
            try:
                user_msg = input("Bạn: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ("exit", "quit", "thoat"):
                    print("Tạm biệt!")
                    break
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
    print(f"Mô hình: {args.model} | Threads: {args.threads}\n")
    print(f"Prompt: {prompt}\n")

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
        "train": lambda: cmd_train(args),
        "predict": lambda: cmd_predict(args),
        "serve": lambda: cmd_serve(args),
        "evaluate": lambda: cmd_evaluate(args),
        "web": lambda: cmd_web(args),
        "finetune": lambda: cmd_finetune(args),
        "rag": lambda: cmd_rag(args),
        "chat": lambda: cmd_chat(args),
        "benchmark": lambda: cmd_benchmark(),
        "scan": lambda: cmd_scan(args),
        "export": lambda: cmd_export(args),
        "studio": lambda: cmd_studio(args),
        "solution": lambda: cmd_solution(args),
        "agent": lambda: cmd_agent(args),
        "run": lambda: cmd_run(args),
        "pack": lambda: cmd_pack(args),
    }
    ham = bang_lenh.get(args.command)
    if ham is not None:
        ham()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
