"""CLI chính cho Vietnamese AI Framework."""

import argparse
import sys

import numpy as np


def _tao_parser():
    """Tạo argument parser cho CLI."""
    parser = argparse.ArgumentParser(
        prog="vai",
        description="Vietnamese AI Framework - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  vai train --data data.csv --model logistic --output model.pkl
  vai predict --model model.pkl --input new_data.csv --output results.csv
  vai serve --model model.pkl --port 8080
  vai info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")

    # === INFO ===
    subparsers.add_parser("info", help="Thông tin framework")

    # === TRAIN ===
    train_parser = subparsers.add_parser("train", help="Huấn luyện mô hình")
    train_parser.add_argument("--data", required=True, help="Đường dẫn file dữ liệu (CSV)")
    train_parser.add_argument("--model", default="logistic", help="Thuật toán (logistic, knn, svm, rung_ngau_nhien, gradient_boosting)")
    train_parser.add_argument("--output", default="model.pkl", help="Đường dẫn lưu mô hình")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ dữ liệu test")
    train_parser.add_argument("--target", default=None, help="Tên cột nhãn (mặc định: cột cuối)")

    # === PREDICT ===
    predict_parser = subparsers.add_parser("predict", help="Dự đoán với mô hình đã huấn luyện")
    predict_parser.add_argument("--model", required=True, help="Đường dẫn file mô hình (.pkl)")
    predict_parser.add_argument("--input", required=True, help="Đường dẫn file dữ liệu đầu vào (CSV)")
    predict_parser.add_argument("--output", default="results.csv", help="Đường dẫn lưu kết quả")

    # === SERVE ===
    serve_parser = subparsers.add_parser("serve", help="Khởi động API server")
    serve_parser.add_argument("--model", required=True, help="Đường dẫn file mô hình (.pkl)")
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

    return parser


def _doc_du_lieu_csv(duong_dan: str, target: str = None):
    """Đọc dữ liệu từ file CSV."""
    try:
        import pandas as pd
    except ImportError:
        print("Lỗi: Cần cài đặt pandas. Chạy: pip install pandas")
        sys.exit(1)

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


def _luu_ket_qua_csv(duong_dan: str, ket_qua):
    """Lưu kết quả dự đoán ra CSV."""
    try:
        import pandas as pd
    except ImportError:
        print("Lỗi: Cần cài đặt pandas. Chạy: pip install pandas")
        sys.exit(1)

    df = pd.DataFrame({"du_doan": ket_qua})
    df.to_csv(duong_dan, index=False)
    print(f"Đã lưu kết quả tại: {duong_dan}")


def cmd_info():
    """Hiển thị thông tin framework."""
    from vietnamese_ai import __version__

    print(f"""
╔══════════════════════════════════════════╗
║       Vietnamese AI Framework            ║
║       Version: {__version__:<24s}║
╚══════════════════════════════════════════╝

Modules:
  - models:       PhanLoai, HoiQuy, PhanCum, MangNron, MoHinhTapHop
  - preprocessing: XuLyVanBan, XuLySo, TaoDacTrung
  - core:         Engine, Pipeline, KiemDinhCheo, TimKiemThamSo
  - utils:        Logger, Metrics, Validator, LuuTai
  - visualization: BieuDo
  - api:          ServerDonGian

Lệnh:
  vai info                          Thông tin framework
  vai train --data FILE --model M   Huấn luyện mô hình
  vai predict --model M --input I   Dự đoán
  vai evaluate --model M --data D   Đánh giá mô hình
  vai serve --model M --port P      Khởi động API server
""")


def cmd_train(args):
    """Huấn luyện mô hình từ CLI."""
    from vietnamese_ai.models.classifier import PhanLoai
    from vietnamese_ai.models.regression import HoiQuy
    from vietnamese_ai.preprocessing.numerical import XuLySo
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
    logger.info(f"Đã lưu mô hình tại: {args.output}")


def cmd_predict(args):
    """Dự đoán từ CLI."""
    from vietnamese_ai.models.base import BaseModel
    from vietnamese_ai.utils.logger import Logger

    logger = Logger("CLI-Predict")
    logger.info(f"Tải mô hình từ: {args.model}")

    mo_hinh = BaseModel.tai(args.model)
    logger.info(f"Loại mô hình: {type(mo_hinh).__name__}")

    try:
        import pandas as pd
        df = pd.read_csv(args.input)
        X = df.values
    except ImportError:
        X = np.loadtxt(args.input, delimiter=",")

    logger.info(f"Dự đoán {len(X)} mẫu")
    ket_qua = mo_hinh.du_doan(X)

    _luu_ket_qua_csv(args.output, ket_qua)
    logger.info("Dự đoán hoàn tất")


def cmd_serve(args):
    """Khởi động API server từ CLI."""
    from vietnamese_ai.api.server import ServerDonGian
    from vietnamese_ai.models.base import BaseModel
    from vietnamese_ai.utils.logger import Logger

    logger = Logger("CLI-Serve")
    logger.info(f"Tải mô hình từ: {args.model}")

    mo_hinh = BaseModel.tai(args.model)
    logger.info(f"Loại mô hình: {type(mo_hinh).__name__}")

    server = ServerDonGian(mo_hinh=mo_hinh, ten="VAI-Server")
    server.chay(host=args.host, port=args.port)


def cmd_evaluate(args):
    """Đánh giá mô hình từ CLI."""
    from vietnamese_ai.models.base import BaseModel
    from vietnamese_ai.utils.logger import Logger
    from vietnamese_ai.utils.metrics import Metrics
    from vietnamese_ai.utils.validators import Validator

    logger = Logger("CLI-Evaluate")
    logger.info(f"Tải mô hình từ: {args.model}")

    mo_hinh = BaseModel.tai(args.model)
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


def main():
    """Entry point cho CLI."""
    parser = _tao_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "info":
        cmd_info()
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
