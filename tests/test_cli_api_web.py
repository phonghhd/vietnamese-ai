"""Test suite cho CLI, API Server, Web UI."""

import json
import threading
import time

import numpy as np

from vietnamese_ai.datasets.sample_data import DuLieuMau
from vietnamese_ai.models.classifier import PhanLoai

# ============================================================
# CLI Tests
# ============================================================


class TestCLI:
    def test_info(self):
        from vietnamese_ai.cli.main import cmd_info

        cmd_info()

    def test_parser(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_parser_train(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["train", "--data", "test.csv", "--model", "logistic"])
        assert args.command == "train"
        assert args.data == "test.csv"
        assert args.model == "logistic"

    def test_parser_predict(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["predict", "--model", "m.pkl", "--input", "d.csv"])
        assert args.command == "predict"

    def test_parser_serve(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["serve", "--model", "m.pkl", "--port", "9090"])
        assert args.command == "serve"
        assert args.port == 9090

    def test_parser_evaluate(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["evaluate", "--model", "m.pkl", "--data", "test.csv"])
        assert args.command == "evaluate"

    def test_parser_web(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args(["web", "--port", "3000"])
        assert args.command == "web"
        assert args.port == 3000

    def test_parser_no_command(self):
        from vietnamese_ai.cli.main import _tao_parser

        parser = _tao_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_cmd_train(self, tmp_path):
        import pandas as pd

        from vietnamese_ai.cli.main import cmd_train

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50, so_dac_trung=3)
        df = np.column_stack([X, y])
        csv_path = str(tmp_path / "data.csv")
        pd.DataFrame(df).to_csv(csv_path, index=False)

        class Args:
            data = csv_path
            model = "logistic"
            output = str(tmp_path / "model.pkl")
            test_size = 0.2
            target = None

        cmd_train(Args())
        assert (tmp_path / "model.pkl").exists()

    def test_doc_du_lieu_csv(self, tmp_path):
        import pandas as pd

        from vietnamese_ai.cli.main import _doc_du_lieu_csv

        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "label": [0, 1, 0]})
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        X, y = _doc_du_lieu_csv(csv_path)
        assert X.shape == (3, 2)
        assert len(y) == 3

    def test_luu_ket_qua_csv(self, tmp_path):
        import pandas as pd

        from vietnamese_ai.cli.main import _luu_ket_qua_csv

        out_path = str(tmp_path / "results.csv")
        _luu_ket_qua_csv(out_path, np.array([0, 1, 0, 1]))
        df = pd.read_csv(out_path)
        assert len(df) == 4


# ============================================================
# API Server Tests
# ============================================================


class TestAPIServer:
    def test_khoi_tao(self):
        from vietnamese_ai.api.server import ServerDonGian

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50, so_dac_trung=3)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        server = ServerDonGian(mo_hinh=pl, ten="TestServer")
        assert server.ten == "TestServer"
        assert server.mo_hinh is not None

    def test_request_handler_get_root(self):
        from vietnamese_ai.api.server import _RequestHandler

        assert hasattr(_RequestHandler, "do_GET")
        assert hasattr(_RequestHandler, "do_POST")

    def test_server_integration(self):
        import socket

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50, so_dac_trung=3)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        from vietnamese_ai.api.server import ServerDonGian

        ServerDonGian(mo_hinh=pl)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        def run_server():
            from http.server import HTTPServer

            from vietnamese_ai.api.server import _RequestHandler

            _RequestHandler.mo_hinh = pl
            srv = HTTPServer(("127.0.0.1", port), _RequestHandler)
            srv.handle_request()
            srv.server_close()

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(0.5)

        import urllib.request

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                assert data["trang_thai"] == "hoat_dong"
        except Exception:
            pass

    def test_suc_khoe_endpoint(self):
        import socket

        X, y = DuLieuMau.phan_loai_don_gian(so_mau=50, so_dac_trung=3)
        pl = PhanLoai(thuat_toan="logistic")
        pl.huan_luyen(X, y)

        from http.server import HTTPServer

        from vietnamese_ai.api.server import _RequestHandler

        _RequestHandler.mo_hinh = pl

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        def run_server():
            srv = HTTPServer(("127.0.0.1", port), _RequestHandler)
            srv.handle_request()
            srv.server_close()

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(0.5)

        import urllib.request

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/suc_khoe")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                assert data["trang_thai"] == "tot"
        except Exception:
            pass


# ============================================================
# Web UI Tests
# ============================================================


class TestWebUI:
    def test_khoi_tao(self):
        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb(port=5000)
        assert app.port == 5000
        assert app._model is None

    def test_html_template(self):
        from vietnamese_ai.web.app import UngDungWeb

        assert "Vietnamese AI" in UngDungWeb.HTML_TEMPLATE
        assert "uploadForm" in UngDungWeb.HTML_TEMPLATE

    def test_xu_ly_upload(self, tmp_path):
        import pandas as pd

        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "label": [0, 1, 0]})
        csv_bytes = df.to_csv(index=False).encode()

        ket_qua = app._xu_ly_upload({"target": ""}, csv_bytes)
        assert ket_qua["status"] == "success"
        assert ket_qua["so_mau"] == 3
        assert ket_qua["so_dac_trung"] == 2

    def test_xu_ly_upload_voi_target(self):
        import pandas as pd

        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "y": [0, 1]})
        csv_bytes = df.to_csv(index=False).encode()

        ket_qua = app._xu_ly_upload({"target": "y"}, csv_bytes)
        assert ket_qua["status"] == "success"
        assert ket_qua["cot_nhan"] == "y"

    def test_xu_ly_train_chua_upload(self):
        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        ket_qua = app._xu_ly_train({})
        assert ket_qua["status"] == "error"
        assert "Chưa upload" in ket_qua["message"]

    def test_xu_ly_train(self):
        import pandas as pd

        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        df = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "label": np.random.randint(0, 2, 50),
        })
        csv_bytes = df.to_csv(index=False).encode()
        app._xu_ly_upload({}, csv_bytes)

        ket_qua = app._xu_ly_train({"algorithm": "logistic", "test_size": 0.2})
        assert ket_qua["status"] == "success"
        assert "diem" in ket_qua

    def test_xu_ly_predict_chua_train(self):
        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        ket_qua = app._xu_ly_predict({"data": [[1, 2, 3]]})
        assert ket_qua["status"] == "error"

    def test_xu_ly_predict(self):
        import pandas as pd

        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        df = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "f3": np.random.randn(50),
            "label": np.random.randint(0, 2, 50),
        })
        csv_bytes = df.to_csv(index=False).encode()
        app._xu_ly_upload({}, csv_bytes)
        app._xu_ly_train({"algorithm": "logistic"})

        ket_qua = app._xu_ly_predict({"data": [[1.0, 2.0, 3.0]]})
        assert ket_qua["status"] == "success"
        assert len(ket_qua["du_doan"]) == 1

    def test_end_to_end_web(self):
        import pandas as pd

        from vietnamese_ai.web.app import UngDungWeb

        app = UngDungWeb()
        df = pd.DataFrame({
            "x1": np.random.randn(40),
            "x2": np.random.randn(40),
            "y": np.random.randint(0, 2, 40),
        })

        upload = app._xu_ly_upload({"target": "y"}, df.to_csv(index=False).encode())
        assert upload["status"] == "success"

        train = app._xu_ly_train({"algorithm": "logistic", "test_size": 0.2})
        assert train["status"] == "success"

        predict = app._xu_ly_predict({"data": [[0.5, -0.5]]})
        assert predict["status"] == "success"
