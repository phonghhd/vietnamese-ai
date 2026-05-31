"""UngDungWeb - Giao diện web no-code/low-code cho Vietnamese AI."""

import json

from vietnamese_ai.utils.logger import Logger


class UngDungWeb:
    """
    Giao diện web no-code/low-code cho Vietnamese AI Framework.

    Chạy web app để:
    - Upload dữ liệu CSV
    - Chọn mô hình và tham số
    - Huấn luyện và đánh giá
    - Dự đoán trực tuyến
    - Tải về mô hình đã huấn luyện

    Sử dụng:
        >>> app = UngDungWeb(port=5000)
        >>> app.chay()

    CLI:
        vai web --port 5000
    """

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vietnamese AI - No-Code ML</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: white; padding: 2rem; text-align: center; }
        .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.9; }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .card h2 { color: #667eea; margin-bottom: 1rem; font-size: 1.2rem; }
        label { display: block; margin-bottom: 0.3rem; font-weight: 600; color: #555; }
        select, input[type="file"], input[type="number"], input[type="text"] {
            width: 100%%; padding: 0.7rem; border: 2px solid #e0e0e0; border-radius: 8px;
            margin-bottom: 1rem; font-size: 1rem; transition: border-color 0.3s;
        }
        select:focus, input:focus { outline: none; border-color: #667eea; }
        button {
            background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);
            color: white; border: none; padding: 0.8rem 2rem; border-radius: 8px;
            font-size: 1rem; cursor: pointer; transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .result { background: #f8f9fa; border-left: 4px solid #667eea; padding: 1rem; border-radius: 0 8px 8px 0; margin-top: 1rem; }
        .result pre { white-space: pre-wrap; font-family: 'Courier New', monospace; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .status { padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; display: inline-block; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .status.loading { background: #fff3cd; color: #856404; }
        #loading { display: none; text-align: center; padding: 2rem; }
        #loading .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { 0%% { transform: rotate(0deg); } 100%% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Vietnamese AI Framework</h1>
        <p>Giao diện No-Code ML - Không cần viết code</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>1. Upload dữ liệu</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <label>File CSV:</label>
                <input type="file" id="dataFile" accept=".csv" required>
                <label>Tên cột nhãn (để trống = cột cuối):</label>
                <input type="text" id="targetCol" placeholder="VD: label, target, class">
                <button type="submit">Upload</button>
            </form>
            <div id="uploadResult"></div>
        </div>

        <div class="card">
            <h2>2. Huấn luyện mô hình</h2>
            <div class="grid">
                <div>
                    <label>Thuật toán:</label>
                    <select id="algorithm">
                        <option value="logistic">Logistic Regression</option>
                        <option value="knn">K-Nearest Neighbors</option>
                        <option value="rung_ngau_nhien">Random Forest</option>
                        <option value="gradient_boosting">Gradient Boosting</option>
                        <option value="tuyen_tinh">Linear Regression</option>
                        <option value="ridge">Ridge Regression</option>
                    </select>
                </div>
                <div>
                    <label>Tỷ lệ test (%%):</label>
                    <input type="number" id="testSize" value="20" min="5" max="50">
                </div>
            </div>
            <button id="trainBtn" onclick="trainModel()" disabled>Huấn luyện</button>
            <div id="trainResult"></div>
        </div>

        <div class="card">
            <h2>3. Dự đoán</h2>
            <label>Nhập dữ liệu (phân tách bằng dấu phẩy):</label>
            <input type="text" id="predictInput" placeholder="VD: 1.0, 2.0, 3.0, 4.0, 5.0">
            <button id="predictBtn" onclick="predict()" disabled>Dự đoán</button>
            <div id="predictResult"></div>
        </div>

        <div id="loading"><div class="spinner"></div><p>Đang xử lý...</p></div>
    </div>

    <script>
        let dataLoaded = false;
        let modelTrained = false;

        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            showLoading(true);
            const formData = new FormData();
            formData.append('file', document.getElementById('dataFile').files[0]);
            formData.append('target', document.getElementById('targetCol').value);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('uploadResult').innerHTML =
                    '<div class="result"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
                if (data.status === 'success') {
                    dataLoaded = true;
                    document.getElementById('trainBtn').disabled = false;
                }
            } catch(err) {
                document.getElementById('uploadResult').innerHTML =
                    '<div class="status error">Lỗi: ' + err.message + '</div>';
            }
            showLoading(false);
        };

        async function trainModel() {
            showLoading(true);
            try {
                const res = await fetch('/api/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        algorithm: document.getElementById('algorithm').value,
                        test_size: parseInt(document.getElementById('testSize').value) / 100
                    })
                });
                const data = await res.json();
                document.getElementById('trainResult').innerHTML =
                    '<div class="result"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
                if (data.status === 'success') {
                    modelTrained = true;
                    document.getElementById('predictBtn').disabled = false;
                }
            } catch(err) {
                document.getElementById('trainResult').innerHTML =
                    '<div class="status error">Lỗi: ' + err.message + '</div>';
            }
            showLoading(false);
        }

        async function predict() {
            showLoading(true);
            try {
                const input = document.getElementById('predictInput').value;
                const values = input.split(',').map(Number);
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ data: [values] })
                });
                const data = await res.json();
                document.getElementById('predictResult').innerHTML =
                    '<div class="result"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
            } catch(err) {
                document.getElementById('predictResult').innerHTML =
                    '<div class="status error">Lỗi: ' + err.message + '</div>';
            }
            showLoading(false);
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
    </script>
</body>
</html>"""

    def __init__(self, port: int = 5000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.logger = Logger("WebApp")
        self._data = None
        self._labels = None
        self._model = None
        self._target_col = None

    def _xu_ly_upload(self, form_data: dict, file_data: bytes) -> dict:
        """Xử lý upload dữ liệu CSV."""
        try:
            import io

            import pandas as pd

            df = pd.read_csv(io.BytesIO(file_data))
            target = form_data.get("target", "").strip()

            if target and target in df.columns:
                self._labels = df[target].values
                self._data = df.drop(columns=[target]).values
                self._target_col = target
            else:
                self._labels = df.iloc[:, -1].values
                self._data = df.iloc[:, :-1].values
                self._target_col = df.columns[-1]

            return {
                "status": "success",
                "so_mau": len(self._data),
                "so_dac_trung": self._data.shape[1],
                "cot_nhan": self._target_col,
                "cac_cot": list(df.columns),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _xu_ly_train(self, params: dict) -> dict:
        """Huấn luyện mô hình."""
        if self._data is None:
            return {"status": "error", "message": "Chưa upload dữ liệu"}

        try:
            from vietnamese_ai.models.classifier import PhanLoai
            from vietnamese_ai.models.regression import HoiQuy
            from vietnamese_ai.preprocessing.numerical import XuLySo
            from vietnamese_ai.utils.validators import Validator

            nhiem_vu = Validator.kiem_tra_nhiem_vu(self._labels)
            test_size = params.get("test_size", 0.2)
            algorithm = params.get("algorithm", "logistic")

            X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(
                self._data, self._labels, ty_le_test=test_size
            )

            if nhiem_vu == "phan_loai":
                self._model = PhanLoai(thuat_toan=algorithm)
            else:
                self._model = HoiQuy(thuat_toan=algorithm)

            self._model.huan_luyen(X_train, y_train)
            diem = self._model.danh_gia(X_test, y_test)

            return {
                "status": "success",
                "nhiem_vu": nhiem_vu,
                "thuat_toan": algorithm,
                "diem": round(float(diem), 4),
                "so_mau_train": len(X_train),
                "so_mau_test": len(X_test),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _xu_ly_predict(self, params: dict) -> dict:
        """Dự đoán với mô hình đã huấn luyện."""
        if self._model is None:
            return {"status": "error", "message": "Chưa huấn luyện mô hình"}

        try:
            import numpy as np

            data = np.array(params.get("data", []))
            if data.ndim == 1:
                data = data.reshape(1, -1)

            ket_qua = self._model.du_doan(data)
            return {
                "status": "success",
                "du_doan": ket_qua.tolist(),
                "so_mau": len(ket_qua),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def chay(self) -> None:
        """Khởi động web server."""
        try:
            from http.server import BaseHTTPRequestHandler, HTTPServer

            app = self

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/" or self.path == "/index.html":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(app.HTML_TEMPLATE.encode("utf-8"))
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)

                    if self.path == "/api/upload":
                        content_type = self.headers.get("Content-Type", "")
                        if "multipart/form-data" in content_type:
                            boundary = content_type.split("boundary=")[1].encode()
                            parts = body.split(b"--" + boundary)
                            file_data = None
                            form_data = {}
                            for part in parts:
                                if b"Content-Disposition" in part:
                                    header_end = part.find(b"\r\n\r\n")
                                    header = part[:header_end].decode()
                                    content = part[header_end + 4 :]
                                    if content.endswith(b"\r\n"):
                                        content = content[:-2]
                                    if 'name="file"' in header:
                                        file_data = content
                                    elif 'name="target"' in header:
                                        form_data["target"] = content.decode()
                            if file_data:
                                ket_qua = app._xu_ly_upload(form_data, file_data)
                            else:
                                ket_qua = {"status": "error", "message": "Không tìm thấy file"}
                        else:
                            ket_qua = {"status": "error", "message": "Sai Content-Type"}

                    elif self.path == "/api/train":
                        params = json.loads(body)
                        ket_qua = app._xu_ly_train(params)

                    elif self.path == "/api/predict":
                        params = json.loads(body)
                        ket_qua = app._xu_ly_predict(params)

                    else:
                        ket_qua = {"status": "error", "message": "Endpoint không tồn tại"}

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(ket_qua, ensure_ascii=False).encode("utf-8"))

                def log_message(self, format, *args):
                    app.logger.info(f"{self.address_string()} - {format % args}")

            server = HTTPServer((self.host, self.port), Handler)
            self.logger.info(f"Web app đang chạy tại http://{self.host}:{self.port}")
            self.logger.info("Mở trình duyệt và truy cập địa chỉ trên")
            server.serve_forever()

        except KeyboardInterrupt:
            self.logger.info("Web app đang tắt...")
