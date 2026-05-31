import os
import subprocess
import time
from typing import Any, Dict, Optional


class NodeLlamaEngine:
    """
    Trình quản lý Edge AI sử dụng node-llama-cpp.
    Cho phép chạy các mô hình GGUF trực tiếp trên thiết bị (Local/Edge).
    Giao tiếp qua HTTP REST API (OpenAI-compatible) do node-llama-cpp cung cấp.
    """

    def __init__(
        self,
        model_path: str,
        port: int = 8080,
        gpu_layers: int = 35,
        context_size: int = 4096,
        auto_start: bool = True,
    ):
        self.model_path = os.path.expanduser(model_path)
        self.port = port
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.server_process: Optional[subprocess.Popen] = None

        import uuid

        self.node_id = str(uuid.uuid4())
        self.private_key = os.urandom(32)

        self.api_base = f"http://127.0.0.1:{self.port}/v1"

        if auto_start:
            self.start_server()

    def start_server(self):
        """Khởi chạy node-llama-cpp server qua npx."""
        if self.server_process is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {self.model_path}")

        try:
            import shutil

            if not shutil.which("npx"):
                raise EnvironmentError("Cần cài đặt Node.js và npx để sử dụng node-llama-cpp.")
        except Exception as e:
            if isinstance(e, EnvironmentError):
                raise e

        cmd = [
            "npx",
            "node-llama-cpp",
            "server",
            "--model",
            self.model_path,
            "--port",
            str(self.port),
            "--gpuLayers",
            str(self.gpu_layers),
            "--contextSize",
            str(self.context_size),
            "--host",
            "127.0.0.1",  # Only allow local connections by default for security
        ]

        print(f"[Edge AI] Đang khởi chạy: {' '.join(cmd)}")

        # Start server process in background
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # Mute stdout in production
            stderr=subprocess.DEVNULL,
        )

        # Đợi server sẵn sàng (ping đến health check API)
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 30):
        try:
            import requests
        except ImportError:
            raise ImportError("Cần cài đặt requests: pip install requests")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Gửi request tới models endpoint thay vì ping
                res = requests.get(f"{self.api_base}/models", timeout=1)
                if res.status_code == 200:
                    print(f"[Edge AI] Server đã sẵn sàng tại cổng {self.port}")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)

        self.stop_server()
        raise TimeoutError(f"Server không thể khởi động sau {timeout} giây.")

    def stop_server(self):
        """Tắt server."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            print("[Edge AI] Server đã được tắt.")

    def sinh_van_ban(
        self, prompt: str, do_dai: int = 256, kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Hàm tương thích với VietnameseLLM interface.
        Sinh văn bản từ Edge model.
        """
        if not self.server_process:
            raise RuntimeError("Server chưa được khởi động. Hãy gọi start_server() trước.")

        try:
            import requests
        except ImportError:
            raise ImportError("Cần cài đặt requests: pip install requests")

        payload = {
            "model": "edge-model",  # Tên model có thể là tuỳ ý vì đang trỏ thẳng local
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": do_dai,
            "temperature": 0.7,
        }

        if kwargs:
            payload.update(kwargs)

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Lỗi sinh văn bản (Edge): {str(e)}"

    def chay_suy_luan_an_toan(self, prompt: str, do_dai: int = 256) -> Dict[str, Any]:
        """Thực thi sinh văn bản và kèm theo bằng chứng mật mã (Cryptographic Proof) bằng HMAC-SHA256"""
        import hashlib
        import hmac

        cau_tra_loi = self.sinh_van_ban(prompt, do_dai=do_dai)
        model_hash = hashlib.sha256(self.model_path.encode("utf-8")).hexdigest()

        # Tạo Payload để ký
        payload = f"{self.node_id}:{prompt}:{cau_tra_loi}:{model_hash}"

        # Ký Payload bằng HMAC-SHA256
        proof = hmac.new(self.private_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

        return {
            "node_id": self.node_id,
            "cau_tra_loi": cau_tra_loi,
            "proof": proof,
            "model_hash": model_hash,
        }

    @classmethod
    def verify_proof(
        cls,
        node_id: str,
        public_key: str,
        cau_hoi: str,
        cau_tra_loi: str,
        proof: str,
        model_hash: str,
    ) -> bool:
        """
        Xác minh bằng chứng từ Edge Node sử dụng HMAC-SHA256.
        (public_key ở đây thực chất là secret_key hệ hexa trong mô hình đối xứng)
        """
        import hashlib
        import hmac

        payload = f"{node_id}:{cau_hoi}:{cau_tra_loi}:{model_hash}"
        try:
            secret_bytes = bytes.fromhex(public_key)
            expected_proof = hmac.new(
                secret_bytes, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            # So sánh thời gian hằng số để chống Timing Attack
            return hmac.compare_digest(expected_proof, proof)
        except Exception:
            return False

    def __del__(self):
        """Dọn dẹp process khi object bị hủy."""
        self.stop_server()
