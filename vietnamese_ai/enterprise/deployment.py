"""CloudDeployment - Triển khai mô hình lên cloud."""

import json
from pathlib import Path

from vietnamese_ai.utils.logger import Logger


class CloudDeployment:
    """
    Triển khai mô hình lên cloud platforms.

    Hỗ trợ:
    - Docker: Tạo Dockerfile + docker-compose cho mô hình
    - AWS: Tạo deployment config cho SageMaker, Lambda
    - GCP: Tạo deployment config cho Vertex AI
    - Azure: Tạo deployment config cho Azure ML

    Sử dụng:
        >>> deploy = CloudDeployment()
        >>> deploy.tao_docker_config(mo_hinh, "my_model")
        >>> deploy.tao_aws_config(mo_hinh, "my_model")
    """

    def __init__(self):
        self.logger = Logger("CloudDeployment")

    def tao_docker_config(
        self,
        ten_model: str,
        duong_dan: str = "deploy",
        port: int = 8080,
    ) -> str:
        """
        Tạo Docker config cho mô hình.

        Args:
            ten_model: Tên mô hình
            duong_dan: Thư mục output
            port: Port

        Returns:
            Đường dẫn thư mục deploy
        """
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE {port}
CMD ["python", "serve.py"]
"""
        (thu_muc / "Dockerfile").write_text(dockerfile)

        serve_py = f"""import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from vietnamese_ai.models.base import BaseModel

mo_hinh = BaseModel.tai("model.pkl")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = np.array(body["data"])
        ket_qua = mo_hinh.du_doan(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({{"du_doan": ket_qua.tolist()}}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({{"model": "{ten_model}", "status": "healthy"}}).encode())

HTTPServer(("0.0.0.0", {port}), Handler).serve_forever()
"""
        (thu_muc / "serve.py").write_text(serve_py)

        docker_compose = f"""version: "3.8"
services:
  {ten_model}:
    build: .
    ports:
      - "{port}:{port}"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
"""
        (thu_muc / "docker-compose.yml").write_text(docker_compose)

        self.logger.info(f"Đã tạo Docker config tại: {thu_muc}")
        return str(thu_muc)

    def tao_aws_config(self, ten_model: str, duong_dan: str = "deploy/aws") -> str:
        """Tạo AWS SageMaker deployment config."""
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": ten_model,
            "framework": "custom",
            "instance_type": "ml.t2.medium",
            "instance_count": 1,
            "endpoint_name": f"{ten_model}-endpoint",
        }
        (thu_muc / "sagemaker_config.json").write_text(json.dumps(config, indent=2))

        self.logger.info(f"Đã tạo AWS config tại: {thu_muc}")
        return str(thu_muc)

    def tao_gcp_config(self, ten_model: str, duong_dan: str = "deploy/gcp") -> str:
        """Tạo GCP Vertex AI deployment config."""
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "model_display_name": ten_model,
            "container_image_uri": "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0",
            "machine_type": "n1-standard-2",
        }
        (thu_muc / "vertex_config.json").write_text(json.dumps(config, indent=2))

        self.logger.info(f"Đã tạo GCP config tại: {thu_muc}")
        return str(thu_muc)

    def tao_azure_config(self, ten_model: str, duong_dan: str = "deploy/azure") -> str:
        """Tạo Azure ML deployment config."""
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "name": ten_model,
            "compute_type": "managed",
            "instance_type": "Standard_DS2_v2",
            "instance_count": 1,
        }
        (thu_muc / "azure_config.json").write_text(json.dumps(config, indent=2))

        self.logger.info(f"Đã tạo Azure config tại: {thu_muc}")
        return str(thu_muc)
