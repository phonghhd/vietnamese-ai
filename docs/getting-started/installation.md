# Cài đặt

## Yêu cầu hệ thống

- Python >= 3.8
- `pip` hoặc `conda`

## Cài đặt cơ bản (Core — chỉ phụ thuộc NumPy)

Cài đặt mặc định chỉ kéo về phụ thuộc tối thiểu cho phần lõi (`Engine`, `Pipeline`, `KiemDinhCheo`, `TimKiemThamSo`, `XuLySo`, `Zero-GPU Edge Engine`, `utils`):

```bash
pip install vietnamese-ai
```

Hoặc từ file production tối thiểu:

```bash
pip install -r requirements.txt
```

## Cài đặt từ mã nguồn

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Cài đặt phần mở rộng tùy chọn (Optional Extensions)

```bash
# Học máy cổ điển & chuỗi thời gian (scikit-learn, pandas)
pip install "vietnamese-ai[classic-ml]"       # hoặc vietnamese-ai[ml]

# Dữ liệu bảng nâng cao (XGBoost, LightGBM)
pip install "vietnamese-ai[tabular]"

# Xử lý ngôn ngữ tự nhiên tiếng Việt (underthesea)
pip install "vietnamese-ai[vietnamese-nlp]"   # hoặc vietnamese-ai[nlp]

# RAG Hành chính / Pháp lý đọc PDF & ảnh scan OCR (pypdf, Pillow, pytesseract)
pip install "vietnamese-ai[legal-rag]"

# Thị giác máy tính & OCR (Pillow, pytesseract)
pip install "vietnamese-ai[vision]"

# Deep Learning (PyTorch) và Fine-Tuning (Transformers, PEFT, TRL, Datasets)
pip install "vietnamese-ai[torch]"
pip install "vietnamese-ai[dl]"

# RAG nâng cao (ChromaDB, Qdrant, PyPDF, Sentence-Transformers, NetworkX)
pip install "vietnamese-ai[rag]"

# Kết nối Cloud LLM APIs (OpenAI, Google GenAI Gemini), Local llama.cpp & Remote GPU Serverless (Modal)
pip install "vietnamese-ai[llm]"
pip install "vietnamese-ai[local-llm]"
pip install "vietnamese-ai[cloud-gpu]"

# API Server hiệu năng cao (FastAPI, Uvicorn, WebSockets) & gRPC
pip install "vietnamese-ai[api]"
pip install "vietnamese-ai[grpc]"

# Xuất mô hình chuẩn công nghiệp (ONNX Runtime, skl2onnx, GGUF)
pip install "vietnamese-ai[export]"

# Tiện ích Mobile / Web3 / Robotics (thử nghiệm)
pip install "vietnamese-ai[mobile]"
pip install "vietnamese-ai[web3]"
pip install "vietnamese-ai[robotics]"

# Đầy đủ các nhóm mở rộng tiêu chuẩn (không gồm các gói cần CUDA/biên dịch riêng như unsloth/llama-cpp-python)
pip install "vietnamese-ai[all]"
```

## Môi trường phát triển (Development)

Phụ thuộc phát triển được tách riêng khỏi `requirements.txt`:

```bash
pip install -r requirements-dev.txt
# hoặc
pip install -e ".[dev,docs]"
```

## Kiểm tra cài đặt

```python
import vietnamese_ai
from vietnamese_ai.utils import QuanLyPhuThuoc

print(vietnamese_ai.__version__)  # 31.0.3
print(QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc())
```

```bash
vai doctor
vai status-map
vai demo
```
