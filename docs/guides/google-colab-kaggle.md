# Hướng dẫn: Train Model trên Google Colab / Kaggle

Hướng dẫn thực tế chạy huấn luyện model trên cloud GPU (Google Colab, Kaggle) với checkpoint lưu trên Hugging Face Hub.

---

## 1. Cài đặt trên Google Colab

Mở [Google Colab](https://colab.research.google.com) và tạo notebook mới.

### Cell 1: Cài đặt framework

```python
!pip install vietnamese-ai[all] -q

# Hoặc cài từ source (mới nhất)
!git clone https://github.com/phonghhd/vietnamese-ai.git
%cd vietnamese-ai
!pip install -e ".[all]" -q

# Cài thêm dependencies cho Colab
!pip install huggingface_hub datasets accelerate -q
```

### Cell 2: Kiểm tra GPU

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Cell 3: Đăng nhập Hugging Face

```python
from huggingface_hub import login
# Lấy token tại https://huggingface.co/settings/tokens
login(token="hf_YOUR_TOKEN_HERE")
```

---

## 2. Cài đặt trên Kaggle

Mở [Kaggle Notebooks](https://www.kaggle.com/code) và tạo notebook mới.

### Bật GPU

1. Vào **Settings** → **Accelerator** → chọn **GPU T4 x2** hoặc **P100**
2. Thêm **Secret** với tên `HF_TOKEN` và giá trị HuggingFace token

### Cell 1: Cài đặt

```python
!pip install vietnamese-ai[all] -q
!pip install huggingface_hub datasets accelerate -q

import os
os.environ["HF_TOKEN"] = os.environ.get("KAGGLE_SECRET_HF_TOKEN", "")
```

### Cell 2: Kiểm tra

```python
import torch
import vietnamese_ai

print(f"Framework: v{vietnamese_ai.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 3. Ví dụ 1: Phân loại văn bản tiếng Việt

### 3.1 Chuẩn bị dữ liệu

```python
import numpy as np
from datasets import load_dataset

# Tải dữ liệu từ HuggingFace
dataset = load_dataset("imdb", split="train[:1000]")  # Demo với 1000 mẫu

# Hoặc tạo dữ liệu tiếng Việt mẫu
van_ban_list = [
    "Sản phẩm rất tốt, tôi rất hài lòng với chất lượng",
    "Dịch vụ quá tệ, tôi sẽ không bao giờ quay lại",
    "Giá cả hợp lý, chất lượng ổn định",
    "Hàng giao chậm, đóng gói cẩu thả",
    "Tuyệt vời, đúng như mô tả, sẽ mua lại",
    "Thất vọng nặng nề, không đúng mong đợi",
] * 50

nhan = [1, 0, 1, 0, 1, 0] * 50  # 1=tích cực, 0=tiêu cực

print(f"Dữ liệu: {len(van_ban_list)} mẫu")
print(f"Phân bố: {sum(nhan)} tích cực, {len(nhan)-sum(nhan)} tiêu cực")
```

### 3.2 Trích xuất đặc trưng

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()

# Chuyển văn bản thành vector TF-IDF
van_ban_clean = [xl.xu_ly_day_du(vb) for vb in van_ban_list]
X = xl.ma_hoa_tfidf(van_ban_clean)
y = np.array(nhan)

print(f"Feature shape: {X.shape}")

# Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3.3 Huấn luyện và đánh giá

```python
from vietnamese_ai import PhanLoai, KiemDinhCheo, AutoML

# So sánh nhiều thuật toán
ket_qua = {}
for thuat_toan in ["logistic", "svm", "rung_ngau_nhien", "gradient_boosting"]:
    model = PhanLoai(thuat_toan=thuat_toan)
    model.huan_luyen(X_train, y_train)
    diem = model.danh_gia(X_test, y_test)
    ket_qua[thuat_toan] = diem
    print(f"{thuat_toan}: {diem:.4f}")

# Chọn model tốt nhất
tot_nhat = max(ket_qua, key=ket_qua.get)
print(f"\nModel tốt nhất: {tot_nhat} ({ket_qua[tot_nhat]:.4f})")

# Cross-validation
model = PhanLoai(thuat_toan=tot_nhat)
kdc = KiemDinhCheo(so_fold=5)
cv_result = kdc.chay(model, X_train, y_train)
print(f"CV: {cv_result['diem_trung_binh']:.4f} ± {cv_result['do_lech_chuan']:.4f}")
```

### 3.4 Lưu model lên HuggingFace

```python
import pickle
import os
from huggingface_hub import HfApi, hf_hub_download

# Lưu model local
model_path = "model_phan_loai.pkl"
model_best = PhanLoai(thuat_toan=tot_nhat)
model_best.huan_luyen(X_train, y_train)
model_best.luu(model_path)

# Upload lên HuggingFace Hub
api = HfApi()
repo_id = "your-username/vietnamese-classifier"  # Thay bằng username của bạn

# Tạo repo (nếu chưa có)
api.create_repo(repo_id, exist_ok=True)

# Upload
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model_phan_loai.pkl",
    repo_id=repo_id,
    repo_type="model",
)
print(f"Đã upload lên: https://huggingface.co/{repo_id}")
```

---

## 4. Ví dụ 2: Pre-train GPT tiếng Việt

### 4.1 Chuẩn bị corpus

```python
from vietnamese_ai import GPTModel, TextDataset, PreTrainer, ModelConfig

# Tải dữ liệu tiếng Việt
corpus = [
    "Trí tuệ nhân tạo là ngành khoa học nghiên cứu cách tạo ra máy tính thông minh.",
    "Học máy cho phép máy tính học từ dữ liệu mà không cần lập trình rõ ràng.",
    "Deep learning sử dụng mạng neural nhiều lớp để học biểu diễn phức tạp.",
    "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu ngôn ngữ người.",
    "Việt Nam đang phát triển mạnh mẽ trong lĩnh vực công nghệ thông tin.",
] * 100  # Lặp lại để có đủ dữ liệu

print(f"Corpus: {len(corpus)} câu, {sum(len(c.split()) for c in corpus)} từ")
```

### 4.2 Tạo dataset

```python
# Tạo TextDataset
dataset = TextDataset(do_dai_window=128, buoc_nhay=64)
dataset.tai_corpus(corpus, vocab_size=2000)

# Chia train/val
train_data, val_data = dataset.chia_du_lieu(ty_le_val=0.1)

print(f"Vocab size: {dataset.kich_thuoc_tu_vung}")
print(f"Train batches: {dataset.so_batch(32)}")
```

### 4.3 Khởi tạo model

```python
# Sử dụng preset config
config = ModelConfig.from_preset("vnlm-tiny")  # Nhỏ cho demo
print(f"Config: {config.to_dict()}")
print(f"Params: {config.so_tham_so_str}")

# Tạo model
model = GPTModel(
    d_model=config.d_model,
    so_dau=config.so_dau,
    d_ff=config.d_ff,
    so_block=config.so_block,
    so_tu_vung=dataset.kich_thuoc_tu_vung,
    do_dai_toi_da=config.do_dai_toi_da,
)
print(f"Model params: {model.thong_ke()}")
```

### 4.4 Huấn luyện

```python
trainer = PreTrainer(
    so_vong=5,
    toc_do_hoc=3e-4,
    warmup_steps=50,
)

# Callback để log
def log_callback(vong, buoc, loss, lr):
    if buoc % 10 == 0:
        print(f"Vòng {vong}, Bước {buoc}: loss={loss:.4f}, lr={lr:.6f}")

ket_qua = trainer.huan_luyen(model, dataset, callback=log_callback)
print(f"\nKết quả:")
print(f"  Train loss: {ket_qua.get('train_loss_min', 'N/A')}")
print(f"  Thời gian: {ket_qua.get('tong_thoi_gian', 0):.1f}s")
```

### 4.5 Sinh văn bản

```python
# Sinh text
prompt = "Trí tuệ nhân tạo"
ket_qua = model.sinh_tiep(
    dataset.ma_hoa(prompt),
    so_token=100,
    nhiet_do=0.8,
    top_k=50,
)
van_ban = dataset.giai_ma(ket_qua)
print(f"Prompt: {prompt}")
print(f"Generated: {van_ban}")
```

### 4.6 Lưu checkpoint lên HuggingFace

```python
import json
import numpy as np
from huggingface_hub import HfApi

# Lưu checkpoint
checkpoint_dir = "checkpoint_gpt"
os.makedirs(checkpoint_dir, exist_ok=True)

# Lưu weights
np.savez_compressed(
    f"{checkpoint_dir}/weights.npz",
    **{f"layer_{i}": w for i, w in enumerate(model._weights)}
)

# Lưu config + metadata
metadata = {
    "model_config": config.to_dict(),
    "vocab_size": dataset.kich_thuoc_tu_vung,
    "train_loss": ket_qua.get("train_loss_min", None),
    "so_vong": 5,
    "framework": f"vietnamese-ai v{vietnamese_ai.__version__}",
}
with open(f"{checkpoint_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Upload lên HuggingFace
api = HfApi()
repo_id = "your-username/vietnamese-gpt"
api.create_repo(repo_id, exist_ok=True)

api.upload_folder(
    folder_path=checkpoint_dir,
    repo_id=repo_id,
    repo_type="model",
)
print(f"Checkpoint đã upload: https://huggingface.co/{repo_id}")
```

---

## 5. Ví dụ 3: Fine-tune với LoRA + SFT

### 5.1 Load model từ HuggingFace

```python
from vietnamese_ai import HuggingFaceWrapper, LoRAPeft, PEFTConfig, SFTTrainer

# Load PhoBERT
hf = HuggingFaceWrapper()
model = hf.tai_model("vinai/phobert-base", nhiem_vu="text-classification", so_lop=2)
print(f"Model loaded: {type(model)}")
```

### 5.2 Áp dụng LoRA

```python
# Config LoRA
peft_config = PEFTConfig(
    phuong_phap="lora",
    rank=16,
    alpha=32,
    target_modules=["query", "value", "key", "output.dense"],
    dropout=0.1,
)

# Áp dụng LoRA
lora = LoRAPeft(peft_config)
model = lora.ap_dung(model)

# Kiểm tra trainable params
stats = lora.thong_ke(model)
print(f"Trainable: {stats['trainable_params']:,} / {stats['total_params']:,}")
print(f"Trainable %: {stats['trainable_pct']:.2f}%")
```

### 5.3 Chuẩn bị dữ liệu instruction

```python
# Dữ liệu instruction-following
du_lieu_train = [
    {
        "instruction": "Tóm tắt văn bản sau",
        "input": "Trí tuệ nhân tạo đang thay đổi thế giới...",
        "output": "AI đang cách mạng hóa nhiều lĩnh vực.",
    },
    {
        "instruction": "Dịch sang tiếng Anh",
        "input": "Xin chào thế giới",
        "output": "Hello world",
    },
    {
        "instruction": "Phân tích cảm xúc",
        "input": "Sản phẩm rất tốt!",
        "output": "Tích cực - người dùng hài lòng với sản phẩm.",
    },
] * 30  # Lặp lại cho demo
```

### 5.4 Fine-tune

```python
trainer = SFTTrainer(
    so_vong=3,
    kich_thuoc_batch=4,
    toc_do_hoc=2e-4,
    gradient_accumulation=4,
    warmup_ratio=0.1,
)

ket_qua = trainer.huan_luyen(model, du_lieu_train)
print(f"Train loss: {ket_qua.get('train_loss', 'N/A')}")
```

### 5.5 Lưu adapter lên HuggingFace

```python
# Lưu LoRA adapter
adapter_dir = "lora_adapter"
lora.luu(model, adapter_dir)

# Upload adapter
api = HfApi()
repo_id = "your-username/phobert-lora-vietnamese"
api.create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path=adapter_dir,
    repo_id=repo_id,
    repo_type="model",
)
print(f"LoRA adapter: https://huggingface.co/{repo_id}")
```

---

## 6. Resume Training từ Checkpoint

### 6.1 Tải checkpoint từ HuggingFace

```python
from huggingface_hub import snapshot_download

# Tải toàn bộ checkpoint
local_dir = snapshot_download(
    repo_id="your-username/vietnamese-gpt",
    repo_type="model",
    local_dir="./checkpoint_restored",
)
print(f"Checkpoint downloaded: {local_dir}")
```

### 6.2 Load model và tiếp tục train

```python
import numpy as np
import json

# Load metadata
with open(f"{local_dir}/metadata.json") as f:
    metadata = json.load(f)

print(f"Previous training: {metadata['so_vong']} vòng")
print(f"Previous loss: {metadata.get('train_loss', 'N/A')}")

# Load weights
weights = np.load(f"{local_dir}/weights.npz")
config = ModelConfig.from_preset("vnlm-tiny")

# Tạo model và load weights
model = GPTModel(
    d_model=config.d_model,
    so_dau=config.so_dau,
    d_ff=config.d_ff,
    so_block=config.so_block,
    so_tu_vung=metadata["vocab_size"],
    do_dai_toi_da=config.do_dai_toi_da,
)

# Load weights vào model
for key, arr in weights.items():
    idx = int(key.split("_")[1])
    if idx < len(model._weights):
        model._weights[idx] = arr

print("Đã load checkpoint, tiếp tục training...")

# Tiếp tục train thêm 5 vòng
trainer = PreTrainer(so_vong=5, toc_do_hoc=1e-4)  # LR nhỏ hơn khi resume
ket_qua_tiep = trainer.huan_luyen(model, dataset)
print(f"New loss: {ket_qua_tiep.get('train_loss_min', 'N/A')}")
```

---

## 7. Ví dụ 4: Train với PyTorch Trainer (GPU)

### 7.1 Sử dụng HuanLuyenPyTorch

```python
from vietnamese_ai import HuanLuyenPyTorch, PhanLoai
import numpy as np

# Dữ liệu lớn
X = np.random.randn(10000, 100).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Trainer với GPU + mixed precision
trainer = HuanLuyenPyTorch(
    thiet_bi="auto",       # Tự động chọn GPU/CPU
    hon_lep=True,           # Mixed precision (FP16)
    gradient_accumulation=2,
    scheduler="cosine",
    early_stopping=5,
)

# Callback
def callback(vong, loss, lr, **kwargs):
    if vong % 5 == 0:
        print(f"Epoch {vong}: loss={loss:.4f}, lr={lr:.6f}")

# Train
ket_qua = trainer.huan_luyen(
    model=None,  # Sẽ tự tạo model
    X_train=X[:8000],
    y_train=y[:8000],
    X_val=X[8000:],
    y_val=y[8000:],
    callback=callback,
)
print(f"Best loss: {ket_qua.get('best_loss', 'N/A')}")
```

### 7.2 Lưu checkpoint

```python
# Lưu checkpoint
trainer.luu_checkpoint("checkpoint_pytorch.pt")

# Upload lên HuggingFace
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoint_pytorch.pt",
    path_in_repo="checkpoints/epoch_10.pt",
    repo_id="your-username/vietnamese-model",
    repo_type="model",
)
```

---

## 8. Tips cho Google Colab / Kaggle

### Quản lý bộ nhớ

```python
import gc
import torch

# Giải phóng bộ nhớ sau khi train
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Lưu tự động mỗi N epochs

```python
class AutoSaveCallback:
    def __init__(self, every_n=5, repo_id=None):
        self.every_n = every_n
        self.repo_id = repo_id
        self.epoch = 0

    def __call__(self, vong, loss, **kwargs):
        self.epoch = vong
        if vong % self.every_n == 0:
            path = f"checkpoint_epoch_{vong}.pkl"
            model.luu(path)
            print(f"Auto-saved: {path}")

            if self.repo_id:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"checkpoints/{path}",
                    repo_id=self.repo_id,
                    repo_type="model",
                )
                print(f"Uploaded to HF: {self.repo_id}")
```

### Tải dữ liệu từ Kaggle

```python
# Nếu chạy trên Kaggle
import os
if os.path.exists("/kaggle/input"):
    # Dữ liệu Kaggle tự động có trong /kaggle/input/
    data_dir = "/kaggle/input/your-dataset"
else:
    # Trên Colab, tải từ Kaggle API
    !pip install kaggle -q
    !kaggle datasets download -d your-dataset -p ./data --unzip
    data_dir = "./data"
```

### Thời gian chạy Colab

```python
import time

bat_dau = time.time()

# ... training code ...

thoi_gian = time.time() - bat_dau
print(f"Thời gian: {thoi_gian/60:.1f} phút")

# Colab miễn phí ~12 giờ, Colab Pro ~24 giờ
if thoi_gian > 10 * 3600:
    print("CẢNH BÁO: Sắp hết thời gian Colab!")
```

---

## 9. Bảng so sánh GPU

| Platform | GPU | VRAM | Thời gian | Giá |
|---|---|---|---|---|
| Colab Free | T4 | 15GB | ~12h | Miễn phí |
| Colab Pro | T4/V100 | 15-40GB | ~24h | $10/tháng |
| Colab Pro+ | A100 | 40-80GB | ~24h | $50/tháng |
| Kaggle Free | T4/P100 | 16GB | ~9h/tuần | Miễn phí |
| Kaggle+ | P100 | 16GB | ~30h/tuần | $10/tháng |

---

## 10. Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| Out of Memory | Giảm `kich_thuoc_batch`, dùng `gradient_accumulation` |
| Colab disconnect | Lưu checkpoint thường xuyên, dùng AutoSave |
| Slow training | Bật `hon_lep=True` (mixed precision), tăng batch size |
| HF upload fail | Kiểm tra token, tạo repo trước khi upload |
| Import error | `!pip install -e ".[all]"` lại |
