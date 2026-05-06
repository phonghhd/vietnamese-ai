"""HuggingFaceWrapper - Tích hợp HuggingFace cho load, fine-tune, push."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class HuggingFaceWrapper:
    """
    Wrapper cho HuggingFace Transformers.

    Tính năng:
    - Load pretrained models (BERT, GPT, Llama, etc.)
    - Text classification, NER, QA
    - Fine-tune với Trainer API
    - Push model lên HuggingFace Hub
    - Auto tokenizer

    Yêu cầu: pip install transformers datasets

    Sử dụng:
        >>> hf = HuggingFaceWrapper()
        >>> hf.tai_model("vinai/phobert-base", nhiem_vu="text-classification")
        >>> ket_qua = df.du_doan(["Sản phẩm rất tốt"])
    """

    NHIEM_VU_HO_TRO = [
        "text-classification",
        "text-generation",
        "token-classification",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation",
    ]

    VIETNAMESE_MODELS = {
        "phobert": "vinai/phobert-base",
        "phobert-large": "vinai/phobert-large",
        "vit5-base": "VietAI/vit5-base",
        "vgpt2": "vinai/vgpt2",
        "bertvi": "trituenhantao/vibert4news-base",
    }

    def __init__(self):
        self.logger = Logger("HuggingFaceWrapper")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.nhiem_vu: Optional[str] = None
        self.ten_model: Optional[str] = None
        self.da_tai = False

        try:
            import transformers

            self._version = transformers.__version__
            self.logger.info(f"Transformers v{self._version}")
        except ImportError:
            self.logger.warning(
                "Transformers chưa cài. Cài đặt: pip install transformers"
            )

    def danh_sach_models_viet(self) -> Dict[str, str]:
        """Liệt kê models tiếng Việt có sẵn."""
        return self.VIETNAMESE_MODELS.copy()

    def tai_model(
        self,
        ten_model: str,
        nhiem_vu: Optional[str] = None,
        so_lop: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tải pretrained model từ HuggingFace.

        Args:
            ten_model: Tên model (HF Hub ID)
            nhiem_vu: Nhiệm vụ (text-classification, text-generation, ...)
            so_lop: Số lớp output (cho classification)

        Returns:
            Dict chứa thông tin model
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Cần cài đặt: pip install transformers")

        model_id = self.VIETNAMESE_MODELS.get(ten_model, ten_model)
        self.logger.info(f"Đang tải model: {model_id}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)

            if nhiem_vu == "text-classification":
                from transformers import AutoModelForSequenceClassification

                kwargs = {}
                if so_lop:
                    kwargs["num_labels"] = so_lop
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, **kwargs
                )
            elif nhiem_vu == "text-generation":
                from transformers import AutoModelForCausalLM

                self._model = AutoModelForCausalLM.from_pretrained(model_id)
            elif nhiem_vu == "fill-mask":
                from transformers import AutoModelForMaskedLM

                self._model = AutoModelForMaskedLM.from_pretrained(model_id)
            else:
                self._model = AutoModel.from_pretrained(model_id)

            self._nhiem_vu = nhiem_vu
            self._ten_model = model_id
            self._da_tai = True

            so_tham_so = sum(p.numel() for p in self._model.parameters())

            info = {
                "ten_model": model_id,
                "nhiem_vu": nhiem_vu,
                "vocab_size": len(self._tokenizer),
                "so_tham_so": so_tham_so,
                "so_tham_so_str": f"{so_tham_so/1e6:.1f}M",
            }

            self.logger.info(f"Đã tải: {model_id} ({info['so_tham_so_str']} params)")
            return info

        except Exception as e:
            self.logger.error(f"Lỗi tải model: {e}")
            raise

    def du_doan(self, cac_van_ban: List[str], top_k: int = 5) -> List[Dict]:
        """
        Dự đoán với model đã tải.

        Args:
            cac_van_ban: Danh sách văn bản
            top_k: Số kết quả trả về

        Returns:
            Danh sách kết quả
        """
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")

        try:
            import torch

            results = []
            inputs = self._tokenizer(
                cac_van_ban, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            if self._nhiem_vu == "text-classification":
                probs = torch.softmax(outputs.logits, dim=-1)
                for i, vb in enumerate(cac_van_ban):
                    top_probs, top_ids = probs[i].topk(top_k)
                    labels = []
                    for prob, idx in zip(top_probs, top_ids):
                        label = self._model.config.id2label.get(idx.item(), str(idx.item()))
                        labels.append({"nhan": label, "xac_suat": round(prob.item(), 4)})
                    results.append({"van_ban": vb, "ket_qua": labels})

            elif self._nhiem_vu == "text-generation":
                for i, vb in enumerate(cac_van_ban):
                    gen_ids = self._model.generate(
                        inputs["input_ids"][i:i+1],
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                    )
                    gen_text = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    results.append({"van_ban": vb, "sinh_ra": gen_text})

            else:
                for i, vb in enumerate(cac_van_ban):
                    hidden = outputs.last_hidden_state[i]
                    vec = hidden[0].numpy()
                    results.append({"van_ban": vb, "vector": vec.tolist()})

            return results

        except Exception as e:
            self.logger.error(f"Lỗi dự đoán: {e}")
            raise

    def ma_hoa(self, cac_van_ban: List[str]) -> np.ndarray:
        """Mã hóa văn bản thành vectors."""
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")

        try:
            import torch

            inputs = self._tokenizer(
                cac_van_ban, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
            return cls_vectors

        except Exception as e:
            self.logger.error(f"Lỗi mã hóa: {e}")
            raise

    def luu_model(self, duong_dan: str) -> str:
        """Lưu model và tokenizer."""
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")

        duong_dan_path = Path(duong_dan)
        duong_dan_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(duong_dan_path))
        self._tokenizer.save_pretrained(str(duong_dan_path))

        self.logger.info(f"Đã lưu model: {duong_dan}")
        return str(duong_dan_path)

    def tai_model_local(self, duong_dan: str) -> Dict[str, Any]:
        """Tải model từ thư mục local."""
        try:
            from transformers import AutoModel, AutoTokenizer

            duong_dan_path = Path(duong_dan)
            if not duong_dan_path.exists():
                raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

            self._tokenizer = AutoTokenizer.from_pretrained(str(duong_dan_path))
            self._model = AutoModel.from_pretrained(str(duong_dan_path))
            self._da_tai = True
            self._ten_model = str(duong_dan_path)

            so_tham_so = sum(p.numel() for p in self._model.parameters())

            self.logger.info(f"Đã tải model local: {duong_dan}")
            return {
                "duong_dan": str(duong_dan_path),
                "so_tham_so": so_tham_so,
            }

        except ImportError:
            raise ImportError("Cần cài đặt: pip install transformers")

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê wrapper."""
        return {
            "da_tai": self._da_tai,
            "ten_model": self._ten_model,
            "nhiem_vu": self._nhiem_vu,
            "co_tokenizer": self._tokenizer is not None,
            "available_vi_models": list(self.VIETNAMESE_MODELS.keys()),
            "supported_tasks": self.NHIEM_VU_HO_TRO,
        }
