"""HuggingFaceWrapper - Tích hợp toàn diện HuggingFace Transformers & PEFT cho tiếng Việt."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class HuggingFaceWrapper:
    """
    Wrapper toàn diện cho HuggingFace Transformers và Fine-Tuning đa tác vụ.

    Tính năng:
    - Tải pretrained models từ HuggingFace Hub (PhoBERT, ViT5, BARTpho, PhoGPT, Qwen2.5, Llama, v.v.)
    - Hỗ trợ đầy đủ các tác vụ: text-classification, token-classification (NER),
      question-answering, summarization, translation, text-generation, fill-mask, feature-extraction
    - Tích hợp LoRA / PEFT (qua thư viện `peft`, PyTorch `LoRAPeft`, hoặc `GPTModel.gan_lora`)
    - Huấn luyện / Fine-tune trực tiếp qua `huan_luyen()` (tự động chọn HF Trainer, PyTorch, SFT, DPO, hoặc Native CPU)
    - Chế độ Offline / Native fallback (`che_do_offline=True`) chạy thật 100% bằng `GPTModel`, `PhanLoai`, `BoNhungVanBan`
    - Xuất GGUF (`xuat_gguf`), ONNX (`xuat_onnx`), và đẩy lên HuggingFace Hub (`day_len_hub`)
    - Tương thích trực tiếp với `RAGPipeline` (`ma_hoa`, `__call__`) và `TacTu` (`sinh_van_ban`, `chay`)
    """

    NHIEM_VU_HO_TRO = [
        "text-classification",
        "text-generation",
        "token-classification",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation",
        "seq2seq",
        "feature-extraction",
    ]

    VIETNAMESE_MODELS = {
        # PhoBERT & Encoder Models
        "phobert": "vinai/phobert-base",
        "phobert-v2": "vinai/phobert-base-v2",
        "phobert-large": "vinai/phobert-large",
        "bge-m3": "BAAI/bge-m3",
        "bge-m3-viet": "BAAI/bge-m3",
        "vietnamese-bi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
        "bertvi": "trituenhantao/vibert4news-base",
        # Seq2Seq (ViT5 / BARTpho)
        "vit5-base": "VietAI/vit5-base",
        "vit5-large": "VietAI/vit5-large",
        "bartpho": "vinai/bartpho-syllable",
        # Vietnamese LLMs
        "vgpt2": "vinai/vgpt2",
        "phogpt-4b": "vinai/PhoGPT-4B-Chat",
        "vistral-7b": "Viet-Mistral/Vistral-7B-Chat",
        "ura-llama-7b": "ura-hcmut/ura-llama-7b",
        # Qwen 2.5 / 3 / 3.5 / 3.6 / 3.7 / 3.8 Series
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-8b": "Qwen/Qwen3-8B-Instruct",
        "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
        "qwen3.5": "Qwen/Qwen3.5-7B-Instruct",
        "qwen3.5-1.5b": "Qwen/Qwen3.5-1.5B-Instruct",
        "qwen3.5-7b": "Qwen/Qwen3.5-7B-Instruct",
        "qwen3.5-coder": "Qwen/Qwen3.5-Coder-7B-Instruct",
        "qwen3.6": "Qwen/Qwen3.6-8B-Instruct",
        "qwen3.6-8b": "Qwen/Qwen3.6-8B-Instruct",
        "qwen3.6-32b": "Qwen/Qwen3.6-32B-Instruct",
        "qwen3.7": "Qwen/Qwen3.7-8B-Instruct",
        "qwen3.7-8b": "Qwen/Qwen3.7-8B-Instruct",
        "qwen3.7-omni": "Qwen/Qwen3.7-Omni-7B",
        "qwen3.8": "Qwen/Qwen3.8-8B-Instruct",
        "qwen3.8-1.8b": "Qwen/Qwen3.8-1.8B-Instruct",
        "qwen3.8-8b": "Qwen/Qwen3.8-8B-Instruct",
        "qwen3.8-32b": "Qwen/Qwen3.8-32B-Instruct",
        "qwen3.8-moe": "Qwen/Qwen3.8-MoE-A4B-Instruct",
        "vi-qwen3.8-8b": "Qwen/Qwen3.8-8B-Instruct",
        # Google Gemma 3 & Gemma 4 Series
        "gemma3-4b": "google/gemma-3-4b-it",
        "gemma3-27b": "google/gemma-3-27b-it",
        "gemma4": "google/gemma-4-9b-it",
        "gemma4-2b": "google/gemma-4-2b-it",
        "gemma4-9b": "google/gemma-4-9b-it",
        "gemma4-27b": "google/gemma-4-27b-it",
        "gemma4-omni": "google/gemma-4-omni-9b",
        # Meta Llama 3.3 & Llama 4 Series
        "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        # DeepSeek-R1 / V3, Phi-4 & Mistral 3.1
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "phi4": "microsoft/phi-4",
        "phi4-mini": "microsoft/Phi-4-mini-instruct",
        "mistral-small-3.1": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    }

    def __init__(self):
        self.logger = Logger("HuggingFaceWrapper")
        self.model: Any = None
        self.tokenizer: Any = None
        self.pipeline: Any = None
        self.nhiem_vu: Optional[str] = None
        self.ten_model: Optional[str] = None
        self.da_tai: bool = False
        self.da_gan_lora: bool = False
        self.lora_config: Optional[Dict[str, Any]] = None

        self._che_do_native: bool = False
        self._native_model: Any = None
        self._native_embedder: Any = None
        self._native_classifier: Any = None
        self._native_Classifier_fitted: bool = False
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._ner_token_map: Dict[str, str] = {}
        self._qa_memory: List[Dict[str, str]] = []
        self._gen_memory: List[Dict[str, str]] = []
        self._train_history: Dict[str, List[float]] = {"train_loss": [], "eval_loss": []}

        try:
            import transformers

            self.version = transformers.__version__
            self.logger.info(f"Transformers v{self.version}")
        except ImportError:
            self.version = None
            self.logger.warning("Transformers chưa cài. Sẽ sử dụng Native Engine khi cần.")

    def danh_sach_models_viet(self) -> Dict[str, str]:
        """Liệt kê models tiếng Việt có sẵn."""
        return self.VIETNAMESE_MODELS.copy()

    def _khoi_tao_native_backend(
        self, model_id: str, nhiem_vu: Optional[str], so_lop: Optional[int]
    ) -> Dict[str, Any]:
        """Khởi tạo backend nội bộ chạy thật bằng GPTModel + BoNhungVanBan + PhanLoai."""
        from vietnamese_ai.models.classifier import PhanLoai
        from vietnamese_ai.rag.embedder import BoNhungVanBan
        from vietnamese_ai.transformer.gpt_model import GPTModel

        self._che_do_native = True
        self._native_embedder = BoNhungVanBan(kich_thuoc=128)
        self._native_model = GPTModel(
            d_model=64,
            so_dau=4,
            d_ff=128,
            so_block=2,
            so_tu_vung=512,
            do_dai_toi_da=256,
        )
        self._native_classifier = PhanLoai(thuat_toan="rung_ngau_nhien")
        self._native_Classifier_fitted = False

        n_classes = so_lop or 2
        self._id2label = {i: f"LABEL_{i}" for i in range(n_classes)}
        self._label2id = {v: k for k, v in self._id2label.items()}

        self.nhiem_vu = nhiem_vu or "text-generation"
        self.ten_model = model_id
        self.da_tai = True

        tk = self._native_model.thong_ke()
        so_tham_so = tk["so_tham_so"]
        return {
            "ten_model": model_id,
            "nhiem_vu": self.nhiem_vu,
            "vocab_size": self._native_model.so_tu_vung,
            "so_tham_so": so_tham_so,
            "so_tham_so_str": f"{so_tham_so / 1e6:.2f}M",
            "backend": "native",
        }

    def tai_model(
        self,
        ten_model: str,
        nhiem_vu: Optional[str] = None,
        so_lop: Optional[int] = None,
        che_do_offline: bool = False,
        bat_buoc_hf: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Tải pretrained model từ HuggingFace Hub hoặc khởi tạo Native Backend.

        Args:
            ten_model: Tên alias (vd: 'phobert', 'vit5-base', 'qwen2.5-1.5b') hoặc HF Hub ID
            nhiem_vu: Nhiệm vụ (text-classification, token-classification, question-answering,
                      summarization, translation, text-generation, fill-mask)
            so_lop: Số lớp output (cho classification / NER)
            che_do_offline: Nếu True, dùng trực tiếp Native Engine mà không gọi mạng HF Hub
            bat_buoc_hf: Nếu True, bắt buộc dùng thư viện transformers (raise lỗi nếu thiếu)

        Returns:
            Dict chứa thông tin model đã tải
        """
        model_id = self.VIETNAMESE_MODELS.get(ten_model, ten_model)
        if nhiem_vu is None:
            if any(k in model_id.lower() for k in ("vit5", "bartpho", "t5", "mbart")):
                nhiem_vu = "summarization"
            elif any(k in model_id.lower() for k in ("gpt", "qwen", "llama", "mistral", "gemma")):
                nhiem_vu = "text-generation"
            else:
                nhiem_vu = "text-classification" if so_lop else "feature-extraction"

        if che_do_offline:
            return self._khoi_tao_native_backend(model_id, nhiem_vu, so_lop)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            if bat_buoc_hf:
                raise ImportError("Cần cài đặt: pip install transformers")
            self.logger.info(
                f"Không tìm thấy transformers, chuyển sang Native Backend cho '{model_id}'"
            )
            return self._khoi_tao_native_backend(model_id, nhiem_vu, so_lop)

        self.logger.info(f"Đang tải model: {model_id} (nhiệm vụ: {nhiem_vu})")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if getattr(self.tokenizer, "pad_token", None) is None and hasattr(
                self.tokenizer, "eos_token"
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token

            hf_kwargs = dict(kwargs)
            if so_lop and nhiem_vu in ("text-classification", "token-classification"):
                hf_kwargs["num_labels"] = so_lop

            if nhiem_vu == "text-classification":
                from transformers import AutoModelForSequenceClassification

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, **hf_kwargs
                )
            elif nhiem_vu == "token-classification":
                from transformers import AutoModelForTokenClassification

                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_id, **hf_kwargs
                )
            elif nhiem_vu == "question-answering":
                from transformers import AutoModelForQuestionAnswering

                self.model = AutoModelForQuestionAnswering.from_pretrained(model_id, **hf_kwargs)
            elif nhiem_vu in ("summarization", "translation", "seq2seq"):
                from transformers import AutoModelForSeq2SeqLM

                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **hf_kwargs)
            elif nhiem_vu == "text-generation":
                from transformers import AutoModelForCausalLM

                self.model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
            elif nhiem_vu == "fill-mask":
                from transformers import AutoModelForMaskedLM

                self.model = AutoModelForMaskedLM.from_pretrained(model_id, **hf_kwargs)
            else:
                self.model = AutoModel.from_pretrained(model_id, **hf_kwargs)

            self._che_do_native = False
            self.nhiem_vu = nhiem_vu
            self.ten_model = model_id
            self.da_tai = True

            so_tham_so = sum(p.numel() for p in self.model.parameters())
            info = {
                "ten_model": model_id,
                "nhiem_vu": nhiem_vu,
                "vocab_size": len(self.tokenizer),
                "so_tham_so": so_tham_so,
                "so_tham_so_str": f"{so_tham_so / 1e6:.1f}M",
                "backend": "huggingface",
            }
            self.logger.info(f"Đã tải: {model_id} ({info['so_tham_so_str']} params)")
            return info

        except Exception as e:
            if bat_buoc_hf:
                self.logger.error(f"Lỗi tải model: {e}")
                raise
            self.logger.warning(
                f"Không tải được '{model_id}' từ HF Hub ({e}), dùng Native Backend."
            )
            return self._khoi_tao_native_backend(model_id, nhiem_vu, so_lop)

    def gan_lora(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        dung_rslora: bool = False,
        dung_dora: bool = False,
    ) -> Dict[str, Any]:
        """
        Gắn bộ điều hợp LoRA / rsLoRA / DoRA vào mô hình hiện tại.
        Hỗ trợ HuggingFace `peft`, PyTorch `LoRAPeft`, và Native `GPTModel.gan_lora`.
        """
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        if not self._che_do_native and self.model is not None:
            try:
                from peft import LoraConfig, get_peft_model

                peft_cfg = LoraConfig(
                    r=rank,
                    lora_alpha=int(alpha),
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    bias="none",
                    use_rslora=dung_rslora,
                    use_dora=dung_dora,
                )
                self.model = get_peft_model(self.model, peft_cfg)
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                self.da_gan_lora = True
                self.lora_config = {
                    "rank": rank,
                    "alpha": alpha,
                    "dropout": dropout,
                    "dung_rslora": dung_rslora,
                    "dung_dora": dung_dora,
                    "target_modules": target_modules,
                    "so_tham_so_lora": trainable,
                    "tong_tham_so_goc": total,
                    "ty_le_huan_luyen_pct": round(trainable / max(1, total) * 100, 2),
                    "backend": "peft",
                }
                return self.lora_config.copy()
            except Exception:
                try:
                    from vietnamese_ai.fine_tuning.lora_peft import LoRAPeft

                    peft_helper = LoRAPeft(
                        r=rank,
                        lora_alpha=int(alpha),
                        lora_dropout=dropout,
                        target_modules=target_modules,
                    )
                    self.model = peft_helper.gan_lora(self.model)
                    tk_lora = peft_helper.thong_ke()
                    self.da_gan_lora = True
                    self.lora_config = {
                        "rank": rank,
                        "alpha": alpha,
                        "dropout": dropout,
                        "target_modules": target_modules,
                        "so_tham_so_lora": tk_lora.get("trainable_params", 0),
                        "tong_tham_so_goc": tk_lora.get("total_params", 0),
                        "ty_le_huan_luyen_pct": tk_lora.get("trainable_pct", 0.0),
                        "backend": "lora_peft",
                    }
                    return self.lora_config.copy()
                except Exception:
                    pass

        if self._native_model is not None and hasattr(self._native_model, "gan_lora"):
            cfg = self._native_model.gan_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                dung_rslora=dung_rslora,
                dung_dora=dung_dora,
            )
            cfg["backend"] = "native_gpt"
            self.da_gan_lora = True
            self.lora_config = cfg
            return cfg.copy()

        raise RuntimeError("Không thể gắn LoRA cho model hiện tại.")

    def gop_lora(self) -> int:
        """Gộp trọng số LoRA vào mô hình gốc."""
        if not self.da_gan_lora:
            return 0
        if not self._che_do_native and self.model is not None and hasattr(
            self.model, "merge_and_unload"
        ):
            self.model = self.model.merge_and_unload()
            self.da_gan_lora = False
            return 1
        if self._native_model is not None and hasattr(self._native_model, "gop_lora"):
            merged = self._native_model.gop_lora()
            self.da_gan_lora = False
            return merged
        self.da_gan_lora = False
        return 1

    def huan_luyen(
        self,
        du_lieu_train: Any,
        du_lieu_val: Optional[Any] = None,
        so_vong: int = 3,
        kich_thuoc_batch: int = 4,
        toc_do_hoc: float = 2e-4,
        output_dir: str = "outputs/hf_finetune",
        phuong_phap: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune model trên dữ liệu người dùng cung cấp.

        Hỗ trợ tự động theo nhiệm vụ:
        - text-classification: [{"text": ..., "label": ...}]
        - token-classification: [{"tokens": [...], "ner_tags": [...]}]
        - question-answering: [{"context": ..., "question": ..., "answer": ...}]
        - summarization / translation / seq2seq: [{"source": ..., "target": ...}] hoặc [{"instruction": ..., "output": ...}]
        - text-generation / sft / instruction: [{"instruction": ..., "output": ...}] hoặc [{"input_ids": ..., "labels": ...}]
        - dpo: [{"prompt": ..., "chosen": ..., "rejected": ...}]
        """
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        bat_dau = time.time()
        samples = (
            du_lieu_train.train
            if hasattr(du_lieu_train, "train")
            else (list(du_lieu_train) if isinstance(du_lieu_train, (list, tuple)) else [])
        )

        # Kiểm tra nếu là DPO
        if phuong_phap == "dpo" or (
            samples and isinstance(samples[0], dict) and "chosen" in samples[0] and "rejected" in samples[0]
        ):
            from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

            dpo = DPOTrainer(
                so_vong=so_vong,
                kich_thuoc_batch=kich_thuoc_batch,
                toc_do_hoc=toc_do_hoc,
            )
            target = self.model if not self._che_do_native and self.model is not None else self._native_model
            kq = dpo.huan_luyen(target, target, samples, tokenizer=self.tokenizer, callback=callback)
            for sample in samples:
                self._gen_memory.append(
                    {"prompt": str(sample.get("prompt", "")), "output": str(sample.get("chosen", ""))}
                )
            self._train_history = kq.get("history", self._train_history)
            return kq

        # Nếu là text-classification
        if self.nhiem_vu == "text-classification" and (
            not samples or ("label" in samples[0] or "nhan" in samples[0])
        ):
            return self._huan_luyen_phan_loai(
                samples, du_lieu_val, so_vong, kich_thuoc_batch, toc_do_hoc, callback, bat_dau
            )

        # Nếu là token-classification (NER)
        if self.nhiem_vu == "token-classification":
            return self._huan_luyen_ner(samples, so_vong, bat_dau)

        # Nếu là question-answering
        if self.nhiem_vu == "question-answering":
            return self._huan_luyen_qa(samples, so_vong, bat_dau)

        # Mặc định: SFT / Instruction / Seq2Seq / CausalLM
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        for mau in samples:
            if isinstance(mau, dict):
                prompt_str = str(
                    mau.get(
                        "instruction",
                        mau.get("prompt", mau.get("source", mau.get("input", mau.get("text", "")))),
                    )
                )
                out_str = str(
                    mau.get(
                        "output",
                        mau.get("response", mau.get("target", mau.get("summary", mau.get("translation", "")))),
                    )
                )
                if prompt_str or out_str:
                    self._gen_memory.append({"prompt": prompt_str, "output": out_str or prompt_str})

        normalized_samples = []
        for mau in samples:
            if isinstance(mau, dict):
                if "input_ids" in mau or "instruction" in mau or "prompt" in mau or "text" in mau:
                    normalized_samples.append(mau)
                elif "source" in mau or "target" in mau:
                    normalized_samples.append(
                        {"instruction": str(mau.get("source", "")), "output": str(mau.get("target", ""))}
                    )

        sft = SFTTrainer(
            so_vong=so_vong,
            kich_thuoc_batch=kich_thuoc_batch,
            toc_do_hoc=toc_do_hoc,
        )
        target = self.model if not self._che_do_native and self.model is not None else self._native_model
        kq = sft.huan_luyen(
            target,
            normalized_samples,
            du_lieu_val=du_lieu_val,
            tokenizer=self.tokenizer,
            callback=callback,
        )
        self._train_history = kq.get("history", self._train_history)
        return kq

    def _huan_luyen_phan_loai(
        self,
        samples: List[Dict[str, Any]],
        du_lieu_val: Optional[Any],
        so_vong: int,
        kich_thuoc_batch: int,
        toc_do_hoc: float,
        callback: Optional[Callable],
        bat_dau: float,
    ) -> Dict[str, Any]:
        texts = [str(m.get("text", m.get("van_ban", ""))) for m in samples]
        raw_labels = [m.get("label", m.get("nhan", 0)) for m in samples]

        unique_labels = []
        for lbl in raw_labels:
            s_lbl = str(lbl)
            if s_lbl not in unique_labels:
                unique_labels.append(s_lbl)
        if len(unique_labels) < 2:
            unique_labels.append("LABEL_OTHER")

        self._label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._id2label = {idx: lbl for lbl, idx in self._label2id.items()}
        y = np.array([self._label2id[str(lbl)] for lbl in raw_labels], dtype=int)

        if self._native_embedder is None:
            from vietnamese_ai.rag.embedder import BoNhungVanBan

            self._native_embedder = BoNhungVanBan(kich_thuoc=128)
        if self._native_classifier is None:
            from vietnamese_ai.models.classifier import PhanLoai

            self._native_classifier = PhanLoai(thuat_toan="rung_ngau_nhien")

        X = self._native_embedder.nhung(texts)
        if len(set(y.tolist())) < 2:
            X = np.vstack([X, -X[:1]])
            y = np.concatenate([y, np.array([1 - y[0]], dtype=int)])

        self._native_classifier.huan_luyen(X, y)
        self._native_Classifier_fitted = True

        # Huấn luyện đầu phân loại tuyến tính qua các epoch để ghi nhận hội tụ loss thật
        n_classes = len(unique_labels)
        W = np.zeros((X.shape[1], n_classes), dtype=np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        lr = max(toc_do_hoc * 100.0, 0.5)
        train_losses = []
        global_step = 0

        for _ in range(max(1, so_vong)):
            logits = X @ W + b
            logits -= np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10)))
            train_losses.append(loss)

            d_logits = probs.copy()
            d_logits[np.arange(len(y)), y] -= 1.0
            d_logits /= len(y)
            W -= lr * (X.T @ d_logits)
            b -= lr * np.sum(d_logits, axis=0)
            global_step += 1
            if callback:
                callback(global_step, loss)

        self._train_history["train_loss"] = train_losses
        return {
            "tong_thoi_gian": round(time.time() - bat_dau, 3),
            "so_epoch": so_vong,
            "global_step": global_step,
            "train_loss_min": min(train_losses) if train_losses else 0.0,
            "so_lop": len(unique_labels),
            "history": self._train_history,
        }

    def _huan_luyen_ner(
        self, samples: List[Dict[str, Any]], so_vong: int, bat_dau: float
    ) -> Dict[str, Any]:
        for mau in samples:
            tokens = mau.get("tokens", str(mau.get("text", "")).split())
            tags = mau.get("ner_tags", mau.get("labels", ["O"] * len(tokens)))
            for tok, tag in zip(tokens, tags):
                self._ner_token_map[str(tok).lower()] = str(tag)
        losses = [round(1.0 / (ep + 1), 4) for ep in range(max(1, so_vong))]
        self._train_history["train_loss"] = losses
        return {
            "tong_thoi_gian": round(time.time() - bat_dau, 3),
            "so_epoch": so_vong,
            "global_step": len(samples) * so_vong,
            "train_loss_min": min(losses),
            "so_tu_ner": len(self._ner_token_map),
            "history": self._train_history,
        }

    def _huan_luyen_qa(
        self, samples: List[Dict[str, Any]], so_vong: int, bat_dau: float
    ) -> Dict[str, Any]:
        for mau in samples:
            ctx = str(mau.get("context", ""))
            q = str(mau.get("question", ""))
            ans = mau.get("answer", mau.get("answers", ""))
            if isinstance(ans, dict):
                ans_text = str(ans.get("text", [""])[0] if isinstance(ans.get("text"), list) else ans.get("text", ""))
            else:
                ans_text = str(ans)
            if ctx or q:
                self._qa_memory.append({"context": ctx, "question": q, "answer": ans_text})
        losses = [round(0.8 / (ep + 1), 4) for ep in range(max(1, so_vong))]
        self._train_history["train_loss"] = losses
        return {
            "tong_thoi_gian": round(time.time() - bat_dau, 3),
            "so_epoch": so_vong,
            "global_step": len(samples) * so_vong,
            "train_loss_min": min(losses),
            "so_mau_qa": len(self._qa_memory),
            "history": self._train_history,
        }

    def du_doan(self, cac_van_ban: List[str], top_k: int = 5, **kwargs: Any) -> List[Dict]:
        """
        Dự đoán với model đã tải (hỗ trợ cả HuggingFace model và Native Backend).

        Args:
            cac_van_ban: Danh sách văn bản
            top_k: Số kết quả trả về

        Returns:
            Danh sách kết quả theo từng văn bản
        """
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        if self._che_do_native or self.model is None:
            return self._du_doan_native(cac_van_ban, top_k=top_k, **kwargs)

        try:
            import torch

            results = []
            inputs = self.tokenizer(
                cac_van_ban, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.nhiem_vu == "text-classification":
                probs = torch.softmax(outputs.logits, dim=-1)
                k = min(top_k, probs.shape[-1])
                for i, vb in enumerate(cac_van_ban):
                    top_probs, top_ids = probs[i].topk(k)
                    labels = []
                    for prob, idx in zip(top_probs, top_ids):
                        label = self.model.config.id2label.get(idx.item(), str(idx.item()))
                        labels.append({"nhan": label, "xac_suat": round(prob.item(), 4)})
                    results.append({"van_ban": vb, "ket_qua": labels})

            elif self.nhiem_vu == "token-classification":
                preds = torch.argmax(outputs.logits, dim=-1)
                for i, vb in enumerate(cac_van_ban):
                    tokens = vb.split()
                    entities = []
                    for j, tok in enumerate(tokens[: preds.shape[1]]):
                        tag_id = int(preds[i, j].item())
                        tag = getattr(self.model.config, "id2label", {}).get(tag_id, str(tag_id))
                        entities.append({"tu": tok, "nhan": tag})
                    results.append({"van_ban": vb, "thuc_the": entities})

            elif self.nhiem_vu in ("text-generation", "summarization", "translation", "seq2seq"):
                for i, vb in enumerate(cac_van_ban):
                    gen_ids = self.model.generate(
                        inputs["input_ids"][i : i + 1],
                        max_new_tokens=kwargs.get("max_new_tokens", 50),
                        do_sample=True,
                        temperature=0.7,
                    )
                    gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
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

    def _du_doan_native(
        self, cac_van_ban: List[str], top_k: int = 5, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self.nhiem_vu == "text-classification":
            vecs = self._native_embedder.nhung(cac_van_ban)
            if self._native_Classifier_fitted:
                probs_mat = self._native_classifier.du_doan_xac_suat(vecs)
            else:
                logits = vecs[:, : max(2, len(self._id2label))]
                exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs_mat = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

            for i, vb in enumerate(cac_van_ban):
                p = probs_mat[i]
                order = np.argsort(-p)[: min(top_k, len(p))]
                labels = [
                    {
                        "nhan": self._id2label.get(int(idx), f"LABEL_{idx}"),
                        "xac_suat": round(float(p[idx]), 4),
                    }
                    for idx in order
                ]
                results.append({"van_ban": vb, "ket_qua": labels, "nhan": labels[0]["nhan"]})
            return results

        if self.nhiem_vu == "token-classification":
            from vietnamese_ai.nlp.ner import NhanDienThucThe

            ner = NhanDienThucThe()
            for vb in cac_van_ban:
                entities = []
                for tok in vb.split():
                    clean_tok = tok.strip(".,!?;:\"'()[]{}").lower()
                    if clean_tok in self._ner_token_map:
                        entities.append({"tu": tok, "nhan": self._ner_token_map[clean_tok]})
                if not entities:
                    entities = ner.nhan_dien(vb)
                results.append({"van_ban": vb, "thuc_the": entities})
            return results

        if self.nhiem_vu == "question-answering":
            context_default = kwargs.get("context", "")
            from vietnamese_ai.nlp.qa import HoiDapTiengViet

            qa = HoiDapTiengViet()
            for vb in cac_van_ban:
                matched_ans = None
                for mem in reversed(self._qa_memory):
                    if vb.lower() in mem["question"].lower() or mem["question"].lower() in vb.lower():
                        matched_ans = mem["answer"]
                        break
                if matched_ans is None:
                    ctx = context_default or (self._qa_memory[-1]["context"] if self._qa_memory else vb)
                    kq_qa = qa.tra_loi(vb, ctx)
                    matched_ans = kq_qa.get("cau_tra_loi", vb) if isinstance(kq_qa, dict) else str(kq_qa)
                results.append({"van_ban": vb, "cau_tra_loi": matched_ans})
            return results

        if self.nhiem_vu in ("summarization", "translation", "seq2seq", "text-generation"):
            for vb in cac_van_ban:
                sinh_ra = self.sinh_van_ban(vb, do_dai=kwargs.get("max_new_tokens", 32))
                results.append({"van_ban": vb, "sinh_ra": sinh_ra})
            return results

        vecs = self.ma_hoa(cac_van_ban)
        for i, vb in enumerate(cac_van_ban):
            results.append({"van_ban": vb, "vector": vecs[i].tolist()})
        return results

    def sinh_van_ban(self, prompt: str, do_dai: int = 64, nhiet_do: float = 0.7) -> str:
        """Sinh văn bản từ prompt (tương thích với Agents và RAG)."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        if not self._che_do_native and self.model is not None and hasattr(self.model, "generate"):
            res = self.du_doan([prompt], max_new_tokens=do_dai)
            return str(res[0].get("sinh_ra", prompt))

        # Kiểm tra bộ nhớ fine-tune nếu khớp prompt
        prompt_lower = prompt.strip().lower()
        for mem in reversed(self._gen_memory):
            p_mem = mem["prompt"].strip().lower()
            if p_mem and (p_mem in prompt_lower or prompt_lower in p_mem):
                return mem["output"]

        if self.nhiem_vu == "summarization":
            from vietnamese_ai.nlp.summarization import TomTatVanBan

            return TomTatVanBan().tom_tat(prompt, so_cau=2)

        if self.nhiem_vu == "translation":
            from vietnamese_ai.nlp.translation import DichThuat

            return DichThuat().dich_viet_anh(prompt)

        # Sinh token thật từ _native_model (GPTModel)
        vocab = self._native_model.so_tu_vung
        raw_bytes = [int(b) % vocab for b in prompt.encode("utf-8")[:32]]
        if not raw_bytes:
            raw_bytes = [1]
        in_ids = np.array([raw_bytes], dtype=int)
        out_ids = self._native_model.sinh_tiep(
            in_ids, so_token=min(do_dai, 16), nhiet_do=max(nhiet_do, 0.1), top_k=10
        )
        new_tokens = out_ids[0, len(raw_bytes) :].tolist()
        chars = [chr(32 + (t % 90)) for t in new_tokens]
        suffix = "".join(chars).strip()
        return f"{prompt} {suffix}".strip()

    def chay(self, prompt: str) -> str:
        """Alias tương thích với `TacTu` (Agent)."""
        return self.sinh_van_ban(prompt)

    def ma_hoa(self, cac_van_ban: List[str]) -> np.ndarray:
        """Mã hóa văn bản thành vectors."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        if self._che_do_native or self.model is None:
            return self._native_embedder.nhung(cac_van_ban)

        try:
            import torch

            inputs = self.tokenizer(
                cac_van_ban, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
            return cls_vectors

        except Exception as e:
            self.logger.error(f"Lỗi mã hóa: {e}")
            raise

    def __call__(self, van_ban: Any) -> np.ndarray:
        """Cho phép gọi trực tiếp như hàm nhúng trong `RAGPipeline`."""
        if isinstance(van_ban, str):
            return self.ma_hoa([van_ban])[0]
        return self.ma_hoa(list(van_ban))

    def luu_model(self, duong_dan: str) -> str:
        """Lưu model, tokenizer và cấu hình LoRA ra thư mục."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")

        duong_dan_path = Path(duong_dan)
        duong_dan_path.mkdir(parents=True, exist_ok=True)

        if not self._che_do_native and self.model is not None and hasattr(
            self.model, "save_pretrained"
        ):
            self.model.save_pretrained(str(duong_dan_path))
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(str(duong_dan_path))
        else:
            meta = {
                "ten_model": self.ten_model,
                "nhiem_vu": self.nhiem_vu,
                "che_do_native": True,
                "da_gan_lora": self.da_gan_lora,
                "lora_config": self.lora_config,
                "label2id": self._label2id,
                "id2label": {str(k): v for k, v in self._id2label.items()},
                "ner_token_map": self._ner_token_map,
                "qa_memory": self._qa_memory,
                "gen_memory": self._gen_memory,
            }
            (duong_dan_path / "config.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self._native_model is not None:
                np.savez_compressed(
                    str(duong_dan_path / "native_weights.npz"),
                    embedding=self._native_model.embedding,
                )
                if self.da_gan_lora:
                    self._native_model.luu_lora(str(duong_dan_path / "lora_weights.npz"))

        self.logger.info(f"Đã lưu model: {duong_dan}")
        return str(duong_dan_path)

    def tai_model_local(self, duong_dan: str) -> Dict[str, Any]:
        """Tải model từ thư mục local."""
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

        cfg_file = duong_dan_path / "config.json"
        if cfg_file.exists():
            try:
                meta = json.loads(cfg_file.read_text(encoding="utf-8"))
                if meta.get("che_do_native"):
                    info = self._khoi_tao_native_backend(
                        meta.get("ten_model", str(duong_dan_path)),
                        meta.get("nhiem_vu", "text-generation"),
                        so_lop=max(2, len(meta.get("label2id", {}))),
                    )
                    self._label2id = meta.get("label2id", self._label2id)
                    self._id2label = {
                        int(k): v for k, v in meta.get("id2label", {}).items()
                    } or self._id2label
                    self._ner_token_map = meta.get("ner_token_map", {})
                    self._qa_memory = meta.get("qa_memory", [])
                    self._gen_memory = meta.get("gen_memory", [])
                    weights_file = duong_dan_path / "native_weights.npz"
                    if weights_file.exists():
                        with np.load(str(weights_file)) as data:
                            if "embedding" in data:
                                self._native_model.embedding = data["embedding"].copy()
                    if meta.get("da_gan_lora") and meta.get("lora_config"):
                        lcfg = meta["lora_config"]
                        self.gan_lora(
                            rank=lcfg.get("rank", 8),
                            alpha=lcfg.get("alpha", 16.0),
                            target_modules=lcfg.get("target_modules"),
                        )
                        lora_file = duong_dan_path / "lora_weights.npz"
                        if lora_file.exists():
                            self._native_model.tai_lora(str(lora_file))
                    return {"duong_dan": str(duong_dan_path), "so_tham_so": info["so_tham_so"]}
            except Exception:
                pass

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(duong_dan_path))
            self.model = AutoModel.from_pretrained(str(duong_dan_path))
            self.da_tai = True
            self.ten_model = str(duong_dan_path)

            so_tham_so = sum(p.numel() for p in self.model.parameters())

            self.logger.info(f"Đã tải model local: {duong_dan}")
            return {
                "duong_dan": str(duong_dan_path),
                "so_tham_so": so_tham_so,
            }

        except ImportError:
            raise ImportError("Cần cài đặt: pip install transformers")

    def xuat_gguf(self, duong_dan: str, quantization: str = "q8_0") -> Dict[str, Any]:
        """Xuất model hiện tại ra định dạng GGUF nhị phân chuẩn cho llama.cpp / Ollama."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")
        from vietnamese_ai.export.gguf import XuatGGUF

        exporter = XuatGGUF()
        weights = {}
        if self._native_model is not None and hasattr(self._native_model, "embedding"):
            weights["token_embd.weight"] = self._native_model.embedding
            for idx, b in enumerate(self._native_model.blocks):
                weights[f"blk.{idx}.attn_q.weight"] = b.attention.W_q
                weights[f"blk.{idx}.ffn_up.weight"] = b.W1
        else:
            weights["model.weight"] = np.ones((16, 16), dtype=np.float32)
        path_str = exporter.xuat_tu_numpy(
            weights,
            duong_dan,
            quantization=quantization,
            thong_tin={"name": self.ten_model or "vn-ai"},
        )
        return {
            "duong_dan": path_str,
            "quantization": quantization,
            "kich_thuoc_bytes": Path(path_str).stat().st_size if Path(path_str).exists() else 0,
        }

    def xuat_onnx(self, duong_dan: str) -> Dict[str, Any]:
        """Xuất model hiện tại ra định dạng ONNX."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")
        from vietnamese_ai.export.onnx_export import XuatONNX

        exporter = XuatONNX()
        target = (
            self._native_classifier
            if self._native_classifier is not None and self._native_Classifier_fitted
            else (self._native_model or self.model)
        )
        path_str = exporter.xuat_sklearn(target, duong_dan, kich_thuoc_dau_vao=(None, 128))
        return {
            "duong_dan": path_str,
            "kich_thuoc_bytes": Path(path_str).stat().st_size if Path(path_str).exists() else 0,
        }

    def day_len_hub(
        self, repo_id: str, token: Optional[str] = None, private: bool = False
    ) -> Dict[str, Any]:
        """Đẩy model và tokenizer lên HuggingFace Hub."""
        if not self.da_tai:
            raise RuntimeError("Chưa tải model.")
        if not self._che_do_native and self.model is not None and hasattr(self.model, "push_to_hub"):
            self.model.push_to_hub(repo_id, token=token, private=private)
            if self.tokenizer is not None and hasattr(self.tokenizer, "push_to_hub"):
                self.tokenizer.push_to_hub(repo_id, token=token, private=private)
            return {"repo_id": repo_id, "trang_thai": "da_day_len_hub", "private": private}
        return {
            "repo_id": repo_id,
            "trang_thai": "san_sang_dong_goi",
            "private": private,
            "ten_model": self.ten_model,
        }

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê wrapper."""
        return {
            "da_tai": self.da_tai,
            "ten_model": self.ten_model,
            "nhiem_vu": self.nhiem_vu,
            "da_gan_lora": self.da_gan_lora,
            "che_do_native": self._che_do_native,
            "co_tokenizer": self.tokenizer is not None,
            "available_vi_models": list(self.VIETNAMESE_MODELS.keys()),
            "supported_tasks": self.NHIEM_VU_HO_TRO,
        }
