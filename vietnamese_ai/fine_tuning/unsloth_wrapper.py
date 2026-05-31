"""UnslothWrapper - Tích hợp Unsloth cho fine-tune LLM nhanh."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger

try:
    import torch

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class UnslothWrapper:
    """
    Wrapper cho Unsloth - fine-tune LLM nhanh gấp 2x.

    Unsloth tối ưu hóa training loop với:
    - Flash Attention 2
    - Gradient Checkpointing tối ưu
    - LoRA/QLoRA native support
    - 8-bit / 4-bit quantization

    Yêu cầu: pip install unsloth

    Sử dụng:
        >>> unsloth = UnslothWrapper()
        >>> unsloth.tai_mo_hinh("unsloth/llama-3-8b-bnb-4bit")
        >>> unsloth.config_lora(r=16, lora_alpha=16)
        >>> unsloth.fine_tune(datasets, so_vong=3)
    """

    UNSLOTH_MODELS = {
        "llama3-8b": "unsloth/llama-3-8b-bnb-4bit",
        "llama3-70b": "unsloth/llama-3-70b-bnb-4bit",
        "mistral-7b": "unsloth/mistral-7b-v0.3-bnb-4bit",
        "qwen2-7b": "unsloth/qwen2-7b-bnb-4bit",
        "gemma-7b": "unsloth/gemma-7b-bnb-4bit",
        "phi3-mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    }

    def __init__(self):
        self.logger = Logger("UnslothWrapper")
        self._model = None
        self._tokenizer = None
        self._lora_config: Dict[str, Any] = {}
        self._co_unsloth = False
        self._da_tai = False
        self._da_lora = False

        try:
            import unsloth  # noqa: F401

            self._co_unsloth = True
            self.logger.info("Unsloth đã sẵn sàng")
        except ImportError:
            self.logger.warning(
                "Unsloth chưa cài. Cài đặt: pip install unsloth "
                "Sử dụng HuggingFaceWrapper làm fallback."
            )

    @property
    def co_unsloth(self) -> bool:
        return self._co_unsloth

    def danh_sach_models(self) -> Dict[str, str]:
        """Liệt kê models hỗ trợ."""
        return self.UNSLOTH_MODELS.copy()

    def tai_mo_hinh(
        self,
        ten_model: str,
        max_seq_length: int = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
    ) -> Dict[str, Any]:
        """
        Tải model với Unsloth.

        Args:
            ten_model: Tên model (HF repo hoặc key từ danh_sach_models)
            max_seq_length: Độ dài sequence tối đa
            dtype: Data type (None = auto, "float16", "bfloat16")
            load_in_4bit: Load 4-bit quantization

        Returns:
            Dict chứa thông tin model
        """
        if not self._co_unsloth:
            raise ImportError("Unsloth chưa cài. Cài đặt: pip install unsloth")

        model_id = self.UNSLOTH_MODELS.get(ten_model, ten_model)

        self.logger.info(f"Đang tải model: {model_id}")

        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            self._model = model
            self._tokenizer = tokenizer
            self._da_tai = True

            info = {
                "ten_model": model_id,
                "max_seq_length": max_seq_length,
                "load_in_4bit": load_in_4bit,
                "vocab_size": len(tokenizer) if tokenizer else 0,
                "thiet_bi": str(next(model.parameters()).device) if model else "unknown",
            }

            self.logger.info(f"Đã tải model: {model_id}")
            return info

        except Exception as e:
            self.logger.error(f"Lỗi tải model: {e}")
            raise

    def config_lora(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ) -> Dict[str, Any]:
        """
        Cấu hình LoRA adapter.

        Args:
            r: Rank của LoRA (8, 16, 32, 64)
            lora_alpha: Scaling factor
            lora_dropout: Dropout rate
            target_modules: Các module áp dụng LoRA
            bias: Bias handling ("none", "all", "lora_only")

        Returns:
            Dict chứa LoRA config
        """
        if not self._da_tai:
            raise RuntimeError("Chưa tải model. Gọi tai_mo_hinh() trước.")

        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        self._lora_config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": bias,
        }

        try:
            from unsloth import FastLanguageModel

            self._model = FastLanguageModel.get_peft_model(
                self._model,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=bias,
            )
            self._da_lora = True

            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self._model.parameters())

            self.logger.info(
                f"LoRA: r={r}, alpha={lora_alpha}, "
                f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.1f}%)"
            )

            return {
                **self._lora_config,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percent": round(100 * trainable / total, 2),
            }

        except Exception as e:
            self.logger.error(f"Lỗi config LoRA: {e}")
            raise

    def fine_tune(
        self,
        datasets: Any,
        so_vong: int = 3,
        batch_size: int = 2,
        gradient_accumulation: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 5,
        max_steps: int = -1,
        logging_steps: int = 1,
        save_steps: int = 100,
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        """
        Fine-tune model với SFTTrainer.

        Args:
            datasets: HuggingFace Dataset hoặc dict
            so_vong: Số epochs
            batch_size: Batch size per device
            gradient_accumulation: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            max_steps: Max training steps (-1 = all)
            logging_steps: Logging interval
            save_steps: Save checkpoint interval
            output_dir: Output directory

        Returns:
            Dict chứa training results
        """
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")
        if not self._da_lora:
            self.logger.warning("Chưa config LoRA. Gọi config_lora() trước.")

        try:
            from transformers import TrainingArguments
            from trl import SFTTrainer

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=so_vong,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                optim="adamw_8bit",
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=datasets,
                args=training_args,
                max_seq_length=2048,
            )

            self.logger.info(f"Bắt đầu fine-tune ({so_vong} epochs)")
            bat_dau = time.time()
            result = trainer.train()
            tong_thoi_gian = time.time() - bat_dau

            self.logger.info(f"Fine-tune hoàn tất ({tong_thoi_gian:.1f}s)")

            return {
                "train_loss": result.training_loss,
                "tong_thoi_gian": round(tong_thoi_gian, 2),
                "so_epoch": so_vong,
                "global_step": result.global_step,
            }

        except ImportError:
            raise ImportError("Cần cài đặt trl: pip install trl")

    def luu_model(self, duong_dan: str) -> str:
        """Lưu model đã fine-tune."""
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")

        duong_dan_path = Path(duong_dan)
        duong_dan_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(duong_dan_path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(duong_dan_path))

        self.logger.info(f"Đã lưu model: {duong_dan}")
        return str(duong_dan_path)

    def xuat_gguf(self, duong_dan: str, quantization: str = "q4_k_m") -> str:
        """
        Xuất model sang GGUF format cho llama.cpp.

        Args:
            duong_dan: Đường dẫn file .gguf
            quantization: Loại quantization (q4_k_m, q5_k_m, q8_0, f16)

        Returns:
            Đường dẫn file GGUF
        """
        if not self._da_tai:
            raise RuntimeError("Chưa tải model.")

        try:
            self._model.save_pretrained_gguf(
                duong_dan,
                self._tokenizer,
                quantization_method=quantization,
            )
            self.logger.info(f"Đã xuất GGUF: {duong_dan} ({quantization})")
            return duong_dan
        except Exception as e:
            self.logger.error(f"Lỗi xuất GGUF: {e}")
            raise

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê wrapper."""
        return {
            "co_unsloth": self._co_unsloth,
            "da_tai": self._da_tai,
            "da_lora": self._da_lora,
            "lora_config": self._lora_config,
            "available_models": list(self.UNSLOTH_MODELS.keys()),
        }
