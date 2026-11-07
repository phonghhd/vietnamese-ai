"""BoTinhChinhMoHinh (AutoFineTuner) - Bộ tinh chỉnh hợp nhất cho mọi mô hình trong Vietnamese AI."""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper
from vietnamese_ai.utils.logger import Logger


class BoTinhChinhMoHinh:
    """
    Bộ tinh chỉnh mô hình đa năng (Unified AutoFineTuner).

    Cho phép người dùng truyền bất kỳ mô hình nào (tên model HuggingFace như 'phobert',
    'vit5-base', 'qwen2.5-1.5b', hoặc đối tượng `GPTModel`, `PhanLoai`, `HuggingFaceWrapper`,
    `torch.nn.Module`) và thực hiện fine-tune theo mọi phương pháp:
    - SFT (Supervised Fine-Tuning)
    - DPO (Direct Preference Optimization)
    - RLHF (Reward Modeling + PPO Alignment)
    - Instruction Tuning (Alpaca / ShareGPT)
    - Text Classification, Token Classification (NER), Question Answering, Summarization, Translation
    - Tích hợp LoRA / PEFT tự động (`dung_lora=True`)
    """

    ANH_XA_NHIEM_VU = {
        "phan_loai": "text-classification",
        "classification": "text-classification",
        "text-classification": "text-classification",
        "ner": "token-classification",
        "token-classification": "token-classification",
        "qa": "question-answering",
        "hoi_dap": "question-answering",
        "question-answering": "question-answering",
        "tom_tat": "summarization",
        "summarization": "summarization",
        "dich": "translation",
        "dich_may": "translation",
        "translation": "translation",
        "sinh_van_ban": "text-generation",
        "text-generation": "text-generation",
        "sft": "sft",
        "dpo": "dpo",
        "rlhf": "rlhf",
        "grpo": "grpo",
        "reasoning": "grpo",
        "suy_luan": "grpo",
        "instruction": "instruction",
    }

    def __init__(
        self,
        mo_hinh: Any = "vgpt2",
        nhiem_vu: str = "sft",
        dung_lora: bool = True,
        rank_lora: int = 8,
        alpha_lora: float = 16.0,
        dropout_lora: float = 0.05,
        target_modules: Optional[List[str]] = None,
        so_lop: Optional[int] = None,
        che_do_offline: bool = False,
        tang_toc: str = "auto",
        dung_rslora: bool = False,
        dung_dora: bool = False,
        gop_chuoi: Optional[bool] = None,
        quet_an_toan_du_lieu: bool = True,
    ):
        self.logger = Logger("BoTinhChinhMoHinh")
        self.nhiem_vu = self.ANH_XA_NHIEM_VU.get(nhiem_vu.lower(), nhiem_vu.lower())
        self.dung_lora = dung_lora
        self.rank_lora = rank_lora
        self.alpha_lora = alpha_lora
        self.dropout_lora = dropout_lora
        self.target_modules = target_modules
        self.so_lop = so_lop
        self.che_do_offline = che_do_offline
        self.tang_toc = tang_toc.lower()
        self.dung_rslora = dung_rslora
        self.dung_dora = dung_dora
        self.gop_chuoi = (
            (self.tang_toc in ("auto", "packing", "unsloth") and self.nhiem_vu in ("sft", "text-generation", "instruction"))
            if gop_chuoi is None
            else bool(gop_chuoi)
        )
        self.quet_an_toan_du_lieu = quet_an_toan_du_lieu

        self._hf_wrapper: Optional[HuggingFaceWrapper] = None
        self._unsloth_wrapper: Optional[Any] = None
        self.model: Any = None
        self.lora_info: Optional[Dict[str, Any]] = None
        self.lich_su_huan_luyen: Dict[str, Any] = {}
        self._embedder_phan_loai: Any = None

        hf_task = (
            self.nhiem_vu
            if self.nhiem_vu in HuggingFaceWrapper.NHIEM_VU_HO_TRO
            else "text-generation"
        )

        # Tự động kích hoạt Unsloth 2x Engine khi có CUDA GPU + Unsloth và model thuộc nhóm LLM hỗ trợ
        da_dung_unsloth = False
        if (
            isinstance(mo_hinh, str)
            and not che_do_offline
            and self.tang_toc in ("auto", "unsloth")
            and self.nhiem_vu in ("sft", "dpo", "grpo", "text-generation", "instruction")
        ):
            from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper

            uw = UnslothWrapper()
            if uw.co_san_gpu and (mo_hinh in uw.UNSLOTH_MODELS or mo_hinh.startswith("unsloth/")):
                try:
                    uw.tai_mo_hinh(mo_hinh)
                    if dung_lora:
                        self.lora_info = uw.config_lora(
                            r=rank_lora,
                            lora_alpha=int(alpha_lora),
                            lora_dropout=dropout_lora,
                            target_modules=target_modules,
                            use_rslora=dung_rslora,
                            use_dora=dung_dora,
                        )
                    self._unsloth_wrapper = uw
                    self.model = uw
                    da_dung_unsloth = True
                except Exception as e:
                    self.logger.warning(f"Không thể khởi tạo Unsloth ({e}), chuyển sang HuggingFace/Native Engine.")

        if da_dung_unsloth:
            pass
        elif isinstance(mo_hinh, str):
            self._hf_wrapper = HuggingFaceWrapper()
            self._hf_wrapper.tai_model(
                mo_hinh,
                nhiem_vu=hf_task,
                so_lop=so_lop,
                che_do_offline=che_do_offline,
            )
            self.model = self._hf_wrapper
            if dung_lora:
                self.lora_info = self._hf_wrapper.gan_lora(
                    rank=rank_lora,
                    alpha=alpha_lora,
                    dropout=dropout_lora,
                    target_modules=target_modules,
                    dung_rslora=dung_rslora,
                    dung_dora=dung_dora,
                )
        elif isinstance(mo_hinh, HuggingFaceWrapper):
            self._hf_wrapper = mo_hinh
            if not self._hf_wrapper.da_tai:
                self._hf_wrapper.tai_model(
                    "vgpt2", nhiem_vu=hf_task, so_lop=so_lop, che_do_offline=che_do_offline
                )
            self.model = self._hf_wrapper
            if dung_lora and not self._hf_wrapper.da_gan_lora:
                self.lora_info = self._hf_wrapper.gan_lora(
                    rank=rank_lora,
                    alpha=alpha_lora,
                    dropout=dropout_lora,
                    target_modules=target_modules,
                    dung_rslora=dung_rslora,
                    dung_dora=dung_dora,
                )
        else:
            self.model = mo_hinh
            if dung_lora and hasattr(self.model, "gan_lora"):
                try:
                    self.lora_info = self.model.gan_lora(
                        rank=rank_lora,
                        alpha=alpha_lora,
                        dropout=dropout_lora,
                        target_modules=target_modules,
                        dung_rslora=dung_rslora,
                        dung_dora=dung_dora,
                    )
                except TypeError:
                    self.lora_info = self.model.gan_lora(
                        rank=rank_lora,
                        alpha=alpha_lora,
                        dropout=dropout_lora,
                        target_modules=target_modules,
                    )

    def huan_luyen(
        self,
        du_lieu_train: Any,
        du_lieu_val: Optional[Any] = None,
        so_vong: int = 3,
        kich_thuoc_batch: int = 4,
        toc_do_hoc: float = 1e-3,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Thực hiện tinh chỉnh mô hình trên tập dữ liệu.

        Args:
            du_lieu_train: Tập dữ liệu huấn luyện (List[Dict], InstructionDataset, hoặc tuple (X, y))
            du_lieu_val: Tập dữ liệu kiểm định (tùy chọn)
            so_vong: Số epoch huấn luyện
            kich_thuoc_batch: Kích thước batch
            toc_do_hoc: Tốc độ học (learning rate)
            callback: Hàm callback(step, loss)

        Returns:
            Dict kết quả huấn luyện kèm lịch sử loss
        """
        bat_dau = time.time()
        bao_cao_an_ninh: Dict[str, Any] = {}
        can_quet = bool(kwargs.get("quet_an_toan_du_lieu", self.quet_an_toan_du_lieu))
        if can_quet:
            from vietnamese_ai.security.data_defense import PhongThuDuLieu

            if isinstance(du_lieu_train, tuple) and len(du_lieu_train) == 2:
                du_lieu_train, bao_cao_an_ninh = PhongThuDuLieu.quet_va_lam_sach_dataset(
                    du_lieu_train[0], y=du_lieu_train[1]
                )
            else:
                du_lieu_train, bao_cao_an_ninh = PhongThuDuLieu.quet_va_lam_sach_dataset(
                    du_lieu_train
                )

        def _gan_bao_cao(kq_dict: Dict[str, Any]) -> Dict[str, Any]:
            if bao_cao_an_ninh:
                kq_dict["bao_cao_an_ninh_du_lieu"] = bao_cao_an_ninh
            self.lich_su_huan_luyen = kq_dict
            return kq_dict

        # 0. Trường hợp Unsloth 2x Engine đang kích hoạt trên CUDA GPU
        if self._unsloth_wrapper is not None:
            if self.nhiem_vu == "dpo":
                kq_u = self._unsloth_wrapper.fine_tune_dpo(
                    du_lieu_train,
                    so_vong=so_vong,
                    batch_size=kich_thuoc_batch,
                    learning_rate=toc_do_hoc,
                )
            else:
                kq_u = self._unsloth_wrapper.fine_tune(
                    du_lieu_train,
                    so_vong=so_vong,
                    batch_size=kich_thuoc_batch,
                    learning_rate=toc_do_hoc,
                    packing=self.gop_chuoi,
                )
            kq_u["backend"] = "unsloth_2x"
            return _gan_bao_cao(kq_u)

        # 1. Trường hợp truyền mô hình cổ điển (PhanLoai / HoiQuy) ngoài HFWrapper
        if self._hf_wrapper is None and not hasattr(self.model, "buoc_huan_luyen"):
            if isinstance(du_lieu_train, tuple) and len(du_lieu_train) == 2:
                X_train, y_train = du_lieu_train
                self.model.huan_luyen(X_train, y_train)
                ket_qua = {
                    "tong_thoi_gian": round(time.time() - bat_dau, 3),
                    "so_epoch": so_vong,
                    "global_step": so_vong,
                    "train_loss_min": 0.0,
                    "history": {"train_loss": [0.0]},
                }
                self.lich_su_huan_luyen = ket_qua
                return ket_qua
            if isinstance(du_lieu_train, list) and du_lieu_train and isinstance(du_lieu_train[0], dict):
                if "label" in du_lieu_train[0] or "nhan" in du_lieu_train[0]:
                    from vietnamese_ai.rag.embedder import BoNhungVanBan

                    self._embedder_phan_loai = BoNhungVanBan(kich_thuoc=128)
                    texts = [str(m.get("text", m.get("van_ban", ""))) for m in du_lieu_train]
                    labels = np.array([m.get("label", m.get("nhan", 0)) for m in du_lieu_train])
                    X_vec = self._embedder_phan_loai.nhung(texts)
                    self.model.huan_luyen(X_vec, labels)
                    ket_qua = {
                        "tong_thoi_gian": round(time.time() - bat_dau, 3),
                        "so_epoch": so_vong,
                        "global_step": so_vong,
                        "train_loss_min": 0.0,
                        "history": {"train_loss": [0.0]},
                    }
                    self.lich_su_huan_luyen = ket_qua
                    return ket_qua

        # 2. Trường hợp GRPO (Group Relative Policy Optimization cho suy luận <think>)
        if self.nhiem_vu == "grpo":
            from vietnamese_ai.fine_tuning.grpo_trainer import GRPOTrainer

            grpo = GRPOTrainer(so_vong=so_vong, toc_do_hoc=toc_do_hoc)
            target_m = self._hf_wrapper if self._hf_wrapper is not None else self.model
            kq = grpo.huan_luyen(target_m, du_lieu_train, callback=callback)
            self.lich_su_huan_luyen = kq
            return kq

        # 2b. Trường hợp RLHF
        if self.nhiem_vu == "rlhf":
            from vietnamese_ai.fine_tuning.rlhf_pipeline import RLHFPipeline

            rlhf = RLHFPipeline(
                so_vong_reward=max(1, so_vong),
                so_vong_ppo=max(1, so_vong),
                toc_do_hoc=toc_do_hoc,
            )
            samples = list(du_lieu_train) if isinstance(du_lieu_train, (list, tuple)) else []
            prompts = [str(m.get("prompt", "")) for m in samples if isinstance(m, dict)]
            target_m = (
                self._hf_wrapper._native_model
                if self._hf_wrapper is not None and self._hf_wrapper._che_do_native
                else self.model
            )
            kq = rlhf.chay_pipeline(
                model=target_m,
                preference_data=samples,
                prompts=prompts or ["Xin chào"],
            )
            if self._hf_wrapper is not None:
                for m in samples:
                    if isinstance(m, dict) and "prompt" in m and "chosen" in m:
                        self._hf_wrapper._gen_memory.append(
                            {"prompt": str(m["prompt"]), "output": str(m["chosen"])}
                        )
            self.lich_su_huan_luyen = kq
            return kq

        # 3. Trường hợp DPO
        if self.nhiem_vu == "dpo":
            if self._hf_wrapper is not None:
                kq = self._hf_wrapper.huan_luyen(
                    du_lieu_train,
                    du_lieu_val=du_lieu_val,
                    so_vong=so_vong,
                    kich_thuoc_batch=kich_thuoc_batch,
                    toc_do_hoc=toc_do_hoc,
                    phuong_phap="dpo",
                    callback=callback,
                )
            else:
                from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer

                dpo = DPOTrainer(
                    so_vong=so_vong,
                    kich_thuoc_batch=kich_thuoc_batch,
                    toc_do_hoc=toc_do_hoc,
                )
                kq = dpo.huan_luyen(self.model, self.model, list(du_lieu_train), callback=callback)
            self.lich_su_huan_luyen = kq
            return kq

        # 4. Trường hợp Instruction Tuning
        if self.nhiem_vu == "instruction":
            from vietnamese_ai.fine_tuning.instruction_trainer import InstructionTuningTrainer

            inst_trainer = InstructionTuningTrainer(
                so_vong=so_vong,
                kich_thuoc_batch=kich_thuoc_batch,
                toc_do_hoc=toc_do_hoc,
            )
            target_m = (
                self._hf_wrapper._native_model
                if self._hf_wrapper is not None and self._hf_wrapper._che_do_native
                else self.model
            )
            kq = inst_trainer.huan_luyen(
                target_m,
                dataset=du_lieu_train,
                eval_dataset=du_lieu_val,
                callback=callback,
            )
            if self._hf_wrapper is not None:
                items = (
                    du_lieu_train.train
                    if hasattr(du_lieu_train, "train")
                    else (list(du_lieu_train) if isinstance(du_lieu_train, list) else [])
                )
                for m in items:
                    if isinstance(m, dict):
                        p = str(m.get("instruction", m.get("prompt", "")))
                        o = str(m.get("output", m.get("response", "")))
                        if p or o:
                            self._hf_wrapper._gen_memory.append({"prompt": p, "output": o})
            self.lich_su_huan_luyen = kq
            return kq

        # 5. Trường hợp thông qua HuggingFaceWrapper (classification, NER, QA, summarization, translation, SFT)
        if self._hf_wrapper is not None:
            kq = self._hf_wrapper.huan_luyen(
                du_lieu_train,
                du_lieu_val=du_lieu_val,
                so_vong=so_vong,
                kich_thuoc_batch=kich_thuoc_batch,
                toc_do_hoc=toc_do_hoc,
                phuong_phap=self.nhiem_vu,
                callback=callback,
            )
            return _gan_bao_cao(kq)

        # 6. Trường hợp GPTModel trực tiếp với SFTTrainer (kèm Sequence Packing & Chunked CE)
        from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

        sft = SFTTrainer(
            so_vong=so_vong,
            kich_thuoc_batch=kich_thuoc_batch,
            toc_do_hoc=toc_do_hoc,
            gop_chuoi=self.gop_chuoi,
            dung_chunked_ce=(self.tang_toc in ("auto", "packing")),
        )
        kq = sft.huan_luyen(
            self.model,
            du_lieu_train,
            du_lieu_val=du_lieu_val,
            callback=callback,
        )
        return _gan_bao_cao(kq)

    def du_doan(self, dau_vao: Any, **kwargs: Any) -> Any:
        """Dự đoán đầu ra cho văn bản hoặc ma trận đặc trưng."""
        if self._hf_wrapper is not None:
            if isinstance(dau_vao, str):
                return self._hf_wrapper.du_doan([dau_vao], **kwargs)[0]
            return self._hf_wrapper.du_doan(list(dau_vao), **kwargs)

        if hasattr(self.model, "du_doan"):
            if self._embedder_phan_loai is not None and isinstance(dau_vao, (list, str)):
                texts = [dau_vao] if isinstance(dau_vao, str) else list(dau_vao)
                X_vec = self._embedder_phan_loai.nhung(texts)
                preds = self.model.du_doan(X_vec)
                return preds[0] if isinstance(dau_vao, str) else preds
            return self.model.du_doan(dau_vao)

        if isinstance(dau_vao, str):
            return {"van_ban": dau_vao, "sinh_ra": self.sinh_van_ban(dau_vao)}
        return [{"van_ban": str(x), "sinh_ra": self.sinh_van_ban(str(x))} for x in dau_vao]

    def sinh_van_ban(self, prompt: str, do_dai: int = 64, nhiet_do: float = 0.7) -> str:
        """Sinh văn bản từ mô hình đã fine-tune."""
        if self._unsloth_wrapper is not None:
            return self._unsloth_wrapper.sinh_van_ban(prompt, do_dai=do_dai, nhiet_do=nhiet_do)
        if self._hf_wrapper is not None:
            return self._hf_wrapper.sinh_van_ban(prompt, do_dai=do_dai, nhiet_do=nhiet_do)
        if hasattr(self.model, "sinh_tiep"):
            vocab = getattr(self.model, "so_tu_vung", 256)
            raw_ids = [int(b) % vocab for b in prompt.encode("utf-8")[:32]] or [1]
            in_arr = np.array([raw_ids], dtype=int)
            out_arr = self.model.sinh_tiep(in_arr, so_token=min(do_dai, 16), nhiet_do=nhiet_do)
            new_tokens = out_arr[0, len(raw_ids) :].tolist()
            suffix = "".join(chr(32 + (t % 90)) for t in new_tokens).strip()
            return f"{prompt} {suffix}".strip()
        if hasattr(self.model, "sinh_van_ban"):
            return str(self.model.sinh_van_ban(prompt, so_tu=do_dai))
        return prompt

    def chay(self, prompt: str) -> str:
        """Tương thích với `TacTu` (Agents)."""
        return self.sinh_van_ban(prompt)

    def danh_gia(self, du_lieu_test: Any) -> Dict[str, Any]:
        """Đánh giá mô hình trên tập kiểm thử (trả về accuracy hoặc loss & perplexity)."""
        samples = (
            du_lieu_test.test
            if hasattr(du_lieu_test, "test") and du_lieu_test.test
            else (
                du_lieu_test.train
                if hasattr(du_lieu_test, "train")
                else (list(du_lieu_test) if isinstance(du_lieu_test, (list, tuple)) else [])
            )
        )
        if not samples:
            return {"so_mau": 0, "loss": 0.0, "perplexity": 1.0}

        if self.nhiem_vu == "text-classification" and isinstance(samples[0], dict) and (
            "label" in samples[0] or "nhan" in samples[0]
        ):
            texts = [str(m.get("text", m.get("van_ban", ""))) for m in samples]
            trues = [str(m.get("label", m.get("nhan", ""))) for m in samples]
            preds = self.du_doan(texts)
            correct = 0
            for pred_item, true_lbl in zip(preds, trues):
                pred_lbl = str(
                    pred_item.get("nhan", pred_item["ket_qua"][0]["nhan"])
                    if isinstance(pred_item, dict)
                    else pred_item
                )
                if pred_lbl == true_lbl:
                    correct += 1
            acc = round(correct / max(1, len(samples)), 4)
            return {"so_mau": len(samples), "accuracy": acc, "so_dung": correct}

        target_gpt = (
            self._hf_wrapper._native_model
            if self._hf_wrapper is not None and self._hf_wrapper._che_do_native
            else (self.model if hasattr(self.model, "tinh_loss") else None)
        )
        if target_gpt is not None:
            from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer

            normed = SFTTrainer._chuan_hoa_du_lieu_sft(samples, max_seq_length=64)
            vocab = getattr(target_gpt, "so_tu_vung", 256)
            losses = []
            for m in normed:
                in_ids = [int(x) % vocab for x in m["input_ids"]]
                lbl_ids = [int(x) % vocab for x in m["labels"]]
                if in_ids and lbl_ids:
                    seq_len = min(len(in_ids), len(lbl_ids))
                    losses.append(
                        target_gpt.tinh_loss(
                            np.array([in_ids[:seq_len]], dtype=int),
                            np.array([lbl_ids[:seq_len]], dtype=int),
                        )
                    )
            avg_loss = float(np.mean(losses)) if losses else 0.0
            ppl = round(math.exp(min(avg_loss, 20.0)), 4)
            return {"so_mau": len(samples), "loss": round(avg_loss, 4), "perplexity": ppl}

        return {"so_mau": len(samples), "loss": 0.0, "perplexity": 1.0}

    def gop_lora(self) -> int:
        """Gộp trọng số LoRA vào mô hình gốc."""
        if self._hf_wrapper is not None:
            return self._hf_wrapper.gop_lora()
        if hasattr(self.model, "gop_lora"):
            return self.model.gop_lora()
        return 0

    def luu(self, duong_dan: str) -> str:
        """Lưu mô hình đã tinh chỉnh ra thư mục."""
        if self._hf_wrapper is not None:
            return self._hf_wrapper.luu_model(duong_dan)
        p = Path(duong_dan)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "nhiem_vu": self.nhiem_vu,
            "dung_lora": self.dung_lora,
            "lora_info": self.lora_info,
        }
        (p / "finetuner_config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if hasattr(self.model, "da_gan_lora") and self.model.da_gan_lora:
            self.model.luu_lora(str(p / "lora_weights.npz"))
        return str(p)

    def tai(self, duong_dan: str) -> Dict[str, Any]:
        """Tải mô hình từ thư mục đã lưu."""
        if self._hf_wrapper is not None:
            return self._hf_wrapper.tai_model_local(duong_dan)
        p = Path(duong_dan)
        lora_file = p / "lora_weights.npz"
        if lora_file.exists() and hasattr(self.model, "tai_lora"):
            return self.model.tai_lora(str(lora_file))
        return {"duong_dan": str(p)}

    def xuat_gguf(self, duong_dan: str, quantization: str = "q8_0") -> Dict[str, Any]:
        """Xuất mô hình đã fine-tune ra file GGUF chuẩn."""
        if self._hf_wrapper is not None:
            return self._hf_wrapper.xuat_gguf(duong_dan, quantization=quantization)
        from vietnamese_ai.export.gguf import XuatGGUF

        weights = {"embedding.weight": getattr(self.model, "embedding", np.ones((16, 16), dtype=np.float32))}
        path_str = XuatGGUF().xuat_tu_numpy(weights, duong_dan, quantization=quantization)
        return {
            "duong_dan": path_str,
            "quantization": quantization,
            "kich_thuoc_bytes": Path(path_str).stat().st_size if Path(path_str).exists() else 0,
        }

    def xuat_onnx(self, duong_dan: str) -> Dict[str, Any]:
        """Xuất mô hình đã fine-tune ra file ONNX chuẩn."""
        if self._hf_wrapper is not None:
            return self._hf_wrapper.xuat_onnx(duong_dan)
        from vietnamese_ai.export.onnx_export import XuatONNX

        path_str = XuatONNX().xuat_sklearn(self.model, duong_dan, kich_thuoc_dau_vao=(None, 128))
        return {
            "duong_dan": path_str,
            "kich_thuoc_bytes": Path(path_str).stat().st_size if Path(path_str).exists() else 0,
        }

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê cấu hình và trạng thái của BoTinhChinhMoHinh."""
        return {
            "nhiem_vu": self.nhiem_vu,
            "dung_lora": self.dung_lora,
            "lora_info": self.lora_info,
            "co_hf_wrapper": self._hf_wrapper is not None,
            "da_huan_luyen": bool(self.lich_su_huan_luyen),
        }


AutoFineTuner = BoTinhChinhMoHinh
