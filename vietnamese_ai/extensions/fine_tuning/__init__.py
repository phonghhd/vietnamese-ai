"""Extension Fine-Tuning - LoRA, QLoRA, DoRA, SFT, DPO, GRPO, RLHF (extra: `dl` / `torch` / `fast`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "AutoFineTuner": ("vietnamese_ai.fine_tuning.auto_finetuner", "AutoFineTuner"),
    "BoTinhChinhMoHinh": ("vietnamese_ai.fine_tuning.auto_finetuner", "BoTinhChinhMoHinh"),
    "InstructionDataset": ("vietnamese_ai.fine_tuning.dataset", "InstructionDataset"),
    "PEFTConfig": ("vietnamese_ai.fine_tuning.peft_config", "PEFTConfig"),
    "HuggingFaceWrapper": ("vietnamese_ai.fine_tuning.hf_wrapper", "HuggingFaceWrapper"),
    "UnslothWrapper": ("vietnamese_ai.fine_tuning.unsloth_wrapper", "UnslothWrapper"),
    "LoRALayer": ("vietnamese_ai.fine_tuning.lora", "LoRALayer"),
    "LoRAAdapter": ("vietnamese_ai.fine_tuning.lora", "LoRAAdapter"),
    "QLoRAAdapter": ("vietnamese_ai.fine_tuning.lora", "QLoRAAdapter"),
    "DoRALayer": ("vietnamese_ai.fine_tuning.lora", "DoRALayer"),
    "LoRAPeft": ("vietnamese_ai.fine_tuning.lora_peft", "LoRAPeft"),
    "HuanLuyenPyTorch": ("vietnamese_ai.fine_tuning.pytorch_trainer", "HuanLuyenPyTorch"),
    "SFTTrainer": ("vietnamese_ai.fine_tuning.sft_trainer", "SFTTrainer"),
    "DPOTrainer": ("vietnamese_ai.fine_tuning.dpo_trainer", "DPOTrainer"),
    "GRPOTrainer": ("vietnamese_ai.fine_tuning.grpo_trainer", "GRPOTrainer"),
    "RewardModel": ("vietnamese_ai.fine_tuning.reward_model", "RewardModel"),
    "RLHFPipeline": ("vietnamese_ai.fine_tuning.rlhf_pipeline", "RLHFPipeline"),
    "InstructionTuningTrainer": (
        "vietnamese_ai.fine_tuning.instruction_trainer",
        "InstructionTuningTrainer",
    ),
    "BoTangTocHuanLuyen": (
        "vietnamese_ai.fine_tuning.training_accelerator",
        "BoTangTocHuanLuyen",
    ),
    "TrainingAccelerator": (
        "vietnamese_ai.fine_tuning.training_accelerator",
        "TrainingAccelerator",
    ),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
