"""RLHFPipeline - Complete RLHF (Reinforcement Learning from Human Feedback)."""

import time
from typing import Any, Dict, List, Optional

from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer
from vietnamese_ai.fine_tuning.reward_model import RewardModel
from vietnamese_ai.fine_tuning.sft_trainer import SFTTrainer
from vietnamese_ai.utils.logger import Logger


class RLHFPipeline:
    """
    Complete RLHF Pipeline.

    Quy trình:
    1. SFT (Supervised Fine-Tuning) trên instruction data
    2. Train Reward Model từ preference pairs
    3. RL optimization (PPO hoặc DPO) với reward signal

    Tính năng:
    - End-to-end RLHF training
    - Modular design (chọn SFT, DPO, hoặc full RLHF)
    - Reward model training
    - KL divergence penalty
    - Training history và metrics

    Sử dụng:
        >>> pipeline = RLHFPipeline()
        >>> pipeline.sft(model, sft_data)
        >>> pipeline.train_reward_model(reward_model, preference_data)
        >>> ket_qua = pipeline.rlhf(model, ref_model, preference_data)
    """

    def __init__(
        self,
        sft_config: Optional[Dict] = None,
        dpo_config: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
    ):
        self.logger = Logger("RLHFPipeline")

        sft_cfg = sft_config or {}
        dpo_cfg = dpo_config or {}
        reward_cfg = reward_config or {}

        self.sft_trainer = SFTTrainer(**sft_cfg)
        self.dpo_trainer = DPOTrainer(**dpo_cfg)
        self.reward_model_trainer = RewardModel(**reward_cfg)

        self._sft_done = False
        self._reward_done = False
        self._rlhf_done = False
        self._results: Dict[str, Any] = {}

    def sft(
        self,
        model: Any,
        du_lieu: List[Dict[str, Any]],
        du_lieu_val: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Bước 1: Supervised Fine-Tuning.

        Args:
            model: Mô hình cần SFT
            du_lieu: Training data [{input_ids, labels}, ...]
            du_lieu_val: Validation data

        Returns:
            Dict chứa SFT results
        """
        self.logger.info("=== Bước 1: Supervised Fine-Tuning (SFT) ===")
        ket_qua = self.sft_trainer.huan_luyen(model, du_lieu, du_lieu_val)
        self._sft_done = True
        self._results["sft"] = ket_qua
        self.logger.info(f"SFT hoàn tất: loss={ket_qua.get('train_loss_min', 0):.4f}")
        return ket_qua

    def train_reward_model(
        self,
        reward_model: Any,
        preference_data: List[Dict[str, Any]],
        so_vong: int = 1,
    ) -> Dict[str, Any]:
        """
        Bước 2: Train Reward Model.

        Args:
            reward_model: Reward model
            preference_data: Preference pairs [{chosen, rejected}, ...]
            so_vong: Số epochs

        Returns:
            Dict chứa RM training results
        """
        self.logger.info("=== Bước 2: Train Reward Model ===")
        ket_qua = self.reward_model_trainer.huan_luyen(
            reward_model, preference_data, so_vong=so_vong
        )
        self._reward_done = True
        self._results["reward_model"] = ket_qua
        self.logger.info(f"Reward Model hoànất: acc={ket_qua.get('final_accuracy', 0):.4f}")
        return ket_qua

    def rlhf(
        self,
        model: Any,
        ref_model: Any,
        preference_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Bước 3: RLHF với DPO.

        Args:
            model: Policy model
            ref_model: Reference model (frozen)
            preference_data: Preference pairs

        Returns:
            Dict chứa RLHF results
        """
        self.logger.info("=== Bước 3: RLHF (DPO) ===")
        ket_qua = self.dpo_trainer.huan_luyen(model, ref_model, preference_data)
        self._rlhf_done = True
        self._results["rlhf"] = ket_qua
        self.logger.info(f"RLHF hoànất: margin={ket_qua.get('final_reward_margin', 0):.4f}")
        return ket_qua

    def chay_day_du(
        self,
        model: Any,
        ref_model: Any,
        reward_model: Any,
        sft_data: List[Dict[str, Any]],
        preference_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline RLHF.

        Args:
            model: Policy model
            ref_model: Reference model
            reward_model: Reward model
            sft_data: SFT training data
            preference_data: Preference data

        Returns:
            Dict chứa tất cả results
        """
        self.logger.info("=== Bắt đầu Full RLHF Pipeline ===")
        bat_dau = time.time()

        self.sft(model, sft_data)
        self.train_reward_model(reward_model, preference_data)
        self.rlhf(model, ref_model, preference_data)

        tong_thoi_gian = time.time() - bat_dau
        self._results["tong_thoi_gian"] = round(tong_thoi_gian, 2)

        self.logger.info(f"Full RLHF Pipeline hoànất ({tong_thoi_gian:.1f}s)")
        return self._results

    def lay_ket_qua(self) -> Dict[str, Any]:
        return self._results.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "sft_done": self._sft_done,
            "reward_done": self._reward_done,
            "rlhf_done": self._rlhf_done,
            "sft_trainer": self.sft_trainer.thong_ke(),
            "dpo_trainer": self.dpo_trainer.thong_ke(),
            "reward_model": self.reward_model_trainer.thong_ke(),
        }
