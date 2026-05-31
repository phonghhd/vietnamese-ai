"""InstructionDataset - Dataset format cho instruction tuning."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

TEMPLATE_ALPACA = {
    "prompt_input": "### Dưới đây là một nhiệm vụ, mô tả thêm context đầu vào. Viết phản hồi hoàn thành yêu cầu.\n\n### Đầu vào:\n{input}\n\n### Nhiệm vụ:\n{instruction}\n\n### Phản hồi:\n",
    "prompt_no_input": "### Dưới đây là một nhiệm vụ. Viết phản hồi hoàn thành yêu cầu.\n\n### Nhiệm vụ:\n{instruction}\n\n### Phản hồi:\n",
}

TEMPLATE_SHAREGPT = {
    "system": "Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt.",
    "roles": ["user", "assistant"],
}


class InstructionDataset:
    """
    Dataset cho instruction tuning.

    Hỗ trợ 2 format phổ biến:
    - Alpaca: {"instruction": ..., "input": ..., "output": ...}
    - ShareGPT: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}

    Tính năng:
    - Load từ JSON/JSONL
    - Convert giữa các format
    - Tokenize với chat template
    - Train/val split
    - Thống kê dataset

    Sử dụng:
        >>> ds = InstructionDataset(che_do="alpaca")
        >>> ds.tai_file("data/alpaca_vi.json")
        >>> ds.chia_du_lieu(ty_le_val=0.1)
        >>> print(ds.thong_ke())
    """

    CHE_DO_HO_TRO = ["alpaca", "sharegpt"]

    def __init__(self, che_do: str = "alpaca"):
        if che_do not in self.CHE_DO_HO_TRO:
            raise ValueError(f"che_do phải là 'alpaca' hoặc 'sharegpt', nhận: '{che_do}'")

        self.che_do = che_do
        self.logger = Logger("InstructionDataset")
        self._du_lieu: List[Dict[str, Any]] = []
        self._du_lieu_train: Optional[List[Dict]] = None
        self._du_lieu_val: Optional[List[Dict]] = None

    def tai_file(self, duong_dan: str) -> int:
        """
        Load dataset từ file JSON hoặc JSONL.

        Args:
            duong_dan: Đường dẫn file

        Returns:
            Số mẫu đã load
        """
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

        if duong_dan.endswith(".jsonl"):
            du_lieu = []
            with open(duong_dan_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        du_lieu.append(json.loads(line))
        else:
            with open(duong_dan_path, "r", encoding="utf-8") as f:
                du_lieu = json.load(f)
                if isinstance(du_lieu, dict):
                    du_lieu = du_lieu.get("data", [du_lieu])

        self._du_lieu = du_lieu
        self.logger.info(f"Đã load {len(du_lieu)} mẫu ({self.che_do})")
        return len(du_lieu)

    def tai_tu_list(self, du_lieu: List[Dict[str, Any]]) -> int:
        """Load dataset từ list of dicts."""
        self._du_lieu = du_lieu
        return len(du_lieu)

    def them_mau(self, mau: Dict[str, Any]) -> None:
        """Thêm một mẫu vào dataset."""
        self._du_lieu.append(mau)

    def chia_du_lieu(self, ty_le_val: float = 0.1, seed: int = 42) -> Dict[str, int]:
        """
        Chia dataset thành train/val.

        Args:
            ty_le_val: Tỷ lệ validation (0-1)
            seed: Random seed

        Returns:
            Dict với so_train, so_val
        """
        if not self._du_lieu:
            raise RuntimeError("Dataset trống. Gọi tai_file() trước.")

        n = len(self._du_lieu)
        n_val = max(1, int(n * ty_le_val))
        n_train = n - n_val

        indices = np.random.RandomState(seed).permutation(n)
        du_lieu_arr = [self._du_lieu[i] for i in indices]

        self._du_lieu_train = du_lieu_arr[:n_train]
        self._du_lieu_val = du_lieu_arr[n_train:]

        self.logger.info(f"Chia dữ liệu: train={n_train}, val={n_val}")
        return {"so_train": n_train, "so_val": n_val}

    @property
    def train(self) -> List[Dict]:
        """Dữ liệu train."""
        if self._du_lieu_train is not None:
            return self._du_lieu_train
        return self._du_lieu

    @property
    def val(self) -> Optional[List[Dict]]:
        """Dữ liệu validation."""
        return self._du_lieu_val

    @property
    def du_lieu(self) -> List[Dict]:
        """Toàn bộ dữ liệu."""
        return self._du_lieu

    def __len__(self) -> int:
        return len(self._du_lieu)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._du_lieu[idx]

    def format_alpaca(self, mau: Dict[str, Any]) -> str:
        """Format một mẫu theo Alpaca template."""
        instruction = mau.get("instruction", "")
        input_data = mau.get("input", "")
        output = mau.get("output", "")

        if input_data:
            prompt = TEMPLATE_ALPACA["prompt_input"].format(
                input=input_data, instruction=instruction
            )
        else:
            prompt = TEMPLATE_ALPACA["prompt_no_input"].format(instruction=instruction)

        return prompt + output

    def format_sharegpt(self, mau: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format một mẫu theo ShareGPT template."""
        conversations = mau.get("conversations", [])
        formatted = []

        formatted.append(
            {
                "role": "system",
                "content": TEMPLATE_SHAREGPT["system"],
            }
        )

        for turn in conversations:
            role = turn.get("from", "human")
            content = turn.get("value", "")

            if role == "human":
                formatted.append({"role": "user", "content": content})
            elif role == "gpt":
                formatted.append({"role": "assistant", "content": content})

        return formatted

    def format_tat_ca(self) -> List[str]:
        """Format toàn bộ dataset thành text."""
        ket_qua = []
        for mau in self._du_lieu:
            if self.che_do == "alpaca":
                ket_qua.append(self.format_alpaca(mau))
            else:
                formatted = self.format_sharegpt(mau)
                text = "\n".join(f"[{m['role']}] {m['content']}" for m in formatted)
                ket_qua.append(text)
        return ket_qua

    def chuyen_doi_sharegpt_sang_alpaca(self) -> List[Dict[str, str]]:
        """Convert ShareGPT sang Alpaca format."""
        if self.che_do != "sharegpt":
            raise RuntimeError("Chỉ convert từ sharegpt")

        ket_qua = []
        for mau in self._du_lieu:
            conversations = mau.get("conversations", [])
            instruction = ""
            output = ""

            for turn in conversations:
                if turn.get("from") == "human":
                    instruction = turn.get("value", "")
                elif turn.get("from") == "gpt":
                    output = turn.get("value", "")

            if instruction and output:
                ket_qua.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                    }
                )

        return ket_qua

    def luu(self, duong_dan: str) -> str:
        """Lưu dataset ra file."""
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        with open(duong_dan_path, "w", encoding="utf-8") as f:
            json.dump(self._du_lieu, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Đã lưu dataset: {duong_dan} ({len(self._du_lieu)} mẫu)")
        return str(duong_dan_path)

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê dataset."""
        if not self._du_lieu:
            return {"so_mau": 0}

        do_dai_instruction = []
        do_dai_output = []

        for mau in self._du_lieu:
            if self.che_do == "alpaca":
                inst = mau.get("instruction", "")
                out = mau.get("output", "")
                do_dai_instruction.append(len(inst.split()))
                do_dai_output.append(len(out.split()))
            else:
                for turn in mau.get("conversations", []):
                    val = turn.get("value", "")
                    if turn.get("from") == "human":
                        do_dai_instruction.append(len(val.split()))
                    else:
                        do_dai_output.append(len(val.split()))

        return {
            "che_do": self.che_do,
            "so_mau": len(self._du_lieu),
            "so_mau_train": len(self._du_lieu_train) if self._du_lieu_train else None,
            "so_mau_val": len(self._du_lieu_val) if self._du_lieu_val else None,
            "do_dai_tb_instruction": round(np.mean(do_dai_instruction), 1)
            if do_dai_instruction
            else 0,
            "do_dai_tb_output": round(np.mean(do_dai_output), 1) if do_dai_output else 0,
            "do_dai_max_instruction": max(do_dai_instruction) if do_dai_instruction else 0,
            "do_dai_max_output": max(do_dai_output) if do_dai_output else 0,
        }
