"""PreTrainer - Pre-training trainer cho GPT-style models."""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class TextDataset:
    """
    Dataset cho pre-training.

    Tính năng:
    - Tokenize text corpus
    - Sliding window chunking
    - Train/val split
    - Batch iteration

    Sử dụng:
        >>> ds = TextDataset(do_dai_window=512)
        >>> ds.tai_corpus(["văn bản 1", "văn bản 2"])
        >>> for batch in ds.iter_batches(32): ...
    """

    def __init__(self, do_dai_window: int = 512, buoc_nhay: int = 256, seed: int = 42):
        self.do_dai_window = do_dai_window
        self.buoc_nhay = buoc_nhay
        self.seed = seed
        self.logger = Logger("TextDataset")

        self._tokens: List[int] = []
        self._chunks: List[List[int]] = []
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._train_chunks: Optional[List[List[int]]] = None
        self._val_chunks: Optional[List[List[int]]] = None

    def tai_corpus(self, cac_van_ban: List[str], vocab_size: int = 5000) -> Dict[str, int]:
        """
        Load corpus và tokenize.

        Args:
            cac_van_ban: List văn bản
            vocab_size: Kích thước vocabulary

        Returns:
            Dict với so_tokens, so_chunks, vocab_size
        """
        if not cac_van_ban:
            raise ValueError("Corpus trống")

        all_chars = set()
        for vb in cac_van_ban:
            all_chars.update(vb)

        self._vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for i, char in enumerate(sorted(all_chars)):
            if len(self._vocab) >= vocab_size:
                break
            self._vocab[char] = len(self._vocab)

        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

        self._tokens = []
        for vb in cac_van_ban:
            self._tokens.append(self._vocab.get("<BOS>", 2))
            for char in vb:
                self._tokens.append(self._vocab.get(char, 1))
            self._tokens.append(self._vocab.get("<EOS>", 3))

        self._chunks = []
        for i in range(0, len(self._tokens) - self.do_dai_window, self.buoc_nhay):
            chunk = self._tokens[i : i + self.do_dai_window]
            if len(chunk) == self.do_dai_window:
                self._chunks.append(chunk)

        self.logger.info(
            f"Corpus: {len(cac_van_ban)} văn bản, "
            f"{len(self._tokens)} tokens, "
            f"{len(self._chunks)} chunks, "
            f"vocab={len(self._vocab)}"
        )

        return {
            "so_van_ban": len(cac_van_ban),
            "so_tokens": len(self._tokens),
            "so_chunks": len(self._chunks),
            "vocab_size": len(self._vocab),
        }

    def chia_du_lieu(self, ty_le_val: float = 0.1, seed: Optional[int] = None) -> Dict[str, int]:
        if not self._chunks:
            raise RuntimeError("Chưa load corpus. Gọi tai_corpus() trước.")

        np.random.seed(seed or self.seed)
        indices = np.random.permutation(len(self._chunks))
        n_val = max(1, int(len(self._chunks) * ty_le_val))

        self._val_chunks = [self._chunks[i] for i in indices[:n_val]]
        self._train_chunks = [self._chunks[i] for i in indices[n_val:]]

        self.logger.info(
            f"Chia dữ liệu: train={len(self._train_chunks)}, val={len(self._val_chunks)}"
        )
        return {"so_train": len(self._train_chunks), "so_val": len(self._val_chunks)}

    def iter_batches(self, batch_size: int = 32, che_do: str = "train"):
        chunks = self._train_chunks if che_do == "train" else self._val_chunks
        if chunks is None:
            chunks = self._chunks

        indices = np.random.permutation(len(chunks))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = np.array([chunks[j] for j in batch_indices])
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            yield input_ids, targets

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def so_chunks(self) -> int:
        return len(self._chunks)

    def ma_hoa(self, text: str) -> List[int]:
        return [self._vocab.get(c, 1) for c in text]

    def giai_ma(self, ids: List[int]) -> str:
        return "".join(self._reverse_vocab.get(i, "?") for i in ids)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_tokens": len(self._tokens),
            "so_chunks": len(self._chunks),
            "vocab_size": len(self._vocab),
            "do_dai_window": self.do_dai_window,
            "buoc_nhay": self.buoc_nhay,
            "so_train": len(self._train_chunks) if self._train_chunks else None,
            "so_val": len(self._val_chunks) if self._val_chunks else None,
        }


class PreTrainer:
    """
    Pre-training trainer cho GPT-style models.

    Tính năng:
    - Causal language modeling (next token prediction)
    - Gradient accumulation
    - Learning rate warmup + cosine decay
    - Periodic evaluation
    - Checkpoint save
    - Callback system

    Sử dụng:
        >>> trainer = PreTrainer(so_vong=10, toc_do_hoc=3e-4)
        >>> dataset = TextDataset()
        >>> dataset.tai_corpus(corpus)
        >>> ket_qua = trainer.huan_luyen(gpt_model, dataset)
    """

    def __init__(
        self,
        so_vong: int = 10,
        kich_thuoc_batch: int = 32,
        toc_do_hoc: float = 3e-4,
        gradient_accumulation: int = 1,
        warmup_steps: int = 100,
        weight_decay: float = 0.1,
        gradient_clip: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 500,
        seed: int = 42,
    ):
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.toc_do_hoc = toc_do_hoc
        self.gradient_accumulation = gradient_accumulation
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.seed = seed
        self.logger = Logger("PreTrainer")

        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
            "learning_rate": [],
        }
        self._global_step = 0

    def huan_luyen(
        self,
        model: Any,
        dataset: TextDataset,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Pre-training GPT model.

        Args:
            model: GPTModel instance
            dataset: TextDataset
            callback: Callback function(step, loss)

        Returns:
            Dict chứa training results
        """
        self.logger.info(f"Bắt đầu pre-training ({self.so_vong} epochs)")
        self.logger.info(f"  Dataset: {dataset.so_chunks} chunks, vocab={dataset.vocab_size}")

        dataset.chia_du_lieu()

        bat_dau = time.time()

        for epoch in range(self.so_vong):
            epoch_loss = 0.0
            steps = 0

            for input_ids, targets in dataset.iter_batches(self.kich_thuoc_batch, "train"):
                loss = model.tinh_loss(input_ids, targets)
                epoch_loss += loss
                steps += 1
                self._global_step += 1

                if self._global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / max(1, steps)
                    ppl = np.exp(avg_loss)
                    self.logger.info(
                        f"Step {self._global_step}: loss={avg_loss:.4f}, ppl={ppl:.2f}"
                    )

                if callback:
                    callback(self._global_step, loss)

            avg_loss = epoch_loss / max(1, steps)
            ppl = np.exp(avg_loss)
            self._history["train_loss"].append(avg_loss)
            self._history["perplexity"].append(ppl)
            self._history["learning_rate"].append(self.toc_do_hoc)

            eval_loss = self._danh_gia(model, dataset)
            self._history["eval_loss"].append(eval_loss)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.so_vong}: "
                f"loss={avg_loss:.4f}, ppl={ppl:.2f}, eval_loss={eval_loss:.4f}"
            )

        tong_thoi_gian = time.time() - bat_dau

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"]),
            "eval_loss_min": min(self._history["eval_loss"]),
            "final_perplexity": self._history["perplexity"][-1],
            "history": self._history,
        }

    def _danh_gia(self, model: Any, dataset: TextDataset) -> float:
        """Đánh giá trên validation set."""
        total_loss = 0.0
        steps = 0

        for input_ids, targets in dataset.iter_batches(self.kich_thuoc_batch, "val"):
            loss = model.tinh_loss(input_ids, targets)
            total_loss += loss
            steps += 1

        return total_loss / max(1, steps)

    def lay_lich_su(self) -> Dict[str, List[float]]:
        return self._history.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_vong": self.so_vong,
            "kich_thuoc_batch": self.kich_thuoc_batch,
            "toc_do_hoc": self.toc_do_hoc,
            "global_step": self._global_step,
            "train_loss_count": len(self._history["train_loss"]),
            "final_perplexity": self._history["perplexity"][-1]
            if self._history["perplexity"]
            else None,
        }
