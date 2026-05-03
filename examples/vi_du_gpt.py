"""Ví dụ: GPT Pre-training."""

import numpy as np

from vietnamese_ai.transformer.gpt_model import GPTModel
from vietnamese_ai.transformer.pretrainer import PreTrainer, TextDataset


def vi_du_gpt_model():
    """Tạo và sử dụng GPT model."""
    model = GPTModel(d_model=32, so_dau=4, d_ff=64, so_block=2, so_tu_vung=200)

    print(f"Model: {model.thong_ke()}")

    input_ids = np.array([[1, 2, 3, 4, 5]])
    logits = model.tien(input_ids)
    print(f"Logits shape: {logits.shape}")

    generated = model.sinh_tiep(input_ids, so_token=10, nhiet_do=0.8)
    print(f"Generated shape: {generated.shape}")

    loss = model.tinh_loss(
        np.array([[1, 2, 3]]),
        np.array([[2, 3, 4]]),
    )
    print(f"Loss: {loss:.4f}")


def vi_du_pretraining():
    """Pre-train GPT model trên corpus tiếng Việt."""
    corpus = [
        "Trí tuệ nhân tạo đang thay đổi cách con người làm việc.",
        "Học máy là một nhánh quan trọng của trí tuệ nhân tạo.",
        "Mạng nơ-ron nhân tạo mô phỏng cách não bộ hoạt động.",
        "Xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng Việt.",
        "Học sâu đã đạt nhiều thành tựu trong nhận dạng hình ảnh.",
    ] * 20

    dataset = TextDataset(do_dai_window=32, buoc_nhay=16)
    ket_qua = dataset.tai_corpus(corpus, vocab_size=200)
    print(f"Corpus: {ket_qua}")

    model = GPTModel(d_model=32, so_dau=4, d_ff=64, so_block=1, so_tu_vung=200)

    trainer = PreTrainer(so_vong=3, kich_thuoc_batch=8, logging_steps=10)
    ket_qua = trainer.huan_luyen(model, dataset)

    print(f"\nKết quả:")
    print(f"  Thời gian: {ket_qua['tong_thoi_gian']:.1f}s")
    print(f"  Train loss: {ket_qua['train_loss_min']:.4f}")
    print(f"  Perplexity: {ket_qua['final_perplexity']:.2f}")

    generated = model.sinh_tiep(np.array([[1]]), so_token=20, nhiet_do=0.8)
    print(f"\nGenerated tokens: {generated[0].tolist()}")


if __name__ == "__main__":
    print("=== GPT Model ===")
    vi_du_gpt_model()
    print("\n=== Pre-training ===")
    vi_du_pretraining()
