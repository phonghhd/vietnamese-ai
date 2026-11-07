"""ModelConfigs - Cấu hình Vietnamese LLM và Danh mục Mô hình Mã nguồn mở Thế hệ mới (Qwen3.5–3.8, Gemma 4, Llama 4, DeepSeek-R1/V3, Phi-4)."""

from typing import Any, Dict, List, Optional


class ModelConfig:
    """
    Cấu hình kiến trúc cho Vietnamese LLM & Danh mục các mô hình Open-Source mới nhất.

    Hỗ trợ:
    - Các cấu hình gốc `vnlm-*` (`vnlm-tiny` đến `vnlm-7b`)
    - Họ **Qwen 3 / 3.5 / 3.6 / 3.7 / 3.8** (`qwen3.5`, `qwen3.6`, `qwen3.7`, `qwen3.8`, `qwen3.8-moe`, `qwen3.5-coder`)
    - Họ **Google Gemma 3 & Gemma 4** (`gemma3-4b`, `gemma3-27b`, `gemma4`, `gemma4-2b`, `gemma4-9b`, `gemma4-27b`, `gemma4-omni`)
    - Họ **Meta Llama 3.3 & Llama 4** (`llama3.3-70b`, `llama4`, `llama4-scout`, `llama4-maverick`)
    - Họ **DeepSeek Reasoning & MoE** (`deepseek-r1`, `deepseek-r1-7b`, `deepseek-r1-32b`, `deepseek-v3`, `deepseek-v3.2`)
    - Họ **Microsoft Phi-4 & Mistral 3.1** (`phi4`, `phi4-mini`, `phi4-multimodal`, `mistral-small-3.1`)
    - Các mô hình chuyên biệt tiếng Việt (`vistral-7b`, `phogpt-4b`, `vi-qwen3.8-8b`, `bge-m3-viet`)
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        # === 1. Cấu hình nội bộ vnlm ===
        "vnlm-tiny": {
            "d_model": 64,
            "so_dau": 2,
            "d_ff": 256,
            "so_block": 2,
            "so_tu_vung": 1000,
            "do_dai_toi_da": 256,
            "dropout": 0.1,
            "ho_mo_hinh": "vnlm",
            "hf_id": "vinai/vgpt2",
            "mo_ta": "Tiny model cho testing (~10M params)",
        },
        "vnlm-small": {
            "d_model": 768,
            "so_dau": 12,
            "d_ff": 3072,
            "so_block": 12,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 2048,
            "dropout": 0.1,
            "ho_mo_hinh": "vnlm",
            "hf_id": "vinai/vgpt2",
            "mo_ta": "Small model 125M params (GPT-2 small)",
        },
        "vnlm-medium": {
            "d_model": 1024,
            "so_dau": 16,
            "d_ff": 4096,
            "so_block": 24,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 2048,
            "dropout": 0.1,
            "ho_mo_hinh": "vnlm",
            "hf_id": "vinai/vgpt2",
            "mo_ta": "Medium model 350M params",
        },
        "vnlm-large": {
            "d_model": 2048,
            "so_dau": 32,
            "d_ff": 8192,
            "so_block": 24,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.1,
            "ho_mo_hinh": "vnlm",
            "hf_id": "vinai/PhoGPT-4B-Chat",
            "mo_ta": "Large model 1.3B params",
        },
        "vnlm-xl": {
            "d_model": 2560,
            "so_dau": 32,
            "d_ff": 10240,
            "so_block": 32,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.1,
            "ho_mo_hinh": "vnlm",
            "hf_id": "vinai/PhoGPT-4B-Chat",
            "mo_ta": "XL model 2.7B params",
        },
        "vnlm-7b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 11008,
            "so_block": 32,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.0,
            "ho_mo_hinh": "vnlm",
            "hf_id": "Viet-Mistral/Vistral-7B-Chat",
            "mo_ta": "7B params (LLaMA-2 style)",
        },
        # === 2. Họ Qwen 3 / Qwen 3.5 / Qwen 3.6 / Qwen 3.7 / Qwen 3.8 ===
        "qwen3-0.6b": {
            "d_model": 1024,
            "so_dau": 16,
            "d_ff": 2816,
            "so_block": 28,
            "so_tu_vung": 151936,
            "do_dai_toi_da": 32768,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3",
            "hf_id": "Qwen/Qwen3-0.6B",
            "mo_ta": "Qwen3 0.6B siêu nhẹ cho thiết bị biên & di động (32K context)",
        },
        "qwen3-8b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 12288,
            "so_block": 36,
            "so_tu_vung": 151936,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3",
            "hf_id": "Qwen/Qwen3-8B-Instruct",
            "mo_ta": "Qwen3 8B Instruct hỗ trợ hybrid thinking & 128K context",
        },
        "qwen3-30b-a3b": {
            "d_model": 2048,
            "so_dau": 32,
            "d_ff": 6144,
            "so_block": 48,
            "so_tu_vung": 151936,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3",
            "hf_id": "Qwen/Qwen3-30B-A3B",
            "mo_ta": "Qwen3 30B-A3B MoE (kích hoạt 3B tham số mỗi token, tốc độ cao)",
        },
        "qwen3.5": {
            "d_model": 3584,
            "so_dau": 28,
            "d_ff": 18944,
            "so_block": 28,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.5",
            "hf_id": "Qwen/Qwen3.5-7B-Instruct",
            "mo_ta": "Qwen3.5 7B Instruct tối ưu đa ngôn ngữ & tiếng Việt (128K context)",
        },
        "qwen3.5-1.5b": {
            "d_model": 1536,
            "so_dau": 12,
            "d_ff": 8960,
            "so_block": 28,
            "so_tu_vung": 151936,
            "do_dai_toi_da": 65536,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.5",
            "hf_id": "Qwen/Qwen3.5-1.5B-Instruct",
            "mo_ta": "Qwen3.5 1.5B Instruct tốc độ cao trên CPU/Edge",
        },
        "qwen3.5-7b": {
            "d_model": 3584,
            "so_dau": 28,
            "d_ff": 18944,
            "so_block": 28,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.5",
            "hf_id": "Qwen/Qwen3.5-7B-Instruct",
            "mo_ta": "Qwen3.5 7B Instruct chuẩn sản xuất đa tác vụ",
        },
        "qwen3.5-coder-7b": {
            "d_model": 3584,
            "so_dau": 28,
            "d_ff": 18944,
            "so_block": 28,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.5",
            "hf_id": "Qwen/Qwen3.5-Coder-7B-Instruct",
            "mo_ta": "Qwen3.5 Coder 7B chuyên sinh mã nguồn & Agent lập trình",
        },
        "qwen3.6": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 32,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.6",
            "hf_id": "Qwen/Qwen3.6-8B-Instruct",
            "mo_ta": "Qwen3.6 8B Instruct nâng cấp khả năng gọi công cụ (Tool Use / MCP)",
        },
        "qwen3.6-8b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 32,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.6",
            "hf_id": "Qwen/Qwen3.6-8B-Instruct",
            "mo_ta": "Qwen3.6 8B Instruct (GQA + RoPE mở rộng 128K)",
        },
        "qwen3.6-32b": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 27648,
            "so_block": 64,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.6",
            "hf_id": "Qwen/Qwen3.6-32B-Instruct",
            "mo_ta": "Qwen3.6 32B Instruct suy luận chuyên sâu & RAG ngữ cảnh dài",
        },
        "qwen3.7": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.7",
            "hf_id": "Qwen/Qwen3.7-8B-Instruct",
            "mo_ta": "Qwen3.7 8B Instruct hỗ trợ Chain-of-Thought ẩn & 256K context",
        },
        "qwen3.7-8b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.7",
            "hf_id": "Qwen/Qwen3.7-8B-Instruct",
            "mo_ta": "Qwen3.7 8B Instruct (Reasoning + Structured JSON Output)",
        },
        "qwen3.7-omni": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.7",
            "hf_id": "Qwen/Qwen3.7-Omni-7B",
            "mo_ta": "Qwen3.7 Omni đa phương thức (Văn bản + Âm thanh + Thị giác)",
        },
        "qwen3.8": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 16384,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.8",
            "hf_id": "Qwen/Qwen3.8-8B-Instruct",
            "mo_ta": "Qwen3.8 8B thế hệ mới nhất (Hybrid Reasoning, Agentic MCP, 256K context)",
        },
        "qwen3.8-1.8b": {
            "d_model": 2048,
            "so_dau": 16,
            "d_ff": 8192,
            "so_block": 24,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.8",
            "hf_id": "Qwen/Qwen3.8-1.8B-Instruct",
            "mo_ta": "Qwen3.8 1.8B siêu tốc cho Edge/Mobile & Fine-tune LoRA nhẹ",
        },
        "qwen3.8-8b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 16384,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.8",
            "hf_id": "Qwen/Qwen3.8-8B-Instruct",
            "mo_ta": "Qwen3.8 8B Instruct tối ưu tiếng Việt & suy luận toán/code",
        },
        "qwen3.8-32b": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 27648,
            "so_block": 64,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.8",
            "hf_id": "Qwen/Qwen3.8-32B-Instruct",
            "mo_ta": "Qwen3.8 32B Instruct hiệu năng cấp doanh nghiệp",
        },
        "qwen3.8-moe": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 16384,
            "so_block": 48,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "qwen3.8",
            "hf_id": "Qwen/Qwen3.8-MoE-A4B-Instruct",
            "mo_ta": "Qwen3.8 Mixture-of-Experts (kích hoạt 4B/token, suy luận cực nhanh)",
        },
        # === 3. Họ Google Gemma 3 & Gemma 4 ===
        "gemma3-4b": {
            "d_model": 2560,
            "so_dau": 16,
            "d_ff": 10240,
            "so_block": 34,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma3",
            "hf_id": "google/gemma-3-4b-it",
            "mo_ta": "Google Gemma 3 4B Instruct đa phương thức (128K context)",
        },
        "gemma3-27b": {
            "d_model": 4608,
            "so_dau": 32,
            "d_ff": 36864,
            "so_block": 46,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma3",
            "hf_id": "google/gemma-3-27b-it",
            "mo_ta": "Google Gemma 3 27B Instruct độ chính xác cao",
        },
        "gemma4": {
            "d_model": 3584,
            "so_dau": 16,
            "d_ff": 14336,
            "so_block": 42,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma4",
            "hf_id": "google/gemma-4-9b-it",
            "mo_ta": "Google Gemma 4 9B Instruct thế hệ mới (Interleaved Attention, 256K context)",
        },
        "gemma4-2b": {
            "d_model": 2048,
            "so_dau": 8,
            "d_ff": 8192,
            "so_block": 26,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma4",
            "hf_id": "google/gemma-4-2b-it",
            "mo_ta": "Google Gemma 4 2B siêu gọn cho thiết bị cá nhân & nhúng",
        },
        "gemma4-9b": {
            "d_model": 3584,
            "so_dau": 16,
            "d_ff": 14336,
            "so_block": 42,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma4",
            "hf_id": "google/gemma-4-9b-it",
            "mo_ta": "Google Gemma 4 9B Instruct cân bằng tốc độ & trí tuệ đa ngôn ngữ",
        },
        "gemma4-27b": {
            "d_model": 4608,
            "so_dau": 32,
            "d_ff": 36864,
            "so_block": 48,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma4",
            "hf_id": "google/gemma-4-27b-it",
            "mo_ta": "Google Gemma 4 27B Instruct suy luận phức tạp & thị giác máy tính",
        },
        "gemma4-omni": {
            "d_model": 3584,
            "so_dau": 16,
            "d_ff": 14336,
            "so_block": 42,
            "so_tu_vung": 262144,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "gemma4",
            "hf_id": "google/gemma-4-omni-9b",
            "mo_ta": "Google Gemma 4 Omni (Text + Vision + Audio đầu-cuối)",
        },
        # === 4. Họ Meta Llama 3.3 & Llama 4 ===
        "llama3.3-70b": {
            "d_model": 8192,
            "so_dau": 64,
            "d_ff": 28672,
            "so_block": 80,
            "so_tu_vung": 128256,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "llama3.3",
            "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
            "mo_ta": "Meta Llama 3.3 70B Instruct chuẩn mã nguồn mở hiệu năng cao",
        },
        "llama4": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 16384,
            "so_block": 48,
            "so_tu_vung": 202048,
            "do_dai_toi_da": 1048576,
            "dropout": 0.0,
            "ho_mo_hinh": "llama4",
            "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "mo_ta": "Meta Llama 4 Scout MoE (17B active, 16 experts, ngữ cảnh 1M+ tokens)",
        },
        "llama4-scout": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 16384,
            "so_block": 48,
            "so_tu_vung": 202048,
            "do_dai_toi_da": 1048576,
            "dropout": 0.0,
            "ho_mo_hinh": "llama4",
            "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "mo_ta": "Meta Llama 4 Scout 17B-16E đa phương thức siêu ngữ cảnh",
        },
        "llama4-maverick": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 16384,
            "so_block": 48,
            "so_tu_vung": 202048,
            "do_dai_toi_da": 1048576,
            "dropout": 0.0,
            "ho_mo_hinh": "llama4",
            "hf_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "mo_ta": "Meta Llama 4 Maverick 17B-128E MoE chuyên suy luận & hội thoại đa ngôn ngữ",
        },
        # === 5. Họ DeepSeek-R1 & DeepSeek-V3 ===
        "deepseek-r1": {
            "d_model": 3584,
            "so_dau": 28,
            "d_ff": 18944,
            "so_block": 28,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "deepseek-r1",
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "mo_ta": "DeepSeek-R1 7B chuyên suy luận chuỗi tư duy (Chain-of-Thought RL)",
        },
        "deepseek-r1-1.5b": {
            "d_model": 1536,
            "so_dau": 12,
            "d_ff": 8960,
            "so_block": 28,
            "so_tu_vung": 151936,
            "do_dai_toi_da": 65536,
            "dropout": 0.0,
            "ho_mo_hinh": "deepseek-r1",
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "mo_ta": "DeepSeek-R1 1.5B suy luận toán & logic siêu nhẹ",
        },
        "deepseek-r1-32b": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 27648,
            "so_block": 64,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "deepseek-r1",
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "mo_ta": "DeepSeek-R1 32B suy luận chuyên sâu tương đương o1",
        },
        "deepseek-v3": {
            "d_model": 7168,
            "so_dau": 128,
            "d_ff": 18432,
            "so_block": 61,
            "so_tu_vung": 129280,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "deepseek-v3",
            "hf_id": "deepseek-ai/DeepSeek-V3",
            "mo_ta": "DeepSeek-V3 Multi-head Latent Attention (MLA) MoE 671B",
        },
        # === 6. Họ Microsoft Phi-4, Mistral 3.1 & Mô hình Tiếng Việt ===
        "phi4": {
            "d_model": 5120,
            "so_dau": 40,
            "d_ff": 17920,
            "so_block": 40,
            "so_tu_vung": 100352,
            "do_dai_toi_da": 16384,
            "dropout": 0.0,
            "ho_mo_hinh": "phi4",
            "hf_id": "microsoft/phi-4",
            "mo_ta": "Microsoft Phi-4 14B xuất sắc về STEM, toán học & lập trình",
        },
        "phi4-mini": {
            "d_model": 3072,
            "so_dau": 24,
            "d_ff": 8192,
            "so_block": 32,
            "so_tu_vung": 200064,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "phi4",
            "hf_id": "microsoft/Phi-4-mini-instruct",
            "mo_ta": "Microsoft Phi-4 Mini 3.8B hỗ trợ gọi hàm & đa ngôn ngữ",
        },
        "mistral-small-3.1": {
            "d_model": 5120,
            "so_dau": 32,
            "d_ff": 32768,
            "so_block": 40,
            "so_tu_vung": 131072,
            "do_dai_toi_da": 131072,
            "dropout": 0.0,
            "ho_mo_hinh": "mistral",
            "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "mo_ta": "Mistral Small 3.1 24B đa phương thức độ trễ thấp",
        },
        "vi-qwen3.8-8b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 16384,
            "so_block": 36,
            "so_tu_vung": 152064,
            "do_dai_toi_da": 262144,
            "dropout": 0.0,
            "ho_mo_hinh": "vietnamese-llm",
            "hf_id": "Qwen/Qwen3.8-8B-Instruct",
            "mo_ta": "Mô hình Qwen3.8 8B tinh chỉnh chuyên sâu cho văn hóa, pháp luật & tiếng Việt",
        },
        "vistral-7b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 14336,
            "so_block": 32,
            "so_tu_vung": 38369,
            "do_dai_toi_da": 32768,
            "dropout": 0.0,
            "ho_mo_hinh": "vietnamese-llm",
            "hf_id": "Viet-Mistral/Vistral-7B-Chat",
            "mo_ta": "Vistral 7B Chat chuyên tiếng Việt",
        },
    }

    def __init__(
        self,
        d_model: int = 768,
        so_dau: int = 12,
        d_ff: int = 3072,
        so_block: int = 12,
        so_tu_vung: int = 32000,
        do_dai_toi_da: int = 2048,
        dropout: float = 0.1,
        ten: str = "custom",
        mo_ta: str = "",
        ho_mo_hinh: str = "custom",
        hf_id: str = "",
    ):
        if d_model % so_dau != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_dau ({so_dau})")

        self.d_model = d_model
        self.so_dau = so_dau
        self.d_ff = d_ff
        self.so_block = so_block
        self.so_tu_vung = so_tu_vung
        self.do_dai_toi_da = do_dai_toi_da
        self.dropout = dropout
        self.ten = ten
        self.mo_ta = mo_ta
        self.ho_mo_hinh = ho_mo_hinh
        self.hf_id = hf_id or ten

    @classmethod
    def from_preset(cls, preset: str, thu_nho_cpu: bool = False) -> "ModelConfig":
        """
        Khởi tạo `ModelConfig` từ tên preset (ví dụ: `'qwen3.8'`, `'gemma4'`, `'deepseek-r1'`, `'vnlm-small'`).
        Nếu `thu_nho_cpu=True`, tự động thu nhỏ chiều tensor để chạy kiểm thử tức thì trên CPU.
        """
        key = preset.lower().strip()
        if key not in cls.PRESETS:
            # Tìm kiếm gần đúng theo tiền tố (ví dụ 'qwen3.8-chat' -> 'qwen3.8')
            for k in cls.PRESETS:
                if key.startswith(k) or k.startswith(key):
                    key = k
                    break
            else:
                raise ValueError(f"Preset '{preset}' không tồn tại. Chọn: {list(cls.PRESETS.keys())}")

        cfg = dict(cls.PRESETS[key])
        if thu_nho_cpu:
            cfg["d_model"] = 64
            cfg["so_dau"] = 4
            cfg["d_ff"] = 128
            cfg["so_block"] = 2
            cfg["so_tu_vung"] = 512
            cfg["do_dai_toi_da"] = 128
        return cls(ten=preset, **cfg)

    @property
    def so_tham_so(self) -> int:
        return (
            self.so_tu_vung * self.d_model
            + self.so_block
            * (self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2 + self.d_model * 4)
            + self.d_model * 2
        )

    @property
    def so_tham_so_str(self) -> str:
        params = self.so_tham_so
        if params >= 1e9:
            return f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.0f}M"
        return f"{params / 1e3:.0f}K"

    @classmethod
    def danh_sach_presets(cls, ho_mo_hinh: Optional[str] = None) -> Dict[str, str]:
        """Liệt kê tất cả presets (có thể lọc theo `ho_mo_hinh` như `'qwen3.8'`, `'gemma4'`)."""
        if ho_mo_hinh:
            q = ho_mo_hinh.lower()
            return {
                k: v["mo_ta"]
                for k, v in cls.PRESETS.items()
                if q in k.lower() or q in str(v.get("ho_mo_hinh", "")).lower()
            }
        return {k: v["mo_ta"] for k, v in cls.PRESETS.items()}

    @classmethod
    def tim_kiem_mo_hinh(cls, tu_khoa: str = "") -> List[Dict[str, Any]]:
        """Tìm kiếm mô hình trong danh mục theo từ khóa tên, họ mô hình hoặc mô tả."""
        q = tu_khoa.lower().strip()
        ket_qua = []
        for k, v in cls.PRESETS.items():
            if not q or q in k.lower() or q in v["mo_ta"].lower() or q in str(v.get("hf_id", "")).lower():
                ket_qua.append({"ten": k, **v})
        return ket_qua

    def to_dict(self) -> Dict[str, Any]:
        """Trả về dict tham số tương thích trực tiếp với `GPTModel(**config.to_dict())`."""
        return {
            "d_model": self.d_model,
            "so_dau": self.so_dau,
            "d_ff": self.d_ff,
            "so_block": self.so_block,
            "so_tu_vung": self.so_tu_vung,
            "do_dai_toi_da": self.do_dai_toi_da,
            "dropout": self.dropout,
        }

    def __repr__(self) -> str:
        return (
            f"ModelConfig(ten='{self.ten}', "
            f"d_model={self.d_model}, so_block={self.so_block}, "
            f"params={self.so_tham_so_str})"
        )
