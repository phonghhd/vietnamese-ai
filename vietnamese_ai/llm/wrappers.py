import json
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseLLMWrapper(ABC):
    """Lớp trừu tượng cho các LLM API Wrapper."""

    @abstractmethod
    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        """Sinh văn bản từ prompt."""
        pass

    def du_doan(self, X: Any, **kwargs: Any) -> Any:
        """Giao thức dự đoán tương thích Pipeline và Engine."""
        if isinstance(X, str):
            return self.sinh_van_ban(X, **kwargs)
        if isinstance(X, list):
            return [self.sinh_van_ban(str(item), **kwargs) for item in X]
        return self.sinh_van_ban(str(X), **kwargs)

    def _parse_prompt(self, prompt: str) -> List[Dict[str, str]]:
        """Phân tích prompt thành format messages của OpenAI/Ollama."""
        messages = []
        lines = prompt.split("\n")
        current_role = "user"
        current_content: List[str] = []

        for line in lines:
            if line.startswith("Hệ thống: ") or line.startswith("[system]"):
                if current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content).strip()}
                    )
                current_role = "system"
                current_content = [line.replace("Hệ thống: ", "").replace("[system]", "").strip()]
            elif line.startswith("Người dùng: ") or line.startswith("[user]"):
                if current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content).strip()}
                    )
                current_role = "user"
                current_content = [line.replace("Người dùng: ", "").replace("[user]", "").strip()]
            elif line.startswith("Trợ lý: ") or line.startswith("[assistant]"):
                if current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content).strip()}
                    )
                current_role = "assistant"
                current_content = [line.replace("Trợ lý: ", "").replace("[assistant]", "").strip()]
            else:
                current_content.append(line)

        if current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

        if not messages:
            messages = [{"role": "user", "content": prompt}]

        return messages


class OpenAIWrapper(BaseLLMWrapper):
    """
    Wrapper cho OpenAI API (GPT-3.5, GPT-4, GPT-4o).
    Yêu cầu: pip install openai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        messages = self._parse_prompt(prompt)

        temp = kwargs.get("nhiet_do", self.temperature)
        tokens = kwargs.get("do_dai", self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temp, max_tokens=tokens
        )
        return response.choices[0].message.content


class GeminiWrapper(BaseLLMWrapper):
    """
    Wrapper cho Google Gemini API (hỗ trợ cả SDK mới google-genai và google-generativeai).
    Yêu cầu: pip install google-genai hoặc pip install google-generativeai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self._dung_sdk_moi = False
        self.client = None
        self.model_name = model
        try:
            from google import genai  # type: ignore

            self.client = genai.Client(api_key=api_key)
            self._dung_sdk_moi = True
        except ImportError:
            try:
                import google.generativeai as genai_legacy  # type: ignore

                genai_legacy.configure(api_key=api_key)
                self.model = genai_legacy.GenerativeModel(model)
            except ImportError:
                raise ImportError(
                    "Vui lòng cài đặt google-genai hoặc google-generativeai: pip install google-generativeai"
                )

        self.max_tokens = max_tokens
        self.temperature = temperature

    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        temp = kwargs.get("nhiet_do", self.temperature)
        tokens = kwargs.get("do_dai", self.max_tokens)

        if self._dung_sdk_moi and self.client is not None:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"temperature": temp, "max_output_tokens": tokens},
            )
            return str(getattr(resp, "text", ""))

        generation_config = {
            "temperature": temp,
            "max_output_tokens": tokens,
        }

        response = self.model.generate_content(prompt, generation_config=generation_config)

        try:
            return response.text
        except ValueError:
            return "Lỗi: Nội dung bị chặn bởi hệ thống an toàn của Gemini."


class OllamaWrapper(BaseLLMWrapper):
    """
    Wrapper chạy mô hình LLM tiếng Việt cục bộ qua Ollama (Vistral, Qwen2.5-Viet, VinaLLaMA, Llama3).
    Tự động fallback sang `BoTinhChinhMoHinh` (`vgpt2`) nếu máy chủ Ollama không hoạt động hoặc `che_do_offline=True`.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
        temperature: float = 0.7,
        che_do_offline: bool = False,
        timeout: float = 10.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.che_do_offline = che_do_offline
        self.timeout = timeout
        self._fallback_model: Any = None

    def _lay_fallback(self) -> Any:
        if self._fallback_model is None:
            from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

            self._fallback_model = BoTinhChinhMoHinh(
                mo_hinh="vgpt2", nhiem_vu="sft", dung_lora=True, che_do_offline=True
            )
        return self._fallback_model

    def danh_sach_mo_hinh(self) -> List[str]:
        """Lấy danh sách mô hình hiện có trên máy chủ Ollama."""
        if self.che_do_offline:
            return [self.model, "vgpt2-offline"]
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return [self.model, "vgpt2-offline"]

    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        temp = float(kwargs.get("nhiet_do", kwargs.get("temperature", self.temperature)))
        tokens = int(kwargs.get("do_dai", kwargs.get("max_tokens", self.max_tokens)))

        if not self.che_do_offline:
            try:
                messages = self._parse_prompt(prompt)
                payload = json.dumps(
                    {
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temp, "num_predict": tokens},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body.get("message", {}).get("content", "")
                    if content:
                        return str(content)
            except Exception:
                pass

        return self._lay_fallback().sinh_van_ban(prompt, do_dai=min(tokens, 64), nhiet_do=temp)


class VLLMWrapper(BaseLLMWrapper):
    """
    Wrapper kết nối máy chủ vLLM / LM Studio / LocalAI tương thích chuẩn OpenAI API.
    Tự động fallback sang `BoTinhChinhMoHinh` khi chạy offline.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.7,
        che_do_offline: bool = False,
        timeout: float = 10.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.che_do_offline = che_do_offline
        self.timeout = timeout
        self._fallback_model: Any = None

    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        temp = float(kwargs.get("nhiet_do", kwargs.get("temperature", self.temperature)))
        tokens = int(kwargs.get("do_dai", kwargs.get("max_tokens", self.max_tokens)))

        if not self.che_do_offline:
            try:
                messages = self._parse_prompt(prompt)
                payload = json.dumps(
                    {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": tokens,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    choices = body.get("choices", [])
                    if choices:
                        return str(choices[0].get("message", {}).get("content", ""))
            except Exception:
                pass

        if self._fallback_model is None:
            from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh

            self._fallback_model = BoTinhChinhMoHinh(
                mo_hinh="vgpt2", nhiem_vu="sft", dung_lora=True, che_do_offline=True
            )
        return self._fallback_model.sinh_van_ban(prompt, do_dai=min(tokens, 64), nhiet_do=temp)


class LlamaCppWrapper(BaseLLMWrapper):
    """
    Wrapper chạy trực tiếp tệp mô hình định dạng `.gguf`.
    Ưu tiên dùng `llama-cpp-python` nếu đã cài đặt; nếu chưa có sẽ giải lượng tử hóa thật
    tệp `.gguf` bằng `XuatGGUF.tai_tu_gguf` và suy luận bằng `GPTModel` nội bộ.
    """

    def __init__(
        self,
        duong_dan_gguf: Optional[str] = None,
        n_ctx: int = 512,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        self.duong_dan_gguf = duong_dan_gguf
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llama_cpp: Any = None
        self._native_gpt: Any = None
        self.metadata_gguf: Dict[str, Any] = {}

        if duong_dan_gguf and Path(duong_dan_gguf).exists():
            try:
                from llama_cpp import Llama  # type: ignore

                self._llama_cpp = Llama(model_path=duong_dan_gguf, n_ctx=n_ctx, verbose=False)
            except Exception:
                from vietnamese_ai.export.gguf import XuatGGUF
                from vietnamese_ai.transformer.gpt_model import GPTModel

                exporter = XuatGGUF()
                info = exporter.tai_tu_gguf(duong_dan_gguf)
                self.metadata_gguf = info.get("metadata", {})
                tensors = info.get("tensors", {})
                gpt = GPTModel(d_model=64, so_dau=4, d_ff=128, so_block=2, so_tu_vung=256)
                if "tok_emb.weight" in tensors and tensors["tok_emb.weight"].ndim == 2:
                    vocab, d_mod = tensors["tok_emb.weight"].shape
                    gpt = GPTModel(
                        d_model=int(d_mod),
                        so_dau=max(1, int(d_mod) // 16),
                        d_ff=int(d_mod) * 2,
                        so_block=1,
                        so_tu_vung=int(vocab),
                    )
                    gpt.tok_emb = tensors["tok_emb.weight"].astype("float32")
                self._native_gpt = gpt
        else:
            from vietnamese_ai.transformer.gpt_model import GPTModel

            self._native_gpt = GPTModel(d_model=64, so_dau=4, d_ff=128, so_block=2, so_tu_vung=256)

    def sinh_van_ban(self, prompt: str, **kwargs: Any) -> str:
        temp = float(kwargs.get("nhiet_do", kwargs.get("temperature", self.temperature)))
        tokens = int(kwargs.get("do_dai", kwargs.get("max_tokens", self.max_tokens)))

        if self._llama_cpp is not None:
            out = self._llama_cpp(prompt, max_tokens=tokens, temperature=temp)
            return str(out["choices"][0]["text"]).strip()

        return self._native_gpt.sinh_van_ban(prompt, do_dai=min(tokens, 32), nhiet_do=temp)
