import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLLMWrapper(ABC):
    """Lớp trừu tượng cho các LLM API Wrapper."""
    
    @abstractmethod
    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        """Sinh văn bản từ prompt."""
        pass

class OpenAIWrapper(BaseLLMWrapper):
    """
    Wrapper cho OpenAI API (GPT-3.5, GPT-4).
    Yêu cầu: pip install openai
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1024, temperature: float = 0.7):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt openai: pip install openai")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def _parse_prompt(self, prompt: str) -> List[Dict[str, str]]:
        """Phân tích prompt thành format messages của OpenAI."""
        # Cách phân tích đơn giản: Chia theo tag [system], [user], [assistant]
        # Nếu không có tag, coi toàn bộ là user
        messages = []
        lines = prompt.split('\n')
        current_role = "user"
        current_content = []
        
        for line in lines:
            if line.startswith("Hệ thống: ") or line.startswith("[system]"):
                if current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "system"
                current_content = [line.replace("Hệ thống: ", "").replace("[system]", "").strip()]
            elif line.startswith("Người dùng: ") or line.startswith("[user]"):
                if current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "user"
                current_content = [line.replace("Người dùng: ", "").replace("[user]", "").strip()]
            elif line.startswith("Trợ lý: ") or line.startswith("[assistant]"):
                if current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "assistant"
                current_content = [line.replace("Trợ lý: ", "").replace("[assistant]", "").strip()]
            else:
                current_content.append(line)
                
        if current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            
        # Fallback nếu parse bị rỗng
        if not messages:
            messages = [{"role": "user", "content": prompt}]
            
        return messages

    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        messages = self._parse_prompt(prompt)
        
        # Override tham số nếu có truyền vào
        temp = kwargs.get("nhiet_do", self.temperature)
        tokens = kwargs.get("do_dai", self.max_tokens)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens
        )
        return response.choices[0].message.content


class GeminiWrapper(BaseLLMWrapper):
    """
    Wrapper cho Google Gemini API.
    Yêu cầu: pip install google-generativeai
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", max_tokens: int = 1024, temperature: float = 0.7):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Vui lòng cài đặt google-generativeai: pip install google-generativeai")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        # Gemini có model.generate_content
        # Với Gemini, ta có thể quăng trực tiếp đoạn hội thoại vì model xử lý context khá tốt.
        
        temp = kwargs.get("nhiet_do", self.temperature)
        tokens = kwargs.get("do_dai", self.max_tokens)
        
        generation_config = {
            "temperature": temp,
            "max_output_tokens": tokens,
        }
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        try:
            return response.text
        except ValueError:
            # Xử lý trường hợp bị chặn bởi filter an toàn
            return "Lỗi: Nội dung bị chặn bởi hệ thống an toàn của Gemini."
