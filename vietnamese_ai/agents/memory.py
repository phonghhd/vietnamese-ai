import json
from typing import Any, Dict, List, Optional


class BoNhoTacTu:
    """
    Quản lý bộ nhớ (lịch sử hội thoại) cho tác tử.
    Hỗ trợ các vai trò: system, user, assistant, tool.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        self.lich_su: List[Dict[str, Any]] = []
        if system_prompt:
            self.them_tin_nhan("system", system_prompt)

    @property
    def system_prompt(self) -> str:
        if self.lich_su and self.lich_su[0]["role"] == "system":
            return self.lich_su[0]["content"]
        return ""

    @system_prompt.setter
    def system_prompt(self, value: str):
        if self.lich_su and self.lich_su[0]["role"] == "system":
            self.lich_su[0]["content"] = value
        else:
            self.lich_su.insert(0, {"role": "system", "content": value})

    def them_tin_nhan(self, vai_tro: str, noi_dung: str, ten_cong_cu: Optional[str] = None):
        """
        Thêm một tin nhắn vào lịch sử.

        Args:
            vai_tro (str): 'system', 'user', 'assistant', 'tool'
            noi_dung (str): Nội dung tin nhắn
            ten_cong_cu (Optional[str]): Tên công cụ (chỉ dùng khi vai_tro='tool')
        """
        if vai_tro not in ["system", "user", "assistant", "tool"]:
            raise ValueError(f"Vai trò '{vai_tro}' không hợp lệ. Phải là system, user, assistant hoặc tool.")

        tin_nhan = {
            "role": vai_tro,
            "content": noi_dung
        }
        if vai_tro == "tool" and ten_cong_cu:
            tin_nhan["name"] = ten_cong_cu

        self.lich_su.append(tin_nhan)

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Trả về toàn bộ lịch sử hiện tại."""
        return self.lich_su

    def xoa_lich_su(self, giu_lai_system: bool = True):
        """Xóa lịch sử, có thể giữ lại system prompt."""
        if giu_lai_system and self.lich_su and self.lich_su[0]["role"] == "system":
            system_msg = self.lich_su[0]
            self.lich_su = [system_msg]
        else:
            self.lich_su = []

    def lay_noi_dung_chuoi(self) -> str:
        """
        Chuyển lịch sử thành một chuỗi duy nhất, thích hợp cho các mô hình không hỗ trợ chat template.
        """
        chuoi = ""
        for msg in self.lich_su:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chuoi += f"Hệ thống: {content}\n"
            elif role == "user":
                chuoi += f"Người dùng: {content}\n"
            elif role == "assistant":
                chuoi += f"Trợ lý: {content}\n"
            elif role == "tool":
                name = msg.get("name", "Unknown")
                chuoi += f"Kết quả công cụ [{name}]: {content}\n"
        return chuoi

    def to_json(self) -> str:
        return json.dumps(self.lich_su, ensure_ascii=False, indent=2)

    def from_json(self, json_str: str):
        self.lich_su = json.loads(json_str)
