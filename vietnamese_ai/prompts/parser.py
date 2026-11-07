"""PhanTichDauRa & SinhCoCauTruc - Phân tích và ép khuôn đầu ra có cấu trúc (JSON Schema, Pydantic, Dataclass) từ LLM."""

import ast
import dataclasses
import json
import re
from typing import Any, Dict, List, Optional, Union


class PhanTichDauRa:
    """
    Phân tích đầu ra có cấu trúc từ LLM.

    Hỗ trợ parse:
    - JSON từ text (kèm tự động sửa lỗi JSON hỏng `sua_json_loi`)
    - Ép khuôn theo `pydantic.BaseModel`, `@dataclass`, hoặc `dict` schema (`phan_tich_theo_schema`)
    - Markdown tables
    - Code blocks
    - Numbered/bulleted lists
    - Key-value pairs
    """

    def phan_tich_json(self, van_ban: str) -> Optional[Any]:
        """
        Parse JSON từ text (tự động tìm JSON block và sửa lỗi cú pháp thường gặp).
        """
        # Thử parse trực tiếp
        try:
            return json.loads(van_ban.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Tìm JSON trong code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", van_ban, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                repaired = self.sua_json_loi(json_match.group(1).strip())
                if repaired is not None:
                    return repaired

        # Tìm { } hoặc [ ]
        brace_match = re.search(r"\{.*\}", van_ban, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                repaired = self.sua_json_loi(brace_match.group(0))
                if repaired is not None:
                    return repaired

        bracket_match = re.search(r"\[.*\]", van_ban, re.DOTALL)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(0))
            except (json.JSONDecodeError, ValueError):
                repaired = self.sua_json_loi(bracket_match.group(0))
                if repaired is not None:
                    return repaired

        return self.sua_json_loi(van_ban)

    def sua_json_loi(self, van_ban: str) -> Optional[Any]:
        """
        Tự động sửa các lỗi JSON phổ biến do LLM sinh ra:
        - Dấu nháy đơn thay vì nháy kép
        - Dấu phẩy thừa ở cuối (`trailing comma`)
        - Hằng số Python `True`/`False`/`None`
        """
        s = van_ban.strip()
        brace = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if brace:
            s = brace.group(1)

        # Xóa dấu phẩy thừa trước } hoặc ]
        s_clean = re.sub(r",\s*([\}\]])", r"\1", s)
        try:
            return json.loads(s_clean)
        except Exception:
            pass

        # Thử ast.literal_eval cho dict/list kiểu Python (nháy đơn, True/False/None)
        try:
            val = ast.literal_eval(s_clean)
            if isinstance(val, (dict, list)):
                return val
        except Exception:
            pass

        # Thêm dấu nháy kép cho khóa chưa đóng nháy: {ten: "An", tuoi: 20}
        s_quoted_keys = re.sub(r"([\{\,]\s*)([a-zA-Z_]\w*)\s*:", r'\1"\2":', s_clean)
        s_quoted_keys = (
            s_quoted_keys.replace(": True", ": true")
            .replace(": False", ": false")
            .replace(": None", ": null")
        )
        try:
            return json.loads(s_quoted_keys)
        except Exception:
            return None

    @staticmethod
    def _ep_kieu_gia_tri(val: Any, kieu: Any) -> Any:
        """Ép kiểu an toàn cho một trường theo schema."""
        origin = getattr(kieu, "__origin__", None)
        if origin is Union:
            args = [a for a in getattr(kieu, "__args__", ()) if a is not type(None)]
            if args:
                kieu = args[0]
                origin = getattr(kieu, "__origin__", None)

        if kieu in (int, "int", "integer"):
            if isinstance(val, (int, float)):
                return int(val)
            nums = re.findall(r"-?\d+", str(val))
            return int(nums[0]) if nums else 0
        if kieu in (float, "float", "number"):
            if isinstance(val, (int, float)):
                return float(val)
            nums = re.findall(r"-?\d+\.?\d*", str(val))
            return float(nums[0]) if nums else 0.0
        if kieu in (bool, "bool", "boolean"):
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("true", "1", "yes", "co", "có", "đúng", "dung")
        if kieu in (list, List, "list", "array") or origin is list:
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return [x.strip() for x in val.split(",") if x.strip()]
            return [val] if val is not None else []
        if kieu in (dict, Dict, "dict", "object") or origin is dict:
            return val if isinstance(val, dict) else {}
        return str(val) if val is not None else ""

    def phan_tich_theo_schema(self, van_ban: str, schema: Any) -> Any:
        """
        Phân tích văn bản LLM và ép khuôn theo `pydantic.BaseModel`, `@dataclass` hoặc `dict` schema.
        """
        raw_data = self.phan_tich_json(van_ban)
        if not isinstance(raw_data, dict):
            raw_data = self.phan_tich_key_value(van_ban)

        # 1. Nếu là Pydantic BaseModel (v2 hoặc v1)
        if isinstance(schema, type) and (
            hasattr(schema, "model_validate") or hasattr(schema, "parse_obj")
        ):
            fields_info = getattr(schema, "model_fields", getattr(schema, "__fields__", {}))
            coerced: Dict[str, Any] = {}
            for fname, finfo in fields_info.items():
                ftype = getattr(finfo, "annotation", getattr(finfo, "outer_type_", str))
                if fname in raw_data:
                    coerced[fname] = self._ep_kieu_gia_tri(raw_data[fname], ftype)
                else:
                    default_val = getattr(finfo, "default", None)
                    if default_val is not None and str(default_val) != "PydanticUndefined":
                        coerced[fname] = default_val
                    else:
                        coerced[fname] = self._ep_kieu_gia_tri(van_ban, ftype)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(coerced)
            return schema.parse_obj(coerced)

        # 2. Nếu là Python @dataclass
        if isinstance(schema, type) and dataclasses.is_dataclass(schema):
            coerced_dc: Dict[str, Any] = {}
            for f in dataclasses.fields(schema):
                if f.name in raw_data:
                    coerced_dc[f.name] = self._ep_kieu_gia_tri(raw_data[f.name], f.type)
                elif f.default is not dataclasses.MISSING:
                    coerced_dc[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore
                    coerced_dc[f.name] = f.default_factory()  # type: ignore
                else:
                    coerced_dc[f.name] = self._ep_kieu_gia_tri(van_ban, f.type)
            return schema(**coerced_dc)

        # 3. Nếu là dict định nghĩa kiểu {"ten": str, "tuoi": int} hoặc JSON Schema {"properties": ...}
        if isinstance(schema, dict):
            if "properties" in schema and isinstance(schema["properties"], dict):
                props = schema["properties"]
                return {
                    k: self._ep_kieu_gia_tri(raw_data.get(k, van_ban), v.get("type", "string"))
                    for k, v in props.items()
                }
            return {
                k: self._ep_kieu_gia_tri(raw_data.get(k, van_ban), kieu)
                for k, kieu in schema.items()
            }

        return raw_data

    def phan_tich_bang(self, van_ban: str) -> List[Dict[str, str]]:
        """
        Parse markdown table.

        Returns:
            [{cot1: gia_tri1, cot2: gia_tri2}, ...]
        """
        lines = van_ban.strip().split("\n")
        table_lines = [
            line.strip()
            for line in lines
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]

        if len(table_lines) < 2:
            return []

        # Header
        header = [c.strip() for c in table_lines[0].split("|")[1:-1]]

        # Bỏ separator line
        rows = []
        for line in table_lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) == len(header):
                row = dict(zip(header, cells))
                rows.append(row)

        return rows

    def phan_tich_danh_sach(self, van_ban: str) -> List[str]:
        """
        Parse numbered or bulleted list.

        Returns:
            Danh sách items
        """
        items = []
        mau = re.compile(r"^\s*(?:\d+[\.\)]\s*|[\-\*\•]\s*)(.+)$", re.MULTILINE)

        for match in mau.finditer(van_ban):
            item = match.group(1).strip()
            if item:
                items.append(item)

        return items

    def phan_tich_code_blocks(self, van_ban: str) -> List[Dict[str, str]]:
        """
        Parse code blocks.

        Returns:
            [{ngon_ngu, code}, ...]
        """
        blocks = []
        mau = re.compile(r"```(\w*)\n?(.*?)\n?```", re.DOTALL)

        for match in mau.finditer(van_ban):
            blocks.append(
                {
                    "ngon_ngu": match.group(1) or "text",
                    "code": match.group(2).strip(),
                }
            )

        return blocks

    def phan_tich_key_value(self, van_ban: str) -> Dict[str, str]:
        """
        Parse key-value pairs.

        Returns:
            {key: value}
        """
        ket_qua = {}
        mau = re.compile(r"^[\-\*]?\s*(.+?):\s*(.+)$", re.MULTILINE)

        for match in mau.finditer(van_ban):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key and value:
                ket_qua[key] = value

        return ket_qua

    def phan_tich_so(self, van_ban: str) -> List[float]:
        """Trích xuất tất cả số từ text."""
        matches = re.findall(r"-?\d+\.?\d*", van_ban)
        return [float(m) for m in matches]

    def trich_cau_tra_loi(self, van_ban: str) -> str:
        """
        Trích xuất câu trả lời chính từ LLM output.

        Tìm phần text quan trọng nhất (bỏ qua reasoning).
        """
        # Tìm sau "Trả lời:" hoặc "Answer:"
        answer_match = re.search(
            r"(?:Trả lời|Answer|Kết quả):\s*(.+)",
            van_ban,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Tìm câu cuối cùng
        cau = re.split(r"[.!?\n]", van_ban.strip())
        cau = [c.strip() for c in cau if c.strip()]
        if cau:
            return cau[-1]

        return van_ban.strip()

    def __repr__(self) -> str:
        return "PhanTichDauRa()"


class SinhCoCauTruc:
    """
    Bộ sinh đầu ra có cấu trúc (Structured Output Generator) kết hợp LLM và `PhanTichDauRa`.

    Cho phép truyền vào `pydantic.BaseModel`, `@dataclass` hoặc `dict` schema,
    tự động tạo prompt hướng dẫn JSON Schema, gọi mô hình sinh văn bản và
    kiểm tra/sửa lỗi kiểu dữ liệu tự động.
    """

    def __init__(self, mo_hinh: Any = None):
        self.mo_hinh = mo_hinh
        self.parser = PhanTichDauRa()

    def _tao_prompt_schema(self, prompt: str, schema: Any) -> str:
        if isinstance(schema, type) and dataclasses.is_dataclass(schema):
            keys = [f.name for f in dataclasses.fields(schema)]
        elif isinstance(schema, type) and hasattr(schema, "model_fields"):
            keys = list(schema.model_fields.keys())
        elif isinstance(schema, dict):
            keys = list(schema.get("properties", schema).keys())
        else:
            keys = []
        danh_sach_khoa = ", ".join(keys)
        return (
            f"{prompt}\n"
            f"Hãy trả lời dưới dạng JSON hợp lệ với các trường: {{{danh_sach_khoa}}}."
        )

    def sinh(self, prompt: str, schema: Any, so_lan_thu: int = 2) -> Any:
        """
        Gọi LLM sinh văn bản và ép khuôn kết quả trả về đúng với `schema`.
        """
        full_prompt = self._tao_prompt_schema(prompt, schema)
        last_err: Optional[Exception] = None

        for _ in range(max(1, so_lan_thu)):
            try:
                if self.mo_hinh is None:
                    raw_out = prompt
                elif hasattr(self.mo_hinh, "sinh_van_ban"):
                    raw_out = str(self.mo_hinh.sinh_van_ban(full_prompt))
                elif callable(self.mo_hinh):
                    raw_out = str(self.mo_hinh(full_prompt))
                else:
                    raw_out = prompt

                # Kết hợp cả thông tin trong prompt gốc nếu model ngắn gọn
                combined = f"{raw_out}\n{prompt}" if "{" not in raw_out else raw_out
                return self.parser.phan_tich_theo_schema(combined, schema)
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise last_err
        return self.parser.phan_tich_theo_schema(prompt, schema)
