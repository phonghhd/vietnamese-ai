# Prompt Engineering

## Tổng quan

Bộ công cụ Prompt Engineering cho tiếng Việt:

- **MauPrompt** - Prompt templates với `{{variables}}`
- **ChuoiPrompt** - Chain-of-thought, few-shot, prompt chains
- **LuongAnToan** - Guardrails, PII detection, content filtering
- **PhanTichDauRa** - Parse JSON, table, list từ LLM output

---

## MauPrompt

Quản lý prompt templates với biến và conditional logic.

### Cơ bản

```python
from vietnamese_ai import MauPrompt

mau = MauPrompt(
    mau="Tóm tắt văn bản sau: {{noi_dung}}\nYêu cầu: {{yeu_cau}}",
    ten="tom_tat",
    mo_ta="Tóm tắt văn bản",
    bien_mac_dinh={"yeu_cau": "Tóm tắt trong 3-5 câu"},
)

prompt = mau.render(noi_dung="Trí tuệ nhân tạo đang phát triển...")
```

### Đăng ký template

```python
MauPrompt.from_template(
    "phan_tich",
    "Phân tích {{chu_de}} trong ngữ cảnh {{ngu_canh}}",
    mo_ta="Phân tích vấn đề",
)

mau = MauPrompt.lay_template("phan_tich")
prompt = mau.render(chu_de="AI", ngu_canh="công nghệ")
```

### Template mặc định

```python
templates = MauPrompt.danh_sach_mau_mac_dinh()
# "tom_tat", "phan_tich", "dich_thuat", "hoi_dap", "sinh_code", "phan_biet"

mau = templates["hoi_dap"]
prompt = mau.render(tai_lieu="Thông tin về AI...", cau_hoi="AI là gì?")
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `mau` | — | Template string với `{{bien}}` |
| `ten` | `"custom"` | Tên template |
| `bien_mac_dinh` | `{}` | Giá trị mặc định cho biến |

---

## ChuoiPrompt

Chain prompt - kết hợp nhiều prompt thành pipeline.

### Chain-of-Thought (CoT)

```python
from vietnamese_ai import ChuoiPrompt

chain = ChuoiPrompt(ham_sinh_mac_dinh=my_llm)

prompt = chain.tao_cot_prompt(
    cau_hoi="Tại sao AI quan trọng?",
    ngu_canh="Trong bối cảnh công nghệ 4.0",
)
# Tự động thêm "Hãy suy nghĩ từng bước: Bước 1, Bước 2, Bước 3"
```

### Few-shot prompting

```python
chain = ChuoiPrompt()
chain.them_few_shot("1 + 1", "2")
chain.them_few_shot("2 + 3", "5")

prompt = chain.tao_few_shot_prompt("4 + 7")
```

### Sequential chain

```python
from vietnamese_ai import MauPrompt, ChuoiPrompt

mau_phan_tich = MauPrompt("Phân tích: {{chu_de}}\nKết quả trước: {{ket_qua_truoc}}")
mau_tom_tat = MauPrompt("Tóm tắt phân tích: {{ket_qua_truoc}}")

chain = ChuoiPrompt()
chain.them_buoc("phan_tich", mau_phan_tich)
chain.them_buoc("tom_tat", mau_tom_tat)

ket_qua = chain.thuc_hien(ham_sinh=my_llm, bien={"chu_de": "AI"})
```

### Conditional branching

```python
chain = ChuoiPrompt()
chain.them_buoc(
    "kiem_tra",
    MauPrompt("Kiểm tra: {{noi_dung}}"),
)
chain.them_buoc(
    "sua_loi",
    MauPrompt("Sửa lỗi: {{ket_qua_truoc}}"),
    dieu_kien=lambda ctx: "lỗi" in ctx.get("ket_qua_truoc", "").lower(),
)
```

---

## LuongAnToan

Guardrails cho LLM output - kiểm tra an toàn và chất lượng.

### Cơ bản

```python
from vietnamese_ai import LuongAnToan

guardrail = LuongAnToan(
    tu_cam=["từ_cấm_1", "từ_cấm_2"],
    toi_da_do_dai=5000,
    chan_pii=True,
)

ket_qua = guardrail.kiem_tra("Nội dung cần kiểm tra")
if ket_qua["an_toan"]:
    print("Nội dung an toàn")
else:
    print(f"Lỗi: {ket_qua['loi']}")
```

### PII Detection

```python
guardrail = LuongAnToan(chan_pii=True)

noi_dung = "Liên hệ Nguyễn Văn A, SĐT: 0912345678, email: a@example.com"
noi_dung_loc, pii = guardrail.loc_pii(noi_dung)

print(noi_dung_loc)
# "Liên hệ Nguyễn Văn A, SĐT: [SĐT đã ẩn], email: [Email đã ẩn]"
print(pii)
# {"so_dien_thoai": ["0912345678"], "email": ["a@example.com"], ...}
```

### Format validation

```python
guardrail = LuongAnToan(dinh_dang_yeu_cau="json")
ket_qua = guardrail.kiem_tra('{"key": "value"}')
# {"an_toan": True, ...}

guardrail = LuongAnToan(dinh_dang_yeu_cau="markdown")
```

### Custom rules

```python
def rule_khong_so(noi_dung):
    if any(c.isdigit() for c in noi_dung):
        return False, "Không được chứa số"
    return True, ""

guardrail = LuongAnToan(rules=[rule_khong_so])
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `tu_cam` | `[]` | Danh sách từ cấm |
| `toi_da_do_dai` | `10000` | Độ dài tối đa |
| `toi_thieu_do_dai` | `0` | Độ dài tối thiểu |
| `chan_pii` | `False` | Bật PII detection |
| `dinh_dang_yeu_cau` | `None` | Định dạng: `json`, `markdown`, `code`, `number` |

---

## PhanTichDauRa

Phân tích đầu ra có cấu trúc từ LLM.

### Parse JSON

```python
from vietnamese_ai import PhanTichDauRa

parser = PhanTichDauRa()

output = """
Đây là kết quả phân tích:
```json
{"diem": 8.5, "nhan_xet": "Tốt"}
```
"""

data = parser.phan_tich_json(output)
# {"diem": 8.5, "nhan_xet": "Tốt"}
```

### Parse bảng

```python
output = """
| Tên | Điểm |
|-----|------|
| A   | 8.5  |
| B   | 9.0  |
"""

rows = parser.phan_tich_bang(output)
# [{"Tên": "A", "Điểm": "8.5"}, {"Tên": "B", "Điểm": "9.0"}]
```

### Parse danh sách

```python
output = """
1. Học máy
2. Học sâu
3. Xử lý ngôn ngữ tự nhiên
"""

items = parser.phan_tich_danh_sach(output)
# ["Học máy", "Học sâu", "Xử lý ngôn ngữ tự nhiên"]
```

### Parse code blocks

```python
output = '```python\nprint("hello")\n```'

blocks = parser.phan_tich_code_blocks(output)
# [{"ngon_ngu": "python", "code": 'print("hello")'}]
```

### Parse key-value

```python
output = """
Tên: Nguyễn Văn A
Tuổi: 25
"""

data = parser.phan_tich_key_value(output)
# {"Tên": "Nguyễn Văn A", "Tuổi": "25"}
```

### Trích xuất câu trả lời

```python
output = "Suy nghĩ: ... Bước 1: ... Bước 2: ... Trả lời: AI là trí tuệ nhân tạo."
cau_tra_loi = parser.trich_cau_tra_loi(output)
# "AI là trí tuệ nhân tạo."
```

---

## Kết hợp Full Pipeline

```python
from vietnamese_ai import MauPrompt, ChuoiPrompt, LuongAnToan, PhanTichDauRa

guardrail = LuongAnToan(chan_pii=True, tu_cam=["spam"])
parser = PhanTichDauRa()
mau = MauPrompt("Phân tích sentiment: {{van_ban}}\nTrả về JSON: {\"sentiment\": \"...\", \"score\": ...}")

prompt = mau.render(van_ban="Sản phẩm rất tốt!")

if guardrail.kiem_tra(prompt)["an_toan"]:
    output = my_llm(prompt)
    data = parser.phan_tich_json(output)
    print(data)
```
