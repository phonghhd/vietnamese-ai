# Hướng Dẫn Sử Dụng EvoNet-Studio, Bảo Mật Tự Vá & Cách Mạng CPU (v21 - v23)

Tài liệu này hướng dẫn bạn cách khai thác toàn bộ sức mạnh của hệ sinh thái Vietnamese AI Framework từ phiên bản v21.0 đến v23.0.

## 1. EvoNet-Studio Super Core (v21.0)
Hệ thống lõi Zero-Dependency giúp quản lý toàn bộ hệ sinh thái như một SaaS thực thụ.

### Orchestrator & DAG Workflow
Điều phối các Tác tử (Agents) hoạt động song song để hoàn thành một nhiệm vụ lớn.

```python
from vietnamese_ai.orchestrator import VOrchestrator
from vietnamese_ai.workflow import VWorkflow

orchestrator = VOrchestrator()
workflow = VWorkflow(name="Phan_Tich_Tai_Chinh")

workflow.add_node("AgentThuThap", agent_thu_thap)
workflow.add_node("AgentPhanTich", agent_phan_tich)
workflow.connect("AgentThuThap", "AgentPhanTich")

orchestrator.run_workflow(workflow)
```

## 2. Agentic Evolution & Self-Healing (v22.0)
Hệ thống v22.0 mang đến khả năng tự nhận thức, tự phòng thủ và tự lập trình cho các Tác tử.

### BlueTeamAgent & Self-Healing Sandbox
Hệ thống tự động phân tích mã độc từ RedTeam và vá lỗ hổng (patching) trực tiếp vào file cấu hình bảo mật `sandbox_rules.json`.

```python
from vietnamese_ai.security.blue_team import BlueTeamAgent

# Kích hoạt Đặc vụ BlueTeam
blue_team = BlueTeamAgent(llm=llm)

# Agent sẽ phân tích payload độc hại và tự động sinh ra luật cấm
# để khóa cứng lỗ hổng trong Sandbox
blue_team.xu_ly_canh_bao(ma_doc="import os; os.system('rm -rf /')", li_do_thanh_cong="Bypass via OS module")
```

### DevOps Agent (Safe Auto-Coding)
Đặc vụ tự động lập trình thông qua 2 lớp bảo vệ: AST Validation và Dry-Run Pipeline. Nếu code bị lỗi, Agent sẽ nhận lỗi và tự động sửa (Self-Correction).

```python
from vietnamese_ai.agents.devops import DevOpsAgent

devops = DevOpsAgent(llm=llm)
devops.chay("Viết một hàm Python tính giai thừa đệ quy và sinh bài test pytest cho nó.")
# Code sẽ được lưu an toàn tại thư mục cách ly `scratch/`
```

## 3. Cuộc Cách Mạng CPU & Năng Lượng (v23.0)
Phiên bản v23.0 tập trung vào việc dân chủ hóa AI, cho phép chạy suy luận và huấn luyện ngay cả trên những máy tính không có GPU đắt tiền.

### EvoKernelCPU (1.58-bit Inference)
Thực thi mạng nơ-ron BitNet 1.58-bit trên CPU sử dụng phép tính thuần Cộng/Trừ (Add-only Matmul). Loại bỏ hoàn toàn phép nhân số thập phân nặng nề.

```python
import torch
from vietnamese_ai.compression.extreme import BitLinear

# Module BitLinear đã tích hợp sẵn tính năng tự động điều phối (Adaptive Dispatch)
# Nếu chạy trên CPU, nó tự động sử dụng EvoKernelCPU cực kỳ nhẹ
layer = BitLinear(in_features=512, out_features=128)
x = torch.randn(1, 512)
out = layer(x) # Chạy bằng CPU Kernel nhanh ngang ngửa GPU!
```

### BitLoRA (Huấn Luyện Trên CPU)
Tinh chỉnh (Fine-tune) mô hình LLM trên thiết bị yếu bằng cách khóa gốc (Base) 1.58-bit và mở rộng Adapter FP32 siêu nhỏ.

```python
from vietnamese_ai.fine_tuning.bitlora import BitLoRALinear

# Gói lớp 1.58-bit vào trong BitLoRA
lora_layer = BitLoRALinear(layer, r=8)

# Tiến hành huấn luyện trực tiếp trên CPU
out = lora_layer(x)
```

### PowerManager (Quản Lý Năng Lượng Thông Minh)
Tự động cảm nhận % Pin và Nhiệt độ để quyết định Precision phù hợp.

```python
from vietnamese_ai.mobile.power_manager import PowerManager

# Nếu rút sạc và pin còn dưới 20%, hệ thống trả về cấu hình:
# precision='1.58-bit', use_bitnet=True để sinh tồn!
config = PowerManager.dieu_tiet_do_chinh_xac()
print(f"Hệ thống khuyên dùng cấu hình: {config['precision']}")
```
