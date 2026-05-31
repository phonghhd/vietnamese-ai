"""Lõi Selective State Space Model (Mamba) bằng PyTorch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    Module Selective State Space Model (Mamba) cơ bản.
    Mô phỏng cơ chế lựa chọn (Selective) để lọc thông tin quan trọng.
    Độ phức tạp tuyến tính O(N) theo độ dài chuỗi, giải quyết bài toán O(N^2) của Transformer.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 1. Chi nhánh Input (In projection)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 1D Convolution để xử lý cục bộ (như Mamba)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. Các tham số Selective (Phụ thuộc vào Input)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Tham số State Space (A log)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (Batch, SeqLen, D_Model)
        Returns:
            output: (Batch, SeqLen, D_Model)
        """
        batch, seqlen, _ = hidden_states.shape

        # 1. In Projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # 2. 1D Convolution
        # Cần chuyển sang shape (Batch, Channels, SeqLen)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen] # Lấy đúng seqlen
        x = x.transpose(1, 2)
        
        # Activation
        x = F.silu(x)

        # 3. Tính Delta, B, C (Selective mechanism)
        # Các trọng số này biến đổi theo từng token
        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        # Softplus cho Delta (để luôn dương)
        delta = F.softplus(self.dt_proj(delta)) # (Batch, SeqLen, D_Inner)

        # Trạng thái A
        A = -torch.exp(self.A_log.float()) # (D_Inner, D_State)

        # 4. Selective Scan (Thuật toán quét tuần tự O(N) trên PyTorch)
        # Thay vì tính Attention O(N^2), ta cập nhật State (h) từ trái qua phải
        y = self._selective_scan_pytorch(x, delta, A, B, C)
        
        # D residual
        y = y + x * self.D
        
        # Nhân Gated (như SwiGLU)
        y = y * F.silu(z)

        # 5. Out Projection
        out = self.out_proj(y)
        return out

    def _selective_scan_pytorch(self, x, delta, A, B, C):
        """
        Thực hiện Scan vòng lặp thuần PyTorch.
        Để đạt tốc độ tối đa trên Production, phần này cần được viết bằng C++ Triton.
        Ở đây dùng Vòng lặp For minh họa cơ chế Toán học O(N) - Chạy tốt trên CPU.
        """
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]

        # Khởi tạo h: (Batch, D_Inner, D_State)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(batch, seqlen, d_inner, device=x.device, dtype=x.dtype)

        # Bước tính rời rạc hóa (Discretization: A_bar, B_bar)
        for t in range(seqlen):
            delta_t = delta[:, t, :].unsqueeze(-1) # (B, D_inner, 1)
            x_t = x[:, t, :].unsqueeze(-1) # (B, D_inner, 1)
            B_t = B[:, t, :].unsqueeze(1) # (B, 1, D_state)
            C_t = C[:, t, :].unsqueeze(1) # (B, 1, D_state)
            
            # Khai triển Euler chuẩn xác
            A_bar = torch.exp(delta_t * A) # (B, D_inner, D_state)
            B_bar_x = delta_t * B_t * x_t # (B, D_inner, D_state)
            
            # Cập nhật State (h_t = A_bar * h_{t-1} + B_bar * x_t)
            h = A_bar * h + B_bar_x
            
            # Tính Output (y_t = C * h_t)
            y[:, t, :] = torch.sum(h * C_t, dim=-1)
            
        return y
