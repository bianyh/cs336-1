import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from .base import Linear



class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x / (1 + torch.exp(-x))
        return x * torch.sigmoid(x)
    

class SwiGLU(nn.Module):
    # 有门控的情况下，最好把d_ff设置成d_model的8/3倍，否则是4倍
    # （因为有门控，维度相同的情况下，参数量变为了3/2，所以保持参数量相同，维度是2/3）
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cpu'
        
        # self.up_weight = nn.Parameter(torch.rand(d_ff, d_model, dtype=self.dtype, device=self.device))
        # self.gate_weight = nn.Parameter(torch.rand(d_ff, d_model, dtype=self.dtype, device=self.device))
        # self.down_weight = nn.Parameter(torch.rand(d_model, d_ff, dtype=self.dtype, device=self.device))

        self.up_linear = Linear(d_model, d_ff, self.device, self.dtype)
        self.gate_linear = Linear(d_model, d_ff, self.device, self.dtype)
        self.down_linear = Linear(d_ff, d_model, self.device, self.dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # up_x = x.to(self.up_weight.device) @ self.up_weight.T

        # gate_x = x.to(self.gate_weight.device) @ self.gate_weight.T

        # in_x = self.silu(gate_x).to(up_x.device) * up_x

        # out_x = in_x.to(self.down_weight.device) @ self.down_weight.T

        up_x = self.up_linear(x)

        gate_x = self.gate_linear(x)

        in_x = self.silu(gate_x) * up_x

        out_x = self.down_linear(in_x)

        return out_x.type_as(x)





if __name__ == '__main__':
    silu = SiLU()
    x = torch.rand(2)
    print(silu(x))