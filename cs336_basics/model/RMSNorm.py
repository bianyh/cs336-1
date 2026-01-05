import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from .base import Linear



class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cpu'
        
        self.weight = nn.Parameter(torch.rand(d_model, dtype=self.dtype, device=self.device))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * self._norm(x.float()).type_as(x).to(self.weight.device)).type_as(x)


if __name__=="__main__":
    model = RMSNorm(2, device='cuda:2')
    print(model)
    x = torch.rand(2, device='cpu')
    print(model(x))
    pass