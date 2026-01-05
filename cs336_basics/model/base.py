import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(self, 
                 in_feature: int, 
                 out_feature: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cuda'
        self.weight = nn.Parameter(torch.rand(out_feature, in_feature, dtype=self.dtype, device=self.device))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.weight.device) @ self.weight.T



class Embedding(nn.Module):
    def __init__(self,
                 num_embedding: int, # 词汇表大小
                 embedding_dim: int = 512, # 嵌入向量的维度
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        # "vocab_size d_model"
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cpu'
        self.weight = nn.Parameter(torch.rand(num_embedding, embedding_dim, dtype=self.dtype, device=self.device))


    # 查找给定令牌ID的嵌入向量
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
    

class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                dim: int) -> torch.Tensor:
        # 首先计算指定维度dim上的最大值
        # keepdim=True确保max_val的形状与x兼容，避免维度广播错误
        max_val = torch.max(x, dim=dim, keepdim=True)[0]

        # 进行数值稳定化处理：减去对应维度的最大值，避免exp计算溢出
        x_stable = x - max_val

        # 计算稳定后的指数值
        exp_x = torch.exp(x_stable)

        # 计算指定维度dim上的指数和
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

        # 归一化得到softmax结果
        softmax_out = exp_x / sum_exp_x

        return softmax_out




if __name__=="__main__":
    embedding = Embedding(2,2)
    print(embedding(0))