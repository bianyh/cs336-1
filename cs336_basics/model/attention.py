import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from .base import SoftMax, Linear
from .RoPE import RoPE


class Attention(nn.Module):
    # 这儿我们默认设置d_k == d_v == d_model / num_heads
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: int = 10000.0,
                 max_seq_len: int = 512,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cuda'

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kv = self.d_model // self.num_heads

        self.q_linear = Linear(self.d_model, self.num_heads * self.d_kv, self.device, self.dtype)
        self.k_linear = Linear(self.d_model, self.num_heads * self.d_kv, self.device, self.dtype)
        self.v_linear = Linear(self.d_model, self.num_heads * self.d_kv, self.device, self.dtype)

        self.o_linear = Linear(self.num_heads * self.d_kv, self.d_model, self.device, self.dtype)

        self.rope = RoPE(theta=theta, max_seq_len=max_seq_len, d_k=self.d_kv, device=self.device)



    def forward(self, 
                x: torch.Tensor,
                use_rope: bool = True,
                token_position = None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.q_linear(x).reshape(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1,2)
        K = self.k_linear(x).reshape(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1,2)
        V = self.v_linear(x).reshape(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1,2)

        if use_rope and token_position is not None:
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)


        # 组装因果掩码mask
        # torch.tril：生成下三角矩阵, 下面是1，上面是0
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, self.num_heads, seq_len, seq_len)

        to_output_x = scaled_dot_product_attention(Q, K, V, causal_mask)

        to_output_x = to_output_x.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.d_kv)

        output = self.o_linear(to_output_x)

        return output





def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    seq_len_q = Q.shape[-2]
    seq_len_k = K.shape[-2]
    batch_dims = Q.shape[:-2] # 除了最后两个维度以外的所有前导维度

    assert K.shape[-1] == d_k

    # 计算注意力分数
    attn_scores = Q @ K.transpose(-1, -2)

    # 缩放
    scale = torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    attn_scores = attn_scores / scale

    # 进行掩码
    if mask is not None:
        assert mask.shape[-1] ==  seq_len_k and mask.shape[-2] == seq_len_q

        attn_scores = attn_scores.masked_fill(~mask, -torch.inf)
    
    softmax = SoftMax()

    attn_weights = softmax(attn_scores, dim=-1)

    output = attn_weights @ V

    return output