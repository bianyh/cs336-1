import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from .attention import Attention
from .MLP import SwiGLU
from .RMSNorm import RMSNorm
from .base import Embedding, Linear, SoftMax
from .RMSNorm import RMSNorm



class Transformer_block(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ffn: int = 1024,
                 max_seq_len: int = 512,
                 theta: int = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cuda'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn

        self.attention = Attention(d_model=self.d_model,
                                   num_heads=self.num_heads,
                                   max_seq_len=max_seq_len,
                                   theta=theta,
                                   device=self.device,
                                   dtype=self.dtype
                                   )
        
        self.mlp = SwiGLU(d_model=self.d_model,
                          d_ff=self.d_ffn,
                          device=self.device,
                          dtype=self.dtype)
        
        self.attn_norm = RMSNorm(d_model=self.d_model,
                                 device=self.device,
                                 dtype=self.dtype)

        self.mlp_norm = RMSNorm(d_model=self.d_model,
                            device=self.device,
                            dtype=self.dtype)
        
    def forward(self, x: torch.Tensor):
        in_x = x

        res_x = x 

        x = self.attn_norm(x)

        # 组装位置
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        base_position = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_position = base_position.unsqueeze(0).expand(batch_size, seq_len)

        x = self.attention(x, use_rope=True, token_position=token_position)

        x = x + res_x.to(x.device)

        res_x = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + res_x.to(x.device)

        return x.type_as(in_x)


class Transformer_lm(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else 'cuda'

        self.embedding = Embedding(vocab_size, d_model, self.device, self.dtype)

        self.transformer_blocks = []
        for i in range(num_layers):
            transformer_block = Transformer_block(d_model, num_heads, d_ff, context_length*10, rope_theta, self.device, self.dtype)
            self.transformer_blocks.append(transformer_block)

        self.norm = RMSNorm(d_model, 0.00001, self.device, self.dtype)

        self.linear = Linear(d_model, vocab_size, self.device, self.dtype)

        # self.softmax = SoftMax()

    # input_ids : [batch_size, seq_len]
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)

        for tf_block in self.transformer_blocks:
            x = tf_block(x)

        x = self.norm(x)

        x = self.linear(x)

        # x = self.softmax(x, dim=-1)

        return x