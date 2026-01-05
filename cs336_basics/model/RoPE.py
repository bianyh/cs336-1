import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn


class RoPE(nn.Module):
    def __init__(self,
                 theta: float = 10000.0,
                 d_k: int = 512, # 每个注意力头的维度，必须是偶数
                 max_seq_len: int = 4096, # 预设的最大序列长度（预计算用）
                 device = None):
        super().__init__()
        # 首先需要断言检查维度是否是偶数
        # 虽然好像维度不是偶数也不是不行。但是姑且我们还是设定为只能是偶数吧
        assert d_k % 2 == 0, f"d_k musk be even, but got {d_k}"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else 'cpu'

        # 开始与计算频率矩阵(theta_i): 形状[d_k//2]
        # 维度索引 i： 0, 1, ..., d_k//2-1
        dim_indices = torch.arange(0, d_k, 2, device=self.device, dtype=torch.float32)
        # 计算每个(theta_i)：1 / (theta ** (2i / d_k))
        self.freqs = 1.0 / (theta ** (dim_indices / d_k))

        # 预计算位置-频率矩阵（pos_freqs）: 形状 [max_seq_len, d_k//2]
        # 位置 m: 0,1,...,max_seq_len-1
        position = torch.arange(0, max_seq_len, device=self.device, dtype=torch.float32)
        # 计算外积：位置×频率 → [max_seq_len, d_k//2]
        self.pos_freqs = torch.outer(position, self.freqs)

        # 得到sin和cos矩阵（核心旋转参数）：形状 [max_seq_len, d_k//2]
        self.cos_mat = torch.cos(self.pos_freqs)
        self.sin_mat = torch.sin(self.pos_freqs)





        
    def forward(self,
                x: torch.Tensor,
                token_position: torch.Tensor):
        # 获取输入形状信息
        x_shape = x.shape
        seq_len_x = x_shape[-2]
        d_k = x_shape[-1]

        # 形状应当是： [batch_size, att_head, seq_len, hide_dim]
        # 也可以不是，总之是这个形状: [..., seq_len, hide_dim]
        # 需要保证 hide_dim 的维度对上了
        assert d_k == self.d_k

        # 验证 token_position 有效性
        assert torch.all(token_position < self.max_seq_len), "token_position exceeds max_seq_len"
        assert torch.all(token_position >= 0), "token_position must be non-negative"

        # 获取对应位置的 cos、sin
        # 形状 [..., seq_len, d_k//2]
        cos = self.cos_mat[token_position]
        sin = self.sin_mat[token_position]

        # 扩展 cos、sin的形状，匹配x的形状： [..., seq_len, hide_dim]
        prefix_dim = x_shape[:-2] # 输入x的前导维度
        num_prefix_dim = len(prefix_dim) # 前导维度的数量
        # 然后动态补全cos、sin的前置维度，使其维度与x一致
        num_cos_prefix_dim = len(cos.shape) - 2
        need_add_dims = num_prefix_dim - num_cos_prefix_dim
        # 循环添加缺失的维度
        for _ in range(need_add_dims):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # 然后把cos、sin动态扩展成x的形状
        # 扩展形状：[*prefix_dims, seq_len_x, d_k//2]
        expand_shape = (*prefix_dim, seq_len_x, d_k//2)
        cos = cos.expand(expand_shape)
        sin = sin.expand(expand_shape)

        # 核心旋转逻辑（与前导维度无关，只看最后两个维度：seq_len, hide_size)
        # 先把x分奇偶：[..., seq_len, d_k//2, 2]
        x_reshape = x.reshape(*x_shape[:-1], self.d_k//2, 2)
        x_even = x_reshape[..., 0]
        x_odd = x_reshape[..., 1]

        # 旋转计算
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_odd * cos + x_even * sin

        # 合并恢复原始状态
        x_rot_reshape = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot_reshape.reshape(*x_shape)

        return x_rot

