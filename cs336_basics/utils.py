import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
import numpy as np
from typing import Tuple
import numpy.typing as npt



def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从1D token数据集中随机采样一批语言模型的输入序列和标签序列。

    核心逻辑：
    1.  每个样本需要长度为`context_length + 1`的连续token（输入占前`context_length`个，标签占后`context_length`个）
    2.  随机采样`batch_size`个有效起始索引（避免序列越界）
    3.  对每个起始索引，提取输入序列`x`和标签序列`y`（y是x的下一个token）
    4.  堆叠为批次张量并转换为`torch.LongTensor`，部署到指定设备

    Args:
        dataset (npt.NDArray): 1D numpy整数数组，存储所有token ID
        batch_size (int): 批次大小（每个批次的样本数量）
        context_length (int): 每个样本的上下文长度（输入序列的长度）
        device (str): PyTorch设备字符串（如'cpu'、'cuda:0'），用于部署张量

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 两个形状均为(batch_size, context_length)的LongTensor
            - 第一个张量：输入序列x（语言模型的输入）
            - 第二个张量：标签序列y（x的下一个token，语言模型的预测目标）
    """
    # 步骤1：计算有效起始索引的最大值（避免采样的序列越界）
    # 每个样本需要context_length + 1个token，因此起始索引i的上限为 len(dataset) - context_length - 1
    max_valid_start_idx = len(dataset) - context_length - 1
    if max_valid_start_idx < 0:
        raise ValueError(f"数据集长度{len(dataset)}小于所需最小长度{context_length + 1}，无法采样")

    # 步骤2：随机采样batch_size个有效起始索引（均匀分布，允许重复采样）
    start_indices = np.random.randint(
        low=0,
        high=max_valid_start_idx + 1,  # np.randint上限是开区间，因此+1包含max_valid_start_idx
        size=batch_size
    )

    # 步骤3：初始化列表存储每个样本的x和y
    x_samples = []
    y_samples = []

    # 步骤4：遍历每个起始索引，提取x和y序列
    for idx in start_indices:
        # 输入序列x：从idx开始，取context_length个token
        x_seq = dataset[idx: idx + context_length]
        # 标签序列y：从idx+1开始，取context_length个token（x的下一个token）
        y_seq = dataset[idx + 1: idx + 1 + context_length]
        # 添加到样本列表
        x_samples.append(x_seq)
        y_samples.append(y_seq)

    # 步骤5：将列表堆叠为numpy数组，再转换为torch.LongTensor（token ID用长整型）
    x_batch = torch.tensor(np.stack(x_samples), dtype=torch.long, device=device)
    y_batch = torch.tensor(np.stack(y_samples), dtype=torch.long, device=device)

    return (x_batch, y_batch)



def gradient_clipping(parameters, max_l2_norm):
    """
    对所有参数的联合梯度进行L2范数裁剪，原地修改参数的梯度（parameter.grad）。

    核心逻辑：
    1.  过滤掉无梯度的参数（p.grad is None）
    2.  计算所有参数梯度的联合L2范数 ∥g∥2
    3.  若 ∥g∥2 > max_l2_norm，按缩放因子 M/(∥g∥2+ϵ) 缩小所有梯度（ϵ=1e-6，数值稳定）
    4.  若 ∥g∥2 ≤ max_l2_norm，不修改梯度

    Args:
        parameters (Iterable[torch.nn.Parameter]): 可迭代的训练参数集合
        max_l2_norm (float): 梯度的最大L2范数（正数）
    """
    # 步骤1：过滤出带有梯度的参数（跳过无梯度的冻结参数等）
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:  # 无带梯度的参数，直接返回
        return

    # 步骤2：计算所有梯度的联合L2范数（先算平方和，再开根号，数值更稳定）
    # 确保张量设备一致性（与梯度同设备，避免CPU/GPU张量不匹配）
    device = params_with_grad[0].grad.device
    total_squared_norm = torch.tensor(0.0, device=device)

    for p in params_with_grad:
        # 计算单个参数梯度的平方和
        param_squared_norm = torch.sum(p.grad ** 2)
        # 累加所有参数梯度的平方和
        total_squared_norm += param_squared_norm

    # 计算联合L2范数
    l2_norm = torch.sqrt(total_squared_norm)

    # 步骤3：定义数值稳定项ϵ（题目要求1e-6，与PyTorch默认一致）
    epsilon = 1e-6

    # 步骤4：判断是否需要裁剪，并原地修改梯度
    if l2_norm > max_l2_norm:
        # 计算缩放因子：M / (∥g∥2 + ϵ)
        scale_factor = max_l2_norm / (l2_norm + epsilon)
        # 对每个参数的梯度进行原地缩放（mul_ 是in-place操作）
        for p in params_with_grad:
            p.grad.mul_(scale_factor)

    # 若l2_norm ≤ max_l2_norm，不做任何修改，直接返回
    return



def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    计算带线性预热的余弦学习率调度下，指定迭代次数的学习率（完全匹配测试用例输出）。

    阶段划分：
    1.  预热阶段（it ≤ warmup_iters）：学习率从0线性增长到max_learning_rate
    2.  余弦衰减阶段（warmup_iters < it ≤ cosine_cycle_iters）：学习率从max_learning_rate
        按余弦函数平滑衰减到min_learning_rate（关键：进度分母为cosine_cycle_iters - warmup_iters）
    3.  保持阶段（it > cosine_cycle_iters）：学习率固定为min_learning_rate

    Args:
        it (int): 当前迭代次数
        max_learning_rate (float): 最大学习率（预热结束后达到的峰值学习率）
        min_learning_rate (float): 最小学习率（余弦衰减结束后保持的最低学习率）
        warmup_iters (int): 线性预热的迭代次数
        cosine_cycle_iters (int): 余弦衰减的总迭代上限（预热+衰减的总步数）

    Returns:
        float: 当前迭代次数对应的学习率
    """
    # 阶段1：线性预热阶段（包含边界it=warmup_iters）
    if it <= warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        # 线性插值，确保与测试用例预热阶段数值完全一致
        lr = max_learning_rate * (it / warmup_iters)
        return lr

    # 阶段3：超出余弦衰减总上限，直接返回最小学习率
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 阶段2：余弦衰减阶段（核心修正：有效衰减步数 = cosine_cycle_iters - warmup_iters）
    effective_cosine_iters = cosine_cycle_iters - warmup_iters
    cosine_it = it - warmup_iters
    # 计算归一化进度（0~1），匹配测试用例的衰减节奏
    progress = cosine_it / effective_cosine_iters
    # 浮点精度截断，避免超出范围
    progress = min(max(progress, 0.0), 1.0)
    # 余弦衰减核心公式（与测试用例完全对齐）
    cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_term

    return lr

def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失（负对数似然），保证数值稳定性并消去冗余log/exp运算。

    参数:
        logits: torch.Tensor
            模型输出的未归一化对数概率，形状为 [batch_dim1, batch_dim2, ..., vocab_size]
            （批量维度在前，最后一维为词汇表大小）。
        targets: torch.Tensor
            目标标签索引，形状为 [batch_dim1, batch_dim2, ...]
            （与logits去掉最后一维的形状完全一致）。

    返回:
        torch.Tensor
            所有样本的平均交叉熵损失，为标量张量。
    """
    # 步骤1：计算每个logits向量的最大值（最后一维，保持维度用于广播）
    # [batch_dims, vocab_size] -> [batch_dims, 1]
    max_logits = logits.max(dim=-1, keepdim=True)[0]

    # 步骤2：计算中心化logits（数值稳定，避免exp溢出）
    centered_logits = logits - max_logits

    # 步骤3：计算log_sum_exp（消去exp后求和再取log，数值稳定）
    # 先exp求和，再log：[batch_dims, 1] -> [batch_dims]
    sum_exp_centered = torch.exp(centered_logits).sum(dim=-1)
    log_sum_exp = torch.log(sum_exp_centered)

    # 步骤4：提取每个target对应的logits值
    # 1. 给targets增加最后一维，形状变为 [batch_dims, 1]
    targets_unsqueezed = targets.unsqueeze(dim=-1)
    # 2. 从logits中提取target对应的值，形状 [batch_dims, 1] -> [batch_dims]
    logits_target = logits.gather(dim=-1, index=targets_unsqueezed).squeeze(dim=-1)
    # 3. 挤压max_logits的最后一维，形状 [batch_dims, 1] -> [batch_dims]
    max_logits_squeezed = max_logits.squeeze(dim=-1)

    # 步骤5：计算每个样本的交叉熵损失（推导后的公式，消去了log/exp）
    # 公式推导：-log(softmax(logits)[target]) = log_sum_exp - (logits_target - max_logits)
    per_sample_loss = log_sum_exp - (logits_target - max_logits_squeezed)

    # 步骤6：计算所有样本的平均损失
    avg_loss = per_sample_loss.mean()

    return avg_loss
    
class AdamW(Optimizer):
        def __init__(
            self,
            params,
            lr: float,  # 对应 α
            betas: tuple[float, float] = (0.9, 0.999),  # 对应 β1, β2
            eps: float = 1e-8,  # 对应 ϵ
            weight_decay: float = 0.0  # 对应 λ
        ):
            # 校验超参数合法性
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta1 value: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta2 value: {betas[1]}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= weight_decay:
                raise ValueError(f"Invalid weight decay value: {weight_decay}")

            # 超参数字典（传给父类）
            defaults = dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
            super().__init__(params, defaults)

        def step(self, closure=None):
            """
            执行单次优化步骤
            Args:
                closure: 可选闭包，用于重新计算损失（默认 None）
            Returns:
                损失值（若 closure 不为 None）
            """
            loss = None
            if closure is not None:
                loss = closure()

            # 遍历所有参数组（支持不同参数不同超参数）
            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                # 遍历参数组内的每个参数
                for p in group['params']:
                    if p.grad is None:
                        continue  # 跳过无梯度的参数

                    g = p.grad.data  # 梯度 g
                    state = self.state[p]  # 获取该参数的状态字典

                    # 初始化状态（首次迭代时）
                    if len(state) == 0:
                        state['m'] = torch.zeros_like(p.data)  # 一阶矩 m
                        state['v'] = torch.zeros_like(p.data)  # 二阶矩 v
                        state['step'] = 0  # 时间步 t（初始为 0，step 后变为 1）

                    # 取出状态变量
                    m = state['m']
                    v = state['v']
                    t = state['step'] + 1  # t 从 1 开始
                    state['step'] = t  # 更新时间步

                    # 1. 更新一阶矩 m（inplace 操作，节省内存）
                    m.mul_(beta1).add_(g, alpha=(1 - beta1))  # m = β1*m + (1-β1)*g

                    # 2. 更新二阶矩 v（inplace 操作，g² 是元素级乘法，用 addcmul_）
                    v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))  # v = β2*v + (1-β2)*g²

                    # 3. 计算偏置校正后的学习率 αt
                    # ========== 核心修正：先将标量转为与参数匹配的Tensor ==========
                    # 匹配参数的dtype和device，避免类型/设备不匹配
                    dtype = p.data.dtype
                    device = p.data.device
                    beta1_tensor = torch.tensor(beta1, dtype=dtype, device=device)
                    beta2_tensor = torch.tensor(beta2, dtype=dtype, device=device)
                    lr_tensor = torch.tensor(lr, dtype=dtype, device=device)
                    # 使用torch.pow进行张量幂运算，结果仍为Tensor
                    beta1_t = torch.pow(beta1_tensor, t)
                    beta2_t = torch.pow(beta2_tensor, t)
                    # 后续运算均为Tensor运算，兼容torch.sqrt
                    sqrt_term = torch.sqrt(1 - beta2_t)
                    denom_term = 1 - beta1_t
                    alpha_t = lr_tensor * (sqrt_term / denom_term)
                    # ========== 修正结束 ==========

                    # 4. 计算分母：√v + ϵ（inplace 操作）
                    denom = v.sqrt().add_(eps)

                    # 5. 梯度更新：θ = θ - αt * m / (√v + ϵ)（inplace 操作）
                    # 注意：alpha_t是Tensor标量，可直接参与张量运算
                    p.data.addcdiv_(m, denom, value=-alpha_t)

                    # 6. 权重衰减：θ = θ - α*λ*θ（inplace 操作，解耦核心）
                    if weight_decay > 0.0:
                        # 转为Tensor避免类型问题
                        weight_decay_tensor = torch.tensor(weight_decay, dtype=dtype, device=device)
                        p.data.mul_(1 - lr_tensor * weight_decay_tensor)

            return loss