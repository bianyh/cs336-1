import torch
import os
from typing import BinaryIO, IO, Union

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
) -> None:
    """
    保存训练检查点，包含模型状态、优化器状态和当前迭代数，支持保存到文件路径或文件类对象。

    核心逻辑：
    1.  提取模型的状态字典（model.state_dict()）：存储所有可学习权重
    2.  提取优化器的状态字典（optimizer.state_dict()）：存储优化器动量、矩估计等状态
    3.  封装为字典对象，包含上述两个状态字典和迭代数
    4.  使用torch.save将封装后的对象保存到目标位置（out）

    Args:
        model (torch.nn.Module): 需要保存的PyTorch模型
        optimizer (torch.optim.Optimizer): 需要保存的PyTorch优化器
        iteration (int): 当前训练迭代数（用于恢复训练时继续计数）
        out (Union[str, os.PathLike, BinaryIO, IO[bytes]]): 保存目标，可以是文件路径（str/PathLike）
            或二进制文件类对象（BinaryIO/IO[bytes]）
    """
    # 封装检查点数据（字典格式，结构清晰，便于后续加载解析）
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    # 保存检查点（torch.save原生支持路径和文件类对象）
    torch.save(checkpoint_data, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点加载模型状态、优化器状态，并返回保存时的迭代数，支持从文件路径或文件类对象加载。

    核心逻辑：
    1.  使用torch.load从源位置（src）加载检查点字典
    2.  用model.load_state_dict恢复模型权重
    3.  用optimizer.load_state_dict恢复优化器状态
    4.  提取并返回保存时的迭代数

    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): 检查点来源，可以是文件路径（str/PathLike）
            或二进制文件类对象（BinaryIO/IO[bytes]）
        model (torch.nn.Module): 需要恢复权重的PyTorch模型
        optimizer (torch.optim.Optimizer): 需要恢复状态的PyTorch优化器

    Returns:
        int: 检查点中保存的训练迭代数（用于恢复训练时继续迭代）
    """
    # 加载检查点数据（自动兼容路径和文件类对象）
    checkpoint_data = torch.load(src, map_location='cpu')  # 先加载到CPU，避免设备不匹配问题（可根据需求调整）

    # 恢复模型状态（严格匹配模型结构，否则会报错，保证权重一致性）
    model.load_state_dict(checkpoint_data['model_state_dict'])

    # 恢复优化器状态（严格匹配优化器配置，否则会报错，保证优化器状态一致性）
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # 提取并返回迭代数
    saved_iteration = checkpoint_data['iteration']

    return saved_iteration