from __future__ import annotations

import argparse
import os
import time
import math
import numpy as np
import torch
from torch import Tensor
import wandb
from tqdm import tqdm

# ====== 你的组件 ======
from model.transformer import Transformer_lm
import utils
from utils import (
    get_batch,
    compute_cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    AdamW
)
from checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from Tokenizer import MyTokenizer

# =========================================================
# 参数解析
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser("Train Transformer Language Model")

    # ---------- data ----------
    parser.add_argument("--input_txt", type=str, default=None,
                        help="Path to raw text file (.txt). If provided, will tokenize and split into train/val.")
    parser.add_argument("--train_data", type=str, default='/home/bianyuhan/LLM Learning/cs336/data/TinyStoriesV2-GPT4-train.txt',
                        help="Path to training data (.bin or memmap). If --input_txt is provided, this is ignored.")
    parser.add_argument("--val_data", type=str, default='/home/bianyuhan/LLM Learning/cs336/data/TinyStoriesV2-GPT4-valid.txt',
                        help="Path to validation data (.bin or memmap). If --input_txt is provided, this is ignored.")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--vocab_file", type=str, default="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/vocab.txt",
                        help="Path to vocab file")
    parser.add_argument("--merges_file", type=str, default="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/merges.txt",
                        help="Path to merges file")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=["<|endoftext|>"],
                        help="Special tokens for tokenizer")

    # ---------- model ----------
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # ---------- training ----------
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # ---------- optimizer / lr ----------
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_iters", type=int, default=100000)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # ---------- system ----------
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Use Weights and Biases for logging")

    return parser.parse_args()

# =========================================================
# Streaming 数据集（动态tokenize）
# =========================================================


class StreamingDataset:
    """
    True sliding-window streaming dataset.

    - Each get_batch() consumes NEW tokens
    - No random sampling
    - No overlapping samples within a batch
    - Deterministic token order (until EOF rewind)
    """

    def __init__(
        self,
        txt_path: str,
        tokenizer,
        buffer_size: int = 4096,
    ):
        self.txt_path = txt_path
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size

        self.buffer = np.empty(0, dtype=np.int32)
        self.pos = 0  # read pointer inside buffer

        self.file = open(self.txt_path, "r", encoding="utf-8")
        self.eof = False

    def _fill_buffer(self):
        """从文件中读取一段文本并填充到 buffer"""
        if self.eof:
            return

        text = self.file.read(self.buffer_size)
        if text == "":
            # 文件读完，rewind
            self.file.seek(0)
            text = self.file.read(self.buffer_size)
            if text == "":
                self.eof = True
                return

        tokens = self.tokenizer.encode(text)
        self.buffer = np.concatenate([self.buffer[self.pos:], np.array(tokens, dtype=np.int32)])
        self.pos = 0

    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device,
    ):
        """
        Return a batch of NEW sliding windows.
        每次调用都会读取 batch_size * context_length 的新 token
        """
        required_tokens = 2 * context_length

        # 如果 buffer 不够，填充
        while len(self.buffer) - self.pos < required_tokens:
            self._fill_buffer()
            if self.eof and len(self.buffer) - self.pos < required_tokens:
                raise StopIteration("End of file reached")

        # 取出一段连续 token
        tokens = self.buffer[self.pos:self.pos + required_tokens]
        self.pos += required_tokens

        return utils.get_batch(tokens, batch_size, context_length, device)





# =========================================================
# 数据准备：从txt文件tokenize并划分train/val
# =========================================================

def prepare_data(input_txt: str, vocab_file: str, merges_file: str, special_tokens: list[str], train_ratio: float = 0.9):
    """
    从原始txt文件加载数据，tokenize，划分train/val，并保存为memmap。
    返回train_data_path和val_data_path。
    """
    print("Loading tokenizer...")
    tokenizer = MyTokenizer.from_files(
        vocab_filepath=vocab_file,
        merges_filepath=merges_file,
        special_tokens=special_tokens
    )

    print("Reading input text...")
    with open(input_txt, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing text...")
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.int32)

    print(f"Total tokens: {len(tokens)}")

    # 划分train/val
    split_idx = int(len(tokens) * train_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")

    # 保存为.bin文件（memmap格式）
    train_data_path = input_txt.replace('.txt', '_train.bin')
    val_data_path = input_txt.replace('.txt', '_val.bin')

    print("Saving train data...")
    train_tokens.tofile(train_data_path)
    print("Saving val data...")
    val_tokens.tofile(val_data_path)

    return train_data_path, val_data_path

# =========================================================
# 验证集评估
# =========================================================

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: StreamingDataset,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    for _ in tqdm(range(eval_iters), desc='验证损失中'):
        x, y = dataset.get_batch(
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = compute_cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

# =========================================================
# 主训练入口
# =========================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.use_wandb:
        wandb.init(project="cs336-training", config=vars(args))

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---------- tokenizer ----------
    tokenizer = None
    if args.input_txt or (args.train_data and args.train_data.endswith('.txt')) or (args.val_data and args.val_data.endswith('.txt')):
        print("Loading tokenizer...")
        tokenizer = MyTokenizer.from_files(
            vocab_filepath=args.vocab_file,
            merges_filepath=args.merges_file,
            special_tokens=args.special_tokens
        )

    # ---------- data ----------
    if args.input_txt is not None:
        print("Preparing data from raw text...")
        train_data_path, val_data_path = prepare_data(
            args.input_txt,
            args.vocab_file,
            args.merges_file,
            args.special_tokens
        )
    else:
        if args.train_data is None or args.val_data is None:
            raise ValueError("Must provide either --input_txt or both --train_data and --val_data")
        train_data_path = args.train_data
        val_data_path = args.val_data

    # 创建streaming数据集
    train_dataset = StreamingDataset(train_data_path, tokenizer)
    val_dataset = StreamingDataset(val_data_path, tokenizer)

    # ---------- model ----------
    model = Transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    # ---------- optimizer ----------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
    )

    # ---------- resume ----------
    start_iter = 0
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    print("======================================")
    print(" Training configuration")
    print("======================================")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================")

    t0 = time.time()

    # =====================================================
    # 训练循环
    # =====================================================
    for it in tqdm(range(start_iter, args.max_iters), desc='训练iter:'):
        # ----- LR schedule -----
        lr = get_lr_cosine_schedule(
            it,
            args.lr_max,
            args.lr_min,
            args.warmup_iters,
            args.cosine_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ----- batch -----
        x, y = train_dataset.get_batch(
            args.batch_size,
            args.context_length,
            device,
        )

        # ----- forward -----
        logits = model(x)
        loss = compute_cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # ----- backward -----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # ----- logging -----
        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(
                f"iter {it:07d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"time {dt:.2f}s"
            )
            if args.use_wandb:
                wandb.log({"loss": loss.item(), "lr": lr, "iter": it})
            t0 = time.time()

        # ----- evaluation & checkpoint -----
        if it > 0 and it % args.eval_interval == 0:
            # train_loss = estimate_loss(
            #     model,
            #     train_dataset,
            #     args.batch_size,
            #     args.context_length,
            #     args.eval_iters,
            #     device,
            # )
            # val_loss = estimate_loss(
            #     model,
            #     val_dataset,
            #     args.batch_size,
            #     args.context_length,
            #     args.eval_iters,
            #     device,
            # )

            # print(
            #     f"[eval] iter {it:07d} | "
            #     f"train_loss {train_loss:.4f} | "
            #     f"val_loss {val_loss:.4f} | "
            #     f"ppl {math.exp(val_loss):.2f}"
            # )

            # if args.use_wandb:
            #     wandb.log({"train_loss": train_loss, "val_loss": val_loss, "ppl": math.exp(val_loss), "iter": it})

            ckpt_path = os.path.join(
                args.ckpt_dir, f"ckpt_iter_{it}.pt"
            )
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished.")

# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    main()
