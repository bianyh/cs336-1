import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
import numpy as np
from typing import Tuple
import numpy.typing as npt
from checkpoint import load_checkpoint


from model.transformer import Transformer_lm
from Tokenizer import MyTokenizer


def generate(prompt: str, max_tokens: int = 10000, temperature: float = 1.0, top_p: float = 1.0, checkpoint_path: str = '/home/bianyuhan/LLM Learning/checkpoints_test/ckpt_iter_8000.pt') -> str:
    """
    Generate completions for a user-provided prompt using the language model.

    Args:
        prompt: The input prompt string.
        max_tokens: Maximum number of tokens to generate.
        temperature: Temperature for softmax scaling. Lower values make output more deterministic.
        top_p: Top-p (nucleus) sampling threshold. Lower values focus on higher probability tokens.
        checkpoint_path: Path to the model checkpoint.

    Returns:
        The generated text including the prompt.
    """
    # Initialize tokenizer
    tokenizer = MyTokenizer.from_files(
        vocab_filepath="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/vocab.txt",
        merges_filepath="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    # Initialize model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = Transformer_lm(
        vocab_size=20000,
        # context_length=1024,
        context_length=256,
        d_model=512,
        # num_layers=12,
        num_layers=4,
        num_heads=16,
        # num_heads=8,
        # d_ff=2048,
        d_ff=1344,
        rope_theta=10000.0,
        device=device
    )

    # Load checkpoint
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # dummy optimizer, not used for generation
    iteration = load_checkpoint(checkpoint_path, model, dummy_optimizer)
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    # Get endoftext token id
    endoftext_id = tokenizer.token_to_ids_vocab[b'<|endoftext|>']

    generated_tokens = 0
    while generated_tokens < max_tokens:
        # Forward pass
        with torch.no_grad():
            logits = model(input_tensor)  # [1, seq_len, vocab_size]

        # Get logits for the next token
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # Apply temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            # Compute probabilities
            probs = F.softmax(next_logits, dim=-1)
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Find cutoff point
            cutoff_mask = cumulative_probs > top_p
            # Set probabilities beyond cutoff to 0
            sorted_probs[cutoff_mask] = 0.0
            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Sample from the truncated distribution
            next_token_idx = torch.multinomial(sorted_probs.squeeze(0), 1).item()
            next_token_id = sorted_indices[0, next_token_idx].item()
        else:
            # Standard sampling
            probs = F.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs.squeeze(0), 1).item()

        # Check for endoftext token
        if next_token_id == endoftext_id:
            break

        # Append to input
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

        generated_tokens += 1

        # Prevent exceeding context length
        if input_tensor.size(1) >= 256:
            break

    # Decode the full sequence
    full_ids = input_tensor.squeeze(0).tolist()
    generated_text = tokenizer.decode(full_ids)

    return generated_text


if __name__ == "__main__":
    # Example usage
    prompt = "I love you"
    result = generate(prompt, max_tokens=1000, temperature=0.8, top_p=0.9, checkpoint_path='/home/bianyuhan/LLM Learning/checkpoints/ckpt_iter_20000.pt')
    print(result)


