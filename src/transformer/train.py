import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
from src.transformer.model import GPTLanguageModel
from src.data.data import get_batch

batch_size = 4
block_size = 64
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(
    device
)
m = model.to(device)

print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
