import os
import time
import torch
import tiktoken
from src.transformer.model import GPTLanguageModel
from src.data.data import get_batch

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")


class TransformerConfig:
    def __init__(self):
        self.batch_size = 16
        self.block_size = 128
        self.max_iters = 2000
        self.eval_interval = 100
        self.learning_rate = 3e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 50
        self.n_embd = 384
        self.n_head = 4
        self.n_layer = 6
        self.dropout = 0.2
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in self.__dict__.items())


config = TransformerConfig()
batch_size = config.batch_size
block_size = config.block_size
max_iters = config.max_iters
eval_interval = config.eval_interval
learning_rate = config.learning_rate
device = config.device
eval_iters = config.eval_iters
n_embd = config.n_embd
n_head = config.n_head
n_layer = config.n_layer
dropout = config.dropout
enc = config.enc
vocab_size = config.vocab_size


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(
    vocab_size,
    config.block_size,
    config.n_embd,
    config.n_head,
    config.n_layer,
    config.dropout,
).to(config.device)
m = model.to(config.device)

print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

training_start_time = time.time()

for iter in range(config.max_iters):
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed {time.time() - training_start_time:.2f}s"
        )

        checkpoint_path = os.path.join(
            "checkpoints", f"checkpoint_transformer_{iter}.pt"
        )
        torch.save(
            {
                iter: iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_name": "cl100k_base",
                "block_size": config.block_size,
            },
            checkpoint_path,
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

print("Sample generation: ")
print(enc.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
