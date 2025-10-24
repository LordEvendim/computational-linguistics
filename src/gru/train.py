import os
import torch
import tiktoken
import torch.nn.functional as F
from src.gru.model import Generator
from src.data.data import get_batch
import time

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")


class GRUConfig:
    def __init__(self):
        self.batch_size = 16
        self.block_size = 128
        self.max_iters = 1000
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_iters = 50
        self.embed_size = 12
        self.hidden_size = 10
        self.num_layers = 1
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in self.__dict__.items())


config = GRUConfig()
torch.manual_seed(42)

print(f"Device: {config.device}")
print(config)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, _ = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Generator(
    vocab_size=config.vocab_size,
    embedding_size=config.embed_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
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

        checkpoint_path = os.path.join("checkpoints", f"checkpoint_gru_{iter}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "iter": iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_name": "cl100k_base",
                "config": config,
            },
            checkpoint_path,
        )

    xb, yb = get_batch("train")
    logits, _ = model(xb)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print("Sample generation:")
print(config.enc.decode(m.generate(context, max_new_tokens=100)[0].tolist()))
