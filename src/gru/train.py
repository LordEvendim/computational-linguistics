import torch
from src.gru.model import Generator
from src.data.data import get_batch
import tiktoken
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(42)

# Parameters
embed_size = 12
hidden_size = 10
num_layers = 1
learning_rate = 1e-3
max_iters = 1000
eval_iters = 100

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

model = Generator(
    vocab_size=vocab_size,
    embedding_size=embed_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
)

m = model.to(device)

print(sum(p.numel() for p in m.parameters() if p.requires_grad), "trainable parameters")

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, _ = m(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss
        out[split] = losses.mean().item()
    m.train()
    return out


for iter in range(max_iters):
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, _ = model(xb)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(m.generate(context, max_new_tokens=100)[0].tolist()))
