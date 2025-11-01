import torch
import os
from lab2.src.transformer.model import GPTLanguageModel
from src.tokenizers.tokenizers import p50k_base, WhitespaceTokenizer, SentencePieceTokenizer
from src.data.data import read_jsonl, get_batch
import time

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

class TransformerConfig:
    def __init__(self, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embd, n_head, n_layer, dropout, vocab_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.vocab_size = vocab_size


def train(encoder, name):
    vocab_size = encoder.vocab_size
    config = TransformerConfig(
        batch_size=16,
        block_size=128,
        max_iters=20000,
        eval_interval=100,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_iters=50,
        n_embd=384,
        n_head=4,
        n_layer=6,
        dropout=0.2,
        vocab_size=vocab_size,
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = get_batch(split)
                X = X.to(config.device)
                Y = Y.to(config.device)
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
        encoder,
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
                "checkpoints", f"checkpoint_transformer_{name}_{iter}.pt"
            )
            torch.save(
                {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_name": name,
                    "block_size": config.block_size,
                },
                checkpoint_path,
            )

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    training_end_time = time.time()
    print(f"Training completed in {training_end_time - training_start_time:.2f} seconds")

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    print("Sample generation: ")
    print(encoder.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

def main():
    corpus = read_jsonl("./datasets/wolne_lektury_corpus_cleaned.jsonl")
    vocab_size = 50000

    print(f"Corpus length: {len(corpus)}")

    tokenizers = {
        "p50k_base": p50k_base(),
        "whitespace": WhitespaceTokenizer(corpus, vocab_size),
        "sentencepiece": SentencePieceTokenizer(vocab_size),
    }

    tokenizers["sentencepiece"].train(corpus)

    for name, tokenizer in tokenizers.items():
        print(f"{name}: {tokenizer.encode('Maria Konopnicka')}")
        print(f"{name}: {tokenizer.decode(tokenizer.encode('Maria Konopnicka'))}")

    for name, tokenizer in tokenizers.items():
        print(f"Training {name}...")
        train(tokenizer, name)
        print(f"Training {name} completed")


if __name__ == "__main__":
    main()