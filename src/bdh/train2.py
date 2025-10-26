import os
import time
from contextlib import nullcontext
import torch
import tiktoken
import bdh
import torch
import json

enc = tiktoken.get_encoding("cl100k_base")
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(filepath):
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
    return "".join(texts)


text = read_jsonl("./datasets/wolne_lektury_corpus_cleaned.jsonl")
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split, block_size=8, batch_size=32):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


BLOCK_SIZE = 128
BATCH_SIZE = 16
MAX_ITERS = 2000
EVAL_INTERVAL = 100
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
CHECKPOINT_DIR = "checkpoints"

USE_COMPILE = True

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in str(device)
    else nullcontext()
)
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype: {dtype}")

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

print(f"Vocabulary size: {vocab_size}")


@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- Main Execution ---
if __name__ == "__main__":
    print(f"\nTraining BDH model on Shakespeare dataset")

    model_config = bdh.BDHConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = bdh.BDH(model_config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nBDH Model initialized with {num_params/1e6:.2f}M parameters")
    print(f"Total parameters: {num_params:,}")

    if USE_COMPILE:
        print(f"\nCompiling the model...")
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully with 'aot_eager' backend.")
        except Exception as e:
            print(
                f"Warning: torch.compile failed with error: {e}\nContinuing without compilation..."
            )
    else:
        print("Compilation disabled, running in eager mode.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"\nStarting training for {MAX_ITERS} iterations...")
    training_start_time = time.time()

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model)
            elapsed_time = time.time() - training_start_time
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                f"time elapsed {elapsed_time:.2f}s"
            )

            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_bdh_{iter}.pt")
            torch.save(
                {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "vocab_name": "cl100k_base",
                    "block_size": BLOCK_SIZE,
                    "config": (
                        model_config.__dict__
                        if hasattr(model_config, "__dict__")
                        else {}
                    ),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Training step
        xb, yb = get_batch("train")

        with ctx:
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    print("\nTraining finished. Generating a sample...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        with ctx:
            generated = model.generate(context, max_new_tokens=500)

    generated_text = enc.decode(generated[0].tolist())
    print("-" * 50)
    print("Sample generation:")
    print(generated_text)
    print("-" * 50)

    # Save
    final_model_path = os.path.join(CHECKPOINT_DIR, f"bdh_shakespeare_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_name": "cl100k_base",
            "block_size": BLOCK_SIZE,
            "config": (
                model_config.__dict__ if hasattr(model_config, "__dict__") else {}
            ),
        },
        final_model_path,
    )
    print(f"\nFinal model saved to {final_model_path}")
    print("Training complete!")
