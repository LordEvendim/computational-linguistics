import os
import time
import torch
import tiktoken

from src.transformer.model import GPTLanguageModel as GPTBaseline

# from src.transformer.model_fa import GPTLanguageModel as GPTFlashAttention
# from src.transformer.model_wa import GPTLanguageModel as GPTWindowedAttention

from src.data.data import get_batch, get_epoch_batches, train_data

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")


class TransformerConfig:
    def __init__(self):
        self.batch_size = 16
        self.block_size = 128
        self.num_epochs = 1
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

        self.use_bf16 = False
        self.use_gradient_checkpointing = False
        self.window_size = None

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in self.__dict__.items())


config = TransformerConfig()
batch_size = config.batch_size
block_size = config.block_size
num_epochs = config.num_epochs
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

dataset_size = len(train_data)
steps_per_epoch = (dataset_size - block_size) // (batch_size * block_size)
max_iters = steps_per_epoch * num_epochs

print("------")
print(f"BF16 Mixed Precision: {config.use_bf16}")
print(f"Gradient Checkpointing: {config.use_gradient_checkpointing}")
print(
    f"Window Size: {config.window_size if config.window_size else 'None (full attention)'}"
)
print(f"Batch size: {config.batch_size}")
print(f"Block size: {config.block_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Dataset size: {dataset_size:,} tokens")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {max_iters}")
print("------")


def print_memory_stats(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(
            f"[{stage}] Memory - Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB, Peak: {max_allocated:.3f}GB"
        )


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config.block_size, config.batch_size)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def compute_perplexity(split="val", num_batches=100):
    model.eval()
    total_loss = 0.0
    for k in range(num_batches):
        X, Y = get_batch(split, config.block_size, config.batch_size)
        X = X.to(device)
        Y = Y.to(device)
        logits, loss = model(X, Y)
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    model.train()
    return perplexity.item(), avg_loss


ModelClass = GPTBaseline
# ModelClass = GPTFlashAttention
# ModelClass = GPTWindowedAttention

model_kwargs = {
    "vocab_size": vocab_size,
    "block_size": config.block_size,
    "n_embd": config.n_embd,
    "n_head": config.n_head,
    "n_layer": config.n_layer,
    "dropout": config.dropout,
}

if config.window_size is not None and ModelClass.__name__ == "GPTWindowedAttention":
    model_kwargs["window_size"] = config.window_size

model = ModelClass(**model_kwargs).to(config.device)

if config.use_gradient_checkpointing:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Cannot enable gradient checkpointing")

m = model.to(config.device)

print(f"\nModel parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print_memory_stats("Initial (after model load)")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

training_start_time = time.time()
step_times = []
epoch_start_time = training_start_time

iter = 0
for epoch in range(num_epochs):
    print(f"\n{'='*80}\nStarting Epoch {epoch + 1}/{num_epochs}\n{'='*80}\n")
    epoch_start_time = time.time()

    for xb, yb in get_epoch_batches("train", config.block_size, config.batch_size):
        step_start_time = time.time()

        if iter % config.eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter} (epoch {epoch + 1}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed {time.time() - training_start_time:.2f}s"
            )
            print_memory_stats(f"Step {iter}")

            checkpoint_path = os.path.join(
                "checkpoints", f"checkpoint_transformer_epoch{epoch + 1}_step{iter}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_name": "cl100k_base",
                    "block_size": config.block_size,
                },
                checkpoint_path,
            )

        if torch.cuda.is_available() and iter == 0:
            torch.cuda.reset_peak_memory_stats()

        if config.use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(xb, yb)
        else:
            logits, loss = model(xb, yb)

        if iter == 0:
            print_memory_stats("After Forward Pass")

        optimizer.zero_grad(set_to_none=True)

        # Backward pass
        loss.backward()
        if iter == 0:
            print_memory_stats("After Backward Pass")

        optimizer.step()

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_times.append(step_duration)

        if iter == 0:
            print_memory_stats("After Optimizer Step (Peak for one training step)")

        if (iter + 1) % 20 == 0:
            mean_step_time = sum(step_times[-20:]) / min(20, len(step_times))
            print(f"[Timing] Mean step time (last 20): {mean_step_time:.4f}s/step")

        iter += 1

    epoch_time = time.time() - epoch_start_time
    print("------")
    print(
        f"EPOCH {epoch + 1} completed in {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)"
    )
    print(f"Mean step time for epoch: {epoch_time/steps_per_epoch:.4f}s/step")
    print("------")

training_end_time = time.time()
total_training_time = training_end_time - training_start_time
print("------")
print("TRAINING COMPLETED")
print(f"Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
print("------")
print(f"Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
print(f"Number of epochs: {num_epochs}")
print(f"Total steps: {max_iters}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Mean time per step: {total_training_time/max_iters:.4f}s/step")
print(f"Mean time per epoch: {total_training_time/num_epochs:.2f}s")
print("------")

print("FINAL EVALUATION")

val_perplexity, val_loss = compute_perplexity(split="val", num_batches=100)
train_perplexity, train_loss = compute_perplexity(split="train", num_batches=100)

print(f"Final Validation Perplexity: {val_perplexity:.4f} (loss: {val_loss:.4f})")
print(f"Final Training Perplexity: {train_perplexity:.4f} (loss: {train_loss:.4f})")
print("------")
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

print("Sample generation: ")
print(enc.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
