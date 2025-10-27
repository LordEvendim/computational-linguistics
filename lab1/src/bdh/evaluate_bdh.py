import json
import torch
import time
import tiktoken
from bdh import BDH, BDHConfig

checkpoint_path = "checkpoint_bdh_9999.pt"
block_size = 512
batch_size = 8
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "datasets/wolne_lektury_corpus_cleaned.jsonl"
enc = tiktoken.get_encoding("cl100k_base")


def load_data():
    print("Loading dataset...")
    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "text" in data:
                texts.append(data["text"])
            elif "content" in data:
                texts.append(data["content"])
            else:
                texts.append(
                    " ".join(str(v) for v in data.values() if isinstance(v, str))
                )

    full_text = "".join(texts)
    tokens = enc.encode(full_text)
    data_tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"Loaded {len(data_tokens):,} tokens from {len(texts)} documents")
    return data_tokens


data_array = load_data()
split_idx = int(0.9 * len(data_array))
train_data = data_array[:split_idx]
val_data = data_array[split_idx:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


print("Evaluating BDH model:")

checkpoint = torch.load(checkpoint_path, map_location=device)

if "config" in checkpoint:
    config_dict = checkpoint["config"]
    config = BDHConfig(**config_dict)
    print(f"Loaded config from checkpoint with vocab_size: {config.vocab_size}")
else:
    config = BDHConfig()
    state_dict_sample = checkpoint["model_state_dict"]
    for key in state_dict_sample.keys():
        if "embed.weight" in key:
            vocab_size = state_dict_sample[key].shape[0]
            config.vocab_size = vocab_size
            print(f"Detected vocab_size: {vocab_size}")
            break

model = BDH(config).to(device)

state_dict = checkpoint["model_state_dict"]
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
iteration = checkpoint.get("iter", "unknown")
print(f"Loaded checkpoint from iteration {iteration}")

num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters ({num_params/1e6:.2f}M)")
print(f"Vocab size: {config.vocab_size}")


def calculate_perplexity(data_split):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_batch(data_split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
    model.train()
    return torch.exp(losses.mean()).item()


def inference_performance(prompt, max_new_tokens=100):
    model.eval()
    context = torch.tensor(
        enc.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens)
    end_time = time.time()

    elapsed_time = end_time - start_time
    output = enc.decode(generated.squeeze(0).tolist())

    model.train()
    return output, elapsed_time


print(f"\nCalculating perplexity on validation set...")
perplexity = calculate_perplexity("val")
print(f"Perplexity: {perplexity:.4f}")

prompt = "To be or not to be"
print(f"\nGenerating text from prompt: '{prompt}'")
output, elapsed_time = inference_performance(prompt, max_new_tokens=1000)
print(f"\nGenerated text:")
print(output)
print(f"\nInference time: {elapsed_time:.4f} seconds")
print(f"Tokens per second: {1000 / elapsed_time:.4f}")
print("Evaluation complete!")
