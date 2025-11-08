import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(filepath):
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
    return texts


corpus = read_jsonl("./datasets/wolne_lektury_corpus_cleaned.jsonl")
text = "".join(corpus)

_encoded_cache = {}


def get_batch(split, enc, block_size=8, batch_size=32):
    cache_key = id(enc)

    if cache_key not in _encoded_cache:
        _encoded_cache[cache_key] = torch.tensor(enc.encode(text), dtype=torch.long)

    data = _encoded_cache[cache_key]
    n = int(0.9 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
