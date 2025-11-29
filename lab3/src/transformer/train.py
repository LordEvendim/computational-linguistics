import math
import os
import random
import time
from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
import wandb

from src.data import df_train, df_val, df_test
from src.transformer.model import GPTLanguageModel, config as model_config
from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, block_size: int):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.labels[idx]
        # For HuggingFace tokenizer:
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        return ids, int(y)


def collate_batch(batch: List[Tuple[List[int], int]], block_size: int, pad_id: int = 0):
    B = len(batch)
    x = torch.full((B, block_size), pad_id, dtype=torch.long)
    y = torch.empty(B, dtype=torch.long)
    for i, (ids, label) in enumerate(batch):
        ids = ids[:block_size]
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i] = label
    return x, y


@torch.no_grad()
def evaluate(model: GPTLanguageModel, loader: DataLoader, device: torch.device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_predictions = []
    all_targets = []
    for idx, targets in loader:
        idx = idx.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, loss = model(idx, targets)
        total_loss += loss.item() * idx.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += idx.size(0)
        all_predictions.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    f1_weighted = f1_score(all_targets, all_predictions, average="weighted")
    f1_macro = f1_score(all_targets, all_predictions, average="macro")

    return avg_loss, accuracy, f1_weighted, f1_macro


def main():
    set_seed(42)

    train_texts = df_train["tekst"].to_list()
    train_labels = df_train["sentyment"].to_list()

    val_texts = df_val["tekst"].to_list()
    val_labels = df_val["sentyment"].to_list()

    unique_labels = sorted(set(train_labels + val_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    train_labels = [label_to_id[label] for label in train_labels]
    val_labels = [label_to_id[label] for label in val_labels]

    num_labels = len(unique_labels)

    print("Dataset loaded:")
    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Test samples: {len(df_test)}")
    print(f"  Number of classes: {num_labels}")
    print(f"  Label mapping: {label_to_id}\n")

    # Use Herbert tokenizer instead of p50k_base
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    block_size = 128  # or model_config["block_size"]
    vocab_size = tokenizer.vocab_size

    # Update model config with Herbert's vocab size
    model_config["vocab_size"] = vocab_size
    model_config["n_classes"] = num_labels

    train_ds = TextClassificationDataset(
        train_texts, train_labels, tokenizer, block_size
    )
    val_ds = TextClassificationDataset(val_texts, val_labels, tokenizer, block_size)

    pin_mem = torch.cuda.is_available()
    batch_size = 8
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_mem,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, block_size, pad_id=0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, block_size, pad_id=0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel(**model_config).to(device)

    wandb.init(
        project="computational-linguistics-lab3",
        name="transformer-from-scratch",
        config={
            "model_type": "transformer-from-scratch",
            "vocab_size": model_config["vocab_size"],
            "block_size": model_config["block_size"],
            "n_embd": model_config["n_embd"],
            "n_head": model_config["n_head"],
            "n_layer": model_config["n_layer"],
            "dropout": model_config["dropout"],
            "n_classes": model_config["n_classes"],
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "epochs": 5,
            "warmup_ratio": 0.06,
            "grad_clip": 1.0,
            "device": str(device),
        },
    )
    wandb.config.update(model_config)

    lr = 2e-4
    weight_decay = 0.01
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, len(train_loader)) * 5
    warmup_steps = max(1, int(0.06 * total_steps))

    def lr_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    global_step = 0
    epochs = 5
    grad_clip = 1.0
    eval_interval = 10

    total_training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        epoch_losses = []
        epoch_accs = []

        for idx, targets in train_loader:
            idx = idx.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            current_lr = lr * lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, loss = model(idx, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            pred = logits.argmax(dim=-1)
            batch_correct = (pred == targets).sum().item()
            batch_count = idx.size(0)
            batch_acc = batch_correct / batch_count

            running_loss += batch_loss * batch_count
            running_correct += batch_correct
            running_count += batch_count
            epoch_losses.append(batch_loss)
            epoch_accs.append(batch_acc)
            global_step += 1

            if global_step % eval_interval == 0:
                recent_loss = np.mean(epoch_losses[-eval_interval:])
                recent_acc = np.mean(epoch_accs[-eval_interval:])
                val_loss, val_acc, val_f1_weighted, val_f1_macro = evaluate(
                    model, val_loader, device
                )

                wandb.log(
                    {
                        "train/loss_step": recent_loss,
                        "train/accuracy_step": recent_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/f1_weighted": val_f1_weighted,
                        "val/f1_macro": val_f1_macro,
                        "learning_rate": current_lr,
                    },
                    step=global_step,
                )
                model.train()

        train_loss = running_loss / max(1, running_count)
        train_acc = running_correct / max(1, running_count)

        val_loss, val_acc, val_f1_weighted, val_f1_macro = evaluate(
            model, val_loader, device
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} F1: {val_f1_weighted:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_epoch": train_loss,
                "train/accuracy_epoch": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1_weighted": val_f1_weighted,
                "val/f1_macro": val_f1_macro,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            },
            step=global_step,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_config,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

            wandb.log({"best_val_acc": best_val_acc}, step=global_step)

    total_training_time = time.time() - total_training_start
    wandb.log({"total_training_time": total_training_time})
    wandb.finish()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))


if __name__ == "__main__":
    main()
