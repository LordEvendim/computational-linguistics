import os
import math
import time
from typing import List
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
import wandb

from src.data import df_train, df_val, df_test
from src.fine_tuned.model import create_lora_model


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 128,
    device: str = None,
    eval_interval: int = 50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    wandb.init(
        project="computational-linguistics-lab3",
        name="herbert-lora-fine-tuned",
        config={
            "model_type": "herbert-lora-fine-tuned",
            "base_model": "allegro/herbert-base-cased",
            "num_labels": len(set(train_labels + val_labels)),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "grad_clip": 1.0,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "device": str(device),
        },
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Starting training with LoRA...")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print(f"Device: {device}\n")

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)) + 0.00002)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    total_training_start = time.time()
    global_step = 0

    def evaluate_validation():
        """Helper function to evaluate on validation set"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                val_loss += loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                val_correct += (predictions == batch["labels"]).sum().item()
                val_total += batch["labels"].size(0)

                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(batch["labels"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_f1_weighted = f1_score(val_true_labels, val_predictions, average="weighted")
        val_f1_macro = f1_score(val_true_labels, val_predictions, average="macro")

        return {
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "val/f1_weighted": val_f1_weighted,
            "val/f1_macro": val_f1_macro,
        }

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_train_losses = []
        epoch_train_accs = []

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Update metrics
            train_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            batch_correct = (predictions == batch["labels"]).sum().item()
            batch_total = batch["labels"].size(0)
            train_correct += batch_correct
            train_total += batch_total

            batch_acc = batch_correct / batch_total
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(batch_acc)

            global_step += 1

            # Log training metrics and evaluate validation at intervals
            if global_step % eval_interval == 0:
                recent_loss = np.mean(epoch_train_losses[-eval_interval:])
                recent_acc = np.mean(epoch_train_accs[-eval_interval:])
                val_metrics = evaluate_validation()

                wandb.log(
                    {
                        "train/loss_step": recent_loss,
                        "train/accuracy_step": recent_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **val_metrics,
                    },
                    step=global_step,
                )
                model.train()  # Switch back to training mode

        # End of epoch: full evaluation
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Full validation evaluation at end of epoch
        val_metrics = evaluate_validation()
        val_acc = val_metrics["val/accuracy"]
        avg_val_loss = val_metrics["val/loss"]
        val_f1 = val_metrics["val/f1_weighted"]

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        # Log epoch-level metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss_epoch": avg_train_loss,
                "train/accuracy_epoch": train_acc,
                **val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
            },
            step=global_step,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            model.save_pretrained("checkpoints/lora-best")
            tokenizer.save_pretrained("checkpoints/lora-best")
            print(f"Saved best model with val_acc: {val_acc:.4f}")

            wandb.log({"best_val_acc": best_val_acc}, step=global_step)

    total_training_time = time.time() - total_training_start
    wandb.log({"total_training_time": total_training_time})
    wandb.finish()

    return model


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

    print("Creating LoRA model...")
    model, tokenizer = create_lora_model(
        num_labels=num_labels,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    train(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=5,
        batch_size=8,
        learning_rate=2e-4,
        max_length=128,
        eval_interval=10,
    )


if __name__ == "__main__":
    main()
