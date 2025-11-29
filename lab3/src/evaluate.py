import os
import time
import json
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from src.data import df_test
from src.transformer.model import GPTLanguageModel
from src.transformer.train import TextClassificationDataset, collate_batch


def count_parameters(model) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def load_transformer_model(
    checkpoint_path: str = "checkpoints/best.pt",
) -> Tuple[GPTLanguageModel, dict, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = GPTLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_acc": checkpoint.get("val_acc"),
    }
    return model, config, metadata


def load_lora_model(
    checkpoint_path: str = "checkpoints/lora-best",
    num_labels: int = 3,
    device: str = None,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            num_labels = json.load(f).get("num_labels", 3)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "allegro/herbert-base-cased", num_labels=num_labels
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def evaluate_transformer_model(model, test_texts, test_labels, device, batch_size=128):
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    block_size = 128

    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, block_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, block_size, pad_id=0),
    )

    all_predictions, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for idx, targets in test_loader:
            idx, targets = idx.to(device), targets.to(device)
            logits, _ = model(idx, targets)
            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    inference_time = time.time() - start_time
    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "f1_weighted": f1_score(all_labels, all_predictions, average="weighted"),
        "inference_time": inference_time,
        "samples": len(all_labels),
        "time_per_sample": inference_time / len(all_labels),
    }


def evaluate_lora_model(
    model, tokenizer, test_texts, test_labels, device, batch_size=128, max_length=128
):
    model = model.to(device).eval()

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    test_labels_tensor = torch.tensor(test_labels)

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

    test_loader = DataLoader(
        TextDataset(test_encodings, test_labels_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    all_predictions, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(**batch).logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    inference_time = time.time() - start_time
    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "f1_weighted": f1_score(all_labels, all_predictions, average="weighted"),
        "inference_time": inference_time,
        "samples": len(all_labels),
        "time_per_sample": inference_time / len(all_labels),
    }


def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def main():
    print("Starting model evaluation...")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    test_texts = df_test["tekst"].to_list()
    test_labels_raw = df_test["sentyment"].to_list()
    unique_labels = sorted(set(test_labels_raw))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    test_labels = [label_to_id[label] for label in test_labels_raw]

    print(
        f"Evaluating on {len(test_texts)} test samples with {len(unique_labels)} classes."
    )
    print(f"Label mapping: {label_to_id}")
    print()

    results = {}

    print("Evaluating from-scratch transformer model...")
    try:
        model, config, metadata = load_transformer_model("checkpoints/best.pt")
        params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)

        print(
            f"Loaded model with {params['total']:,} total parameters ({params['trainable']:,} trainable), size: {model_size_mb:.2f} MB"
        )
        if metadata.get("epoch"):
            print(f"Model was trained for {metadata['epoch']} epochs")
        if metadata.get("val_acc"):
            print(
                f"Best validation accuracy during training: {metadata['val_acc']:.4f}"
            )
        print()

        print("Running evaluation on test set...")
        eval_results = evaluate_transformer_model(
            model, test_texts, test_labels, device
        )
        results["transformer"] = {
            **eval_results,
            "params": params,
            "model_size_mb": model_size_mb,
        }

        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"F1 (macro): {eval_results['f1_macro']:.4f}")
        print(f"F1 (weighted): {eval_results['f1_weighted']:.4f}")
        print(
            f"Inference time: {eval_results['inference_time']:.2f}s ({eval_results['time_per_sample'] * 1000:.2f}ms per sample)"
        )
        print()

    except FileNotFoundError as e:
        print(f"Could not find checkpoint: {e}")
        print("Skipping transformer model evaluation.")
        print()
    except Exception as e:
        print(f"Error evaluating transformer model: {e}")
        print()

    print("Evaluating fine-tuned LoRA model...")
    try:
        model, tokenizer = load_lora_model("checkpoints/lora-best")
        params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)

        print(
            f"Loaded model with {params['total']:,} total parameters ({params['trainable']:,} trainable), size: {model_size_mb:.2f} MB"
        )
        print()

        print("Running evaluation on test set...")
        eval_results = evaluate_lora_model(
            model, tokenizer, test_texts, test_labels, device
        )
        results["lora"] = {
            **eval_results,
            "params": params,
            "model_size_mb": model_size_mb,
        }

        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"F1 (macro): {eval_results['f1_macro']:.4f}")
        print(f"F1 (weighted): {eval_results['f1_weighted']:.4f}")
        print(
            f"Inference time: {eval_results['inference_time']:.2f}s ({eval_results['time_per_sample'] * 1000:.2f}ms per sample)"
        )
        print()

    except FileNotFoundError as e:
        print(f"Could not find checkpoint: {e}")
        print("Skipping LoRA model evaluation.")
        print()
    except Exception as e:
        print(f"Error evaluating LoRA model: {e}")
        print()

    if "transformer" in results and "lora" in results:
        print("Comparison summary:")
        print()
        t, lora = results["transformer"], results["lora"]
        print(
            f"Accuracy:           Transformer: {t['accuracy']:.4f}  |  LoRA: {lora['accuracy']:.4f}"
        )
        print(
            f"F1 (macro):          Transformer: {t['f1_macro']:.4f}  |  LoRA: {lora['f1_macro']:.4f}"
        )
        print(
            f"F1 (weighted):       Transformer: {t['f1_weighted']:.4f}  |  LoRA: {lora['f1_weighted']:.4f}"
        )
        print(
            f"Inference time:      Transformer: {t['inference_time']:.2f}s  |  LoRA: {lora['inference_time']:.2f}s"
        )
        print(
            f"Time per sample:     Transformer: {t['time_per_sample'] * 1000:.2f}ms  |  LoRA: {lora['time_per_sample'] * 1000:.2f}ms"
        )
        print(
            f"Total parameters:    Transformer: {t['params']['total']:,}  |  LoRA: {lora['params']['total']:,}"
        )
        print(
            f"Trainable params:    Transformer: {t['params']['trainable']:,}  |  LoRA: {lora['params']['trainable']:,}"
        )
        print(
            f"Model size:          Transformer: {t['model_size_mb']:.2f} MB  |  LoRA: {lora['model_size_mb']:.2f} MB"
        )
        print()

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
