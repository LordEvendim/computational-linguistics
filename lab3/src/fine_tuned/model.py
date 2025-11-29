from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def create_lora_model(
    num_labels: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        "allegro/herbert-base-cased",
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "query",
            "key",
            "value",
            "dense",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    model.print_trainable_parameters()

    return model, tokenizer
