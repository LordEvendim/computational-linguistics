import torch
import time
import torch.nn.functional as F
from src.transformer.model import GPTLanguageModel
from src.transformer.train import TransformerConfig
from src.gru.train import GRUConfig
from src.gru.model import Generator
from src.data.data import get_batch

transformer_path = "checkpoints/checkpoint_transformer_1000.pt"
gru_path = "checkpoints/checkpoint_gru_1000.pt"

transformer_config = TransformerConfig()
gru_config = GRUConfig()

transformer_model = GPTLanguageModel(
    transformer_config.vocab_size,
    transformer_config.block_size,
    transformer_config.n_embd,
    transformer_config.n_head,
    transformer_config.n_layer,
    transformer_config.dropout,
).to(transformer_config.device)

gru_model = Generator(
    vocab_size=gru_config.vocab_size,
    embedding_size=gru_config.embed_size,
    hidden_size=gru_config.hidden_size,
    num_layers=gru_config.num_layers,
).to(gru_config.device)

evaluated_models = {
    "Transformer": (
        transformer_model,
        transformer_config,
        transformer_path,
        transformer_config.enc,
        "transformer",
    ),
    "GRU": (gru_model, gru_config, gru_path, gru_config.enc, "gru"),
}

for model_name, (
    model,
    config,
    model_path,
    enc,
    model_type,
) in evaluated_models.items():
    print(f"Evaluating {model_name} model:")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from iteration {checkpoint.get('iter', 'unknown')}")

    def calculate_perplexity(data_split):
        model.eval()
        losses = torch.zeros(config.eval_iters)
        with torch.no_grad():
            for k in range(config.eval_iters):
                X, Y = get_batch(data_split)

                if model_type == "transformer":
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                else:
                    logits, _ = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                    losses[k] = loss.item()

        model.train()
        return torch.exp(losses.mean()).item()

    def inference_performance(prompt, max_new_tokens=100):
        model.eval()
        context = torch.tensor(
            [enc.encode(prompt)], dtype=torch.long, device=config.device
        )

        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=max_new_tokens)
        end_time = time.time()

        elapsed_time = end_time - start_time
        output = enc.decode(generated[0].tolist())

        model.train()
        return output, elapsed_time

    print(f"\nCalculating perplexity on validation set...")
    perplexity = calculate_perplexity("val")
    print(f"Perplexity: {perplexity:.4f}")

    prompt = "Pewnego razu młody książe "
    print(f"\nGenerating text from prompt: '{prompt}'")
    output, elapsed_time = inference_performance(prompt, max_new_tokens=1000)

    print(f"\nGenerated text:")
    print(output)
    print(f"\nInference time: {elapsed_time:.4f} seconds")
    print(f"Tokens per second: {1000 / elapsed_time:.4f}")

print("Evaluation complete!")
