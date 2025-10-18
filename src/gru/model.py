import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, use_softmax=False, hidden=None, lengths=None):
        if not isinstance(use_softmax, bool):
            raise TypeError("use_softmax must be a bool; pass other args by keyword")
        embedded = self.embedding(inputs)

        if lengths is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )

        output, hidden = self.gru(embedded, hidden)

        if lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        batch_size, sequence_len, hidden_size = output.shape
        output = output.contiguous().view(batch_size * sequence_len, hidden_size)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.classifier(output).view(batch_size, sequence_len, -1)

        return (F.softmax(output, dim=2), hidden) if use_softmax else (output, hidden)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        hidden = None
        for _ in range(max_new_tokens):
            # get the predictions
            logits, hidden = self(idx, hidden=hidden)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
