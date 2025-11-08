from collections import Counter
import sentencepiece as spm
import tempfile
import tiktoken
import shutil
import re
import os


class p50k_base:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("p50k_base")

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.encoding.n_vocab


class WhitespaceTokenizer:
    def __init__(self, corpus: list[str], vocab_size: int):
        self.vocab_size = vocab_size

        counter = Counter(token for doc in corpus for token in self._split(doc))

        self.vocab = [
            word
            for word, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[
                : self.vocab_size - 1
            ]
        ] + ["<UNK>"]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        tokens = self._split(text)
        return [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens
        ]

    def decode(self, tokens: list[int]) -> str:
        result = []
        for token in tokens:
            word = self.vocab[token]
            if result and word not in ".,!?;:":
                result.append(" ")
            result.append(word)
        return "".join(result)

    def _split(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+|[.,!?;:]", text)
        return tokens

    def get_vocab_size(self) -> int:
        return len(self.vocab)


class SentencePieceTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.model_path = "models/sentencepiece.spm"
        self.model: spm.SentencePieceProcessor | None = None

    def train(self, corpus: list[str]):
        cleaned_corpus = [text.strip("\ufeff") for text in corpus]

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".txt", newline="\n"
        )
        temp_file.write("\n".join(cleaned_corpus))
        temp_file.flush()
        temp_file.close()

        spm.SentencePieceTrainer.train(
            input=temp_file.name,
            model_prefix=self.model_path.replace(".spm", ""),
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            input_sentence_size=-1,
            num_threads=4,
        )

        self.model = spm.SentencePieceProcessor()
        self.model.load(self.model_path.replace(".spm", "") + ".model")

    def save(self, model_path: str):
        if self.model is None:
            print("Train or load the model first")
            return

        current_base = self.model_path.replace(".spm", "")
        target_base = model_path.replace(".spm", "")

        # Create directory if needed
        os.makedirs(os.path.dirname(target_base) or ".", exist_ok=True)

        # Copy .model and .vocab files
        for ext in [".model", ".vocab"]:
            src = current_base + ext
            dst = target_base + ext
            if os.path.exists(src):
                shutil.copy2(src, dst)

        print(f"Model saved to {target_base}.model")

    def load(self, model_path: str):
        self.model = spm.SentencePieceProcessor()
        self.model.load(model_path)

    def encode(self, text: str) -> list[int]:
        if self.model is None:
            print("Train or load the model first")
            return []

        return self.model.encode(text)

    def decode(self, tokens: list[int]) -> str:
        if self.model is None:
            print("Train or load the model first")
            return ""

        return self.model.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.model.get_piece_size() if self.model else 0
