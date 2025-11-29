import tiktoken


class p50k_base:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("p50k_base")

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.encoding.n_vocab
