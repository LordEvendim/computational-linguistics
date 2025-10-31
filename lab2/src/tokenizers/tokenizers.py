import sentencepiece as spm
from collections import Counter
import tiktoken
import re
import tempfile

class p50k_base:
  def __init__(self):
    self.encoding = tiktoken.get_encoding("p50k_base")

  def encode(self, text):
    return self.encoding.encode(text)

  def decode(self, tokens):
    return self.encoding.decode(tokens)

class WhitespaceTokenizer:
  def __init__(self, corpus: list[str], vocab_size: int):
    self.vocab_size = vocab_size
    
    counter = Counter(word for doc in corpus for word in doc.split())

    self.vocab = [word for word, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 1]] + ["<UNK>"]
    self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

  def encode(self, text: str) -> list[int]:
    tokens = self._split(text)
    return [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens]

  def decode(self, tokens: list[int]) -> str:
    # tutaj natrafiam na problem opisany w atykule dołączonym do zadania. Problem w utratą informacji o znaku spacji.
    return " ".join(self.vocab[token] for token in tokens).replace("<UNK>", "").replace(" . ", ". ").replace(" , ", ", ")

  def _split(self, text: str) -> list[str]:
    return [token for token in re.split(r'\W+', text) if token]

class SentencePieceTokenizer:
  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size
    self.model_path = "models/sentencepiece.spm"
    self.model = None

  def train(self, corpus: list[str]):
      cleaned_corpus = [text.strip('\ufeff') for text in corpus]
      
      temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt', newline='\n')
      temp_file.write("\n".join(cleaned_corpus))
      temp_file.flush()
      temp_file.close()
      
      spm.SentencePieceTrainer.train(
          input=temp_file.name,
          model_prefix=self.model_path.replace('.spm', ''),
          vocab_size=self.vocab_size,
          character_coverage=0.9995,
          model_type="bpe",
          input_sentence_size=-1,
          num_threads=4,
      )
      
      self.model = spm.SentencePieceProcessor()
      self.model.load(self.model_path.replace('.spm', '') + '.model')

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