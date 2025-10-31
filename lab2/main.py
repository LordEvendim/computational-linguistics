from src.tokenizers.tokenizers import p50k_base, WhitespaceTokenizer, SentencePieceTokenizer
from src.data.data import read_jsonl

def main():
    corpus = read_jsonl("./datasets/wolne_lektury_corpus_cleaned.jsonl")
    vocab_size = 50000

    print(f"Corpus length: {len(corpus)}")

    tokenizers = {
        "p50k_base": p50k_base(),
        "whitespace": WhitespaceTokenizer(corpus, vocab_size),
        "sentencepiece": SentencePieceTokenizer(vocab_size),
    }

    tokenizers["sentencepiece"].train(corpus)

    for name, tokenizer in tokenizers.items():
        print(f"{name}: {tokenizer.encode('Maria Konopnicka')}")
        print(f"{name}: {tokenizer.decode(tokenizer.encode('Maria Konopnicka'))}")


if __name__ == "__main__":
    main()
