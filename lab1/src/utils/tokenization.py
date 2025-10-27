import zstandard
import json
import tiktoken


def streaming_tokenizer_zst(file_path, encoding_name="cl100k_base"):
    encoder = tiktoken.get_encoding(encoding_name)

    dctx = zstandard.ZstdDecompressor()

    with open(file_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode("utf-8")
            for line in text_stream.splitlines():
                try:
                    data = json.loads(line)
                    text = data.get("text", None)

                    if text:
                        tokens = encoder.encode(text)
                        yield tokens

                except json.JSONDecodeError:
                    print(f"Skipping malformed line: {line[:50]}...")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue


ENCODING = "cl100k_base"

encoder = tiktoken.get_encoding(ENCODING)
token_stream = streaming_tokenizer_zst(
    "datasets/plwiki.jsonl.zst", encoding_name=ENCODING
)

for _ in range(10):
    print("-----------")
    tokens = next(token_stream)
    text = encoder.decode(tokens)
    print(f"Decoded text: {text}")
