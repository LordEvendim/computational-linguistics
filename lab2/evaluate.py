from datetime import time
import torch
from src.data.data import read_jsonl
from src.tokenizers.tokenizers import p50k_base, WhitespaceTokenizer, SentencePieceTokenizer
from src.transformer.model import GPTLanguageModel
import math
import re

def compute_perplexity(model, text, encoder, device):
  model.eval()
  tokens = encoder.encode(text)

  words = split(text)
  num_words = len(words)
  num_chars = len(text)
  
  total_loss = 0.0
  num_tokens = 0
  
  block_size = model.block_size

  with torch.no_grad():
      for i in range(0, len(tokens) - 1, block_size):
          end_idx = min(i + block_size, len(tokens) - 1)
          x = torch.tensor(tokens[i:end_idx], dtype=torch.long, device=device).unsqueeze(0)
          y = torch.tensor(tokens[i+1:end_idx+1], dtype=torch.long, device=device).unsqueeze(0)
          
          if x.size(1) == 0:
              break
              
          logits, loss = model(x, y)
          
          total_loss += loss.item() * x.size(1)
          num_tokens += x.size(1)
  
  word_ppl = math.exp(total_loss / num_words) if num_words > 0 else float('inf')
  char_ppl = math.exp(total_loss / num_chars) if num_chars > 0 else float('inf')
  
  return word_ppl, char_ppl, num_tokens

def split(text):
  return re.split(r'\W+', text)

def count_words_in_dictionary(encoder, words):
    words_in_dict = 0
    
    for word in words:
        if not word:
            continue
            
        tokens = encoder.encode(word)
        
        if len(tokens) == 1:
            decoded = encoder.decode(tokens)
            if decoded.strip().lower() == word.strip().lower():
                words_in_dict += 1
    
    return words_in_dict

def compute_inference_time(model, text, encoder, device, num_runs=10):
    model.eval()
    tokens = encoder.encode(text)
    
    max_len = min(len(tokens) - 1, model.block_size)
    x = torch.tensor(tokens[:max_len], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(tokens[1:max_len+1], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(3):
            model(x, y)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            logits, loss = model(x, y)
            if device == "cuda":
                torch.cuda.synchronize()
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    tokens_per_second = x.size(1) / avg_inference_time if avg_inference_time > 0 else 0
    
    return avg_inference_time, tokens_per_second

def compute_efficiency(encoder, text, device="cpu"):
    words = split(text)
    
    start_time = time.time()
    tokens = encoder.encode(text)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    throughput = len(tokens) / elapsed_time if elapsed_time > 0 else 0
    
    avg_tokens_per_word = len(tokens) / len(words) if len(words) > 0 else 0
    
    words_in_dict = count_words_in_dictionary(encoder, words)
    words_in_dict_pct = (words_in_dict / len(words) * 100) if len(words) > 0 else 0
    
    return {
        'throughput': throughput,
        'avg_tokens_per_word': avg_tokens_per_word,
        'words_in_dict': words_in_dict,
        'words_in_dict_pct': words_in_dict_pct,
        'total_words': len(words),
        'total_tokens': len(tokens),
    }

def compute_oov_statistics(tokenizer, text):
    words = tokenizer._split(text)
    words = [w for w in words if w]

    oov_count = 0
    for word in words:
        if word not in tokenizer.word_to_idx:
            oov_count += 1

    total_words = len(words)
    oov_percentage = (oov_count / total_words * 100) if total_words > 0 else 0

    unique_words = set(words)
    unique_oov = sum(1 for word in unique_words if word not in tokenizer.word_to_idx)

    return {
        'oov_words': oov_count,
        'oov_percentage': oov_percentage,
        'total_words': total_words,
        'unique_oov_words': unique_oov,
        'unique_total_words': len(unique_words),
        'unique_oov_percentage': (unique_oov / len(unique_words) * 100) if len(unique_words) > 0 else 0,
        'vocab_size': tokenizer.vocab_size
    }

def analyze_sample_text(text, tokenizers):
    words = [w for w in split(text) if w]
    num_words = len(words)
    
    print(f"\n")
    print(f"Sample Text ({num_words} words):")
    print(f"{'-'*50}")
    print(f"{text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"\n")
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\n{name.upper()} Tokenizer:")
        print(f"{'-'*50}")
        
        tokens = tokenizer.encode(text)
        
        decoded = tokenizer.decode(tokens)
        
        num_tokens = len(tokens)
        tokens_per_word = num_tokens / num_words if num_words > 0 else 0
        
        words_direct = 0
        for word in words:
            word_tokens = tokenizer.encode(word)
            if len(word_tokens) == 1:
                word_decoded = tokenizer.decode(word_tokens).strip()
                if word_decoded.lower() == word.lower() or (isinstance(tokenizer, WhitespaceTokenizer) and word in tokenizer.word_to_idx):
                    words_direct += 1
        
        words_direct_pct = (words_direct / num_words * 100) if num_words > 0 else 0
        
        print(f"-> Token IDs (first 10): {tokens[:10]}")
        
        if isinstance(tokenizer, WhitespaceTokenizer):
            token_strings = [tokenizer.vocab[t] for t in tokens[:20]]
            print(f"-> Tokens (first 20): {' | '.join(token_strings)}")
        else:
            token_strings = []
            for t in tokens[:20]:
                try:
                    token_str = tokenizer.decode([t])
                    token_str = repr(token_str)[1:-1]
                    token_strings.append(token_str)
                except:
                    token_strings.append(f"[{t}]")
            print(f"-> Tokens (first 20): {' | '.join(token_strings)}")
        
        print("\n")
        print("Metrics:")
        print(f"-> Total tokens:             {num_tokens}")
        print(f"-> Total words:              {num_words}")
        print(f"-> Tokens per word:          {tokens_per_word}")
        print(f"-> Words encoded directly:   {words_direct} / {num_words} ({words_direct_pct}%)")
        
        results[name] = {
            'num_tokens': num_tokens,
            'num_words': num_words,
            'tokens_per_word': tokens_per_word,
            'words_direct': words_direct,
            'words_direct_pct': words_direct_pct,
            'tokens': tokens,
            'decoded': decoded
        }
    
    print(f"\n")
    print("COMPARISON SUMMARY:")
    print(f"{'-'*50}")
    print(f"{'Tokenizer':<20} {'Tokens':<10} {'Tok/Word':<12} {'Direct Encoded':<20}")
    print(f"{'-'*50}")
    for name, result in results.items():
        print(f"{name:<20} {result['num_tokens']:<10} {result['tokens_per_word']:<12.2f} {result['words_direct']}/{result['num_words']} ({result['words_direct_pct']:.1f}%)")
    print(f"\n")
    
    return results

def perform_qualitative_analysis(tokenizers, corpus):
    print("\n")
    print("QUALITATIVE ANALYSIS - SAMPLE TEXTS")
    print("\n")

    samples = []
    
    sample1_text = ""
    for doc in corpus[:50]:
        sample1_text += " " + doc
        if len(split(sample1_text)) >= 50:
            break
    samples.append(("Sample 1", sample1_text.strip()))
    
    mid_point = len(corpus) // 2
    sample2_text = ""
    for doc in corpus[mid_point:mid_point+50]:
        sample2_text += " " + doc
        if len(split(sample2_text)) >= 50:
            break
    samples.append(("Sample 2", sample2_text.strip()))
    
    sample3_text = ""
    for doc in corpus[-50:]:
        sample3_text += " " + doc
        if len(split(sample3_text)) >= 50:
            break
    samples.append(("Sample 3", sample3_text.strip()))
    
    all_results = {}
    for sample_name, sample_text in samples:
        print(f"\n")
        print(f"{sample_name}")
        print(f"{'-'*50}")
        results = analyze_sample_text(sample_text, tokenizers)
        all_results[sample_name] = results
    
    print(f"\n")
    print("OVERALL SUMMARY ACROSS ALL SAMPLES")
    
    for tokenizer_name in tokenizers.keys():
        avg_tokens_per_word = sum(all_results[sample][tokenizer_name]['tokens_per_word'] 
                                  for sample in all_results) / len(all_results)
        avg_direct_pct = sum(all_results[sample][tokenizer_name]['words_direct_pct'] 
                            for sample in all_results) / len(all_results)
        
        print(f"{tokenizer_name.upper()}:")
        print(f"-> Average tokens per word:        {avg_tokens_per_word:.2f}")
        print(f"-> Average direct encoding:        {avg_direct_pct:.2f}%")
    
    return all_results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    corpus = read_jsonl("./datasets/wolne_lektury_corpus_cleaned.jsonl")
    test_text = " ".join(corpus[:100])
    
    efficiency_text = " ".join(corpus[len(corpus) * 0.9:])
    
    print(f"Efficiency test text size: {len(efficiency_text.encode('utf-8')) / (1024*1024):.2f} MB")
    
    vocab_size = 50000
    
    tokenizers = {
        "p50k_base": p50k_base(),
        "whitespace": WhitespaceTokenizer(corpus, vocab_size),
        "sentencepiece": SentencePieceTokenizer(vocab_size),
    }
    
    models = {
        "p50k_base": "checkpoints/checkpoint_transformer_p50k_base_20000.pt",
        "whitespace": "checkpoints/checkpoint_transformer_whitespace_20000.pt",
        "sentencepiece": "checkpoints/checkpoint_transformer_sentencepiece_20000.pt",
    }
    
    print(f"Evaluating on {len(split(test_text))} words, {len(test_text)} characters\n")
    print("=" * 80)
    
    for model_name, model_path in models.items():
        print(f"\nTokenizer: {model_name}")
        print("-" * 40)
        
        checkpoint = torch.load(model_path, map_location=device)
        encoder = tokenizers[model_name]
        
        model = GPTLanguageModel(
            vocab_size=encoder.vocab_size,
            block_size=checkpoint['block_size'],
            n_embd=384,
            n_head=4,
            n_layer=6,
            dropout=0.2,
        ).to(device)


        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Perplexity metrics
        word_ppl, char_ppl, num_tokens = compute_perplexity(model, test_text, encoder, device)
        
        print(f"-> Word-level perplexity:      {word_ppl:.2f}")
        print(f"-> Character-level perplexity: {char_ppl:.2f}")
        print(f"-> Number of tokens:           {num_tokens}")
        print(f"-> Tokens per word:            {num_tokens / len(split(test_text)):.2f}")
        
        # Efficiency metrics
        print("\n")
        print("Efficiency Metrics (on 1MB+ test text):")
        efficiency = compute_efficiency(encoder, efficiency_text, device)
        print(f"-> Tokenizer throughput:     {efficiency['throughput']:.2f} tokens/sec")
        print(f"-> Avg tokens per word:      {efficiency['avg_tokens_per_word']:.2f}")
        print(f"-> Words in dictionary:      {efficiency['words_in_dict']} / {efficiency['total_words']} ({efficiency['words_in_dict_pct']:.2f}%)")
        
        # Inference time
        print("\n")
        avg_time, model_throughput = compute_inference_time(model, test_text, encoder, device)
        print(f"-> Inference Metrics:")
        print(f"-> Avg inference time:       {avg_time*1000:.2f} ms/batch")
        print(f"-> Model throughput:         {model_throughput:.2f} tokens/sec")
    
        if model_name == "whitespace":
            print("\n")
            print("OOV Statistics (Whitespace Tokenizer):")
            oov_stats = compute_oov_statistics(encoder, efficiency_text)
            print(f"-> OOV words:                {oov_stats['oov_words']} / {oov_stats['total_words']} ({oov_stats['oov_percentage']:.2f}%)")
            print(f"-> Unique OOV words:         {oov_stats['unique_oov_words']} / {oov_stats['unique_total_words']} ({oov_stats['unique_oov_percentage']:.2f}%)")
            print(f"-> Vocabulary size:          {oov_stats['vocab_size']}")

    print("\n" + "=" * 80)

    # Qualitative analysis
    perform_qualitative_analysis(tokenizers, corpus)

if __name__ == "__main__":
    main()