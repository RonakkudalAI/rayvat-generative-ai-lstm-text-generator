import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import re
import torch
import torch.nn.functional as F

from model import LSTMTextGenerator


def load_model_checkpoint(checkpoint_path="outputs/best_model.pt", device=None):
    """
    Loads saved PyTorch model checkpoint, metadata, and hyperparameter configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    config = checkpoint["config"]

    model = LSTMTextGenerator(
        vocab_size=metadata["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.2)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, metadata, config, device


def top_k_top_p_filtering(logits, top_k=40, top_p=0.9, filter_value=-float("Inf")):
    """
    Filters a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits


def generate_text(
    model,
    metadata,
    seed_text="to be or",
    gen_length=100,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    device=None
):
    """
    Generates new word-level text autoregressively given a word seed sequence.
    Predicts one word -> appends it -> predicts next word -> repeats.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word2idx = metadata.get("word2idx", metadata.get("token2idx"))
    idx2word = metadata.get("idx2word", metadata.get("idx2token"))
    seq_length = metadata["seq_length"]
    unk_idx = word2idx.get("<unk>", 0)

    # Clean & tokenize seed prompt (lowercasing and removing punctuation)
    clean_seed = re.sub(r"[^\w\s]", "", seed_text.lower())
    seed_words = clean_seed.split()

    if len(seed_words) == 0:
        seed_words = ["to", "be", "or"]

    # Convert seed words to indices using word2idx (fallback to <unk> if unseen)
    encoded_seed = [word2idx.get(w, unk_idx) for w in seed_words]

    # Pad or truncate seed to seq_length
    if len(encoded_seed) < seq_length:
        padding = [unk_idx] * (seq_length - len(encoded_seed))
        current_seq = padding + encoded_seed
    else:
        current_seq = encoded_seed[-seq_length:]

    generated_words = list(seed_words)
    input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    model.eval()
    hidden = None

    with torch.no_grad():
        for _ in range(gen_length):
            logits, hidden = model(input_tensor, hidden)
            
            # Apply Temperature scaling
            scaled_logits = logits / max(temperature, 1e-5)

            # Apply Top-k / Top-p filtering
            filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=top_k, top_p=top_p)

            # Calculate Softmax Probabilities over word vocabulary
            probs = F.softmax(filtered_logits, dim=-1)

            # Sample next word index using multinomial sampling
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = idx2word.get(next_idx, "<unk>")

            generated_words.append(next_word)

            # Update input sequence sliding window
            current_seq = current_seq[1:] + [next_idx]
            input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    # Reconstruct readable string by joining generated word tokens with spaces
    generated_text = " ".join(generated_words)
    return generated_text


def generate_sample_suite(checkpoint_path="outputs/best_model.pt", output_file="outputs/generation_samples.txt"):
    """
    Generates a suite of sample text outputs across multiple word seeds and temperatures.
    Saves report to outputs/generation_samples.txt.
    """
    model, metadata, config, device = load_model_checkpoint(checkpoint_path)
    
    seeds = [
        "to be or",
        "the king",
        "my lord",
        "shall i compare",
        "first citizen"
    ]
    temperatures = [0.2, 0.7, 1.2]

    report = []
    report.append("=====================================================")
    report.append("  WORD-LEVEL LSTM TEXT GENERATOR - SAMPLE EVALUATION  ")
    report.append("=====================================================\n")

    for seed in seeds:
        report.append(f"--- SEED: '{seed}' ---")
        for temp in temperatures:
            text = generate_text(
                model=model,
                metadata=metadata,
                seed_text=seed,
                gen_length=80,
                temperature=temp,
                top_k=40,
                top_p=0.9,
                device=device
            )
            report.append(f"\n[Temperature: {temp}]")
            report.append(text)
            report.append("-" * 40)
        report.append("\n")

    full_report_text = "\n".join(report)
    print(full_report_text)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    print(f"\nSample generation suite saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Word-Level LSTM Model")
    parser.add_argument("--seed", type=str, default="to be or", help="Seed prompt word string")
    parser.add_argument("--length", type=int, default=100, help="Number of words to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (0.2, 0.7, 1.2)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering")
    parser.add_argument("--suite", action="store_true", help="Generate full benchmark evaluation suite")
    args = parser.parse_args()

    if args.suite:
        generate_sample_suite()
    else:
        model, metadata, config, device = load_model_checkpoint()
        output = generate_text(
            model=model,
            metadata=metadata,
            seed_text=args.seed,
            gen_length=args.length,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        print("\n--- GENERATED WORD TEXT ---")
        print(output)
