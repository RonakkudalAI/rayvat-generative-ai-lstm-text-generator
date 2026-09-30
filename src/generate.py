import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import re
import torch
import torch.nn.functional as F

from model import LSTMTextGenerator


def load_model_checkpoint(checkpoint_path="models/best_model.pt", device=None):
    """
    Loads saved model checkpoint, vocabulary metadata, and configuration settings.
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


def generate_text(
    model,
    metadata,
    seed_text="to be or not",
    num_words=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    device=None
):
    """
    Autoregressively generates new text given a starting seed prompt.
    Iterative Loop: Predict next word -> append -> predict next word -> repeat num_words times.
    Temperature scaling controls distribution randomness:
    - Temperature 0.5: Conservative / Structured
    - Temperature 0.8: Balanced Coherence & Creativity (Recommended)
    - Temperature 1.0+: Exploratory / High Entropy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_to_idx = metadata["word_to_idx"]
    idx_to_word = metadata["idx_to_word"]
    seq_length = metadata["seq_length"]
    unk_id = word_to_idx["<UNK>"]

    # Preprocess seed text
    clean_seed = re.sub(r"[^\w\s]", "", seed_text.lower())
    seed_tokens = clean_seed.split()

    if len(seed_tokens) == 0:
        seed_tokens = ["to", "be", "or", "not"]

    # Convert seed tokens to IDs
    encoded_seed = [word_to_idx.get(w, unk_id) for w in seed_tokens]

    # Pad or truncate seed sequence to seq_length
    if len(encoded_seed) < seq_length:
        padding = [0] * (seq_length - len(encoded_seed))  # 0 is <PAD>
        current_seq = padding + encoded_seed
    else:
        current_seq = encoded_seed[-seq_length:]

    generated_words = list(seed_tokens)
    input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    model.eval()
    hidden = None

    with torch.no_grad():
        for _ in range(num_words):
            logits, hidden = model(input_tensor, hidden)

            # Apply Temperature scaling to logits
            scaled_logits = logits / max(temperature, 1e-5)

            # Calculate Softmax probabilities over word vocabulary
            probs = F.softmax(scaled_logits, dim=-1)

            # Sample next word index from multinomial distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = idx_to_word.get(next_idx, "<UNK>")

            # Skip adding raw <PAD> or <UNK> tags into final string output for clean readability
            if next_word not in ["<PAD>", "<UNK>"]:
                generated_words.append(next_word)
            else:
                # Fallback to a common high-probability word if UNK/PAD is sampled
                generated_words.append("the")

            # Update input sequence (slide window)
            current_seq = current_seq[1:] + [next_idx]
            input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    # Join word tokens with spaces
    return " ".join(generated_words)


def generate_sample_suite(checkpoint_path="models/best_model.pt", output_file="outputs/generated_samples.txt"):
    """
    Generates a suite of sample texts across multiple seed prompts and temperatures.
    Saves outputs to outputs/generated_samples.txt.
    """
    model, metadata, config, device = load_model_checkpoint(checkpoint_path)

    seeds = [
        "to be or not",
        "the king",
        "my lord",
        "shall i compare",
        "first citizen"
    ]
    temperatures = [0.5, 0.8, 1.0]

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
                num_words=80,
                temperature=temp,
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
    parser = argparse.ArgumentParser(description="Generate Word-Level Text with Trained LSTM Model")
    parser.add_argument("--seed", type=str, default="to be or not", help="Seed prompt string")
    parser.add_argument("--num_words", type=int, default=100, help="Number of words to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (e.g. 0.5, 0.8, 1.0)")
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
            num_words=args.num_words,
            temperature=args.temp,
            device=device
        )
        print("\n--- GENERATED WORD TEXT ---")
        print(output)
