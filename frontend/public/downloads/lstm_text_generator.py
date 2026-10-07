"""Shakespeare word-level LSTM text generator.

Install: pip install torch
Train:   python lstm_text_generator.py --epochs 12
Generate from a saved checkpoint:
         python lstm_text_generator.py --generate-only --seed "to be or not"

The public-domain source is downloaded on the first training run. Training writes
best_lstm.pt and samples.txt locally. No pretrained model is bundled.
"""

import argparse
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def tokenize(text):
    """Lowercase, remove punctuation, and split into word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def load_text(path):
    if not path.exists():
        print(f"Downloading public-domain Shakespeare text to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


class SequenceDataset(Dataset):
    """Sliding input window paired with the following word index."""

    def __init__(self, encoded, sequence_length):
        self.tokens = encoded
        self.length = sequence_length

    def __len__(self):
        return max(0, len(self.tokens) - self.length)

    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index:index + self.length], dtype=torch.long)
        y = torch.tensor(self.tokens[index + self.length], dtype=torch.long)
        return x, y


class WordLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_size, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x):
        embedded = self.embedding(x)
        hidden, _ = self.lstm(embedded)
        return self.output(hidden[:, -1, :])  # raw logits for CrossEntropyLoss


@torch.no_grad()
def generate(model, seed, word_to_id, id_to_word, sequence_length, words, temperature, device):
    """Predict the next word repeatedly, sampling from temperature-scaled softmax."""
    model.eval()
    tokens = tokenize(seed)
    if not tokens:
        raise ValueError("Seed must contain at least one alphabetic word.")
    result = list(tokens)
    unknown = word_to_id["<unk>"]
    for _ in range(words):
        context = [word_to_id.get(t, unknown) for t in result[-sequence_length:]]
        context = [unknown] * (sequence_length - len(context)) + context
        logits = model(torch.tensor([context], device=device))[0] / temperature
        # Avoid emitting the unknown token in generated output.
        logits[unknown] = float("-inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        result.append(id_to_word[next_id])
    return " ".join(result)


def train(args, device):
    text = load_text(args.data)
    if args.max_chars:
        text = text[:args.max_chars]
    tokens = tokenize(text)
    if len(tokens) < args.sequence_length * 3:
        raise ValueError("Dataset is too small for the chosen sequence length.")

    # Split chronologically before constructing windows to prevent leakage.
    split = int(len(tokens) * 0.9)
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    counts = Counter(train_tokens)  # vocabulary learns only from training data
    vocabulary = ["<unk>"] + [word for word, _ in counts.most_common(args.vocab_size - 1)]
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    unknown = word_to_id["<unk>"]
    encode = lambda words: [word_to_id.get(word, unknown) for word in words]
    train_data = SequenceDataset(encode(train_tokens), args.sequence_length)
    val_data = SequenceDataset(encode(val_tokens), args.sequence_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = WordLSTM(len(vocabulary), args.embedding_dim, args.hidden_size, args.layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    stale_epochs = 0
    print(f"Training on {len(train_data):,} windows; validating on {len(val_data):,}. Vocabulary: {len(vocabulary):,} words.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item() * len(y)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_total += criterion(model(x), y).item() * len(y)
        val_loss = val_total / len(val_data)
        print(f"Epoch {epoch:02d}: train loss {train_total / len(train_data):.4f} | val loss {val_loss:.4f}")
        if val_loss < best_loss - 0.001:
            best_loss = val_loss
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(), "vocabulary": vocabulary,
                "sequence_length": args.sequence_length,
                "embedding_dim": args.embedding_dim, "hidden_size": args.hidden_size,
                "layers": args.layers, "dropout": args.dropout,
                "best_val_loss": best_loss,
            }, args.checkpoint)
            print(f"  Saved improved checkpoint: {args.checkpoint}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation loss stopped improving.")
                break
    return args.checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("shakespeare.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("best_lstm.pt"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--seed", default=None, help="One custom seed, or omit for three samples")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-chars", type=int, default=250000, help="0 uses full corpus")
    parser.add_argument("--words", type=int, default=55)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.vocab_size < 2 or args.sequence_length < 1 or args.batch_size < 1:
        parser.error("vocabulary, sequence length, and batch size must be positive")

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not args.generate_only:
        train(args, device)
    if not args.checkpoint.exists():
        parser.error(f"No checkpoint at {args.checkpoint}; train first without --generate-only")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = WordLSTM(len(vocabulary), checkpoint["embedding_dim"], checkpoint["hidden_size"],
                     checkpoint["layers"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    seeds = [args.seed] if args.seed else ["to be or not", "my lord i pray", "the king is"]
    outputs = []
    for seed in seeds:
        output = generate(model, seed, word_to_id, vocabulary, checkpoint["sequence_length"],
                          args.words, args.temperature, device)
        outputs.append(f"Seed: {seed}\nGenerated: {output}\n")
    result = "\n".join(outputs)
    print("\n" + result)
    Path("samples.txt").write_text(result, encoding="utf-8")
    print("Samples saved to samples.txt")


if __name__ == "__main__":
    main()
