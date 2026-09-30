import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import ssl
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

# Bypass SSL certificate verification issues on Anaconda Windows setups
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_dataset(data_dir="data", filename="raw_text.txt"):
    """
    Downloads Shakespeare text dataset (Complete Works / Tiny Shakespeare) if not present locally.
    Dataset URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading Shakespeare dataset from {SHAKESPEARE_URL}...")
        context = ssl._create_unverified_context()
        req = urllib.request.urlopen(SHAKESPEARE_URL, context=context)
        with open(file_path, "wb") as f:
            f.write(req.read())
        print(f"Dataset saved to {file_path}")
    else:
        print(f"Dataset already exists at {file_path}")
    return file_path


def load_and_preprocess_text(file_path, lowercase=True, remove_punctuation=True):
    """
    Reads text file, converts to lowercase, and removes punctuation.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Keep letters, numbers, and whitespace (spaces/newlines)
        text = re.sub(r"[^\w\s]", "", text)

    return text


def build_vocab(tokens, min_freq=1):
    """
    Builds word-level vocabulary mappings from list of word tokens.
    Includes special '<unk>' token at index 0 for unknown/OOV words.
    
    Returns:
    - word2idx: dict mapping word -> index
    - idx2word: dict mapping index -> word
    - vocab_size: total number of words in vocabulary
    """
    # Count word frequencies
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # Filter by minimum frequency and sort
    filtered_words = [w for w, c in freq.items() if c >= min_freq]
    filtered_words.sort()

    # Reserve index 0 for <unk>
    word2idx = {"<unk>": 0}
    idx2word = {0: "<unk>"}

    for idx, word in enumerate(filtered_words, start=1):
        word2idx[word] = idx
        idx2word[idx] = word

    vocab_size = len(word2idx)
    return word2idx, idx2word, vocab_size


class TextDataset(Dataset):
    """
    PyTorch Dataset for Word-Level Next-Token Prediction.
    Input sequence X: sequence of `seq_length` previous words (default 20 words).
    Target token y: scalar index of next word.
    """
    def __init__(self, encoded_tokens, seq_length=20, step=1):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []

        # Create sliding window sequence pairs
        for i in range(0, len(encoded_tokens) - seq_length, step):
            self.inputs.append(encoded_tokens[i : i + seq_length])
            self.targets.append(encoded_tokens[i + seq_length])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def prepare_data(
    data_path=None,
    seq_length=20,
    batch_size=64,
    val_split=0.1,
    level="word",
    lowercase=True,
    remove_punct=True,
    step=1
):
    """
    Full preprocessing pipeline:
    1. Downloads/reads Shakespeare corpus.
    2. Lowercases and removes punctuation.
    3. Tokenizes into word tokens.
    4. Constructs word -> index vocabulary with <unk>.
    5. Formulates sliding input-target sequence pairs (X=20 words, y=next word).
    6. Performs 90/10 train/validation split and returns PyTorch DataLoaders.
    """
    if data_path is None or not os.path.exists(data_path):
        data_path = download_dataset()

    raw_text = load_and_preprocess_text(data_path, lowercase=lowercase, remove_punctuation=remove_punct)

    if level == "word":
        tokens = raw_text.split()
    elif level == "char":
        tokens = list(raw_text)
    else:
        raise ValueError("level must be 'word' or 'char'")

    word2idx, idx2word, vocab_size = build_vocab(tokens)
    encoded = [word2idx.get(t, word2idx["<unk>"]) for t in tokens]

    dataset = TextDataset(encoded, seq_length=seq_length, step=step)

    # 90% Train / 10% Validation split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Sequential split to preserve narrative text sequence continuity
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "token2idx": word2idx,  # Alias for backward compatibility
        "idx2token": idx2word,  # Alias for backward compatibility
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "level": level,
        "total_tokens": len(tokens),
        "train_size": train_size,
        "val_size": val_size
    }

    return train_loader, val_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, meta = prepare_data(seq_length=20, batch_size=64, val_split=0.1)
    print("Word-Level Data Preprocessing Complete!")
    print(f"Vocabulary Size (Word-level): {meta['vocab_size']:,}")
    print(f"Total Words in Corpus: {meta['total_tokens']:,}")
    print(f"Train Sequences: {meta['train_size']:,}, Validation Sequences: {meta['val_size']:,}")
    print(f"Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")
