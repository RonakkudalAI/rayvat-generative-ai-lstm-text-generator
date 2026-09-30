import os
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import download_dataset, load_and_clean_text, build_vocabulary


class TextDataset(Dataset):
    """
    PyTorch Dataset for Word-Level Next-Token Prediction.
    Given a sequence of `seq_length` previous words (X), target (y) is the next word index.
    """
    def __init__(self, encoded_tokens, seq_length=30, step=1):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []

        # Construct sliding window pairs (X: seq_length words, y: next word)
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
    seq_length=30,
    batch_size=64,
    val_split=0.1,
    max_vocab_size=15000,
    step=1
):
    """
    Full Data Preparation Pipeline:
    1. Loads cleaned text file.
    2. Tokenizes into words.
    3. Builds word_to_idx and idx_to_word dictionaries with <PAD> (0) and <UNK> (1).
    4. Performs sequential 90/10 train/validation split on raw token list (preserving text continuity).
    5. Formulates sliding window sequence pairs.
    6. Returns PyTorch DataLoaders and metadata.
    """
    if data_path is None or not os.path.exists(data_path):
        data_path = download_dataset()

    raw_text = load_and_clean_text(data_path)
    tokens = raw_text.split()

    # Build vocabulary with reserved <PAD> (0) and <UNK> (1)
    word_to_idx, idx_to_word, vocab_size = build_vocabulary(tokens, max_vocab_size=max_vocab_size)

    # Encode token sequence to IDs
    unk_id = word_to_idx["<UNK>"]
    encoded = [word_to_idx.get(t, unk_id) for t in tokens]

    # Sequential Train / Validation split (90% train, 10% validation)
    split_idx = int(len(encoded) * (1 - val_split))
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    # Create Datasets
    train_dataset = TextDataset(train_tokens, seq_length=seq_length, step=step)
    val_dataset = TextDataset(val_tokens, seq_length=seq_length, step=step)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "total_tokens": len(tokens),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset)
    }

    return train_loader, val_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, meta = prepare_data(seq_length=30, batch_size=64, val_split=0.1)
    print("Dataset Preparation Complete:")
    print(f"Total Words in Corpus: {meta['total_tokens']:,}")
    print(f"Vocabulary Size: {meta['vocab_size']:,}")
    print(f"Train Sequences: {meta['train_size']:,}, Validation Sequences: {meta['val_size']:,}")
    print(f"Train Batches: {len(train_loader)}, Validation Batches: {len(val_loader)}")
