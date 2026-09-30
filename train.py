import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_preprocessing import prepare_data
from model import LSTMTextGenerator


class EarlyStopping:
    """
    Early Stopping monitor to halt training when validation loss stops improving over patience epochs.
    Automatically saves best model checkpoint to save_path.
    """
    def __init__(self, patience=4, min_delta=0.001, save_path="outputs/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def check(self, val_loss, model, metadata, config):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, metadata, config)
            return True
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, model, metadata, config):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "config": config,
            "best_loss": self.best_loss
        }
        torch.save(checkpoint, self.save_path)
        print(f"Checkpoint saved: Best Validation Loss = {self.best_loss:.4f} (Perplexity = {math.exp(self.best_loss):.2f})")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical overflow protection
    return avg_loss, accuracy, perplexity


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, accuracy, perplexity


def plot_history(history, save_path="outputs/training_history.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 5))

    # Loss & Perplexity Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title("Word-Level LSTM Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(history["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2)
    plt.title("Word-Level LSTM Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history visualization saved to {save_path}")


def run_training(
    epochs=10,
    seq_length=20,
    batch_size=64,
    learning_rate=0.002,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2,
    patience=4,
    save_path="outputs/best_model.pt",
    history_plot_path="outputs/training_history.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Prepare Word-Level Dataset with 90/10 Split
    train_loader, val_loader, metadata = prepare_data(
        seq_length=seq_length,
        batch_size=batch_size,
        val_split=0.1,
        level="word",
        lowercase=True,
        remove_punct=True
    )

    vocab_size = metadata["vocab_size"]
    config = {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "level": "word"
    }

    # Initialize Word-Level LSTM Generator
    model = LSTMTextGenerator(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, save_path=save_path)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_ppl": [], "val_ppl": []}

    print("\n=======================================================")
    print(" Starting Word-Level LSTM Model Training ")
    print(f" Vocabulary: {vocab_size:,} words | Seq Length: {seq_length} words")
    print(f" Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    print("=======================================================\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_ppl = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.1f}, Acc: {train_acc:.4f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.1f}, Acc: {val_acc:.4f})",
            flush=True
        )

        # Check Early Stopping
        is_best = early_stopping.check(val_loss, model, metadata, config)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining Completed in {total_time/60:.2f} minutes.")
    
    # Save History Curves
    plot_history(history, save_path=history_plot_path)

    return model, metadata, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word-Level LSTM Text Generator")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length in words")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        patience=args.patience
    )
