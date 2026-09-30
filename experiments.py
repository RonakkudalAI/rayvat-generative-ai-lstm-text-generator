import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_preprocessing import prepare_data
from model import LSTMTextGenerator
from generate import generate_text


def run_single_experiment(exp_name, epochs=5, seq_length=20, num_layers=1, hidden_dim=128, embed_dim=128, batch_size=64, lr=0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Running Word-Level Experiment: {exp_name}")
    print(f" Config: Layers={num_layers}, Hidden={hidden_dim}, SeqLen={seq_length} words")
    print(f"=======================================================")

    train_loader, val_loader, metadata = prepare_data(
        seq_length=seq_length,
        batch_size=batch_size,
        val_split=0.1,
        level="word",
        lowercase=True,
        remove_punct=True
    )

    vocab_size = metadata["vocab_size"]
    model = LSTMTextGenerator(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        t_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_count += targets.size(0)
        avg_t_loss = t_loss / t_count

        # Validation
        model.eval()
        v_loss = 0.0
        v_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                v_loss += loss.item() * inputs.size(0)
                v_count += targets.size(0)
        avg_v_loss = v_loss / v_count
        val_ppl = math.exp(min(avg_v_loss, 20))

        history["train_loss"].append(avg_t_loss)
        history["val_loss"].append(avg_v_loss)
        history["val_ppl"].append(val_ppl)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f} (PPL: {val_ppl:.1f})")

    train_time = time.time() - start_time

    # Generate sample output
    sample_output = generate_text(
        model=model,
        metadata=metadata,
        seed_text="to be or",
        gen_length=60,
        temperature=0.7,
        device=device
    )

    return {
        "name": exp_name,
        "history": history,
        "final_val_loss": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "train_time": train_time,
        "sample": sample_output
    }


def run_all_experiments(save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    
    experiments = [
        {"name": "1-Layer Word LSTM (SeqLen=10)", "num_layers": 1, "hidden_dim": 128, "seq_length": 10},
        {"name": "2-Layer Word LSTM (SeqLen=10)", "num_layers": 2, "hidden_dim": 256, "seq_length": 10},
        {"name": "2-Layer Word LSTM (SeqLen=20)", "num_layers": 2, "hidden_dim": 256, "seq_length": 20},
    ]

    results = []
    for exp in experiments:
        res = run_single_experiment(
            exp_name=exp["name"],
            epochs=5,
            seq_length=exp["seq_length"],
            num_layers=exp["num_layers"],
            hidden_dim=exp["hidden_dim"]
        )
        results.append(res)

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res["history"]["val_loss"], label=f"{res['name']} (Val Loss)", linewidth=2)

    plt.title("Bonus Word-Level LSTM Architecture Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "architecture_experiments.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save Markdown Summary Report
    summary_path = os.path.join(save_dir, "experiment_report.md")
    report_lines = [
        "# Bonus Word-Level Architecture & Hyperparameter Experimentation Report\n",
        "| Architecture Name | Num Layers | Hidden Dim | Seq Length (words) | Final Val Loss | Val Perplexity | Training Time (s) |",
        "|---|---|---|---|---|---|---|"
    ]

    for res, exp in zip(results, experiments):
        report_lines.append(
            f"| {res['name']} | {exp['num_layers']} | {exp['hidden_dim']} | {exp['seq_length']} | {res['final_val_loss']:.4f} | {res['final_val_ppl']:.1f} | {res['train_time']:.1f}s |"
        )

    report_lines.append("\n## Generated Word Text Samples per Architecture\n")
    for res in results:
        report_lines.append(f"### {res['name']}")
        report_lines.append(f"```text\n{res['sample']}\n```\n")

    report_content = "\n".join(report_lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n--- WORD-LEVEL EXPERIMENTS COMPLETE ---")
    print(f"Comparison Plot saved to: {plot_path}")
    print(f"Experiment Report saved to: {summary_path}")


if __name__ == "__main__":
    run_all_experiments()
