import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class LSTMTextGenerator(nn.Module):
    """
    Word-Level PyTorch LSTM Text Generation Neural Network.
    
    Architecture:
    Input Token IDs (Sequence of N words)
            │
            ▼
    [nn.Embedding] (vocab_size -> embed_dim=128, padding_idx=0)
            │
            ▼
    [nn.LSTM Layer 1 & 2] (input_size=128, hidden_size=256, num_layers=2, dropout=0.2)
            │
            ▼
    [nn.Dropout] (p=0.2)
            │
            ▼
    [nn.Linear] (hidden_size=256 -> vocab_size)
            │
            ▼
    Vocabulary Logits (Unnormalized Log Probabilities)
    
    Technical Detail on Softmax:
    - During training, CrossEntropyLoss takes unnormalized logits directly because it internally
      combines LogSoftmax and NLLLoss for maximum numerical stability.
    - During inference, Softmax with Temperature scaling is applied in generate.py.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2, pad_idx=0):
        super(LSTMTextGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding Layer with <PAD> token padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Multi-layer Stacked LSTM
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        # Dropout Regularization Layer
        self.dropout = nn.Dropout(dropout)

        # Output Projection Layer (Linear)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass for next-word prediction.
        
        Args:
        - x: LongTensor of shape (batch_size, seq_length) containing word token IDs.
        - hidden: Optional tuple (h_0, c_0) containing previous LSTM hidden state.
        
        Returns:
        - logits: Tensor of shape (batch_size, vocab_size) with unnormalized class scores.
        - hidden: Updated tuple (h_n, c_n) containing current hidden states.
        """
        # Embed word indices: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        # Forward pass through stacked LSTM: (batch_size, seq_length, hidden_dim)
        if hidden is None:
            out, hidden = self.lstm(embedded)
        else:
            out, hidden = self.lstm(embedded, hidden)

        # Extract output of the final sequence position (last word in window)
        final_output = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply dropout
        final_output = self.dropout(final_output)

        # Project to vocabulary size logits: (batch_size, vocab_size)
        logits = self.fc(final_output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """
        Initializes zero hidden state (h_0, c_0) for LSTM execution.
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_vocab_size = 12850
    model = LSTMTextGenerator(vocab_size=dummy_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2).to(device)
    
    # Test dummy batch of 32 sequences of 30 words
    dummy_input = torch.randint(0, dummy_vocab_size, (32, 30)).to(device)
    logits, _ = model(dummy_input)
    print("LSTM Text Generator Model Initialized Successfully!")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
