import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class LSTMTextGenerator(nn.Module):
    """
    Word-Level LSTM Text Generation Neural Network built with PyTorch.
    
    Architecture & Flow:
    1. Embedding Layer (nn.Embedding):
       - Maps word token indices (range 0 to vocab_size-1) into dense 128-dimensional continuous vector embeddings.
    2. Stacked LSTM Layers (nn.LSTM):
       - 2 stacked LSTM layers with hidden dimension 256 and dropout 0.2.
       - Processes sequential word representations to capture long-term narrative dependencies.
    3. Dropout Regularization (nn.Dropout):
       - Regularizes final hidden representations to prevent co-adaptation and overfitting.
    4. Dense Output Head (nn.Linear):
       - Linear projection layer mapping 256-dim hidden state to unnormalized logits over vocabulary size (vocab_size).
    
    Note on Softmax / CrossEntropyLoss:
    - During training, PyTorch's nn.CrossEntropyLoss is applied directly to unnormalized logits
      because it internally combines LogSoftmax and NLLLoss for numerical stability.
    - During inference generation (generate.py), Softmax is applied to scale logits with Temperature
      before sampling next words.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(LSTMTextGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

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

        # Final Linear Projection Layer mapping hidden_dim -> vocab_size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass for next-word prediction.
        
        Args:
        - x: LongTensor of shape (batch_size, seq_length) containing word token indices.
        - hidden: Optional tuple (h_0, c_0) containing previous hidden state.
        
        Returns:
        - logits: Tensor of shape (batch_size, vocab_size) with unnormalized log-probabilities.
        - hidden: Updated tuple (h_n, c_n) containing current hidden states.
        """
        # Embed word indices: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        # Forward pass through stacked LSTM: (batch_size, seq_length, hidden_dim)
        if hidden is None:
            out, hidden = self.lstm(embedded)
        else:
            out, hidden = self.lstm(embedded, hidden)

        # Extract output from the final sequence step (last word in the context window)
        final_output = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply dropout regularization
        final_output = self.dropout(final_output)

        # Project to word vocabulary logits: (batch_size, vocab_size)
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
    dummy_vocab_size = 12849
    model = LSTMTextGenerator(vocab_size=dummy_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2).to(device)
    
    # Test dummy input sequence of 20 words
    dummy_input = torch.randint(0, dummy_vocab_size, (32, 20)).to(device)
    logits, _ = model(dummy_input)
    print("Word-Level LSTM Model Initialized Successfully!")
    print(f"Input Sequence Batch Shape: {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
