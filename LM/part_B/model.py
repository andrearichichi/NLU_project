import torch.nn as nn
import warnings

# Applies the same dropout mask across time steps (variational dropout).
class LockedDropout(nn.Module):
    
    # Initializes the locked dropout module with the given dropout probability.
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    # Applies locked dropout during training; returns input unchanged during evaluation.
    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x

        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        return x * mask


# LSTM-based language model with optional dropout and weight tying.
class LM_LSTM(nn.Module):
    
    # Initializes the model with embedding, LSTM, and classifier layers.
    def __init__(self, emb_dim, hid_dim, vocab_size, pad_idx=0,
                 use_dropout=False, dropout=0.2, weight_tying=False):
        super().__init__()

        if weight_tying and emb_dim != hid_dim:
            warnings.warn("Weight tying enabled: overriding hidden_dim to match emb_dim")
            hid_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lockdrop = LockedDropout(dropout) if use_dropout else nn.Identity()
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.classifier = nn.Linear(hid_dim, vocab_size)

        if weight_tying:
            self.classifier.weight = self.embedding.weight

        self.weight_tying = weight_tying

    # Forward pass through the model: embedding, locked dropout, LSTM, and classifier.
    def forward(self, x):
        x = self.embedding(x)       
        x = self.lockdrop(x)        
        x, _ = self.lstm(x)         
        x = self.lockdrop(x)        
        x = self.classifier(x)      
        return x.permute(0, 2, 1)   
