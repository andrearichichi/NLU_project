# model.py
import torch
import torch.nn as nn

# Basic single-layer RNN for language modeling with an embedding and linear output layer.
class LM_RNN(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.classifier = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)       
        x, _ = self.rnn(x)          
        x = self.classifier(x)      
        return x.permute(0, 2, 1)   


# LSTM-based model with optional dropout after embedding and before the output layer.
class LM_LSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, pad_idx=0,
                 use_dropout=False, drop_emb=0.3, drop_out=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop_emb = nn.Dropout(drop_emb) if use_dropout else nn.Identity()
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.drop_out = nn.Dropout(drop_out) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)        
        x = self.drop_emb(x)
        x, _ = self.lstm(x)          
        x = self.drop_out(x)
        x = self.classifier(x)       
        return x.permute(0, 2, 1)    
