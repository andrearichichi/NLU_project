import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# This model is designed for joint intent detection and slot filling. 
# It uses a single BiLSTM (or LSTM) encoder to process the input and produces two separate outputs: 
# one for the intent classification and one for the slot tags per token.
class ModelIAS(nn.Module):
    
    # Initializes all the layers of the model: embedding, dropout, LSTM encoder, 
    # and the two output classifiers for slot and intent. Handles optional bidirectionality and dropout settings.
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len,
                 n_layer=1, pad_index=0, bidirectional=False, dropout_p=0.0):
        super(ModelIAS, self).__init__()
        
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout_p)

        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = hid_size * 2 if bidirectional else hid_size
        self.slot_out = nn.Linear(lstm_output_size, out_slot)
        self.intent_out = nn.Linear(lstm_output_size, out_int)

    # Processes a batch of utterances with lengths, applies embedding and LSTM, and outputs slot 
    # and intent predictions using packed sequences to handle padding.
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)             
        utt_emb = self.dropout(utt_emb)

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, _) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True) 

        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=-1) 
        else:
            last_hidden = h_n[-1]  # [B, H]

        slots = self.slot_out(utt_encoded)  
        slots = slots.permute(0, 2, 1)      
        intent = self.intent_out(last_hidden)  

        return slots, intent
