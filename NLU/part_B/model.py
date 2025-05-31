import torch
import torch.nn as nn
from transformers import BertModel

# BERT-based model for joint intent classification and slot labeling.
class JointBERT(nn.Module):
    
    # Initializes the model with a BERT encoder and two output classifiers.
    def __init__(self, model_name, num_slot_labels, num_intent_labels, dropout=0.1):
        super(JointBERT, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_labels)
        
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slot_labels)


    # Performs a forward pass through BERT and the classification heads. Computes loss if labels are provided.
    def forward(self, input_ids, attention_mask, slot_labels=None, intent_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        sequence_output = outputs.last_hidden_state     
        pooled_output = outputs.pooler_output           

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)    
        intent_logits = self.intent_classifier(pooled_output)  

        output = {
            "slot_logits": slot_logits,
            "intent_logits": intent_logits
        }

        if slot_labels is not None and intent_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            slot_loss = loss_fct(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
            intent_loss = loss_fct(intent_logits, intent_label)
            total_loss = slot_loss + intent_loss
            output["loss"] = total_loss

        return output
