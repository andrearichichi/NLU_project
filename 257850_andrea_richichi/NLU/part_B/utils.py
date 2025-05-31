import json 
import torch
import json
from collections import Counter
from sklearn.model_selection import train_test_split

# Loads a JSON dataset from the given file path.
def load_data(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Splits the training set into train and dev, keeping rare intents only in train.
def split_train_dev(train_json_path, portion=0.1, seed=42):
    
    with open(train_json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    intents = [x['intent'] for x in raw_data]
    count_y = Counter(intents)

    inputs = []
    labels = []
    mini_train = []

    for idx, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(raw_data[idx])
            labels.append(y)
        else:
            mini_train.append(raw_data[idx])

    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=seed,
        shuffle=True,
        stratify=labels
    )

    X_train.extend(mini_train)

    return X_train, X_dev

# Builds slot and intent vocabularies from all datasets.
class NLULabels:
    def __init__(self, train_data, dev_data=None, test_data=None, pad_token=0):
        all_data = train_data + (dev_data or []) + (test_data or [])

        intents = [ex['intent'] for ex in all_data]
        slots = [slot for ex in all_data for slot in ex['slots'].split()]

        self.slot2id = self._build_vocab(slots, pad=True, pad_token=pad_token)
        self.intent2id = self._build_vocab(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    # Constructs vocabularies (optionally with padding token).
    def _build_vocab(self, items, pad=True, pad_token=0):
        vocab = {}
        if pad:
            vocab['pad'] = pad_token
        for item in sorted(set(items)):
            if item not in vocab:
                vocab[item] = len(vocab)
        return vocab

# Prepares a tokenized dataset with aligned slot and intent labels.
class NLUDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, slot2id, intent2id, max_len=128):
        self.input_ids = []
        self.attention_mask = []
        self.slot_labels = []
        self.intent_labels = []

        for entry in dataset:
            words = entry['utterance'].split()
            slots = entry['slots'].split()
            intent = entry['intent']

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding='max_length',
                truncation=True,
                max_length=max_len
            )

            word_ids = encoding.word_ids()
            aligned_slots = []

            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_slots.append(-100)  # special token
                elif word_idx != prev_word_idx:
                    aligned_slots.append(slot2id[slots[word_idx]])
                else:
                    aligned_slots.append(-100)  # sub-token
                prev_word_idx = word_idx

            self.input_ids.append(torch.tensor(encoding['input_ids']))
            self.attention_mask.append(torch.tensor(encoding['attention_mask']))
            self.slot_labels.append(torch.tensor(aligned_slots))
            self.intent_labels.append(torch.tensor(intent2id[intent]))

    # Returns dataset size.
    def __len__(self):
        return len(self.input_ids)

    # Returns a dictionary with input tensors for a given sample.
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'slot_labels': self.slot_labels[idx],
            'intent_label': self.intent_labels[idx]
        }
