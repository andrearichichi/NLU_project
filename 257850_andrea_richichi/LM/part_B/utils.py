# utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Reads a text file line by line and appends <eos> to each sentence.
def read_file(path, eos="<eos>"):
    with open(path, "r") as f:
        return [line.strip() + " " + eos for line in f]

# Creates token-to-ID and ID-to-token vocab mappings from a corpus.
class Lang:
    def __init__(self, corpus, specials=[]):
        self.word2id = self._build_vocab(corpus, specials)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def _build_vocab(self, corpus, specials=[]):
        vocab = {token: idx for idx, token in enumerate(specials)}
        idx = len(vocab)
        for line in corpus:
            for word in line.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

# Custom dataset that builds input-target pairs from sentences and maps them to token IDs.
class PennTreeBank(Dataset):
    def __init__(self, lines, lang):
        self.inputs = [line.split()[:-1] for line in lines]
        self.targets = [line.split()[1:] for line in lines]
        self.inputs_ids = self.to_ids(self.inputs, lang)
        self.targets_ids = self.to_ids(self.targets, lang)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {
            'source': torch.LongTensor(self.inputs_ids[i]),
            'target': torch.LongTensor(self.targets_ids[i])
        }

    def to_ids(self, sequences, lang):
        return [[lang.word2id[token] for token in seq] for seq in sequences]

# Pads and batches sequences dynamically for use in a DataLoader.
def collate_fn(samples, pad_token):
    def pad_batch(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), pad_token)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded, sum(lengths)

    samples.sort(key=lambda x: len(x['source']), reverse=True)
    srcs = [s['source'] for s in samples]
    tgts = [s['target'] for s in samples]
    srcs_pad, _ = pad_batch(srcs)
    tgts_pad, total_tokens = pad_batch(tgts)

    return {
        'source': srcs_pad.to(device),
        'target': tgts_pad.to(device),
        'number_tokens': total_tokens
    }

# Builds train, dev, and test datasets using PennTreeBank and Lang.
def create_dataset(train, dev, test, lang):
    return PennTreeBank(train, lang), PennTreeBank(dev, lang), PennTreeBank(test, lang)

# Wraps datasets in DataLoaders with padding-aware collation.
def create_dataloader(train, dev, test, lang, batch=40):
    cfn = partial(collate_fn, pad_token=lang.word2id["<pad>"])
    return (
        DataLoader(train, batch_size=batch + 40, collate_fn=cfn, shuffle=True),
        DataLoader(dev, batch_size=batch, collate_fn=cfn),
        DataLoader(test, batch_size=batch, collate_fn=cfn)
    )
