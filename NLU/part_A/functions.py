import os
import json
import random
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Lang, IntentsAndSlots, collate_fn
from model import ModelIAS
from conll import evaluate


# Sets proper initial values for LSTM and Linear layers to improve training stability.
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [torch.nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [torch.nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


# Goes through the training data, computes intent and slot losses, applies backpropagation and gradient clipping.
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array

# Runs evaluation on dev or test data, returning slot F1, intent accuracy, and losses.
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
        
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
       
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# Opens a JSON file and returns its content as Python data structures.
def load_data(path):
    with open(path) as f:
        return json.load(f)

# Loads raw data, builds vocabularies, splits into train/dev/test, and returns DataLoaders and Lang object.
def prepare_data(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    PAD_TOKEN = 0

    tmp_train_raw = load_data(os.path.join('dataset', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'test.json'))

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    labels, inputs, mini_train = [], [], []

    for i, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[i])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[i])

    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=config["portion"],
        stratify=labels, random_state=seed, shuffle=True
    )
    X_train.extend(mini_train)

    train_raw = X_train
    dev_raw = X_dev


    w2id = {'pad': PAD_TOKEN, 'unk': 1}
    slot2id = {'pad': PAD_TOKEN}
    intent2id = {}

    for split in [train_raw, dev_raw, test_raw]:
        for ex in split:
            for w in ex['utterance'].split():
                if w not in w2id and split == train_raw:
                    w2id[w] = len(w2id)
            for slot in ex['slots'].split():
                if slot not in slot2id:
                    slot2id[slot] = len(slot2id)
            if ex['intent'] not in intent2id:
                intent2id[ex['intent']] = len(intent2id)

    words = sum([x['utterance'].split() for x in train_raw], [])
    slots = set(sum([x['slots'].split() for x in train_raw + dev_raw + test_raw], []))
    intents = set([x['intent'] for x in train_raw + dev_raw + test_raw])
    lang = Lang(words, intents, slots, cutoff=config["cutoff"])

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"]+64, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

    return PAD_TOKEN, lang, train_loader, dev_loader, test_loader

# Handles full experiment: model setup, training with early stopping, and testing over multiple runs.
def run_experiment(model_type, runs, config):
    print(f"\n CONFIG: {model_type}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if model_type == 'LSTM':
        bidirectional = False
        dropout_p = 0.0
    elif model_type == 'BiLSTM':
        bidirectional = True
        dropout_p = 0.0
    elif model_type == 'BiLSTM_dropout':
        bidirectional = True
        dropout_p = 0.3

    slot_f1s = []
    intent_accuracies = []

    for run in range(runs):
        print(f"\n Run {run + 1}/{runs}")
        seed = config["seed"] + run
        PAD_TOKEN, lang, train_loader, dev_loader, test_loader = prepare_data(config, seed)

        model = ModelIAS(
            hid_size=config["hidden_dim"],
            out_slot=len(lang.slot2id),
            out_int=len(lang.intent2id),
            emb_size=config["embedding_dim"],
            vocab_len=len(lang.word2id),
            pad_index=PAD_TOKEN,
            bidirectional=bidirectional,
            dropout_p=dropout_p
        ).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        best_f1 = 0
        patience = config["patience"]

        for epoch in tqdm(range(1, config["num_epochs"] + 1)):
            train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=config["clip"])

            results_dev, intent_res, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            f1 = results_dev['total']['f']
            # print(f" Epoch {epoch}: Dev Slot F1 = {f1:.4f} | Dev Intent Acc = {intent_res['accuracy']:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                patience = config["patience"]

                model_path = f"best_model_run{run + 1}_{model_type}.pt"
                torch.save(model.state_dict(), model_path)
            else:
                patience -= 1

            if patience <= 0:
                print(" Early stopping triggered.")
                break


        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print(f"Run {run + 1}: Slot F1 = {results_test['total']['f']:.4f} | Intent Acc = {intent_test['accuracy']:.4f}")
        slot_f1s.append(results_test['total']['f'])
        intent_accuracies.append(intent_test['accuracy'])

    slot_f1_mean = np.mean(slot_f1s)
    slot_f1_std = np.std(slot_f1s)
    intent_acc_mean = np.mean(intent_accuracies)
    intent_acc_std = np.std(intent_accuracies)

    print(f"\n [{model_type}] Mean ± Standard Deviation over {runs} runs:")
    print(f"Slot F1       → {slot_f1_mean:.3f} ± {slot_f1_std:.3f}")
    print(f"Intent Acc    → {intent_acc_mean:.3f} ± {intent_acc_std:.3f}")


