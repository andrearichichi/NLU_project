import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pprint import pprint

from conll import evaluate as conll_evaluate
from NLU.part_B.utils import (
    load_data,
    split_train_dev,
    print_intent_distribution,
    NLULabels,
    NLUDataset
)
from NLU.part_B.model import JointBERT

# Loads and preprocesses the dataset: splits data, builds label mappings.
def preprocess_data(dataset_path):
    test_raw = load_data(os.path.join(dataset_path, 'test.json'))

    train_raw, dev_raw = split_train_dev(os.path.join(dataset_path, 'train.json'))
    test_raw = load_data(os.path.join(dataset_path, 'test.json'))

    labels = NLULabels(train_raw, dev_raw, test_raw)

    slot2id = labels.slot2id
    intent2id = labels.intent2id
    id2slot = labels.id2slot
    id2intent = labels.id2intent

    return train_raw, dev_raw, test_raw, slot2id, intent2id, id2slot, id2intent

# Runs a full training + evaluation pipeline for one run, including model setup and dataloaders.
def run_experiment(config, run_id):
    print(f"\n Starting run {run_id}/{config['total_runs']}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw, dev_raw, test_raw, slot2id, intent2id, id2slot, id2intent = preprocess_data(config["dataset_path"])

    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])

    train_dataset = NLUDataset(train_raw, tokenizer, slot2id, intent2id)
    dev_dataset   = NLUDataset(dev_raw, tokenizer, slot2id, intent2id)
    test_dataset  = NLUDataset(test_raw, tokenizer, slot2id, intent2id)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=config["batch_size"])
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"])

    model = JointBERT(
        model_name=config["model_name"],
        num_slot_labels=len(slot2id),
        num_intent_labels=len(intent2id),
        dropout=config.get("dropout", 0.1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    config["save_path"] = f"best_model_run{run_id}.pt"

    train_loop(model, train_loader, dev_loader, optimizer, config, device, id2slot, id2intent)

    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()

    print(f"\nEvaluation on TEST set (run {run_id})")
    test_metrics = evaluate(model, test_loader, device, id2slot, id2intent)
    print(f"Test Intent Accuracy: {test_metrics['intent_acc']:.4f}")
    print(f"Test Slot F1 Score: {test_metrics['slot_f1']:.4f}")

    return test_metrics["intent_acc"], test_metrics["slot_f1"]

# Trains the model for multiple epochs with early stopping based on dev performance.
def train_loop(model, train_loader, dev_loader, optimizer, config, device, id2slot, id2intent):
    best_score = 0.0
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, slot_labels, intent_label)

            loss = output["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # avg_loss = total_loss / len(train_loader)
        # print(f" Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

        val_metrics = evaluate(model, dev_loader, device, id2slot, id2intent)

        # print(f" Dev Intent Accuracy: {val_metrics['intent_acc']:.4f}")
        # print(f" Dev Slot F1 Score: {val_metrics['slot_f1']:.4f}")
        

        score = (val_metrics['intent_acc'] + val_metrics['slot_f1']) / 2
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), config["save_path"])
        else:
            patience += 1
            print(f"Early stopping patience: {patience}/{config['patience']}")
            if patience >= config["patience"]:
                print("Early stopping triggered.")
                break


# Evaluates the model on a dataset and returns intent accuracy and slot F1 score.
def evaluate(model, data_loader, device, id2slot, id2intent):
    model.eval()

    intent_preds, intent_trues = [], []
    slot_preds, slot_trues = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            output = model(input_ids, attention_mask)
            logits_intent = output["intent_logits"]
            logits_slot = output["slot_logits"]

            pred_intents = logits_intent.argmax(dim=1).cpu().tolist()
            true_intents = intent_label.cpu().tolist()

            pred_slots = logits_slot.argmax(dim=-1).cpu().tolist()
            true_slots = slot_labels.cpu().tolist()

            for p, t in zip(pred_intents, true_intents):
                intent_preds.append(p)
                intent_trues.append(t)

            for pred_seq, true_seq in zip(pred_slots, true_slots):
                true_labels = []
                pred_labels = []
                for t_id, p_id in zip(true_seq, pred_seq):
                    if t_id != -100:
                        true_labels.append(id2slot[t_id])
                        pred_labels.append(id2slot[p_id])
                slot_trues.append(true_labels)
                slot_preds.append(pred_labels)

    intent_acc = accuracy_score(intent_trues, intent_preds)

    ref_struct = [[('_W', 'X', tag) for tag in seq] for seq in slot_trues]
    hyp_struct = [[('_W', 'X', tag) for tag in seq] for seq in slot_preds]

    slot_f1 = conll_evaluate(ref_struct, hyp_struct)['total']['f']

    return {
        "intent_acc": intent_acc,
        "slot_f1": slot_f1,
        "intent_preds": intent_preds,
        "intent_trues": intent_trues,
        "slot_preds": slot_preds,
        "slot_trues": slot_trues
    }
