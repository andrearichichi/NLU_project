import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shutil

from LM.part_A.model import LM_RNN, LM_LSTM
from LM.part_A.utils import Lang, create_dataset, create_dataloader

# Initialize weights (RNN/LSTM + Linear)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.RNN, nn.LSTM)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

# Full training loop with early stopping and evaluation
def train_loop(model, train_loader, dev_loader, test_loader, optimizer,
               criterion_train, criterion_eval, config, run_id):

    best_ppl = float("inf")
    wait = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch["source"])
            loss = criterion_train(output, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()

            total_loss += loss.item() * batch["number_tokens"]
            total_tokens += batch["number_tokens"]

        avg_train_loss = total_loss / total_tokens
        val_ppl, val_loss = evaluate(dev_loader, model, criterion_eval)

        print(f"[Epoch {epoch}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Dev Loss: {val_loss:.4f} | Dev PPL: {val_ppl:.4f}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_model = copy.deepcopy(model).to("cpu")
            wait = config["patience"]
        else:
            wait -= 1

        if wait == 0:
            print(f" Early stopping at epoch {epoch}")
            break

    # Evaluate best model on test set
    torch.save(best_model.state_dict(), f"bin/best_model_run{run_id}.pt")
    best_model.to(config["device"])
    final_ppl, _ = evaluate(test_loader, best_model, criterion_eval)
    return final_ppl

# Evaluate model and return (PPL, avg loss)
def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            output = model(batch["source"])
            loss = criterion(output, batch["target"])
            tokens = batch["number_tokens"]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.item()
            total_loss += loss.item()
            total_tokens += tokens

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return ppl, avg_loss

# Run experiment for 1 or more runs
def run_experiment(config):
    if os.path.exists("bin"):
        shutil.rmtree("bin")
    os.makedirs("bin")

    lang = Lang(config["train_data"], ["<pad>", "<eos>"])
    train_set, dev_set, test_set = create_dataset(
        config["train_data"], config["dev_data"], config["test_data"], lang)
    train_loader, dev_loader, test_loader = create_dataloader(
        train_set, dev_set, test_set, lang)

    pad_idx = lang.word2id["<pad>"]
    crit_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    crit_eval = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")

    results = []

    for run_id in range(config["runs"]):
        model = LM_LSTM(
            config["emb_size"], config["hid_size"], len(lang.word2id),
            pad_idx=pad_idx, use_dropout=config["use_dropout"]
        ).to(config["device"]) if config["architecture"] == "LSTM" else LM_RNN(
            config["emb_size"], config["hid_size"], len(lang.word2id),
            pad_idx=pad_idx
        ).to(config["device"])

        initialize_weights(model)

        optimizer = (
            optim.SGD(model.parameters(), lr=config["lr"])
            if config["optimizer"] == "SGD"
            else optim.AdamW(model.parameters(), lr=config["lr"])
        )

        final_ppl = train_loop(
            model, train_loader, dev_loader, test_loader,
            optimizer, crit_train, crit_eval,
            config, run_id
        )
        print(f" Run {run_id + 1}/{config['runs']} completed. Test PPL: {round(final_ppl, 3)}") 
        results.append(final_ppl)

    # Final summary
    if config["runs"] == 1:
        print(f"\n Test PPL: {round(results[0], 3)}")
    else:
        print(f"\n Final Test PPL: {round(np.mean(results), 3)} ± {round(np.std(results), 2)}")
