import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shutil
from tqdm import tqdm

from model import LM_LSTM
from utils import Lang, create_dataset, create_dataloader

# Initializes LSTM and Linear weights using standard initialization techniques.
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.LSTM,)):
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


# Performs the training loop with early stopping and NTASGD support.
def train_loop(model, optimizer, train_loader, dev_loader,
               crit_train, crit_eval, clip, patience, epochs):
    best_ppl = float('inf')
    wait = patience
    avg_wait = None
    best_model = None

    for epoch in range(1, epochs): 
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch["source"])
            loss = crit_train(output, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            ntokens = batch["number_tokens"]
            ntokens = ntokens.item() if isinstance(ntokens, torch.Tensor) else ntokens
            total_loss += loss.item() * ntokens
            total_tokens += ntokens

        train_loss = total_loss / total_tokens
        val_ppl, val_loss = evaluate(dev_loader, model, crit_eval)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Dev Loss: {val_loss:.4f} | Dev PPL: {val_ppl:.4f}")

        if isinstance(optimizer, NTASGD):
            optimizer.update(val_ppl)
            if optimizer.averaging and avg_wait is None:
                avg_wait = optimizer.nonmono

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_model = copy.deepcopy(model).to("cpu")
            if isinstance(optimizer, NTASGD) and optimizer.averaging:
                avg_wait = optimizer.nonmono
            else:
                wait = patience
        else:
            if isinstance(optimizer, NTASGD):
                if optimizer.averaging:
                    if avg_wait is not None:
                        avg_wait -= 1
                else:
                    wait -= 1
                    if wait == 0:
                        print(f" Switching to averaging instead of early stopping at epoch {epoch}")
                        optimizer.avg_params = [p.data.clone() for p in model.parameters()]
                        optimizer.averaging = True
                        optimizer.n_avg = 1
                        avg_wait = optimizer.nonmono
            else:
                wait -= 1

        
        if isinstance(optimizer, NTASGD):
            if optimizer.averaging and avg_wait == 0:
                print(f"⏹️ Early stopping post-averaging at epoch {epoch}")
                break
        else:
            if wait == 0:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    if isinstance(optimizer, NTASGD):
        optimizer.assign_average()

    return best_model

# Orchestrates the full training pipeline for one or more runs, including model saving and evaluation.
def run_experiment(train_data, dev_data, test_data,
                   lr, runs, epochs, clip, patience,
                   device, hid_size, emb_size,
                    optimizer_type, use_dropout,
                   weight_tying, dropout):

    if os.path.exists("bin"):
        shutil.rmtree("bin")
    os.makedirs("bin")

    lang = Lang(train_data, ["<pad>", "<eos>"])
    train_set, dev_set, test_set = create_dataset(train_data, dev_data, test_data, lang)
    train_loader, dev_loader, test_loader = create_dataloader(train_set, dev_set, test_set, lang)

    crit_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    crit_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction="sum")

    results = []

    for run_id in range(runs):
        model = LM_LSTM(emb_size, hid_size, len(lang.word2id),
                        pad_idx=lang.word2id["<pad>"],
                        use_dropout=use_dropout,
                        weight_tying=weight_tying,
                        dropout=dropout).to(device)

        initialize_weights(model)

        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_type == "NTASGD":
            optimizer = NTASGD(model, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        best_model = train_loop(model, optimizer, train_loader, dev_loader,
                                crit_train, crit_eval, clip, patience, epochs)

        if best_model is not None:
            best_model.to(device)
            torch.save(best_model.state_dict(), f"bin/bestmodel_{run_id+1}.pt")
            final_ppl, _ = evaluate(test_loader, best_model, crit_eval)
            results.append(final_ppl)

    
    if runs > 1:
        print(f"\n Final Test PPL: {round(np.mean(results), 3)} ± {round(np.std(results), 2)}")
    elif runs == 1 and results:
        print(f"\n Final Test PPL: {round(results[0], 3)}")


# Evaluates the model on the provided data loader and returns perplexity and loss.
def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            output = model(batch["source"])
            loss = criterion(output, batch["target"])

            ntokens = batch["number_tokens"]
            ntokens = ntokens.item() if isinstance(ntokens, torch.Tensor) else ntokens

            total_loss += loss.item()
            total_tokens += ntokens

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return ppl, avg_loss

# Custom optimizer that implements Non-monotonically Triggered Averaged SGD (NT-ASGD).
class NTASGD:
    
    # Initializes the NTASGD optimizer with model parameters and NT trigger settings.
    def __init__(self, model, lr, nonmono=5):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.lr = lr
        self.nonmono = nonmono
        self.best_val = float('inf')
        self.bad_count = 0
        self.averaging = False
        self.avg_params = None
        self.n_avg = 0

    # Applies a single SGD optimization step.
    def step(self):
        self.optimizer.step()

    # Resets the gradients of the model parameters.
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Updates the optimizer state based on validation loss and manages weight averaging.
    def update(self, val_loss):
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.bad_count = 0
        else:
            self.bad_count += 1

        if self.bad_count >= self.nonmono and not self.averaging:
            print("NT-ASGD: starting weight averaging")
            self.avg_params = [p.data.clone() for p in self.model.parameters()]
            self.averaging = True
            self.n_avg = 1

        elif self.averaging and self.avg_params is not None:
            self.n_avg += 1
            for avg, p in zip(self.avg_params, self.model.parameters()):
                avg.mul_((self.n_avg - 1) / self.n_avg).add_(p.data / self.n_avg)

    # Returns the current learning rate.
    def assign_average(self):
        if self.averaging and self.avg_params is not None:
            for avg, p in zip(self.avg_params, self.model.parameters()):
                p.data.copy_(avg)
