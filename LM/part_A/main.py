import argparse
import torch
from LM.part_A.functions import run_experiment
from LM.part_A.utils import read_file

# Predefined configurations (model type, optimizer, learning rate, dropout)
PREDEFINED_CONFIGS = {
    "rnn_sgd": {
        "architecture": "RNN",
        "optimizer": "SGD",
        "use_dropout": False,
        "lr": 1
    },
    "lstm_sgd": {
        "architecture": "LSTM",
        "optimizer": "SGD",
        "use_dropout": False,
        "lr": 3
    },
    "lstm_sgd_dropout": {
        "architecture": "LSTM",
        "optimizer": "SGD",
        "use_dropout": True,
        "lr": 3
    },
    "lstm_adamw_dropout": {
        "architecture": "LSTM",
        "optimizer": "AdamW",
        "use_dropout": True,
        "lr": 0.001
    }
}

# Shared training hyperparameters
DEFAULT_CONFIG = {
    "epochs": 200,
    "patience": 3,
    "clip": 5,
    "hid_size": 200,
    "emb_size": 300,
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "runs": 1
}

# Merge config, load data, and run the experiment
def launch(config_key, runs_override=None):
    if config_key not in PREDEFINED_CONFIGS:
        print(f" Config '{config_key}' not found. Defaulting to 'lstm_adamw_dropout'.")
        config_key = "lstm_adamw_dropout"

    config = {**DEFAULT_CONFIG, **PREDEFINED_CONFIGS[config_key]}

    if runs_override is not None:
        config["runs"] = runs_override

    # Load datasets
    config["train_data"] = read_file("dataset/ptb.train.txt")
    config["dev_data"]   = read_file("dataset/ptb.valid.txt")
    config["test_data"]  = read_file("dataset/ptb.test.txt")

    # Print configuration summary
    print(f" Config: {config_key}")
    print(f" Architecture: {config['architecture']} | Optimizer: {config['optimizer']} | Dropout: {config['use_dropout']} | LR: {config['lr']} | Runs: {config['runs']}")

    run_experiment(config)

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lstm_adamw_dropout",
                        help="Choose one of: rnn_sgd, lstm_sgd, lstm_sgd_dropout, lstm_adamw_dropout")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs to execute (overrides default)")

    args = parser.parse_args()
    launch(args.config, runs_override=args.runs)
