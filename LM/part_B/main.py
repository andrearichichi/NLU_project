# main.py
import argparse
import torch
from functions import run_experiment
from utils import read_file

# Predefined configurations for different model architectures and optimizers
PREDEFINED_CONFIGS = {
    "lstm_sgd_wt": {
        "architecture": "LSTM",
        "optimizer": "SGD",
        "use_dropout": False,
        "weight_tying": True,
        "lr": 3.0,
        "dropout": 0.0,
        "emb_size": 300,
        "hid_size": 300,
        "clip": 5.0,
        "patience": 3
    },
    "lstm_sgd_wt_vdrop": {
        "architecture": "LSTM",
        "optimizer": "SGD",
        "use_dropout": True,
        "weight_tying": True,
        "lr": 3.0,
        "dropout": 0.2,
        "emb_size": 300,
        "hid_size": 300,
        "clip": 5.0,
        "patience": 3
    },
    "lstm_ntasgd_wt_vdrop": {
        "architecture": "LSTM",
        "optimizer": "NTASGD",
        "use_dropout": True,
        "weight_tying": True,
        "lr": 30.0,
        "dropout": 0.4,
        "emb_size": 400,
        "hid_size": 400,
        "clip": 0.25,
        "patience": 3
    }
}

# Shared training hyperparameters
CONFIG = {
    "epochs": 200,
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "runs": 1  
}

# Loads the configuration, prepares the dataset, and launches the training experiment.
def launch(config_key):
    if config_key not in PREDEFINED_CONFIGS:
        print(f"Configuration '{config_key}' not found. Using default 'lstm_sgd_wt'.")
        config_key = "lstm_sgd_wt"

    conf = PREDEFINED_CONFIGS[config_key]

    print(f"Selected configuration: {config_key}")
    print(f"Architecture: {conf['architecture']} | Optimizer: {conf['optimizer']} | Weight Tying: {conf['weight_tying']} | Dropout: {conf['use_dropout']} (value={conf['dropout']}) | Learning Rate: {conf['lr']}")

    train = read_file("dataset/ptb.train.txt")
    valid = read_file("dataset/ptb.valid.txt")
    test = read_file("dataset/ptb.test.txt")
    
    run_experiment(
        train_data=train,
        dev_data=valid,
        test_data=test,
        lr=conf["lr"],
        runs=CONFIG["runs"],
        epochs=CONFIG["epochs"],
        clip=conf["clip"],
        patience=conf["patience"],
        device=CONFIG["device"],
        hid_size=conf["hid_size"],
        emb_size=conf["emb_size"],
        optimizer_type=conf["optimizer"],
        use_dropout=conf["use_dropout"],
        weight_tying=conf["weight_tying"],
        dropout=conf["dropout"]
    )

# Parses command-line arguments and starts the experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lstm_sgd_wt", help="Name of the configuration to run")
    parser.add_argument("--runs", type=int, help="Number of runs to execute")
    args = parser.parse_args()

    if args.runs:
        CONFIG["runs"] = args.runs

    launch(args.config)
