# BERT Joint NLU – Usage Guide

To run the project, simply execute:

python main.py

All configurations are set directly in main.py inside the config dictionary.
No command-line arguments are needed.

Defaults:
config = {
    "dataset_path": "dataset",
    "model_name": "bert-base-uncased",
    "batch_size": 64, (128 train batch size)
    "lr": 5e-5,
    "dropout": 0.1,
    "num_epochs": 10,
    "patience": 3,
    "total_runs": 3
}
