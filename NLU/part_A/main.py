import argparse
from functions import run_experiment

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['LSTM', 'BiLSTM', 'BiLSTM_dropout', 'all'], default='LSTM')
parser.add_argument('--runs', type=int, default=3, help=" Number of runs to repeat the experiment")
args = parser.parse_args()

# Hyperparameters
config = {
    "batch_size": 64,
    "embedding_dim": 300,
    "hidden_dim": 200,
    "lr": 0.0005,
    "num_epochs": 200,
    "patience": 3,
    "seed": 42,
    "portion": 0.1,
    "clip": 5.0,
    "cutoff": 0,  
}

# Run experiment
if args.config == "all":
    for model_type in ["LSTM", "BiLSTM", "BiLSTM_dropout"]:
        run_experiment(model_type, args.runs, config)
else:
    run_experiment(args.config, args.runs, config)
