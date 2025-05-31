# Sequence Labeling - Usage Guide

Run the project with:

python main.py --config <CONFIG> --runs <N_RUNS>

Available configs:
- LSTM
- BiLSTM
- BiLSTM_dropout
- all → runs all

Defaults:
- --runs defaults to 3 if not specified
- if --config all is used, each config is run N times (default 3)
