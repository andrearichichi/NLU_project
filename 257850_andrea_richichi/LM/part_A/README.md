# Language Modeling – Usage Guide

Run the project with:

python main.py --config <CONFIG> --runs <N_RUNS>

Available configs:
- rnn_sgd
- lstm_sgd
- lstm_sgd_dropout
- lstm_adamw_dropout

Defaults:
- --runs defaults to 1 if not specified
- If --config is not recognized, defaults to lstm_adamw_dropout.
- For each run, the model is trained with early stopping and the best test perplexity is reported.

Example:

python main.py --config lstm_adamw_dropout --runs 5
