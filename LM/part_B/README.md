
# AWD-LSTM Language Model (PTB)

This project implements 3 variants of an LSTM-based language model on the Penn Treebank (PTB) dataset.

## 🔧 How to Run

Make sure the following dataset files are available:

```
dataset/ptb.train.txt
dataset/ptb.valid.txt
dataset/ptb.test.txt
```

Run one of the following configurations:

```bash
python main.py --config lstm_sgd_wt
python main.py --config lstm_sgd_wt_vdrop
python main.py --config lstm_ntasgd_wt_vdrop
```

Each run will print training and evaluation loss and perplexity.
