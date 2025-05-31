# NLU Course Project: Language Modeling & Understanding

**Author:** Andrea Richichi
**University of Trento**

## Overview

This repository includes two independent projects developed for the NLU course:

* **Lab 4 – Language Modeling:**
  RNN and LSTM models trained on Penn Treebank for next-word prediction. Explores regularization and optimization techniques (e.g., dropout, weight tying, NT-ASGD).

* **Lab 5 – Natural Language Understanding:**
  Joint intent classification and slot filling on the ATIS dataset. Compares LSTM-based models and fine-tuned BERT.

## Results Summary

| Task                  | Best Model     | Metric   | Score      |
| --------------------- | -------------- | -------- | ---------- |
| Language Modeling     | LSTM + NT-ASGD | Test PPL | **94.33**  |
| Intent Classification | BERT           | Accuracy | **97.36%** |
| Slot Filling          | BERT           | F1 Score | **95.38%** |

## Structure

```
.
├── LM/        # Lab 4: Language Modeling
├── NLU/       # Lab 5: NLU (Intent + Slot)
└── README.md  # Main project readme
```

Each subdirectory contains its own README with setup and usage instructions.

## Requirements

* Python 3.x
* PyTorch
* Transformers (HuggingFace)

