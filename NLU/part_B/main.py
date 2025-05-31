import numpy as np
from NLU.part_B.functions import run_experiment

config = {
    "dataset_path": "dataset",
    "model_name": "bert-base-uncased",
    "batch_size": 64,
    "lr": 5e-5,
    "dropout": 0.1,
    "num_epochs": 20,
    "patience": 3,
    "total_runs": 3  
}

intent_accuracies = []
slot_f1s = []

for run_id in range(1, config["total_runs"] + 1):
    intent_acc, slot_f1 = run_experiment(config, run_id)
    intent_accuracies.append(intent_acc)
    slot_f1s.append(slot_f1)

slot_f1_mean = np.mean(slot_f1s)
slot_f1_std = np.std(slot_f1s)
intent_acc_mean = np.mean(intent_accuracies)
intent_acc_std = np.std(intent_accuracies)

print(f"\n [BERT] Mean ± Standard Deviation over {config['total_runs']} runs:")
print(f"Slot F1       → {slot_f1_mean:.3f} ± {slot_f1_std:.3f}")
print(f"Intent Acc    → {intent_acc_mean:.3f} ± {intent_acc_std:.3f}")
