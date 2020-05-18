import calibration as cal
import torch
import torch.nn.functional as F
import numpy as np


def ece(logits, labels, n_bins=15, display=False):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    if display: print("Num\tConf\tAcc")
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        avg_confidence_in_bin = 0
        accuracy_in_bin = 0
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        if display: print(f"{in_bin.sum()}\t{avg_confidence_in_bin:.2f}\t{accuracy_in_bin:.2f}")
    return ece

D = torch.load("logits_and_labels.pt")
logits = D["logits"]
labels = D["labels"]

probs = F.softmax(logits, dim=1).numpy()
print(logits.shape)
# print(np.amax(logits[:100], axis=1))
my_ece = cal.lower_bound_scaling_top_ce(probs, labels.numpy(), p=1, debias=False)
ece = ece(logits, labels)
print(my_ece)
print(ece)
