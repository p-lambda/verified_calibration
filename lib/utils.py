
import bisect
from typing import List, Tuple, NewType, TypeVar
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Define data types.

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')

eps = 1e-6


# Functions the produce bins from data.

def split(sequence: List[T], parts: int) -> List[List[T]]:
    assert parts <= len(sequence)
    part_size = int(np.ceil(len(sequence) * 1.0 / parts))
    assert part_size * parts >= len(sequence)
    assert (part_size - 1) * parts < len(sequence)
    return [sequence[i:i + part_size] for i in range(0, len(sequence), part_size)]


def get_equal_bins(probs: List[float], num_bins: int=10) -> Bins:
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs = sorted(probs)
    binned_data = split(sorted_probs, num_bins)
    bins: Bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    return bins


def get_equal_prob_bins(probs: List[float], num_bins: int=10) -> Bins:
    return [i * 1.0 / num_bins for i in range(1, num_bins + 1)]


def get_discrete_bins(data: List[float]) -> Bins:
    sorted_values = sorted(np.unique(data))
    bins = []
    for i in range(len(sorted_values) - 1):
        mid = (sorted_values[i] + sorted_values[i+1]) / 2.0
        bins.append(mid)
    bins.append(1.0)
    return bins


# Functions that bin data.

def get_bin(pred_prob: float, bins: List[float]) -> int:
    """Get the index of the bin that pred_prob belongs in."""
    assert 0.0 <= pred_prob <= 1.0
    return bisect.bisect_left(bins, pred_prob)


def bin(data: Data, bins: Bins):
    return fast_bin(data, bins)


def fast_bin(data, bins):
    prob_label = np.array(data)
    bin_indices = np.searchsorted(bins, prob_label[:, 0])
    bin_sort_indices = np.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
    binned_data = np.split(prob_label[bin_sort_indices], splits)
    return binned_data


def equal_bin(data: Data, num_bins : int) -> BinnedData:
    sorted_probs = sorted(data)
    return split(sorted_probs, num_bins)


# Calibration error estimators.

def difference_mean(data : Data) -> float:
    """Returns average pred_prob - average label."""
    data = np.array(data)
    ave_pred_prob = np.mean(data[:, 0])
    ave_label = np.mean(data[:, 1])
    return ave_pred_prob - ave_label


def get_bin_probs(binned_data: BinnedData) -> List[float]:
    bin_sizes = list(map(len, binned_data))
    num_data = sum(bin_sizes)
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    assert(abs(sum(bin_probs) - 1.0) < eps)
    return list(bin_probs)


def plugin_ce(binned_data: BinnedData, power=2) -> float:
    def bin_error(data: Data):
        if len(data) == 0:
            return 1.0
        return abs(difference_mean(data)) ** power
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors) ** (1.0 / power)


def unbiased_square_ce(binned_data: BinnedData) -> float:
    # Note, this is not the l2 CE. It does not take the square root.
    def bin_error(data: Data):
        if len(data) < 2:
            return 1.0
        biased_estimate = abs(difference_mean(data)) ** 2
        label_values = list(map(lambda x: x[1], data))
        mean_label = np.mean(label_values)
        variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        return biased_estimate - variance
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors)


def unbiased_l2_ce(binned_data: BinnedData) -> float:
    return max(unbiased_square_ce(binned_data), 0.0) ** 0.5


# MSE Estimators.

def eval_top_mse(probs, logits, labels):
    correct = (get_top_predictions(logits) == labels)
    return np.mean(np.square(probs - correct))


def eval_marginal_mse(probs, logits, labels):
    assert probs.shape == logits.shape
    k = logits.shape[1]
    labels_one_hot = get_labels_one_hot(np.array(labels), k)
    return np.mean(np.square(probs - labels_one_hot)) * probs.shape[1] / 2.0


# Bootstrap utilities.

def resample(data: List[T]) -> List[T]:
    indices = np.random.choice(list(range(len(data))), size=len(data), replace=True)
    return [data[i] for i in indices]


def bootstrap_uncertainty(data: List[T], functional, estimator=None, alpha=10.0, 
                          num_samples=1000) -> Tuple[float, float]:
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    if estimator is None:
        estimator = functional
    estimate = estimator(data)
    plugin = functional(data)
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return (plugin + estimate - np.percentile(bootstrap_estimates, 100 - alpha / 2.0),
            plugin + estimate - np.percentile(bootstrap_estimates, 50),
            plugin + estimate - np.percentile(bootstrap_estimates, alpha / 2.0))


def bootstrap_std(data: List[T], estimator=None, num_samples=100) -> Tuple[float, float]:
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return np.std(bootstrap_estimates)


# Re-Calibration utilities.

def get_platt_scaler(model_probs, labels):
    clf = LogisticRegression(C=1e10, solver='lbfgs')
    eps = 1e-12
    model_probs = model_probs.astype(dtype=np.float64)
    model_probs = np.expand_dims(model_probs, axis=-1)
    model_probs = np.clip(model_probs, eps, 1 - eps)
    model_probs = np.log(model_probs / (1 - model_probs))
    clf.fit(model_probs, labels)
    def calibrator(probs):
        x = np.array(probs, dtype=np.float64)
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * clf.coef_[0] + clf.intercept_
        output = 1 / (1 + np.exp(-x))
        return output
    return calibrator


def get_histogram_calibrator(model_probs, values, bins):
    binned_values = [[] for _ in range(len(bins))]
    for prob, value in zip(model_probs, values):
        bin_idx = get_bin(prob, bins)
        binned_values[bin_idx].append(float(value))
    def safe_mean(values, bin_idx):
        if len(values) == 0:
            if bin_idx == 0:
                return float(bins[0]) / 2.0
            return float(bins[bin_idx] + bins[bin_idx - 1]) / 2.0
        return np.mean(values)
    bin_means = [safe_mean(values, bidx) for values, bidx in zip(binned_values, range(len(bins)))]
    bin_means = np.array(bin_means)
    def calibrator(probs):
        indices = np.searchsorted(bins, probs)
        return bin_means[indices]
    return calibrator


def get_discrete_calibrator(model_probs, bins):
    return get_histogram_calibrator(model_probs, model_probs, bins)


# Utils to load and save files.

def save_test_logits_labels(dataset, model, filename):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    logits = model.predict(x_test) 
    pickle.dump((logits, y_test), open(filename, "wb"))


def load_test_logits_labels(filename):
    logits, labels = pickle.load(open(filename, "rb"))
    if len(labels.shape) > 1:
        labels = labels[:, 0]
    indices = np.random.choice(list(range(len(logits))), size=len(logits), replace=False)
    logits = np.array([logits[i] for i in indices])
    labels = np.array([labels[i] for i in indices])
    return logits, labels


def get_top_predictions(logits):
    return np.argmax(logits, 1)


def get_top_probs(logits):
    return np.max(logits, 1)


def get_accuracy(logits, labels):
    return sum(labels == predictions) * 1.0 / len(labels)


def get_labels_one_hot(labels, k):
    assert np.min(labels) == 0
    assert np.max(labels) == k - 1
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros((num_labels, k))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return labels_one_hot
