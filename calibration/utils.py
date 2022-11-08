
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
    array_splits = np.array_split(sequence, parts)
    splits = [list(l) for l in array_splits]
    assert len(splits) == parts
    return splits

def get_equal_bins(probs: List[float], num_bins: int=10) -> Bins:
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs = sorted(probs)
    if num_bins > len(sorted_probs):
        num_bins = len(sorted_probs)
    binned_data = split(sorted_probs, num_bins)
    bins: Bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    bins = sorted(list(set(bins)))
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


# User facing functions to measure calibration error.

def get_top_calibration_error_uncertainties(probs, labels, p=2, alpha=0.1):
    return get_calibration_error_uncertainties(probs, labels, p, alpha, mode='top-label')


def get_calibration_error_uncertainties(probs, labels, p=2, alpha=0.1, mode='marginal'):
    """Get confidence intervals for the calibration error.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        [lower, mid, upper]: 1-alpha confidence intervals produced by bootstrap resampling.
        [lower, upper] represents the confidence interval. mid represents the median of
        the bootstrap estimates. When p is not 2 (e.g. for the ECE where p = 1), this
        can be used as a debiased estimate as well.
    """
    data = list(zip(probs, labels))
    def ce_functional(data):
        probs, labels = zip(*data)
        return get_calibration_error(probs, labels, p, debias=False, mode=mode)
    [lower, mid, upper] = bootstrap_uncertainty(data, ce_functional, num_samples=100, alpha=alpha)
    return [lower, mid, upper]


def get_top_calibration_error(probs, labels, p=2, debias=True):
    return get_calibration_error(probs, labels, p, debias, mode='top-label')


def get_calibration_error(probs, labels, p=2, debias=True, mode='marginal'):
    """Get the calibration error.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
        The method first uses heuristics to check if the values came from a scaling
        method or binning method, and then calls the corresponding function. For
        more explicit control, use lower_bound_scaling_ce or get_binning_ce.
    """
    if is_discrete(probs):
        return get_binning_ce(probs, labels, p, debias, mode=mode)
    else:
        return lower_bound_scaling_ce(probs, labels, p, debias, mode=mode)


def lower_bound_scaling_top_ce(probs, labels, p=2, debias=True, num_bins=15,
                               binning_scheme=get_equal_bins):
    return lower_bound_scaling_ce(probs, labels, p, debias, num_bins, binning_scheme,
                                  mode='top-label')


def lower_bound_scaling_ce(probs, labels, p=2, debias=True, num_bins=15,
                           binning_scheme=get_equal_bins, mode='marginal'):
    """Lower bound the calibration error of a model with continuous outputs.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        num_bins: Integer number of bins used to estimate the calibration error.
        binning_scheme: A function that takes in a list of probabilities and number of bins,
            and outputs a list of bins. See get_equal_bins, get_equal_prob_bins for examples.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated lower bound for calibration error, a floating point value.
        For scaling methods we cannot estimate the calibration error, but only a
        lower bound.
    """
    return _get_ce(probs, labels, p, debias, num_bins, binning_scheme, mode=mode)


def get_binning_top_ce(probs, labels, p=2, debias=True, mode='marginal'):
    return get_binning_ce(probs, labels, p, debias, mode='top-label')


def get_binning_ce(probs, labels, p=2, debias=True, mode='marginal'):
    """Estimate the calibration error of a binned model.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
    """
    return _get_ce(probs, labels, p, debias, None, binning_scheme=get_discrete_bins, mode=mode)


def get_ece(probs, labels, debias=False, num_bins=15, mode='top-label'):
    """Get ECE as computed by Guo et al."""
    return lower_bound_scaling_ce(probs, labels, p=1, debias=debias, num_bins=num_bins,
                                  binning_scheme=get_equal_prob_bins, mode=mode)


def get_ece_em(probs, labels, debias=False, num_bins=15, mode='top-label'):
    """Get ECE, but use equal mass binning."""
    return lower_bound_scaling_ce(probs, labels, p=1, debias=debias, num_bins=num_bins,
                                  binning_scheme=get_equal_bins, mode=mode)


def get_selective_stats(probs, correct):
    """Return area under coverage-accuracy curve, and acc for 10% most confident predictions."""
    # Sort in descending order.
    probs = np.array(probs)
    correct = np.array(correct)
    sort_indices = np.argsort(-probs)
    sorted_correct = correct[sort_indices]
    accs = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
    coverage_acc_area = np.mean(accs)
    acc_percentile_90 = accs[int(0.1 * len(sorted_correct))]
    return coverage_acc_area, acc_percentile_90


def _get_ce(probs, labels, p, debias, num_bins, binning_scheme, mode='marginal'):
    def ce_1d(probs, labels):
        assert probs.shape == labels.shape
        assert len(probs.shape) == 1
        data = list(zip(probs, labels))
        if binning_scheme == get_discrete_bins:
            assert(num_bins is None)
            bins = binning_scheme(probs)
        else:
            bins = binning_scheme(probs, num_bins=num_bins)
        if p == 2 and debias:
            return unbiased_l2_ce(bin(data, bins))
        elif debias:
            return normal_debiased_ce(bin(data, bins), power=p)
        else:
            return plugin_ce(bin(data, bins), power=p)
    if mode != 'marginal' and mode != 'top-label':
        raise ValueError("mode must be 'marginal' or 'top-label'.")
    probs = np.array(probs)
    labels = np.array(labels)
    if not(np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('labels should an integer numpy array.')
    if len(labels.shape) != 1:
        raise ValueError('labels should be a 1D numpy array.')
    if probs.shape[0] != labels.shape[0]:
        raise ValueError('labels and probs should have the same number of entries.')
    if len(probs.shape) == 1:
        # If 1D (2-class setting), compute the regular calibration error.
        if np.min(labels) < 0 or np.max(labels) > 1:
            raise ValueError('If probs is 1D, each label should be 0 or 1.')
        return ce_1d(probs, labels)
    elif len(probs.shape) == 2:
        if np.min(labels) < 0 or np.max(labels) > probs.shape[1] - 1:
            raise ValueError('labels should be between 0 and num_classes - 1.')
        if mode == 'marginal':
            labels_one_hot = get_labels_one_hot(labels, k=probs.shape[1])
            assert probs.shape == labels_one_hot.shape
            marginal_ces = []
            for k in range(probs.shape[1]):
                cur_probs = probs[:, k]
                cur_labels = labels_one_hot[:, k]
                marginal_ces.append(ce_1d(cur_probs, cur_labels) ** p)
            return np.mean(marginal_ces) ** (1.0 / p)
        elif mode == 'top-label':
            preds = get_top_predictions(probs)
            correct = (preds == labels).astype(probs.dtype)
            confidences = get_top_probs(probs)
            return ce_1d(confidences, correct)
    else:
        raise ValueError('probs should be a 1D or 2D numpy array.')


def is_discrete(probs):
    probs = np.array(probs)
    if len(probs.shape) == 1:
        return enough_duplicates(probs)
    elif len(probs.shape) == 2:
        for k in range(probs.shape[1]):
            if not enough_duplicates(probs[:, k]):
                return False
        return True
    else:
        raise ValueError('probs must be a 1D or 2D numpy array.')


def enough_duplicates(array):
    # TODO: instead check that we have at least 2 values in each bin.
    num_bins = get_discrete_bins(array)
    if len(num_bins) < array.shape[0] / 4.0:
        return True
    return False


# Functions that bin data.

def get_bin(pred_prob: float, bins: List[float]) -> int:
    """Get the index of the bin that pred_prob belongs in."""
    assert 0.0 <= pred_prob <= 1.0
    assert bins[-1] == 1.0
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
            return 0.0
        return abs(difference_mean(data)) ** power
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors) ** (1.0 / power)


def unbiased_square_ce(binned_data: BinnedData) -> float:
    # Note, this is not the l2 CE. It does not take the square root.
    def bin_error(data: Data):
        if len(data) < 2:
            return 0.0
            # raise ValueError('Too few values in bin, use fewer bins or get more data.')
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


def normal_debiased_ce(binned_data : BinnedData, power=1, resamples=1000) -> float:
    bin_sizes = np.array(list(map(len, binned_data)))
    if np.min(bin_sizes) <= 1:
        raise ValueError('Every bin must have at least 2 points for debiased estimator. '
                         'Try adding the argument debias=False to your function call.')
    label_means = np.array(list(map(lambda l: np.mean([b for a, b in l]), binned_data)))
    label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
    model_vals = np.array(list(map(lambda l: np.mean([a for a, b in l]), binned_data)))
    assert(label_means.shape == (len(binned_data),))
    assert(model_vals.shape == (len(binned_data),))
    ce = plugin_ce(binned_data, power=power)
    bin_probs = get_bin_probs(binned_data)
    resampled_ces = []
    for i in range(resamples):
        label_samples = np.random.normal(loc=label_means, scale=label_stddev)
        # TODO: we can also correct the bias for the model_vals, although this is
        # smaller.
        diffs = np.power(np.abs(label_samples - model_vals), power)
        cur_ce = np.power(np.dot(bin_probs, diffs), 1.0 / power)
        resampled_ces.append(cur_ce)
    mean_resampled = np.mean(resampled_ces)
    bias_corrected_ce = 2 * ce - mean_resampled
    return bias_corrected_ce


# MSE Estimators.

def eval_top_mse(calibrated_probs, probs, labels):
    correct = (get_top_predictions(probs) == labels)
    return np.mean(np.square(calibrated_probs - correct))


def eval_marginal_mse(calibrated_probs, probs, labels):
    assert calibrated_probs.shape == probs.shape
    k = probs.shape[1]
    labels_one_hot = get_labels_one_hot(np.array(labels), k)
    return np.mean(np.square(calibrated_probs - labels_one_hot)) * calibrated_probs.shape[1] / 2.0


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


def precentile_bootstrap_uncertainty(data: List[T], functional, estimator=None, alpha=10.0,
                                     num_samples=1000) -> Tuple[float, float]:
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    if estimator is None:
        estimator = functional
    plugin = functional(data)
    estimate = estimator(data)
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    bias = 2 * np.percentile(bootstrap_estimates, 50) - plugin - estimate
    return (np.percentile(bootstrap_estimates, alpha / 2.0) - bias,
            np.percentile(bootstrap_estimates, 50) - bias,
            np.percentile(bootstrap_estimates, 100 - alpha / 2.0) - bias)


def bootstrap_std(data: List[T], estimator=None, num_samples=100) -> Tuple[float, float]:
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return np.std(bootstrap_estimates)


# Re-Calibration utilities.

def get_platt_scaler(model_probs, labels, get_clf=False):
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
    if get_clf:
        return calibrator, clf
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

def save_test_probs_labels(dataset, model, filename):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    probs = model.predict(x_test) 
    pickle.dump((probs, y_test), open(filename, "wb"))


def load_test_probs_labels(filename):
    probs, labels = pickle.load(open(filename, "rb"))
    if len(labels.shape) > 1:
        labels = labels[:, 0]
    indices = np.random.choice(list(range(len(probs))), size=len(probs), replace=False)
    probs = np.array([probs[i] for i in indices])
    labels = np.array([labels[i] for i in indices])
    return probs, labels


def get_top_predictions(probs):
    return np.argmax(probs, 1)


def get_top_probs(probs):
    return np.max(probs, 1)


def get_accuracy(probs, labels):
    return sum(labels == predictions) * 1.0 / len(labels)


def get_labels_one_hot(labels, k):
    assert np.min(labels) >= 0
    assert np.max(labels) <= k - 1
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros((num_labels, k))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return labels_one_hot
