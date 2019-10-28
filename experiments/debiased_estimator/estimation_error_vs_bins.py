
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os
import lib.utils as utils


np.random.seed(0)  # Keep results consistent.

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='data/cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')
parser.add_argument('--platt_data_size', default=1000, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--bin_data_size', default=2000, type=int,
                    help='Number of examples to use for binning.')
parser.add_argument('--num_bins', default=100, type=int,
                    help='Bins to test estimators with.')


def compare_estimators(logits, labels, platt_data_size, bin_data_size, num_bins,
                       ver_base_size=2000, ver_size_increment=1000, num_resamples=1000,
                       save_prefix='./saved_files/debiased_estimator/'):
    # Convert logits to prediction, probs.
    predictions = utils.get_top_predictions(logits)
    probs = utils.get_top_probs(logits)
    correct = (predictions == labels)
    # Platt scale on first chunk of data
    platt = utils.get_platt_scaler(probs[:platt_data_size], correct[:platt_data_size])
    platt_probs = platt(probs)
    estimator_names = ['biased', 'unbiased']
    estimators = [lambda x: utils.plugin_ce(x) ** 2, utils.unbiased_square_ce]

    bins = utils.get_equal_bins(
        platt_probs[:platt_data_size+bin_data_size], num_bins=num_bins)
    binner = utils.get_discrete_calibrator(
        platt_probs[platt_data_size:platt_data_size+bin_data_size], bins)
    verification_probs = binner(platt_probs[platt_data_size+bin_data_size:])
    verification_correct = correct[platt_data_size+bin_data_size:]
    verification_data = list(zip(verification_probs, verification_correct))
    verification_sizes = list(range(ver_base_size, len(verification_probs) + 1,
                              ver_size_increment))
    # We want to compare the two estimators when varying the number of samples.
    # However, a single point of comparison does not tell us much about the estimators.
    # So we use resampling - we resample from the test set many times, and run the estimators
    # on the resamples. We stores these values. This gives us a sense of the range of values
    # the estimator might output.
    # So estimates[i][j][k] stores the estimate when using estimator i, with verification_sizes[j]
    # samples, in the k-th resampling.
    estimates = np.zeros((len(estimators), len(verification_sizes), num_resamples))
    # We also store the certified estimates. These represent the upper bounds we get using
    # each estimator. They are computing using the std-dev of the estimator estimated by
    # Bootstrap.
    cert_estimates = np.zeros((len(estimators), len(verification_sizes), num_resamples))
    for ver_idx, verification_size in zip(range(len(verification_sizes)), verification_sizes):
        for k in range(num_resamples):
            # Resample
            indices = np.random.choice(list(range(len(verification_data))),
                                       size=verification_size, replace=True)
            cur_verification_data = [verification_data[i] for i in indices]
            cur_verification_probs = [verification_probs[i] for i in indices]
            bins = utils.get_discrete_bins(cur_verification_probs)
            # Compute estimates for each estimator.
            for i in range(len(estimators)):
                def estimator(data):
                    binned_data = utils.bin(data, bins)
                    return estimators[i](binned_data)
                cur_estimate = estimator(cur_verification_data)
                estimates[i][ver_idx][k] = cur_estimate
                # cert_resampling_estimates[j].append(
                #   cur_estimate + utils.bootstrap_std(cur_verification_data, estimator, num_samples=20))

    estimates = np.sort(estimates, axis=-1)
    lower_bound = int(0.1 * num_resamples)
    median = int(0.5 * num_resamples)
    upper_bound = int(0.9 * num_resamples)
    lower_estimates = estimates[:, :, lower_bound]
    upper_estimates = estimates[:, :, upper_bound]
    median_estimates = estimates[:, :, median]

    # We can also compute the MSEs of the estimator.
    bins = utils.get_discrete_bins(verification_probs)
    true_calibration = utils.plugin_ce(utils.bin(verification_data, bins)) ** 2
    print(true_calibration)
    print(np.sqrt(np.mean(estimates[1, -1, :])))
    errors = np.abs(estimates - true_calibration)
    accumulated_errors = np.mean(errors, axis=-1)
    error_bars_90 = 1.645 * np.std(errors, axis=-1) / np.sqrt(num_resamples)
    print(accumulated_errors)
    plt.errorbar(
        verification_sizes, accumulated_errors[0], yerr=[error_bars_90[0], error_bars_90[0]],
        barsabove=True, color='red', capsize=4, label='plugin')
    plt.errorbar(
        verification_sizes, accumulated_errors[1], yerr=[error_bars_90[1], error_bars_90[1]],
        barsabove=True, color='blue', capsize=4, label='debiased')
    plt.ylabel("MSE of Squared Calibration Error")
    plt.xlabel("Number of Samples")
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_name = save_prefix + "curve_" + str(num_bins)
    plt.ylim(bottom=0.0)
    plt.savefig(save_name)

    plt.clf()
    plt.ylabel("Number of estimates")
    plt.xlabel("Absolute deviation from ground truth")
    bins = np.linspace(np.min(errors[:, 0, :]), np.max(errors[:, 0, :]), 40)
    plt.hist(errors[0][0], bins, alpha=0.5, label='plugin')
    plt.hist(errors[1][0], bins, alpha=0.5, label='debiased')
    plt.legend(loc='upper right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=num_resamples))
    plt.tight_layout()
    save_name = save_prefix + "histogram_" + str(num_bins)
    plt.savefig(save_name)


if __name__ == "__main__":
    args = parser.parse_args()
    logits, labels = utils.load_test_logits_labels(args.logits_file)
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/debiased_estimator/'):
        os.mkdir('./saved_files/debiased_estimator/')
    save_prefix = './saved_files/debiased_estimator/'
    compare_estimators(logits, labels, args.platt_data_size, args.bin_data_size, args.num_bins,
                       save_prefix=save_prefix)
