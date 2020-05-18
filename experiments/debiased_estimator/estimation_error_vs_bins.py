
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import calibration as cal


np.random.seed(0)  # Keep results consistent.

parser = argparse.ArgumentParser()
parser.add_argument('--probs_file', default='data/cifar_probs.dat', type=str,
                    help='Name of file to load probs, labels pair.')
parser.add_argument('--platt_data_size', default=2000, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--bin_data_size', default=2000, type=int,
                    help='Number of examples to use for binning.')
parser.add_argument('--num_bins', default=100, type=int,
                    help='Bins to test estimators with.')


def compare_scaling_binning_squared_ce(
    probs, labels, platt_data_size, bin_data_size, num_bins, ver_base_size=2000,
    ver_size_increment=1000, max_ver_size=7000, num_resamples=1000,
    save_prefix='./saved_files/debiased_estimator/', lp=2,
    Calibrator=cal.PlattBinnerTopCalibrator):
    calibrator = Calibrator(num_calibration=platt_data_size, num_bins=num_bins)
    calibrator.train_calibration(probs[:platt_data_size], labels[:platt_data_size])
    predictions = cal.get_top_predictions(probs)
    correct = (predictions == labels).astype(np.int32)
    verification_correct = correct[bin_data_size:]
    verification_probs = calibrator.calibrate(probs[bin_data_size:])
    verification_sizes = list(range(ver_base_size, 1 + min(max_ver_size, len(verification_probs)),
                              ver_size_increment))
    estimators = [lambda p, l: cal.get_calibration_error(p, l, p=lp, debias=False) ** lp,
                lambda p, l: cal.get_calibration_error(p, l, p=lp, debias=True) ** lp]
    estimates = get_estimates(
        estimators, verification_probs, verification_correct, verification_sizes,
        num_resamples)
    true_calibration = cal.get_calibration_error(verification_probs, verification_correct, p=lp, debias=False) ** lp
    print(true_calibration)
    print(np.sqrt(np.mean(estimates[1, -1, :])))
    errors = np.abs(estimates - true_calibration)
    plot_mse_curve(errors, verification_sizes, num_resamples, save_prefix, num_bins)
    plot_histograms(errors, num_resamples, save_prefix, num_bins)


def compare_scaling_ce(
    probs, labels, platt_data_size, bin_data_size, num_bins, ver_base_size=2000,
    ver_size_increment=1000, max_ver_size=7000, num_resamples=1000,
    save_prefix='./saved_files/debiased_estimator/', lp=1, Calibrator=cal.PlattTopCalibrator):
    calibrator = Calibrator(num_calibration=platt_data_size, num_bins=num_bins)
    calibrator.train_calibration(probs[:platt_data_size], labels[:platt_data_size])
    predictions = cal.get_top_predictions(probs)
    correct = (predictions == labels).astype(np.int32)
    verification_correct = correct[bin_data_size:]
    verification_probs = calibrator.calibrate(probs[bin_data_size:])
    verification_sizes = list(range(ver_base_size, 1 + min(max_ver_size, len(verification_probs)),
                              ver_size_increment))
    binning_probs = calibrator.calibrate(probs[:bin_data_size])
    bins = cal.get_equal_bins(binning_probs, num_bins=num_bins)
    def plugin_estimator(p, l):
        data = list(zip(p, l))
        binned_data = cal.bin(data, bins)
        return cal.plugin_ce(binned_data, power=lp)
    def debiased_estimator(p, l):
        data = list(zip(p, l))
        binned_data = cal.bin(data, bins)
        if lp == 2:
            return cal.unbiased_l2_ce(binned_data)
        else:
            return cal.normal_debiased_ce(binned_data, power=lp)
    estimators = [plugin_estimator, debiased_estimator]
    estimates = get_estimates(
        estimators, verification_probs, verification_correct, verification_sizes,
        num_resamples)
    true_calibration = plugin_estimator(verification_probs, verification_correct)
    print(true_calibration)
    print(np.sqrt(np.mean(estimates[1, -1, :])))
    errors = np.abs(estimates - true_calibration)
    plot_mse_curve(errors, verification_sizes, num_resamples, save_prefix, num_bins)
    plot_histograms(errors, num_resamples, save_prefix, num_bins)


def get_estimates(estimators, verification_probs, verification_labels, verification_sizes,
                  num_resamples=1000):
    # We want to compare the two estimators when varying the number of samples.
    # However, a single point of comparison does not tell us much about the estimators.
    # So we use resampling - we resample from the test set many times, and run the estimators
    # on the resamples. We stores these values. This gives us a sense of the range of values
    # the estimator might output.
    # So estimates[i][j][k] stores the estimate when using estimator i, with verification_sizes[j]
    # samples, in the k-th resampling.
    estimates = np.zeros((len(estimators), len(verification_sizes), num_resamples))
    for ver_idx, verification_size in zip(range(len(verification_sizes)), verification_sizes):
        for k in range(num_resamples):
            # Resample
            indices = np.random.choice(list(range(len(verification_probs))),
                                       size=verification_size, replace=True)
            cur_verification_probs = [verification_probs[i] for i in indices]
            cur_verification_correct = [verification_labels[i] for i in indices]
            for i in range(len(estimators)):
                estimates[i][ver_idx][k] = estimators[i](cur_verification_probs,
                                                         cur_verification_correct)

    estimates = np.sort(estimates, axis=-1)
    return estimates


def plot_mse_curve(errors, verification_sizes, num_resamples, save_prefix, num_bins):
    plt.clf()
    errors = np.square(errors)
    accumulated_errors = np.mean(errors, axis=-1)
    error_bars_90 = 1.645 * np.std(errors, axis=-1) / np.sqrt(num_resamples)
    print(accumulated_errors)
    plt.errorbar(
        verification_sizes, accumulated_errors[0], yerr=[error_bars_90[0], error_bars_90[0]],
        barsabove=True, color='red', capsize=4, label='plugin')
    plt.errorbar(
        verification_sizes, accumulated_errors[1], yerr=[error_bars_90[1], error_bars_90[1]],
        barsabove=True, color='blue', capsize=4, label='debiased')
    plt.ylabel("MSE of Calibration Error")
    plt.xlabel("Number of Samples")
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_name = save_prefix + "curve_" + str(num_bins)
    plt.ylim(bottom=0.0)
    plt.savefig(save_name)


def plot_histograms(errors, num_resamples, save_prefix, num_bins):
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


def cifar_experiments():
    probs, labels = cal.load_test_probs_labels('data/cifar_probs.dat')
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/debiased_estimator/'):
        os.mkdir('./saved_files/debiased_estimator/')
    save_prefix = './saved_files/debiased_estimator/'
    compare_scaling_binning_squared_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=100,
        save_prefix=save_prefix+"cifar_scaling_binning_")
    compare_scaling_binning_squared_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=15,
        save_prefix=save_prefix+"cifar_scaling_binning_")
    compare_scaling_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=100,
        save_prefix=save_prefix+"cifar_scaling_ece_")
    compare_scaling_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=15,
        save_prefix=save_prefix+"cifar_scaling_ece_")


def imagenet_experiments():
    probs, labels = cal.load_test_probs_labels('data/imagenet_probs.dat')
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/debiased_estimator/'):
        os.mkdir('./saved_files/debiased_estimator/')
    save_prefix = './saved_files/debiased_estimator/'
    compare_scaling_binning_squared_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=100,
        save_prefix=save_prefix+"imnet_scaling_binning_")
    compare_scaling_binning_squared_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=15,
        save_prefix=save_prefix+"imnet_scaling_binning_")
    compare_scaling_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=100,
        save_prefix=save_prefix+"imnet_scaling_ece_")
    compare_scaling_ce(
        probs, labels, platt_data_size=3000, bin_data_size=3000, num_bins=15,
        save_prefix=save_prefix+"imnet_scaling_ece_")


def parse_input():
    args = parser.parse_args()
    probs, labels = cal.load_test_probs_labels(args.probs_file)
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/debiased_estimator/'):
        os.mkdir('./saved_files/debiased_estimator/')
    save_prefix = './saved_files/debiased_estimator/'
    compare_scaling_binning_squared_ce(
        probs, labels, args.platt_data_size, args.bin_data_size, args.num_bins,
        save_prefix=save_prefix)


if __name__ == "__main__":
    cifar_experiments()
    imagenet_experiments()
