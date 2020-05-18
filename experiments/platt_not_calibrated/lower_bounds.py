
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import calibration as cal

# Keep the results consistent.
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--probs_path', default='data/cifar_probs.dat', type=str,
                    help='Name of file to load probs, labels pair.')
parser.add_argument('--calibration_data_size', default=1000, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--bin_data_size', default=1000, type=int,
                    help='Number of examples to use for binning.')
parser.add_argument('--plot_save_file', default='lower_bound_plot.png', type=str,
                    help='File to save lower bound plot.')
parser.add_argument('--binning', default='equal_bins', type=str,
                    help='The binning strategy to use.')
parser.add_argument('--lp', default=2, type=int,
                    help='Use the lp-calibration error.')
parser.add_argument('--bins_list', default="2, 4, 8, 16, 32, 64, 128",
                    type=lambda s: [int(t) for t in s.split(',')],
                    help='Bin sizes to evaluate calibration error at.')
parser.add_argument('--num_samples', default=1000, type=int,
                    help='Number of resamples for bootstrap confidence intervals.')


def lower_bound_experiment(probs, labels, calibration_data_size, bin_data_size, bins_list,
                           save_name='cmp_est', binning_func=cal.get_equal_bins, lp=2,
                           num_samples=1000):
    # Shuffle the probs and labels.
    np.random.seed(0)  # Keep results consistent.
    indices = np.random.choice(list(range(len(probs))), size=len(probs), replace=False)
    probs = [probs[i] for i in indices]
    labels = [labels[i] for i in indices]
    predictions = cal.get_top_predictions(probs)
    probs = cal.get_top_probs(probs)
    correct = (predictions == labels)
    print('num_correct: ', sum(correct))
    # Platt scale on first chunk of data
    platt = cal.get_platt_scaler(probs[:calibration_data_size], correct[:calibration_data_size])
    platt_probs = platt(probs)
    lower, middle, upper = [], [], []
    for num_bins in bins_list:
        bins = binning_func(
            platt_probs[:calibration_data_size+bin_data_size], num_bins=num_bins)
        verification_probs = platt_probs[calibration_data_size+bin_data_size:]
        verification_correct = correct[calibration_data_size+bin_data_size:]
        verification_data = list(zip(verification_probs, verification_correct))
        def estimator(data):
            binned_data = cal.bin(data, bins)
            return cal.plugin_ce(binned_data, power=lp)
        print('estimate: ', estimator(verification_data))
        estimate_interval = cal.bootstrap_uncertainty(
            verification_data, estimator, num_samples=1000)
        lower.append(estimate_interval[0])
        middle.append(estimate_interval[1])
        upper.append(estimate_interval[2])
        print('interval: ', estimate_interval)
    # Plot the results.
    lower_errors = np.array(middle) - np.array(lower)
    upper_errors = np.array(upper) - np.array(middle)
    plt.clf()
    font = {'family' : 'normal', 'size': 18}
    rc('font', **font)
    plt.errorbar(
        bins_list, middle, yerr=[lower_errors, upper_errors],
        barsabove=True, fmt = 'none', color='black', capsize=4)
    plt.scatter(bins_list, middle, color='black')
    plt.xlabel(r"No. of bins")
    if lp == 2:
        plt.ylabel("Calibration error")
    else:
        plt.ylabel("l%d Calibration error" % lp)
    plt.xscale('log', basex=2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(save_name)


def cifar_experiment(savefile, binning_func=cal.get_equal_bins, lp=2):
    np.random.seed(0)
    calibration_data_size = 1000
    bin_data_size = 1000
    probs, labels = cal.load_test_probs_labels('cifar_probs.dat')
    lower_bound_experiment(probs, labels, calibration_data_size, bin_data_size,
                           bins_list=[2, 4, 8, 16, 32, 64, 128], save_name=savefile,
                           binning_func=binning_func, lp=lp)


def imagenet_experiment(savefile, binning_func=cal.get_equal_bins, lp=2):
    np.random.seed(0)
    calibration_data_size = 20000
    bin_data_size = 5000
    probs, labels = cal.load_test_probs_labels('imagenet_probs.dat')
    lower_bound_experiment(probs, labels, calibration_data_size, bin_data_size,
                           bins_list=[2, 4, 8, 16, 32, 64, 128, 256, 512], save_name=savefile,
                           binning_func=binning_func, lp=lp)


if __name__ == "__main__":
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/platt_not_calibrated/'):
        os.mkdir('./saved_files/platt_not_calibrated/')
    prefix = './saved_files/platt_not_calibrated/'
    args = parser.parse_args()

    if args.binning == 'equal_prob_bins':
        binning = cal.get_equal_prob_bins
    else:
        binning = cal.get_equal_bins

    probs, labels = cal.load_test_probs_labels(args.probs_path)
    print(args.bins_list)
    lower_bound_experiment(
        probs, labels, args.calibration_data_size, args.bin_data_size, args.bins_list,
        save_name=prefix+args.plot_save_file, binning_func=binning, lp=args.lp,
        num_samples=args.num_samples)
