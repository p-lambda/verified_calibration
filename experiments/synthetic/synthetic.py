
import argparse
import numpy as np
import calibration as cal
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

# Keep the results consistent.
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='vary_n_a1_b0', type=str,
                    help='Name of the experiment to run.')


def platt_function(a, b):
    """Return a (vectorized) platt function f: [0, 1] -> [0, 1] parameterized by a, b."""
    def eval(x):
        x = np.log(x / (1 - x))
        x = a * x + b
        return 1 / (1 + np.exp(-x))
    return np.vectorize(eval)


def noisy_platt_function(a, b, eps, l, u, num_bins=100000):
    """Return a (vectorized) noisy platt function f: [l, u] -> [0, 1] parameterized by a, b.

    Let g denote the Platt function parameterized by a, b. f accepts input x, where l <= x <= u.
    The range [l, u] is split into num_bins equally spaced intervals. If x is in interval j,
    f(x) = g(x) + noise[j], where noise[j] is sampled to be {-eps, +eps} with equal probability.
    f is vectorized, that is, it can also accept a numpy array as argument, and will apply f
    to each element in the array."""
    def platt(x):
        x = np.log(x / (1 - x))
        x = a * x + b
        return 1 / (1 + np.exp(-x))
    assert(1 - eps >= platt(l) >= eps)
    assert(1 - eps >= platt(u) >= eps)
    noise = (np.random.binomial(1, np.ones(num_bins + 1) * 0.5) * 2 - 1) * eps
    def eval(x):
        assert l <= x <= u
        b = np.floor((x - l) / (u - l) * num_bins).astype(np.int32)
        assert(np.all(b <= num_bins))
        b -= (b == num_bins)
        return platt(x) + noise[b]
    return np.vectorize(eval)


def sample(f, z_dist, n):
    """Returns ([z_1, ..., z_n], [y_1, ..., y_n]) where z_i ~ z_dist, y_i ~ Bernoulli(f(z_i))."""
    zs = z_dist(size=n)
    ps = f(zs)
    ys = np.random.binomial(1, p=ps)
    return (zs, ys)


def evaluate_l2ce(f, calibrator, z_dist, n):
    """Returns the calibration error of the calibrator on z_dist, f using n samples."""
    zs = z_dist(size=n)
    ps = f(zs)
    phats = calibrator.calibrate(zs)
    bins = cal.get_discrete_bins(phats)
    data = list(zip(phats, ps))
    binned_data = cal.bin(data, bins)
    return cal.plugin_ce(binned_data) ** 2


def evaluate_mse(f, calibrator, z_dist, n):
    """Returns the MSE of the calibrator on z_dist, f using n samples."""
    zs = z_dist(size=n)
    ps = f(zs)
    phats = calibrator.calibrate(zs)
    return np.mean(np.square(ps - phats))


def get_errors(f, Calibrators, z_dist, nb_args, num_trials, num_evaluation,
               evaluate_error=evaluate_l2ce):
    """Get the errors (+std-devs) of calibrators for each (n, b) in nb_args."""
    means = np.zeros((len(Calibrators), len(nb_args)))
    std_devs = np.zeros((len(Calibrators), len(nb_args)))
    for i, Calibrator in zip(range(len(Calibrators)), Calibrators):
        for j, (num_calibration, num_bins) in zip(range(len(nb_args)), nb_args):
            current_errors = []
            for k in range(num_trials):
                zs, ys = sample(f, z_dist=z_dist, n=num_calibration)
                calibrator = Calibrator(num_calibration=num_calibration, num_bins=num_bins)
                calibrator.train_calibration(zs, ys)
                error = evaluate_error(f, calibrator, z_dist, n=num_evaluation)
                assert(error >= 0.0)
                current_errors.append(error)
            means[i][j] = np.mean(current_errors)
            std_devs[i][j] = np.std(current_errors) / np.sqrt(num_trials)
    return means, std_devs


def sweep_n_platt(a, b, save_file, base_n=500, max_n_multiplier=9, bins=10,
                  num_trials=1000, num_evaluation=1000):
    f = platt_function(a, b)
    Calibrators = [cal.PlattCalibrator,
                   cal.HistogramCalibrator,
                   cal.PlattBinnerCalibrator]
    names = ['scaling', 'binning', 'scaling-binning']
    dist = np.random.uniform
    nb_args = [(base_n * i, bins) for i in range(1, max_n_multiplier)]
    means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
    pickle.dump((names, nb_args, means, stddevs), open(save_file, "wb"))


def sweep_b_platt(a, b, save_file, n=2000, base_bins=10, max_bin_multiplier=9,
                  num_trials=1000, num_evaluation=1000):
    f = platt_function(a, b)
    Calibrators = [cal.PlattCalibrator,
                   cal.HistogramCalibrator,
                   cal.PlattBinnerCalibrator]
    names = ['scaling', 'binning', 'scaling-binning']
    dist = np.random.uniform
    nb_args = [(n, base_bins * i) for i in range(1, max_bin_multiplier)]
    means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
    pickle.dump((names, nb_args, means, stddevs), open(save_file, "wb"))


def sweep_n_noisy_platt(a, b, save_file, base_n=500, max_n_multiplier=9, bins=10,
                        num_trials=1000, num_evaluation=100):
    l, u = 0.25, 0.75  # Probably needs to change.
    f = noisy_platt_function(a, b, 0.02, 0.25, 0.75)  # Probably needs to change.
    Calibrators = [cal.PlattCalibrator,
                   cal.HistogramCalibrator,
                   cal.PlattBinnerCalibrator]
    names = ['scaling', 'binning', 'scaling-binning']
    def dist(size):
        return np.random.uniform(low=l, high=u, size=size)
    nb_args = [(base_n * i, bins) for i in range(1, max_n_multiplier)]
    means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
    pickle.dump((names, nb_args, means, stddevs), open(save_file, "wb"))


# Plots 1/eps^2 against n.
def plot_sweep_n(load_file, save_file):
    # TODO: add legends.
    (names, nb_args, means, stddevs) = pickle.load(open(load_file, "rb"))
    error_bars_90 = (1.645 / (np.square(means))) * stddevs
    plt.clf()
    def plot_inv_err_n(method_means, method_error_bars_90, color, calibrator_name):
        plt.errorbar([n for (n, b) in nb_args], 1 / method_means, color=color,
                     yerr=[method_error_bars_90, method_error_bars_90],
                     barsabove=True, capsize=4, label=calibrator_name)
        plt.ylabel("1 / epsilon^2")
        plt.xlabel("n (number of calibration points)")
        plt.tight_layout()
        plt.savefig(save_file+'_'+calibrator_name)
    colors = ['red', 'green', 'blue']
    for i in range(len(names)):
        plt.clf()
        plot_inv_err_n(means[i], error_bars_90[i], color=colors[i],
                       calibrator_name=names[i])
    print('calibrators', names)
    print('nb_args', nb_args)
    print('means', means)
    print('stddevs', stddevs)
    # # TODO: include method names to avoid confusion.
    # for i in range(means.shape[0]):
    #     factor_1_3 = divide(means[i][1], means[i][3], stddevs[i][1], stddevs[i][3])
    #     print('Method', i, 'going from', nb_args[1], nb_args[3], 'error goes', factor_1_3)


def plot_noisy_eps_n(load_file, save_file, skip_binning=True):
    (names, nb_args, means, stddevs) = pickle.load(open(load_file, "rb"))
    error_bars_90 = 1.645 * stddevs
    plt.clf()
    def plot_err(method_means, method_error_bars_90, color, calibrator_name):
        plt.errorbar([n for (n, b) in nb_args], method_means, color=color,
             yerr=[method_error_bars_90, method_error_bars_90],
             barsabove=True, capsize=4, label=calibrator_name)
        plt.ylabel("epsilon^2")
        plt.xlabel("n (number of calibration points)")
    colors = ['red', 'green', 'blue']
    for i in range(len(names)):
        if names[i] == 'binning' and skip_binning:
            pass
        else:
            plot_err(means[i], error_bars_90[i], color=colors[i],
                     calibrator_name=names[i])
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(save_file)
    print('calibrators', names)
    print('nb_args', nb_args)
    print('means', means)
    print('stddevs', stddevs)


def plot_sweep_b(load_file, save_file):
    # TODO: Add legends.
    (names, nb_args, means, stddevs) = pickle.load(open(load_file, "rb"))
    error_bars_90 = (1.645 / (np.square(means))) * stddevs
    def plot_inv_err_b(method_means, method_error_bars_90, color, calibrator_name):
        plt.errorbar([b for (n, b) in nb_args], 1 / method_means, color=color,
                     yerr=[method_error_bars_90, method_error_bars_90],
                     barsabove=True, capsize=4, label=calibrator_name)
        plt.ylabel("1 / epsilon^2")
        plt.xlabel("b (number of bins)")
        plt.tight_layout()
        plt.savefig(save_file+'_'+calibrator_name)
    colors = ['red', 'green', 'blue']
    for i in range(len(names)):
        plt.clf()
        plot_inv_err_b(means[i], error_bars_90[i], color=colors[i],
                       calibrator_name=names[i])
    print('calibrators', names)
    print('nb_args', nb_args)
    print('means', means)
    print('stddevs', stddevs)


def divide(mu1, mu2, sigma1, sigma2):
    # Use delta method to compute confidence intervals for division.
    mu = mu1 / mu2
    sigma = np.sqrt(1.0 / (mu2 ** 2) * (sigma1 ** 2) + (mu1 ** 2) / (mu2 ** 4) * (sigma2 ** 2))
    return mu, sigma


def plot_curve(f, save_file, l=1e-8, u=1.0-1e-8):
    xs = np.arange(l, u, 1 / 1000.0)
    ys = f(xs)
    plt.clf()
    plt.plot(xs, ys)
    plt.ylabel("P(Y = 1 | z)")
    plt.xlabel("z")
    plt.tight_layout()
    parent = Path(save_file).parent
    plt.savefig(save_file)


if __name__ == "__main__":
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/synthetic/'):
        os.mkdir('./saved_files/synthetic/')
    prefix = './saved_files/synthetic/'
    args = parser.parse_args()

    if args.experiment_name == 'vary_n_a1_b0':
        f = platt_function(1, 0)
        plot_curve(f, prefix+'curve_vary_n_a1_b0')
        sweep_n_platt(1, 0, prefix+'vary_n_a1_b0')
        plot_sweep_n(load_file=prefix+'vary_n_a1_b0',
                     save_file=prefix+'vary_n_a1_b0')
    elif args.experiment_name == 'vary_b_a1_b0':
        f = platt_function(1, 0)
        plot_curve(f, prefix+'curve_vary_b_a1_b0')
        sweep_b_platt(1, 0, prefix+'vary_b_a1_b0')
        plot_sweep_b(load_file=prefix+'vary_b_a1_b0',
                     save_file=prefix+'vary_b_a1_b0')
    elif args.experiment_name == 'vary_n_a2_b1':
        f = platt_function(2, 1)
        plot_curve(f, prefix+'curve_vary_n_a2_b1')
        sweep_n_platt(2, 1, prefix+'vary_n_a2_b1')
        plot_sweep_n(load_file=prefix+'vary_n_a2_b1',
                     save_file=prefix+'vary_n_a2_b1')
    elif args.experiment_name == 'vary_b_a2_b1':
        f = platt_function(2, 1)
        plot_curve(f, prefix+'curve_vary_b_a2_b1')
        sweep_b_platt(2, 1, prefix+'vary_b_a2_b1')
        plot_sweep_b(load_file=prefix+'vary_b_a2_b1',
                     save_file=prefix+'vary_b_a2_b1')
    elif args.experiment_name == 'noisy_vary_n_a2_b1':
        f = noisy_platt_function(2, 1, eps=0.02, l=0.25, u=0.75)
        plot_curve(f, prefix+'noisy_curve_vary_n_a2_b1', l=0.25, u=0.75)
        sweep_n_noisy_platt(2, 1, prefix+'noisy_vary_n_a2_b1')
        plot_noisy_eps_n(load_file=prefix+'noisy_vary_n_a2_b1',
                         save_file=prefix+'noisy_error_plot_vary_n_a2_b1')
