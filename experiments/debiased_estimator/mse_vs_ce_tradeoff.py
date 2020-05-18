import numpy as np
import pickle
import bisect
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import lib.utils as utils


np.random.seed(0)  # Keep results consistent.


def calibrate_marginals_experiment(probs, labels, k):
    num_calib = 3000
    num_bin = 3000
    num_cert = 4000
    assert(probs.shape[0] == num_calib + num_bin + num_cert)
    num_bins = 100
    bootstrap_samples = 100
    # First split by label? To ensure equal class numbers? Do this later.
    labels = utils.get_labels_one_hot(labels[:], k)
    mse = np.mean(np.square(labels - probs))
    print('original mse is ', mse)
    calib_probs = probs[:num_calib, :]
    calib_labels = labels[:num_calib, :]
    bin_probs = probs[num_calib:num_calib + num_bin, :]
    bin_labels = labels[num_calib:num_calib + num_bin, :]
    cert_probs = probs[num_calib + num_bin:, :]
    cert_labels = labels[num_calib + num_bin:, :]
    mses = []
    unbiased_ces = []
    biased_ces = []
    std_unbiased_ces = []
    std_biased_ces = []
    for num_bins in range(10, 101, 10):
        # Train a platt scaler and binner.
        platts = []
        platt_binners_equal_points = []
        for l in range(k):
            platt_l = utils.get_platt_scaler(calib_probs[:, l], calib_labels[:, l])
            platts.append(platt_l)
            cal_probs_l = platt_l(calib_probs[:, l])
            bins_l = utils.get_equal_bins(cal_probs_l, num_bins=num_bins)
            cal_bin_probs_l = platt_l(bin_probs[:, l])
            platt_binner_l = utils.get_discrete_calibrator(cal_bin_probs_l, bins_l)
            platt_binners_equal_points.append(platt_binner_l)

        # Write a function that takes data and outputs the mse, ce
        def get_mse_ce(probs, labels, ce_est):
            mses = []
            ces = []
            probs = np.array(probs)
            labels = np.array(labels)
            for l in range(k):
                cal_probs_l = platt_binners_equal_points[l](platts[l](probs[:, l]))
                data = list(zip(cal_probs_l, labels[:, l]))
                bins_l = utils.get_discrete_bins(cal_probs_l)
                binned_data = utils.bin(data, bins_l)
                # probs = platts[l](probs[:, l])
                # for p in [1, 5, 10, 20, 50, 85, 88.5, 92, 94, 96, 98, 100]:
                #     print(np.percentile(probs, p))
                # import time
                # time.sleep(100)
                # print('lengths')
                # print([len(d) for d in binned_data])
                ces.append(ce_est(binned_data))
                mses.append(np.mean([(prob - label) ** 2 for prob, label in data]))
            return np.mean(mses), np.mean(ces)

        def plugin_ce_squared(data):
            probs, labels = zip(*data)
            return get_mse_ce(probs, labels, lambda x: utils.plugin_ce(x) ** 2)[1]
        def mse(data):
            probs, labels = zip(*data)
            return get_mse_ce(probs, labels, lambda x: utils.plugin_ce(x) ** 2)[0]
        def unbiased_ce_squared(data):
            probs, labels = zip(*data)
            return get_mse_ce(probs, labels, utils.unbiased_square_ce)[1]

        mse, unbiased_ce = get_mse_ce(
            cert_probs, cert_labels, utils.unbiased_square_ce)
        mse, biased_ce = get_mse_ce(
            cert_probs, cert_labels, lambda x: utils.plugin_ce(x) ** 2)
        mses.append(mse)
        unbiased_ces.append(unbiased_ce)
        biased_ces.append(biased_ce)
        print('biased ce: ', np.sqrt(biased_ce))
        print('mse: ', mse)
        print('improved ce: ', np.sqrt(unbiased_ce))
        data = list(zip(list(cert_probs), list(cert_labels)))
        std_biased_ces.append(
            utils.bootstrap_std(data, plugin_ce_squared, num_samples=bootstrap_samples))
        std_unbiased_ces.append(
            utils.bootstrap_std(data, unbiased_ce_squared, num_samples=bootstrap_samples))

    std_multiplier = 1.3  # For one sided 90% confidence interval.
    upper_unbiased_ces = list(map(lambda p: np.sqrt(p[0] + std_multiplier * p[1]),
                                  zip(unbiased_ces, std_unbiased_ces)))
    upper_biased_ces = list(map(lambda p: np.sqrt(p[0] + std_multiplier * p[1]),
                                zip(biased_ces, std_biased_ces)))
    # Get points on the Pareto curve, and plot them.
    def get_pareto_points(data):
        pareto_points = []
        def dominated(p1, p2):
            return p1[0] >= p2[0] and p1[1] >= p2[1]
        for datum in data:
            num_dominated = sum(map(lambda x: dominated(datum, x), data))
            if num_dominated == 1:
                pareto_points.append(datum)
        return pareto_points
    print(get_pareto_points(list(zip(upper_unbiased_ces, mses, list(range(5, 101, 5))))))
    print(get_pareto_points(list(zip(upper_biased_ces, mses, list(range(5, 101, 5))))))
    plot_unbiased_ces, plot_unbiased_mses = zip(*get_pareto_points(list(zip(upper_unbiased_ces, mses))))
    plot_biased_ces, plot_biased_mses = zip(*get_pareto_points(list(zip(upper_biased_ces, mses))))
    plt.title("MSE vs Calibration Error")
    plt.scatter(plot_unbiased_ces, plot_unbiased_mses, c='red', marker='o', label='Ours')
    plt.scatter(plot_biased_ces, plot_biased_mses, c='blue', marker='s', label='Plugin')
    plt.legend(loc='upper left')
    plt.ylim(0.0, 0.013)
    plt.xlabel("Squared Calibration Error")
    plt.ylabel("Mean-Squared Error")
    plt.tight_layout()
    save_name = "./saved_files/debiased_estimator/mse_vs_ce"
    plt.savefig(save_name)


if __name__ == "__main__":
    if not os.path.exists('./saved_files'):
        os.mkdir('./saved_files')
    if not os.path.exists('./saved_files/debiased_estimator/'):
        os.mkdir('./saved_files/debiased_estimator/')

    probs, labels = utils.load_test_probs_labels('data/cifar_probs.dat')
    predictions = np.argmax(probs, 1)
    probabilities = np.max(probs, 1)
    accuracy = np.mean(labels[:] == predictions)
    print('accuracy is ' + str(accuracy))
    calibrate_marginals_experiment(probs, labels, k=10)
