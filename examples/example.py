
import lib.utils as utils
import numpy as np
from lib.calibrators import PlattBinnerCalibrator


def main():
    # Make synthetic dataset.
    np.random.seed(0)  # Keep results consistent.
    num_points = 1000
    (zs, ys) = synthetic_data_1d(num_points=num_points)

    # Estimate a lower bound on the calibration error.
    # Here z_i is the confidence of the uncalibrated model, y_i is the true label.
    data = list(zip(zs, ys))
    bins = utils.get_equal_bins(zs, num_bins=10)
    l2_calibration_error = utils.unbiased_l2_ce(utils.bin(data, bins))
    print("Uncalibrated model L2 calibration error is %.2f%%" % (100 * l2_calibration_error))

    # Use Platt binning to train a recalibrator.
    calibrator = PlattBinnerCalibrator(num_points, num_bins=10)
    calibrator.train_calibration(np.array(zs), ys)

    # # Measure the calibration error of recalibrated model.
    (test_zs, test_ys) = synthetic_data_1d(num_points=num_points)
    calibrated_zs = list(calibrator.calibrate(test_zs))
    data = list(zip(calibrated_zs, test_ys))
    bins = utils.get_discrete_bins(test_zs)
    l2_calibration_error = utils.unbiased_l2_ce(utils.bin(data, bins))
    print("Scaling-binning L2 calibration error is %.2f%%" % (100 * l2_calibration_error))

    # Compute calibration error and confidence interval.
    def estimate_ce(data, estimator):
        zs = [z for z, y in data]
        binned_data = utils.bin(data, utils.get_discrete_bins(zs))
        return estimator(binned_data)
    functional = lambda data: estimate_ce(data, lambda x: utils.plugin_ce(x))
    [lower, _, upper] = utils.bootstrap_uncertainty(data, functional, num_samples=100)
    print("  Confidence interval is [%.2f%%, %.2f%%]" % (100 * lower, 100 * upper))

    # Advanced: boostrap can be used to debias the l1-calibration error as well.
    # Unlike for L2-CE this is a heuristic, which does not (yet) come with a formal guarantee.
    functional = lambda data: estimate_ce(data, lambda x: utils.plugin_ce(x, power=1))
    [lower, mid, upper] = utils.bootstrap_uncertainty(data, functional, num_samples=100)
    print("Debiased estimate of L1 calibration error is %.2f%%" % (100 * mid))
    print("  Confidence interval is [%.2f%%, %.2f%%]" % (100 * lower, 100 * upper))


# Helper functions used to generate synthetic data.

def synthetic_data_1d(num_points):
    f_true = platt_function(1, 1)
    return sample(f_true, np.random.uniform, num_points)

def platt_function(a, b):
    """Return a (vectorized) platt function f: [0, 1] -> [0, 1] parameterized by a, b."""
    def eval(x):
        x = np.log(x / (1 - x))
        x = a * x + b
        return 1 / (1 + np.exp(-x))
    return np.vectorize(eval)

def sample(f, z_dist, n):
    """Returns ([z_1, ..., z_n], [y_1, ..., y_n]) where z_i ~ z_dist, y_i ~ Bernoulli(f(z_i))."""
    zs = list(z_dist(size=n))
    ps = f(zs)
    ys = list(np.random.binomial(1, p=ps))
    return (zs, ys)

if __name__ == "__main__":
    main()
