"""More complete example showing features of the calibration library.

For more advanced users our calibration library is fairly customizable.

We offer a variety of different ways to estimate the calibration
error (e.g. plugin vs debiased), and can measure ECE, or the standard l2 error.

The user can also choose a different binning scheme, or write their own binning
scheme. For example, in the literature some people use equal probability binning,
splitting the interval [0, 1] into B equal parts. Others split the interval [0, 1]
into B bin so that each bin has an equal number of data points. Alternative
binning schemes are also possible.
"""


import calibration
import numpy as np


def main():
    # Make synthetic dataset.
    np.random.seed(0)  # Keep results consistent.
    num_points = 1000
    (zs, ys) = synthetic_data_1d(num_points=num_points)

    # Estimate a lower bound on the calibration error.
    # Here z_i is the confidence of the uncalibrated model, y_i is the true label.
    # In simple_example.py we used get_calibration_error, but for advanced users
    # we recommend using the more explicit lower_bound_scaling_ce to have
    # more control over functionality, and be explicit about the semantics -
    # that we are only estimating a lower bound.
    l2_calibration_error = calibration.lower_bound_scaling_ce(zs, ys)
    print("Uncalibrated model l2 calibration error is > %.2f%%" % (100 * l2_calibration_error))

    # We can break this down into multiple steps. 1. We choose a binning scheme,
    # 2. we bin the data, and 3. we measure the calibration error.
    # Each of these steps can be customized, and users can substitute the component
    # with their own code.
    data = list(zip(zs, ys))
    bins = calibration.get_equal_bins(zs, num_bins=10)
    l2_calibration_error = calibration.unbiased_l2_ce(calibration.bin(data, bins))
    print("Uncalibrated model l2 calibration error is > %.2f%%" % (100 * l2_calibration_error))

    # Use Platt binning to train a recalibrator.
    calibrator = calibration.PlattBinnerCalibrator(num_points, num_bins=10)
    calibrator.train_calibration(np.array(zs), ys)

    # Measure the calibration error of recalibrated model.
    # In this case we have a binning model, so we can estimate the true calibration error.
    # Again, for advanced users we recommend being explicit and using get_binning_ce instead
    # of get_calibration_error.
    (test_zs, test_ys) = synthetic_data_1d(num_points=num_points)
    calibrated_zs = list(calibrator.calibrate(test_zs))
    l2_calibration_error = calibration.get_binning_ce(calibrated_zs, test_ys)
    print("Scaling-binning l2 calibration error is %.2f%%" % (100 * l2_calibration_error))

    # As above we can break this down into 3 steps. Notice here we have a binning model,
    # so we use get_discrete_bins to get all the bins (all possible values the model
    # outputs).
    data = list(zip(calibrated_zs, test_ys))
    bins = calibration.get_discrete_bins(calibrated_zs)
    binned = calibration.bin(data, bins)
    l2_calibration_error = calibration.unbiased_l2_ce(calibration.bin(data, bins))
    print("Scaling-binning l2 calibration error is %.2f%%" % (100 * l2_calibration_error))

    # Compute calibration error and confidence interval.
    # In the simple_example.py we just called get_calibration_error_uncertainties.
    # This function uses the bootstrap to estimate confidence intervals.
    # The bootstrap first requires us to define the functional we are trying to
    # estimate, and then resamples the data multiple times to estimate confidence intervals.
    def estimate_ce(data, estimator):
        zs = [z for z, y in data]
        binned_data = calibration.bin(data, calibration.get_discrete_bins(zs))
        return estimator(binned_data)
    functional = lambda data: estimate_ce(data, lambda x: calibration.plugin_ce(x))
    [lower, _, upper] = calibration.bootstrap_uncertainty(data, functional, num_samples=100)
    print("  Confidence interval is [%.2f%%, %.2f%%]" % (100 * lower, 100 * upper))

    # Advanced: boostrap can be used to debias the l1-calibration error (ECE) as well.
    # This is a heuristic, which does not (yet) come with a formal guarantee.
    functional = lambda data: estimate_ce(data, lambda x: calibration.plugin_ce(x, power=1))
    [lower, mid, upper] = calibration.bootstrap_uncertainty(data, functional, num_samples=100)
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
