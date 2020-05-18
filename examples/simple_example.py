"""Mneasuring calibration and calibrating a model for binary classification."""

import numpy as np
import calibration


def main():
    # Make synthetic dataset.
    np.random.seed(0)  # Keep results consistent.
    num_points = 1000
    (zs, ys) = synthetic_data_1d(num_points=num_points)

    # Estimate a lower bound on the calibration error.
    # Here z_i is the confidence of the uncalibrated model, y_i is the true label.
    calibration_error = calibration.get_calibration_error(zs, ys)
    print("Uncalibrated model calibration error is > %.2f%%" % (100 * calibration_error))

    # Estimate the ECE.
    ece = calibration.get_ece(zs, ys)
    print("Uncalibrated model ECE is > %.2f%%" % (100 * ece))

    # Use Platt binning to train a recalibrator.
    calibrator = calibration.PlattBinnerCalibrator(num_points, num_bins=10)
    calibrator.train_calibration(np.array(zs), ys)

    # Measure the calibration error of recalibrated model.
    (test_zs, test_ys) = synthetic_data_1d(num_points=num_points)
    calibrated_zs = calibrator.calibrate(test_zs)
    calibration_error = calibration.get_calibration_error(calibrated_zs, test_ys)
    print("Scaling-binning L2 calibration error is %.2f%%" % (100 * calibration_error))

    # Get confidence intervals for the calibration error.
    [lower, _, upper] = calibration.get_calibration_error_uncertainties(calibrated_zs, test_ys)
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
