"""Mneasuring calibration and calibrating a model for multiclass classification."""

import numpy as np
import calibration
import scipy


def main():
    # Make synthetic dataset.
    np.random.seed(0)  # Keep results consistent.
    num_points = 10000
    d = 10
    (zs, ys) = synthetic_data(num_points=num_points, d=d)

    # Estimate a lower bound on the calibration error.
    # Here z_i are the per-class confidences of the uncalibrated model, y_i is the true label.
    calibration_error = calibration.get_calibration_error(zs, ys)
    print("Uncalibrated model calibration error is > %.2f%%" % (100 * calibration_error))

    # Use Platt binning to train a recalibrator.
    calibrator = calibration.PlattBinnerMarginalCalibrator(num_points, num_bins=10)
    calibrator.train_calibration(zs, ys)

    # Measure the calibration error of recalibrated model.
    (test_zs, test_ys) = synthetic_data(num_points=num_points, d=d)
    calibrated_zs = calibrator.calibrate(test_zs)
    calibration_error = calibration.get_calibration_error(calibrated_zs, test_ys)
    print("Scaling-binning L2 calibration error is %.2f%%" % (100 * calibration_error))

    # Get confidence intervals for the calibration error.
    [lower, _, upper] = calibration.get_calibration_error_uncertainties(calibrated_zs, test_ys)
    print("  Confidence interval is [%.2f%%, %.2f%%]" % (100 * lower, 100 * upper))


def synthetic_data(num_points, d):
    true_probs = np.random.dirichlet([1] * d, size=num_points)
    samples = vectorized_sample(true_probs, np.array(list(range(d))))
    model_probs = sharpen(true_probs, T=5)
    return (model_probs, samples)

def vectorized_sample(probs, items):
    s = probs.cumsum(axis=1)
    r = np.random.rand(probs.shape[0])
    r = np.expand_dims(r, axis=-1)
    r = np.tile(r, (1, probs.shape[1]))
    k = (s < r).sum(axis=1)
    return items[k]

def sharpen(probs, T):
    probs = np.log(np.clip(probs, 1e-6, 1-1e-6))
    probs = probs * T
    return scipy.special.softmax(probs, axis=1)

if __name__ == "__main__":
    main()