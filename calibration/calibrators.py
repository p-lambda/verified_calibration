
import numpy as np

from . import utils


class HistogramCalibrator:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        bins = utils.get_equal_bins(zs, num_bins=self._num_bins)
        self._calibrator = utils.get_histogram_calibrator(zs, ys, bins)

    def calibrate(self, zs):
        return self._calibrator(zs)


class PlattBinnerCalibrator:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        self._platt = utils.get_platt_scaler(zs, ys)
        platt_probs = self._platt(zs)
        bins = utils.get_equal_bins(platt_probs, num_bins=self._num_bins)
        self._discrete_calibrator = utils.get_discrete_calibrator(platt_probs, bins)

    def calibrate(self, zs):
        platt_probs = self._platt(zs)
        return self._discrete_calibrator(platt_probs)


class PlattCalibrator:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        self._platt = utils.get_platt_scaler(zs, ys)

    def calibrate(self, zs):
        return self._platt(zs)


class HistogramTopCalibrator:

    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, probs, labels):
        assert(len(probs) >= self._num_calibration)
        top_probs = utils.get_top_probs(probs)
        predictions = utils.get_top_predictions(probs)
        correct = (predictions == labels)
        bins = utils.get_equal_bins(top_probs, num_bins=self._num_bins)
        self._calibrator = utils.get_histogram_calibrator(
            top_probs, correct, bins)

    def calibrate(self, probs):
        top_probs = utils.get_top_probs(probs)
        return self._calibrator(top_probs)


class PlattBinnerTopCalibrator:

    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, probs, labels):
        assert(len(probs) >= self._num_calibration)
        predictions = utils.get_top_predictions(probs)
        top_probs = utils.get_top_probs(probs)
        correct = (predictions == labels)
        self._platt = utils.get_platt_scaler(
            top_probs, correct)
        platt_probs = self._platt(top_probs)
        bins = utils.get_equal_bins(platt_probs, num_bins=self._num_bins)
        self._discrete_calibrator = utils.get_discrete_calibrator(
            platt_probs, bins)

    def calibrate(self, probs):
        top_probs = self._platt(utils.get_top_probs(probs))
        return self._discrete_calibrator(top_probs)


class PlattTopCalibrator:

    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, probs, labels):
        assert(len(probs) >= self._num_calibration)
        predictions = utils.get_top_predictions(probs)
        top_probs = utils.get_top_probs(probs)
        correct = (predictions == labels)
        self._platt = utils.get_platt_scaler(
            top_probs, correct)

    def calibrate(self, probs):
        return self._platt(utils.get_top_probs(probs))


class IdentityTopCalibrator:

    def __init__(self, num_calibration, num_bins):
        pass

    def train_calibration(self, probs, labels):
        pass

    def calibrate(self, probs):
        return utils.get_top_probs(probs)


class HistogramMarginalCalibrator:

    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, probs, labels):
        """Train a calibrator given probs and labels.

        Args:
            probs: A sequence of dimension (n, k) where n is the number of
                data points, and k is the number of classes, representing
                the output probabilities/confidences of the uncalibrated
                model.
            labels: A sequence of length n, where n is the number of data points,
                representing the ground truth label for each data point.
        """
        assert(len(probs) >= self._num_calibration)
        probs = np.array(probs)
        self._k = probs.shape[1]  # Number of classes.
        assert self._k == np.max(labels) - np.min(labels) + 1
        labels_one_hot = utils.get_labels_one_hot(np.array(labels), self._k)
        self._calibrators = []
        for c in range(self._k):
            # For each class c, get the probabilities the model output for that class, and whether
            # the data point was actually class c, or not.
            probs_c = probs[:, c]
            labels_c = labels_one_hot[:, c]
            bins = utils.get_equal_bins(probs_c, num_bins=self._num_bins)
            calibrator_c = utils.get_histogram_calibrator(probs_c, labels_c, bins)
            self._calibrators.append(calibrator_c)

    def calibrate(self, probs):
        probs = np.array(probs)
        assert self._k == probs.shape[1]
        calibrated_probs = np.zeros(probs.shape)
        for c in range(self._k):
            probs_c = probs[:, c]
            calibrated_probs[:, c] = self._calibrators[c](probs_c)
        return calibrated_probs


class PlattBinnerMarginalCalibrator:

    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, probs, labels):
        """Train a calibrator given probs and labels.

        Args:
            probs: A sequence of dimension (n, k) where n is the number of
                data points, and k is the number of classes, representing
                the output probabilities/confidences of the uncalibrated
                model.
            labels: A sequence of length n, where n is the number of data points,
                representing the ground truth label for each data point.
        """
        assert(len(probs) >= self._num_calibration)
        probs = np.array(probs)
        self._k = probs.shape[1]  # Number of classes.
        assert self._k == np.max(labels) - np.min(labels) + 1
        labels_one_hot = utils.get_labels_one_hot(np.array(labels), self._k)
        assert labels_one_hot.shape == probs.shape
        self._platts = []
        self._calibrators = []
        for c in range(self._k):
            # For each class c, get the probabilities the model output for that class, and whether
            # the data point was actually class c, or not.
            probs_c = probs[:, c]
            labels_c = labels_one_hot[:, c]
            platt_c = utils.get_platt_scaler(probs_c, labels_c)
            self._platts.append(platt_c)
            platt_probs_c = platt_c(probs_c)
            bins = utils.get_equal_bins(platt_probs_c, num_bins=self._num_bins)
            calibrator_c = utils.get_discrete_calibrator(platt_probs_c, bins)
            self._calibrators.append(calibrator_c)


    def calibrate(self, probs):
        probs = np.array(probs)
        assert self._k == probs.shape[1]
        calibrated_probs = np.zeros(probs.shape)
        for c in range(self._k):
            probs_c = probs[:, c]
            platt_probs_c = self._platts[c](probs_c)
            calibrated_probs[:, c] = self._calibrators[c](platt_probs_c)
        return calibrated_probs
