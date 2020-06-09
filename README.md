# Uncertainty Calibration Library

This repository contains library code to measure the calibration error of models, including confidence intervals computed by Bootstrap resampling, and code to recalibrate models. Our functions estimate the calibration error and ECE more accurately than prior "plugin" estimators and we provide Bootstrap confidence intervals. See [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155) for more details.

Motivating example for uncertainty calibration: Calster and Vickers 2015 train a random forest that takes in features such as tumor size and presence of teratoma, and tries to predict the probability a patient has testicular cancer. They note that for a large number of patients, the model predicts around a 20% chance of cancer. In reality, around 40% of these patients had cancer. This underestimation can lead to doctors prescribing the wrong treatment---in a situation where many lives are at stake.

*The high level point here is that the uncertainties that models output matter, not just the model's accuracy*. Calibration is a popular way to measure the quality of a model's uncertainties, and recalibration is a way to take an existing model and correct its uncertainties to make them better.


## Installation

```python
pip3 install uncertainty-calibration
```

The calibration library requires python 3.6 or higher at the moment because we make use of the Python 3 optional typing mechanism.
If your project requires an earlier of version of python, and you wish to use our library, please contact us.


## Overview

Measuring the calibration error of a model is as simple as:

```python
import calibration as cal
calibration_error = cal.get_calibration_error(model_probs, labels)
```

Recalibrating a model is very simple as well. Recalibration requires a small labeled dataset, on which we train a recalibrator:

```python
calibrator = cal.PlattBinnerMarginalCalibrator(num_points, num_bins=10)
calibrator.train_calibration(model_probs, labels)
```

Now whenever the model outputs a prediction, we pass it through the calibrator to produce better probabilities.

```python
calibrated_probs = cal.calibrate(test_probs)
```

Our library makes it very easy to measure confidence intervals on the calibration error as well, using bootstrap resamples.

```python
[lower, _, upper] = cal.get_calibration_error_uncertainties(model_probs, labels)
```


## Examples

You can find complete code examples in the examples folder. Refer to:
- examples/simple_example.py for a simple example in the binary classification setting.
- examples/multiclass_example.py for the multiclass (more than 2 classes) setting.
- examples/advanced_example.py --- our calibration library also exposes a more customizable interface for advanced users.


## Citation

If you find this library useful please consider citing our paper:

    @inproceedings{kumar2019calibration,
      author = {Ananya Kumar and Percy Liang and Tengyu Ma},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      title = {Verified Uncertainty Calibration},
      year = {2019},
    }


## Advanced: ECE, Debiasing, and Top-Label Calibration Error

By default, our library measure per-class, root-mean-squared calibration error, and uses the techniques in [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155) to accurately estimate the calibration error. However, we also support measuring the ECE [(Guo et al)](https://arxiv.org/abs/1706.04599) and using older, less accurate, ways of estimating the calibration error.

To measure the ECE as in [Guo et al](https://arxiv.org/abs/1706.04599):

```python
calibration_error = cal.get_ece(model_probs, labels)
```

To estimate it more accurately, and use a more stable way of binning, use:

```python
calibration_error = cal.get_top_calibration_error(model_probs, labels, p=1)
```

Multiclass calibration vs ECE / Top-label: When measuring the calibration error of a multiclass model, we can measure the calibration error of all classes (per-class calibration error), or of the top prediction. As an example, imagine that a medical diagnosis system says there is a 50% chance a patient has a benign tumor, a 10% chance she has an aggressive form of cancer, and a 40% chance she has one of a long list of other conditions. We would like the system to be calibrated on each of these predictions (especially cancer!), and not just the top prediction of a benign tumor. [Nixon et al](https://arxiv.org/abs/1909.10155), [Kumar et al](https://arxiv.org/abs/1909.10155), and [Kull et al](https://arxiv.org/abs/1910.12656) measure per-class calibration to account for this.


## Questions, bugs, and contributions

Please feel free to ask us questions, submit bug reports, or contribute push requests.
Feel free to submit a brief description of a push request before you implement it to get feedback on it, or see how it can fit into our project.


## Verified Uncertainty Calibration paper

This repository also contains code for the NeurIPS 2019 (Spotlight) paper [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155)

In our paper, we show that:
- The calibration error of methods like Platt scaling and temperature scaling are typically underestimated, and cannot be easily measured.
- We propose an efficient recalibration method where the calibration error can be measured.
- We show that we can estimate the calibration error with fewer samples (than the standard method) using an estimator from the meteorological literature.


## Experiments

The experiments folder contains experiments for the paper.

We have 4 sets of experiments:
- Showing the Platt scaling is less calibrated than reported (Section 3)
- Comparing the scaling binning calibrator with histogram binning on CIFAR-10 and ImageNet (Section 4)
- Synthetic experiments to validate our theoretical bounds (Section 4)
- Experiments showing the debiased estimator can estimate calibration error with fewer samples than standard estimator (Section 5)
Running each experiment saves plots in the corresponding folder in saved_files

See our CodaLab worksheet https://worksheets.codalab.org/worksheets/0xb6d027ee127e422989ab9115726c5411 which contains all the experiment runs and the exact code used to produce them.
