
# Verified Uncertainty Calibration

Code for the NeurIPS 2019 (Spotlight) paper [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155)

In our paper, we show that:
- The calibration error of methods like Platt scaling is typically underestimated, and cannot be easily measured.
- We propose an efficient recalibration method where the calibration error can be measured.
- We show that we can estimate the calibration error with fewer samples (than the standard method) using an estimator from the meteorological literature.


## Calibration Library

This repository contains library code to measure the calibration error of models, including confidence intervals computed by Bootstrap, and code to recalibrate models.

Start with examples/example.py which walks through how to recalibrate a model and estimate its calibration error, including confidence intervals.
Ensure you're using Python 3 when running all our code:

`export PYTHONPATH="."; python3 examples/example.py`


## Experiments

The experiments folder contains experiments for the paper.
We have 4 sets of experiments:
- Showing the Platt scaling is less calibrated than reported (Section 3)
- Comparing the scaling binning calibrator with histogram binning on CIFAR-10 and ImageNet (Section 4)
- Synthetic experiments to validate our theoretical bounds (Section 4)
- Experiments showing the debiased estimator can estimate calibration error with fewer samples than standard estimator (Section 5)
Running each experiment saves plots in the corresponding folder in saved_files

`export PYTHONPATH="."`

Then run the experiment code, for example:
`python3 experiments/scaling_binning_calibrator/compare_calibrators.py`

More detailed instructions coming soon.

