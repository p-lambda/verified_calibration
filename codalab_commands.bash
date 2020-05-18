
# Lower bound experiments (Section 3).

# CIFAR:

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/cifar_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128" --lp=2 \
--calibration_data_size=1000 --bin_data_size=1000 --plot_save_file=l2_lower_bound_cifar_plot.png \
--binning=equal_bins --num_samples=1000' \
--request-queue tag=nlp -n cifar_l2_lower_bound

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/cifar_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128" --lp=1 \
--calibration_data_size=1000 --bin_data_size=1000 --plot_save_file=l1_lower_bound_cifar_plot.png \
--binning=equal_bins --num_samples=1000' \
--request-queue tag=nlp -n cifar_l1_lower_bound

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/cifar_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128" --lp=1 \
--calibration_data_size=1000 --bin_data_size=1000 --plot_save_file=prob_lower_bound_cifar_plot.png \
--binning=equal_prob_bins --num_samples=1000' \
--request-queue tag=nlp -n cifar_prob_lower_bound

# ImageNet:

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/imagenet_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128, 256, 512" --lp=2 \
--calibration_data_size=20000 --bin_data_size=5000 --plot_save_file=l2_lower_bound_imnet_plot.png \
--binning=equal_bins --num_samples=1000' \
--request-queue tag=nlp -n imagenet_l2_lower_bound

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/imagenet_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128, 256, 512" --lp=1 \
--calibration_data_size=20000 --bin_data_size=5000 --plot_save_file=l1_lower_bound_imnet_plot.png \
--binning=equal_bins --num_samples=1000' \
--request-queue tag=nlp -n imagenet_l1_lower_bound

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/platt_not_calibrated/lower_bounds.py \
--probs_path=data/imagenet_probs.dat --bins_list="2, 4, 8, 16, 32, 64, 128, 256, 512" --lp=1 \
--calibration_data_size=20000 --bin_data_size=5000 --plot_save_file=prob_lower_bound_imnet_plot.png \
--binning=equal_prob_bins --num_samples=1000' \
--request-queue tag=nlp -n imagenet_prob_lower_bound


# Comparing calibrators experiment (Section 4).

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/scaling_binning_calibrator/compare_calibrators.py' \
--request-queue tag=nlp -n compare_calibrators


# Synthetic experiments (Section 4).

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/synthetic/synthetic.py \
--experiment_name=vary_n_a1_b0' \
--request-queue tag=nlp -n vary_n_a1_b0

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/synthetic/synthetic.py \
--experiment_name=vary_b_a1_b0' \
--request-queue tag=nlp -n vary_b_a1_b0

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/synthetic/synthetic.py \
--experiment_name=vary_n_a2_b1' \
--request-queue tag=nlp -n vary_n_a2_b1

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/synthetic/synthetic.py \
--experiment_name=vary_b_a2_b1' \
--request-queue tag=nlp -n vary_b_a2_b1

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/synthetic/synthetic.py \
--experiment_name=noisy_vary_n_a2_b1' \
--request-queue tag=nlp -n noisy_vary_n_a2_b1


# Comparing plugin with debiased calibration error estimators (Section 5).

cl run :experiments :calibration :data \
'export PYTHONPATH="."; \
python3 experiments/debiased_estimator/estimation_error_vs_bins.py' \
--request-queue tag=nlp -n estimator_comparison