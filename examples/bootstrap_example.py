
import calibration as cal
import numpy as np

# Keep the results consistent.
np.random.seed(0)

def mean_test():
    p = 0.9
    num_trials = 100
    num_samples = 100
    bootstrap_valid = 0
    means = []
    for i in range(num_trials):
        samples = np.random.binomial(n=1, p=p, size=num_samples)
        means.append(np.mean(samples))
    #     lower, _, upper = cal.bootstrap_uncertainty(
    #         list(samples), np.mean, num_samples=1000)
    #     if lower <= p <= upper:
    #         bootstrap_valid += 1
    # print("Valid percent is {}".format(float(bootstrap_valid) / num_trials))
    print(np.mean(means), np.std(means))

def ce_test():
    x_low, x_high = 0.5, 0.9
    error = 0.04
    ce = error ** 2
    # The x values will be uniformly distributed. 
    num_trials = 1000
    num_samples = 30000
    bootstrap_valid = 0
    num_bins = 1
    total_len = 0.0
    sum_est = 0.0
    stds = []
    for num_bins in [1, 2, 4, 8, 16, 32, 64]:
        # Define the estimator.
        bin_xs = np.random.uniform(size=num_samples * 2, low=x_low, high=x_high)
        bins = cal.get_equal_bins(bin_xs, num_bins=num_bins)
        def estimate_unbiased_ce(data):
            binned_data = cal.bin(data, bins)
            return np.sqrt(max(0.0, cal.unbiased_square_ce(binned_data)))
        ces = []
        for i in range(num_trials):
            xs = np.random.uniform(size=num_samples, low=x_low, high=x_high)
            ys = np.random.binomial(n=1, p=xs-error)
            data = list(zip(xs, ys))
            ces.append(estimate_unbiased_ce(data))
        print(np.mean(ces), np.std(ces))
        stds.append(np.std(ces))
    print(stds)
    # def estimate_plugin_ce(data):
    #     binned_data = cal.bin(data, bins)
    #     return cal.plugin_ce(binned_data) ** 2
    # for i in range(num_trials):
    #     xs = np.random.uniform(size=num_samples, low=x_low, high=x_high)
    #     ys = np.random.binomial(n=1, p=xs+error)
    #     data = list(zip(xs, ys))
    #     lower, mid, upper = cal.precentile_bootstrap_uncertainty(
    #         data, estimate_plugin_ce, estimate_plugin_ce, num_samples=1000)
    #     # est = estimate_unbiased_ce(data)
    #     # mid = est
    #     # print('mean', np.mean(ys), np.mean(xs))
    #     # print(abs(mid - ce) / ce, mid, ce)
    #     est = estimate_plugin_ce(data)
    #     sum_est += est
    #     total_len += upper - lower
    #     if lower <= ce <= upper:
    #         bootstrap_valid += 1
    #     else:
    #         print('est', est)
    #         print('interval', lower, upper)
    # print("Valid percent is {}".format(float(bootstrap_valid) / num_trials))
    # print("Average length is {}".format(total_len / num_trials))
    # print("Average est is {}".format(sum_est / num_trials))

if __name__ == "__main__":
    ce_test()
