import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from chapter_8 import bootstrap


def main(n_samples, n_boot_samples, n_hist_bins, confidence, n_repeats, dist_sampler, estimator, true_val):
    np.random.seed(0)

    n_contain1 = 0
    n_contain2 = 0
    n_contain3 = 0
    for i_repeat in range(n_repeats):
        x = dist_sampler(n_samples)

        e = estimator(x)
        #print('mu: %s \t sigma: %s \t skew: %s' % (mu, sigma, skew))

        x_boot = bootstrap.generate_boot_samples(x, n_boot_samples, stats.skew)
        ci1 = bootstrap.conf_interval_percentile(e, x_boot, confidence)
        ci2 = bootstrap.conf_interval_pivot(e, x_boot, confidence)
        ci3 = bootstrap.conf_interval_normal(e, x_boot, confidence)
        print('%s\t %s\t%s' % (str(ci1), str(ci2), str(ci3)))

        #count, bins, ignored = plt.hist(x, n_hist_bins)
        #plt.show()

        if ci1[0] < true_val < ci1[1]:
            n_contain1 += 1

        if ci2[0] < true_val < ci2[1]:
            n_contain2 += 1

        if ci3[0] < true_val < ci3[1]:
            n_contain3 += 1

    contain_ratio1 = n_contain1 / n_repeats
    contain_ratio2 = n_contain2 / n_repeats
    contain_ratio3 = n_contain3 / n_repeats

    print('\n\n#########################################')
    print('contain_ratio: %s \t %s \t %s' % (contain_ratio1, contain_ratio2, contain_ratio3))


main(n_samples=40, n_boot_samples=100, n_hist_bins=10, confidence=0.95, n_repeats=20,
     dist_sampler=lambda x: np.exp(np.random.standard_normal(x)), estimator=stats.skew, true_val=(np.exp(1)+2)*np.sqrt(np.exp(1) - 1))