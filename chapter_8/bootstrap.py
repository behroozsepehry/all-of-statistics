import numpy as np
from scipy import stats


def generate_boot_samples(x, n_samples, estimator):
    n = x.size
    e_arr = np.zeros(n_samples)
    for i_iteration in range(n_samples):
        i_boot = np.random.randint(0, n, size=x.shape)
        x_boot = x[i_boot]
        e_arr[i_iteration] = estimator(x_boot)
    return e_arr


def conf_interval_normal(t, t_samples, confidence):
    alpha2 = (1. - confidence) / 2.
    se = np.sqrt(np.var(t_samples))
    ci = (t+se*stats.norm.ppf(alpha2), t+se*stats.norm.ppf(1.-alpha2))
    return ci


def conf_interval_percentile(t, t_samples, confidence):
    n = t_samples.size
    alpha2 = (1. - confidence) / 2.
    quantile_index = int(n * alpha2)
    t_samples_sorted = np.sort(t_samples)
    conf_interval = (t_samples_sorted[quantile_index], t_samples_sorted[-quantile_index])
    return conf_interval


def conf_interval_pivot(t, t_samples, confidence):
    ci1 = conf_interval_percentile(t, t_samples, confidence)
    ci2 = (2*t-ci1[1], 2*t-ci1[0])
    return ci2