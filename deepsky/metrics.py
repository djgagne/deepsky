import numpy as np


def brier_score(observations, forecasts):
    return np.mean((forecasts - observations) ** 2)


def brier_skill_score(observations, forecasts):
    bs_climo = brier_score(observations, observations.mean())
    bs = brier_score(observations, forecasts)
    return 1.0 - bs / bs_climo


def roc_auc(observations, forecasts):
    fore_prob = forecasts * 100
    thresholds = np.unique(fore_prob.astype(np.int64))
    pod = np.zeros(thresholds.size)
    pofd = np.zeros(thresholds.size)
    obs_bin = observations > 0
    no_obs_bin = ~obs_bin
    pos_count = np.count_nonzero(observations)
    neg_count = observations.size - pos_count
    for t, threshold in enumerate(thresholds):
        pos_fore = (fore_prob >= threshold)
        pod[t] = np.count_nonzero(pos_fore & obs_bin)
        pofd[t] = np.count_nonzero(pos_fore & no_obs_bin)
    pod /= pos_count
    pofd /= neg_count
    return -np.trapz(pod, pofd)
