import numpy as np


def brier_score(observations, forecasts):
    return np.mean((forecasts - observations) ** 2)


def brier_skill_score(observations, forecasts):
    bs_climo = brier_score(observations, observations.mean())
    bs = brier_score(observations, forecasts)
    return 1.0 - bs / bs_climo
