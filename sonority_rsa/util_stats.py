"""Statistical utilities shared by reporting and plotting code."""

import math
import warnings

import numpy as np
from scipy.stats import t


def score_statistics(values, confidence=0.95):
    """
    Summarize finite scores with a Student's t interval around their mean.

    Returns mean, sample standard deviation, standard error of the mean,
    confidence bounds, and total/valid counts. Non-finite values are omitted.
    At least two finite values are needed to estimate dispersion and bounds.
    """
    validate_confidence(confidence, warn=False)
    numbers = np.asarray(list(values), dtype=float)
    valid = numbers[np.isfinite(numbers)]
    statistics = {
        'mean': float('nan'),
        'sd': float('nan'),
        'sem': float('nan'),
        'mean_ci_lower': float('nan'),
        'mean_ci_upper': float('nan'),
        'n_total': int(numbers.size),
        'n_valid': int(valid.size),
    }
    if not valid.size:
        return statistics

    mean = float(np.mean(valid))
    statistics['mean'] = mean
    if valid.size == 1:
        return statistics

    sd = float(np.std(valid, ddof=1))
    sem = sd / math.sqrt(valid.size)
    alpha = (1 - confidence) / 2
    critical = float(t.ppf(1 - alpha, df=valid.size - 1))
    margin = critical * sem
    statistics.update({
        'sd': sd,
        'sem': sem,
        'mean_ci_lower': mean - margin,
        'mean_ci_upper': mean + margin,
    })
    return statistics


def validate_confidence(confidence, warn=True):
    """Validate a fractional confidence level."""
    if not 0 < confidence < 1:
        raise ValueError(
            'confidence must be greater than 0 and less than 1')
    if warn and confidence < 0.9:
        warnings.warn(
            f'confidence ({confidence}) is below 0.90; this is an '
            'unexpected confidence level', stacklevel=3)
