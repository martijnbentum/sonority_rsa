"""Statistical summaries shared by reporting and plotting code."""

import math

import numpy as np
from scipy.stats import t


def mean_confidence_interval(values, confidence=0.95):
    """
    Return the mean and Student's t confidence interval for finite values.

    Non-finite values are omitted. With no finite values, all three results
    are NaN. With one finite value, the mean is returned with NaN bounds
    because its standard error cannot be estimated.
    """
    if not 0 < confidence < 1:
        raise ValueError('confidence must be greater than 0 and less than 1')

    numbers = np.asarray(list(values), dtype=float)
    valid = numbers[np.isfinite(numbers)]
    if not valid.size:
        return float('nan'), float('nan'), float('nan')

    mean = float(np.mean(valid))
    if valid.size == 1:
        return mean, float('nan'), float('nan')

    standard_error = float(np.std(valid, ddof=1) / math.sqrt(valid.size))
    alpha = (1 - confidence) / 2
    critical = float(t.ppf(1 - alpha, df=valid.size - 1))
    margin = critical * standard_error
    return mean, mean - margin, mean + margin
