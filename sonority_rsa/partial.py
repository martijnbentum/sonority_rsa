"""Partial RSA scoring."""

import numpy as np
from scipy.stats import pearsonr, rankdata

from sonority_rsa.rdm import upper_triangle


def partial_spearman_rsa(model_rdm, target_rdm, control_rdm):
    """
    Compare model and target RDMs while controlling for a third RDM.

    The strict upper triangles are average-ranked, then model and target
    ranks are independently residualized against control ranks with an
    intercept. The returned value is the Pearson correlation between those
    residuals.

    model_rdm: square model distance matrix
    target_rdm: square target distance matrix
    control_rdm: square control distance matrix
    """
    model_upper = np.asarray(upper_triangle(model_rdm), dtype=float)
    target_upper = np.asarray(upper_triangle(target_rdm), dtype=float)
    control_upper = np.asarray(upper_triangle(control_rdm), dtype=float)

    lengths = {len(model_upper), len(target_upper), len(control_upper)}
    if len(lengths) != 1:
        raise ValueError('RDM upper triangles must have equal length')

    triangles = (model_upper, target_upper, control_upper)
    if any(not np.all(np.isfinite(values)) for values in triangles):
        return np.nan

    model_ranks = rankdata(model_upper, method='average')
    target_ranks = rankdata(target_upper, method='average')
    control_ranks = rankdata(control_upper, method='average')
    if _lacks_variation(control_ranks):
        return np.nan

    design = np.column_stack((np.ones(len(control_ranks)), control_ranks))
    model_residuals = _residualize(model_ranks, design)
    target_residuals = _residualize(target_ranks, design)
    if (_lacks_variation(model_residuals)
            or _lacks_variation(target_residuals)):
        return np.nan

    return pearsonr(model_residuals, target_residuals).statistic


def _residualize(values, design):
    """Return OLS residuals for values regressed on a design matrix."""
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _lacks_variation(values):
    """Return whether values are empty or numerically constant."""
    if len(values) == 0:
        return True
    return np.allclose(values, values[0])
