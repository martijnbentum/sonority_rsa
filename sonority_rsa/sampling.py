"""RSA scores from repeated syllable subsets drawn without replacement."""

import warnings

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from sonority_rsa.partial import partial_spearman_rsa
from sonority_rsa.rdm import (correlation_rdm, intensity_rdm, sonority_rdm,
    spearman_rsa)

MIN_SUBSET_SIZE = 30
PARTIAL_RESULT_NAMES = ['sonority_partial_rsa', 'intensity_partial_rsa']
PARTIAL_INVALID_REASONS = [
    'undefined_sonority_partial_rsa',
    'undefined_intensity_partial_rsa',
    'undefined_both_partial_rsa',
]


def sample_syllables(syllable_population, subset_size, rng):
    """
    Draw a subset of distinct syllables from a fetched population.

    Sampling is without replacement, so every syllable in a single
    subset is unique and no exact-duplicate rows enter the RDMs. Across
    subsets the same syllable may reappear. Uses one rng.choice call of
    size subset_size, so a run is fully replayable from the population
    order and the seed (see replay_sampled_keys).

    syllable_population: SyllablePopulation or sequence of SyllableData;
        a plain sequence is supported for one-off sampling
    subset_size: number of syllables to sample (at least 30 and no larger
        than the population size)
    rng: numpy random generator
    """
    sampled = _draw_syllables(syllable_population, subset_size, rng)
    vectors = np.concatenate([syllable.vectors for syllable in sampled])
    sonority = np.concatenate([syllable.sonority for syllable in sampled])
    keys = [syllable.key for syllable in sampled]
    return vectors, sonority, keys


def _draw_syllables(syllable_population, subset_size, rng):
    """Draw and return SyllableData objects after validating the sample."""
    if subset_size < MIN_SUBSET_SIZE:
        raise ValueError(
            f'subset_size must be at least {MIN_SUBSET_SIZE} syllables')
    if len(syllable_population) == 0:
        raise ValueError('cannot sample from an empty syllable population')
    if subset_size > len(syllable_population):
        raise ValueError(
            f'subset_size ({subset_size}) exceeds the population size '
            f'({len(syllable_population)}); sampling without replacement '
            'needs subset_size <= population size')

    indices = rng.choice(len(syllable_population), size=subset_size,
        replace=False)
    return [syllable_population[int(index)] for index in indices]


def compute_rsa_scores(syllable_population, subset_size, n_subsets,
        random_state=None, return_diagnostics=False):
    """
    Compute one RSA score per syllable subset for a fetched population.

    Draws n_subsets subsets of subset_size distinct syllables (without
    replacement within a subset) and returns the RSA score of each.

    syllable_population: SyllablePopulation for one layer; required so
        progress and warning messages can identify the layer
    subset_size: number of syllables per subset
    n_subsets: number of subsets to draw
    random_state: optional integer seed or numpy random generator
    return_diagnostics: return (scores, diagnostics) when true; diagnostics
        contains invalid-subset counts and one reason per subset
    """
    results, diagnostics = _compute_rsa_results(syllable_population,
        subset_size, n_subsets, random_state=random_state)
    if return_diagnostics:
        return results['rsa'], diagnostics
    return results['rsa']


def _compute_rsa_results(syllable_population, subset_size, n_subsets,
        random_state=None, random_baseline_state=None,
        compute_intensity_baseline=False):
    """
    Compute observed and optionally random-baseline RSA scores.

    random_state controls syllable sampling. random_baseline_state controls
    one independent sonority shuffle per subset; when it is None, only
    observed RSA scores are computed. The two generators remain separate so
    baseline shuffles cannot change the sampled syllables.
    compute_intensity_baseline adds intensity RSA and sonority/intensity
    correlations for the same sampled phone rows, plus model–sonority RSA
    controlling for intensity and model–intensity RSA controlling for
    sonority.
    """
    if n_subsets <= 0:
        raise ValueError('n_subsets must be positive')
    if subset_size < MIN_SUBSET_SIZE:
        raise ValueError(
            f'subset_size must be at least {MIN_SUBSET_SIZE} syllables')
    if subset_size == len(syllable_population):
        if n_subsets > 1:
            raise ValueError(
                f'layer {syllable_population.layer}: subset_size '
                f'({subset_size}) equals the population size, so '
                'n_subsets must be 1')
    elif subset_size / len(syllable_population) >= 0.5:
        warnings.warn(
            f'layer {syllable_population.layer}: subset_size '
            f'({subset_size}) is at least 50% of the population '
            f'({len(syllable_population)}). Subsets are sampled '
            'independently and may overlap, so draws may have limited '
            'diversity; consider a larger population or smaller subset size',
            stacklevel=2)

    rng = make_rng(random_state)
    baseline_rng = (make_rng(random_baseline_state)
        if random_baseline_state is not None else None)
    results = {'rsa': []}
    if baseline_rng is not None:
        results['random_baseline_rsa'] = []
    if compute_intensity_baseline:
        results.update({
            'intensity_rsa': [],
            'sonority_intensity_correlation': [],
            'sonority_intensity_rdm_correlation': [],
            'sonority_partial_rsa': [],
            'intensity_partial_rsa': [],
        })
    invalid_subsets = {'single_sonority_class': 0,
        'undefined_vector_distance': 0}
    invalid_reasons = []
    intensity_invalid_subsets = {'single_intensity_value': 0,
        'undefined_intensity_rsa': 0}
    intensity_invalid_reasons = []
    partial_invalid_subsets = {reason: 0
        for reason in PARTIAL_INVALID_REASONS}
    partial_invalid_reasons = []

    for _ in tqdm(range(n_subsets), desc=f'layer {syllable_population.layer}',
            leave=False):
        sampled = _draw_syllables(syllable_population, subset_size, rng)
        vectors = np.concatenate([s.vectors for s in sampled])
        sonority = np.concatenate([s.sonority for s in sampled])
        intensity = None
        if compute_intensity_baseline:
            if any(s.intensity is None for s in sampled):
                raise RuntimeError(
                    'intensity baseline requested before intensity values '
                    'were attached to the syllable population')
            intensity = np.concatenate([s.intensity for s in sampled])

        model_rdm = correlation_rdm(vectors)
        sonority_predictor_rdm = None
        if np.ptp(sonority) == 0:
            results['rsa'].append(float('nan'))
            if baseline_rng is not None:
                results['random_baseline_rsa'].append(float('nan'))
            invalid_subsets['single_sonority_class'] += 1
            invalid_reasons.append('single_sonority_class')
        else:
            sonority_predictor_rdm = sonority_rdm(sonority)
            score = spearman_rsa(model_rdm, sonority_predictor_rdm)
            if np.isnan(score):
                invalid_subsets['undefined_vector_distance'] += 1
                invalid_reasons.append('undefined_vector_distance')
            else:
                invalid_reasons.append(None)
            results['rsa'].append(score)
            if baseline_rng is not None:
                shuffled = baseline_rng.permutation(sonority)
                baseline_score = spearman_rsa(model_rdm,
                    sonority_rdm(shuffled))
                results['random_baseline_rsa'].append(baseline_score)

        if compute_intensity_baseline:
            if np.ptp(intensity) == 0:
                results['intensity_rsa'].append(float('nan'))
                results['sonority_intensity_correlation'].append(float('nan'))
                results['sonority_intensity_rdm_correlation'].append(
                    float('nan'))
                _append_invalid_partial_results(results,
                    partial_invalid_subsets, partial_invalid_reasons,
                    'undefined_both_partial_rsa')
                intensity_invalid_subsets['single_intensity_value'] += 1
                intensity_invalid_reasons.append('single_intensity_value')
                continue
            intensity_predictor_rdm = intensity_rdm(intensity)
            intensity_score = spearman_rsa(model_rdm,
                intensity_predictor_rdm)
            results['intensity_rsa'].append(intensity_score)
            if np.isnan(intensity_score):
                intensity_invalid_subsets['undefined_intensity_rsa'] += 1
                intensity_invalid_reasons.append('undefined_intensity_rsa')
            else:
                intensity_invalid_reasons.append(None)

            if sonority_predictor_rdm is None:
                value_correlation = float('nan')
                rdm_correlation = float('nan')
            else:
                value_correlation = spearmanr(sonority, intensity).statistic
                rdm_correlation = spearman_rsa(sonority_predictor_rdm,
                    intensity_predictor_rdm)
            results['sonority_intensity_correlation'].append(
                value_correlation)
            results['sonority_intensity_rdm_correlation'].append(
                rdm_correlation)
            if sonority_predictor_rdm is None:
                _append_invalid_partial_results(results,
                    partial_invalid_subsets, partial_invalid_reasons,
                    'undefined_both_partial_rsa')
            else:
                _append_partial_results(results, model_rdm,
                    sonority_predictor_rdm, intensity_predictor_rdm,
                    partial_invalid_subsets, partial_invalid_reasons)

    n_nan = int(np.isnan(results['rsa']).sum())
    if n_nan:
        warnings.warn(
            f'layer {syllable_population.layer}: {n_nan} of {n_subsets} RSA '
            'scores are NaN (a sampled subset may have one sonority class, '
            'or a constant or collinear vector may make correlation distance '
            'undefined); summarize_rsa_scores will ignore them',
            stacklevel=2)

    diagnostics = {'invalid_subsets': invalid_subsets,
        'invalid_reasons': invalid_reasons}
    if compute_intensity_baseline:
        diagnostics.update({
            'intensity_invalid_subsets': intensity_invalid_subsets,
            'intensity_invalid_reasons': intensity_invalid_reasons,
            'partial_invalid_subsets': partial_invalid_subsets,
            'partial_invalid_reasons': partial_invalid_reasons,
        })
    return results, diagnostics


def _append_partial_results(results, model_rdm, sonority_predictor_rdm,
        intensity_predictor_rdm, invalid_subsets, invalid_reasons):
    """Compute paired partial scores and apply paired invalidation."""
    sonority_score = partial_spearman_rsa(model_rdm,
        sonority_predictor_rdm, intensity_predictor_rdm)
    intensity_score = partial_spearman_rsa(model_rdm,
        intensity_predictor_rdm, sonority_predictor_rdm)
    sonority_valid = np.isfinite(sonority_score)
    intensity_valid = np.isfinite(intensity_score)
    if sonority_valid and intensity_valid:
        results['sonority_partial_rsa'].append(sonority_score)
        results['intensity_partial_rsa'].append(intensity_score)
        invalid_reasons.append(None)
        return

    if not sonority_valid and not intensity_valid:
        reason = 'undefined_both_partial_rsa'
    elif not sonority_valid:
        reason = 'undefined_sonority_partial_rsa'
    else:
        reason = 'undefined_intensity_partial_rsa'
    _append_invalid_partial_results(results, invalid_subsets,
        invalid_reasons, reason)


def _append_invalid_partial_results(results, invalid_subsets,
        invalid_reasons, reason):
    """Append a paired invalid partial-RSA result and its reason."""
    for metric in PARTIAL_RESULT_NAMES:
        results[metric].append(float('nan'))
    invalid_subsets[reason] += 1
    invalid_reasons.append(reason)


def summarize_rsa_scores(scores_by_layer, ci=0.95):
    """
    Summarize RSA scores per layer.

    mean_rsa and the confidence interval are computed over the non-NaN
    scores only; n_subsets is the number of subsets drawn and
    n_subsets_valid is how many were non-NaN (the effective sample behind
    mean/CI).

    scores_by_layer: dict mapping layer to a list of RSA scores
    ci: confidence level as a fraction, e.g. 0.95 for a 95% interval;
        values below 0.90 emit a warning
    """
    _validate_ci(ci)
    return _summarize_rsa_scores(scores_by_layer, ci)


def _summarize_rsa_scores(scores_by_layer, ci):
    """Summarize scores after the confidence level has been validated."""
    alpha = (1 - ci) / 2
    rows = []

    for layer in sorted(scores_by_layer):
        rsa = np.asarray(scores_by_layer[layer], dtype=float)
        valid = rsa[~np.isnan(rsa)]
        rows.append({
            'layer': layer,
            'mean_rsa': float(np.mean(valid)) if valid.size else float('nan'),
            'ci_lower': (float(np.percentile(valid, 100 * alpha))
                if valid.size else float('nan')),
            'ci_upper': (float(np.percentile(valid, 100 * (1 - alpha)))
                if valid.size else float('nan')),
            'n_subsets': len(rsa),
            'n_subsets_valid': int(valid.size),
        })

    return rows


def _validate_ci(ci, warn=True):
    """Validate a fractional confidence level and optionally warn if low."""
    if not 0 < ci < 1:
        raise ValueError('ci must be greater than 0 and less than 1')
    if warn and ci < 0.9:
        warnings.warn(
            f'ci ({ci}) is below 0.90; this is an unexpected confidence '
            'interval level', stacklevel=3)


def replay_sampled_keys(syllable_keys, seed, subset_size, n_subsets):
    """
    Recompute the syllable keys drawn in each subset of a past run.

    Mirrors the rng calls of compute_rsa_scores exactly: one
    rng.choice(n, size=subset_size, replace=False) draw per subset over
    the population order. Inputs come from a run log layer entry.

    syllable_keys: population syllable keys in fetched order
    seed: the layer seed recorded in the run log
    subset_size: number of syllables per subset
    n_subsets: number of subsets to draw
    """
    rng = np.random.default_rng(seed)
    subsets = []
    for _ in range(n_subsets):
        indices = rng.choice(len(syllable_keys), size=subset_size,
            replace=False)
        subsets.append([syllable_keys[int(index)] for index in indices])
    return subsets


def make_rng(random_state):
    """
    Build a numpy random generator.

    random_state: None, integer seed, or existing generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state

    return np.random.default_rng(random_state)
