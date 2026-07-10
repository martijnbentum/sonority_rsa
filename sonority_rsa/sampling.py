"""RSA scores from repeated syllable subsets drawn without replacement."""

import warnings

import numpy as np
from tqdm import tqdm

from sonority_rsa.rdm import compute_sonority_rsa


def sample_syllables(syllable_population, subset_size, rng):
    """
    Draw a subset of distinct syllables from a fetched population.

    Sampling is without replacement, so every syllable in a single
    subset is unique and no exact-duplicate rows enter the RDMs. Across
    subsets the same syllable may reappear. Uses one rng.choice call of
    size subset_size, so a run is fully replayable from the population
    order and the seed (see replay_sampled_keys).

    syllable_population: SyllablePopulation (or sequence of SyllableData)
    subset_size: number of syllables to sample (<= population size)
    rng: numpy random generator
    """
    if subset_size <= 0:
        raise ValueError('subset_size must be positive')
    if len(syllable_population) == 0:
        raise ValueError('cannot sample from an empty syllable population')
    if subset_size > len(syllable_population):
        raise ValueError(
            f'subset_size ({subset_size}) exceeds the population size '
            f'({len(syllable_population)}); sampling without replacement '
            'needs subset_size <= population size')

    indices = rng.choice(len(syllable_population), size=subset_size,
        replace=False)
    sampled = [syllable_population[int(index)] for index in indices]

    vectors = np.concatenate([syllable.vectors for syllable in sampled])
    sonority = np.concatenate([syllable.sonority for syllable in sampled])
    keys = [syllable.key for syllable in sampled]
    return vectors, sonority, keys


def compute_rsa_scores(syllable_population, subset_size, n_subsets,
        random_state=None):
    """
    Compute one RSA score per syllable subset for a fetched population.

    Draws n_subsets subsets of subset_size distinct syllables (without
    replacement within a subset) and returns the RSA score of each.

    syllable_population: SyllablePopulation for one layer
    subset_size: number of syllables per subset
    n_subsets: number of subsets to draw
    random_state: optional integer seed or numpy random generator
    """
    if n_subsets <= 0:
        raise ValueError('n_subsets must be positive')
    if subset_size == len(syllable_population):
        warnings.warn(
            f'layer {syllable_population.layer}: subset_size ({subset_size}) '
            'equals the population size, so every subset is the full '
            'population and all RSA scores will be identical (zero '
            'across-subset variability); use subset_size < population size',
            stacklevel=2)

    rng = make_rng(random_state)
    scores = []

    for _ in tqdm(range(n_subsets), desc=f'layer {syllable_population.layer}',
            leave=False):
        vectors, sonority, _ = sample_syllables(syllable_population,
            subset_size, rng)
        if np.ptp(sonority) == 0:
            # backstop: fetch_syllable_data already rejects a population
            # with a single sonority class, so a subset with no sonority
            # variation should be unreachable; guard it rather than emit a
            # silent NaN from the constant sonority RDM
            raise ValueError(
                f'layer {syllable_population.layer}: a sampled subset has a '
                'single sonority class, so the sonority RDM has no '
                'variation; this should not occur for a population with '
                'multiple classes')
        scores.append(compute_sonority_rsa(vectors, sonority))

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        warnings.warn(
            f'layer {syllable_population.layer}: {n_nan} of {n_subsets} RSA '
            'scores are NaN (a constant or collinear vector makes '
            'correlation distance undefined); summarize_rsa_scores will '
            'ignore them', stacklevel=2)

    return scores


def summarize_rsa_scores(scores_by_layer, ci=95):
    """
    Summarize RSA scores per layer.

    mean_rsa and the confidence interval are computed over the non-NaN
    scores only; n_subsets is the number of subsets drawn and
    n_subsets_valid is how many were non-NaN (the effective sample behind
    mean/CI).

    scores_by_layer: dict mapping layer to a list of RSA scores
    ci: percentile confidence interval width
    """
    alpha = (100 - ci) / 2
    rows = []

    for layer in sorted(scores_by_layer):
        rsa = np.asarray(scores_by_layer[layer], dtype=float)
        valid = rsa[~np.isnan(rsa)]
        rows.append({
            'layer': layer,
            'mean_rsa': float(np.mean(valid)) if valid.size else float('nan'),
            'ci_lower': (float(np.percentile(valid, alpha))
                if valid.size else float('nan')),
            'ci_upper': (float(np.percentile(valid, 100 - alpha))
                if valid.size else float('nan')),
            'n_subsets': len(rsa),
            'n_subsets_valid': int(valid.size),
        })

    return rows


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
