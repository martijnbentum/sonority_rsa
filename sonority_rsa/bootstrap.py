"""Bootstrap RSA over fetched syllable populations."""

import warnings

import numpy as np
from tqdm import tqdm

from sonority_rsa.rdm import compute_sonority_rsa


def sample_syllables(population, n_syllables, rng):
    """
    Draw a subset of distinct syllables from a fetched population.

    Sampling is without replacement, so every syllable in a single draw
    is unique and no exact-duplicate rows enter the RDMs. Across draws
    the same syllable may reappear. Uses one rng.choice call of size
    n_syllables, so a run is fully replayable from the population order
    and the seed (see replay_sampled_keys).

    population: SyllablePopulation (or sequence of SyllableData)
    n_syllables: number of syllables to sample (<= population size)
    rng: numpy random generator
    """
    if n_syllables <= 0:
        raise ValueError('n_syllables must be positive')
    if len(population) == 0:
        raise ValueError('cannot sample from an empty population')
    if n_syllables > len(population):
        raise ValueError(
            f'n_syllables ({n_syllables}) exceeds the population size '
            f'({len(population)}); sampling without replacement needs '
            'n_syllables <= population size')

    indices = rng.choice(len(population), size=n_syllables, replace=False)
    sampled = [population[int(index)] for index in indices]

    vectors = np.concatenate([syllable.vectors for syllable in sampled])
    sonority = np.concatenate([syllable.sonority for syllable in sampled])
    keys = [syllable.key for syllable in sampled]
    return vectors, sonority, keys


def compute_bootstrap(population, n_syllables, n_bootstraps,
        random_state=None):
    """
    Compute bootstrap RSA scores for one fetched layer population.

    population: SyllablePopulation for one layer
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    random_state: optional integer seed or numpy random generator
    """
    if n_bootstraps <= 0:
        raise ValueError('n_bootstraps must be positive')
    if n_syllables == len(population):
        warnings.warn(
            f'layer {population.layer}: n_syllables ({n_syllables}) equals '
            'the population size, so every subsample is the full population '
            'and all RSA scores will be identical (zero across-run '
            'variability); use n_syllables < population size', stacklevel=2)

    rng = make_rng(random_state)
    scores = []

    for _ in tqdm(range(n_bootstraps), desc=f'layer {population.layer}',
            leave=False):
        vectors, sonority, _ = sample_syllables(population, n_syllables, rng)
        scores.append(compute_sonority_rsa(vectors, sonority))

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        warnings.warn(
            f'layer {population.layer}: {n_nan} of {n_bootstraps} bootstrap '
            'RSA scores are NaN (a constant or collinear vector makes '
            'correlation distance undefined); summarize_bootstrap will '
            'ignore them', stacklevel=2)

    return scores


def summarize_bootstrap(scores_by_layer, ci=95):
    """
    Summarize bootstrap RSA scores per layer.

    scores_by_layer: dict mapping layer to a list of RSA scores
    ci: percentile confidence interval width
    """
    alpha = (100 - ci) / 2
    rows = []

    for layer in sorted(scores_by_layer):
        rsa = np.asarray(scores_by_layer[layer], dtype=float)
        rows.append({
            'layer': layer,
            'mean_rsa': float(np.nanmean(rsa)),
            'ci_lower': float(np.nanpercentile(rsa, alpha)),
            'ci_upper': float(np.nanpercentile(rsa, 100 - alpha)),
            'n_bootstraps': len(rsa),
        })

    return rows


def replay_sampled_keys(syllable_keys, seed, n_syllables, n_bootstraps):
    """
    Recompute the syllable keys drawn in each bootstrap of a past run.

    Mirrors the rng calls of compute_bootstrap exactly: one
    rng.choice(n, size=n_syllables, replace=False) draw per bootstrap
    over the population order. Inputs come from a run log layer entry.

    syllable_keys: population syllable keys in fetched order
    seed: the layer seed recorded in the run log
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(syllable_keys), size=n_syllables,
            replace=False)
        draws.append([syllable_keys[int(index)] for index in indices])
    return draws


def make_rng(random_state):
    """
    Build a numpy random generator.

    random_state: None, integer seed, or existing generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state

    return np.random.default_rng(random_state)
