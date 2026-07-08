"""Bootstrap RSA over syllable IDs."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonority_rsa.rdm import compute_sonority_rsa


def sample_syllables(df, n_syllables, rng):
    """
    Sample syllable IDs with replacement and return their frame rows.

    df: prepared frame table for one or more layers
    n_syllables: number of syllable IDs to sample
    rng: numpy random generator
    """
    syllable_ids = df['syllable_id'].drop_duplicates().to_numpy()
    if n_syllables <= 0:
        raise ValueError('n_syllables must be positive')
    if len(syllable_ids) == 0:
        raise ValueError('cannot sample from an empty frame table')

    sampled_ids = rng.choice(syllable_ids, size=n_syllables, replace=True)
    sampled_parts = []

    for syllable_id in sampled_ids:
        sampled_parts.append(df[df['syllable_id'] == syllable_id])

    return pd.concat(sampled_parts, ignore_index=True)


def compute_bootstrap_for_layer(df, layer, n_syllables, n_bootstraps,
        random_state=None):
    """
    Compute bootstrap RSA scores for one layer.

    df: prepared frame table
    layer: layer value to analyze
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    random_state: optional random seed or numpy generator
    """
    layer_df = df[df['layer'] == layer].reset_index(drop=True)
    if layer_df.empty:
        raise ValueError(f'layer has no rows: {layer!r}')
    if n_bootstraps <= 0:
        raise ValueError('n_bootstraps must be positive')

    rng = _make_rng(random_state)
    records = []

    for bootstrap in tqdm(range(n_bootstraps), desc=f'layer {layer}',
            leave=False):
        sampled_df = sample_syllables(layer_df, n_syllables, rng)
        vectors = np.stack(sampled_df['vector'].to_numpy())
        sonority = sampled_df['sonority'].to_numpy()
        score = compute_sonority_rsa(vectors, sonority)
        records.append({
            'layer': layer,
            'bootstrap': bootstrap,
            'rsa': score,
        })

    return pd.DataFrame(records)


def compute_bootstrap_by_layer(df, n_syllables, n_bootstraps,
        random_state=None):
    """
    Compute bootstrap RSA scores independently for each layer.

    df: prepared frame table
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    random_state: optional random seed or numpy generator
    """
    rng = _make_rng(random_state)
    records = []

    for layer in sorted(df['layer'].unique()):
        layer_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        layer_scores = compute_bootstrap_for_layer(
            df,
            layer,
            n_syllables,
            n_bootstraps,
            random_state=layer_seed,
        )
        records.append(layer_scores)

    if not records:
        return pd.DataFrame(columns=['layer', 'bootstrap', 'rsa'])

    return pd.concat(records, ignore_index=True)


def summarize_bootstrap(scores, ci=95):
    """
    Summarize bootstrap RSA scores by layer.

    scores: DataFrame with layer, bootstrap, and rsa columns
    ci: percentile confidence interval width
    """
    if scores.empty:
        return pd.DataFrame(columns=[
            'layer',
            'mean_rsa',
            'ci_lower',
            'ci_upper',
            'n_bootstraps',
        ])

    alpha = (100 - ci) / 2
    rows = []

    for layer, layer_scores in scores.groupby('layer', sort=True):
        rsa = layer_scores['rsa'].to_numpy(dtype=float)
        rows.append({
            'layer': layer,
            'mean_rsa': np.nanmean(rsa),
            'ci_lower': np.nanpercentile(rsa, alpha),
            'ci_upper': np.nanpercentile(rsa, 100 - alpha),
            'n_bootstraps': len(rsa),
        })

    return pd.DataFrame(rows)


def _make_rng(random_state):
    """
    Build a numpy random generator.

    random_state: None, integer seed, or existing generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state

    return np.random.default_rng(random_state)
