import numpy as np
import pandas as pd

from sonority_rsa.bootstrap import (compute_bootstrap_by_layer,
    sample_syllables, summarize_bootstrap)


def test_bootstrap_sampling_preserves_duplicate_syllables():
    df = pd.DataFrame({
        'syllable_id': ['s1', 's1', 's2'],
        'phone': ['p', 'a', 'm'],
        'sonority': [0, 5, 2],
        'layer': [0, 0, 0],
        'vector': [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        ],
    })
    rng = FixedRng(['s1', 's1'])

    sampled = sample_syllables(df, n_syllables=2, rng=rng)

    assert sampled['syllable_id'].tolist() == ['s1', 's1', 's1', 's1']


def test_output_contains_one_summary_row_per_layer(toy_frames):
    scores = compute_bootstrap_by_layer(
        toy_frames,
        n_syllables=3,
        n_bootstraps=2,
        random_state=1,
    )
    summary = summarize_bootstrap(scores)

    assert sorted(summary['layer'].tolist()) == [0, 1]
    assert summary['n_bootstraps'].tolist() == [2, 2]


class FixedRng:

    def __init__(self, values):
        self.values = values

    def choice(self, values, size, replace=True):
        return np.asarray(self.values)
