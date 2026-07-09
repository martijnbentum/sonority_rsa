"""Interactive analysis helpers for sonority RSA."""

from pathlib import Path

import pandas as pd

from sonority_rsa.bootstrap import (compute_bootstrap_by_layer,
    summarize_bootstrap)
from sonority_rsa.data import load_frame_table
from sonority_rsa.extract import build_frame_table


def run_analysis(frames, n_syllables, n_bootstraps, random_state=None):
    """
    Run bootstrap RSA by layer on a frame table.

    frames: frame table DataFrame, or path to a Parquet cache file
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    random_state: optional integer seed or numpy random generator
    """
    if not isinstance(frames, pd.DataFrame):
        frames = load_frame_table(frames)
    scores = compute_bootstrap_by_layer(
        frames,
        n_syllables=n_syllables,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    summary = summarize_bootstrap(scores)
    summary['n_syllables'] = n_syllables
    return summary, scores


def run_analysis_from_stores(phrases, model_name, layers, n_syllables,
        n_bootstraps, collar=500, random_state=None):
    """
    Extract a frame table from stored embeddings and run bootstrap RSA.

    phrases: iterable of phraser Phrase objects with stored embeddings
    model_name: registered echoframe model name (e.g. 'wav2vec2')
    layers: list of hidden-state layers to analyze
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    collar: milliseconds of context stored around the phrase
    random_state: optional integer seed or numpy random generator
    """
    frames = build_frame_table(phrases, model_name, layers, collar=collar)
    summary, scores = run_analysis(
        frames,
        n_syllables=n_syllables,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    return summary, scores, frames


def save_analysis(summary, scores, out):
    """
    Save summary and raw bootstrap scores.

    summary: summary DataFrame from run_analysis
    scores: raw bootstrap scores DataFrame from run_analysis
    out: output directory
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'summary.csv', index=False)
    scores.to_csv(out / 'bootstrap_scores.csv', index=False)


def display_analysis(summary, scores, n=10):
    """
    Display analysis output in IPython, falling back to plain text.

    summary: summary DataFrame from run_analysis
    scores: raw bootstrap scores DataFrame from run_analysis
    n: number of raw score rows to preview
    """
    try:
        from IPython.display import display
    except ImportError:
        print(summary.to_string(index=False))
        print(scores.head(n).to_string(index=False))
        return

    display(summary)
    display(scores.head(n))
