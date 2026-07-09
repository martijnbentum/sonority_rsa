"""Interactive analysis helpers for sonority RSA."""

from pathlib import Path

from sonority_rsa.bootstrap import (compute_bootstrap_by_layer,
    summarize_bootstrap)
from sonority_rsa.data import load_frame_table


def run_analysis(path, n_syllables, n_bootstraps, random_state=None):
    """
    Load a frame table and run bootstrap RSA by layer.

    path: CSV or Parquet frame table
    n_syllables: number of sampled syllables per bootstrap
    n_bootstraps: number of bootstrap repetitions
    random_state: optional integer seed or numpy random generator
    """
    df = load_frame_table(path)
    scores = compute_bootstrap_by_layer(
        df,
        n_syllables=n_syllables,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    summary = summarize_bootstrap(scores)
    summary['n_syllables'] = n_syllables
    return summary, scores


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
