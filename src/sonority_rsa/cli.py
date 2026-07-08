"""Command-line interface for sonority RSA."""

import argparse
from pathlib import Path

from sonority_rsa.bootstrap import (compute_bootstrap_by_layer,
    summarize_bootstrap)
from sonority_rsa.data import load_frame_table


def main(argv=None):
    """
    Run bootstrap RSA from the command line.

    argv: optional command-line argument list
    """
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_frame_table(args.frames)
    scores = compute_bootstrap_by_layer(
        df,
        n_syllables=args.n_syllables,
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )
    summary = summarize_bootstrap(scores)
    summary['n_syllables'] = args.n_syllables

    summary.to_csv(out / 'summary.csv', index=False)
    scores.to_csv(out / 'bootstrap_scores.csv', index=False)


def parse_args(argv=None):
    """
    Parse command-line arguments.

    argv: optional command-line argument list
    """
    parser = argparse.ArgumentParser(
        description='Bootstrap RSA for wav2vec frame geometry and sonority.',
    )
    parser.add_argument('frames', help='input CSV or Parquet frame table')
    parser.add_argument('--n-syllables', type=int, required=True,
        help='number of syllables sampled per bootstrap')
    parser.add_argument('--n-bootstraps', type=int, required=True,
        help='number of bootstrap repetitions')
    parser.add_argument('--out', required=True,
        help='output directory for summary and bootstrap score CSV files')
    parser.add_argument('--random-state', type=int, default=None,
        help='optional integer random seed')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main()
