"""Run subset-sampled RSA from phraser/echoframe stores and log the run."""

import csv
import datetime
import json
import secrets
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from sonority_rsa.sampling import (compute_rsa_scores, make_rng,
    replay_sampled_keys, summarize_rsa_scores)
from sonority_rsa.fetch import fetch_syllable_data

SUMMARY_COLUMNS = ['run_id', 'layer', 'mean_rsa', 'ci_lower', 'ci_upper',
    'n_subsets', 'n_subsets_valid', 'subset_size']
SCORE_COLUMNS = ['run_id', 'layer', 'subset', 'rsa', 'invalid_reason']


def run_analysis(syllables, model_name, layers, echoframe_store,
        subset_size, n_subsets, collar=500, random_state=42, ci=0.95,
        verbose=False):
    """
    Fetch syllable populations per layer and run subset-sampled RSA.

    Returns (summary, scores, log): summary rows per layer, raw scores
    per layer, and a run log that makes every sampled subset replayable
    (see replay_sampled_keys and log_sampled_keys).

    A layer with no usable data (no stored embedding for any phrase, or
    every syllable skipped) is dropped: it is left out of summary and
    scores and recorded under the log's 'failed_layers' key with the
    reason. Each layer consumes exactly one seed draw whether or not it
    fails, so a dropped layer does not shift the seeds of the others.
    Raises ValueError only when every requested layer fails.

    This assumes the same usable syllable population is available for every
    requested layer. A subset-size validation error is therefore treated as
    a configuration error for the whole run rather than a failed layer.

    syllables: list of phraser Syllable objects with linked phones
    model_name: registered echoframe model name (e.g. 'wav2vec2')
    layers: iterable of hidden-state layers to analyze
    echoframe_store: echoframe Store holding the hidden states
    subset_size: number of syllables per subset
    n_subsets: number of subsets to draw
    collar: milliseconds of context stored around the phrase
    random_state: integer master seed (default 42), logged for replay
    ci: confidence level as a fraction, e.g. 0.95 for a 95% interval
    verbose: print each layer's skip report (off by default; the same
        counts are recorded per layer in the run log regardless)
    """
    layers = list(layers)
    seed = int(random_state)
    rng = make_rng(seed)
    scores, layer_logs, failed_layers = {}, {}, {}

    for layer in layers:
        layer_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        try:
            syllable_population = fetch_syllable_data(syllables, model_name,
                layer, echoframe_store, collar=collar, verbose=verbose)
        except ValueError as error:
            failed_layers[str(layer)] = {'seed': layer_seed,
                'reason': str(error)}
            warnings.warn(f'layer {layer} dropped: {error}', stacklevel=2)
            continue
        layer_scores, diagnostics = compute_rsa_scores(
            syllable_population, subset_size, n_subsets,
            random_state=layer_seed, return_diagnostics=True)
        if not np.isfinite(layer_scores).any():
            failed_layers[str(layer)] = {
                'seed': layer_seed,
                'reason': 'every sampled subset produced an invalid RSA '
                    'score',
                'invalid_subsets': diagnostics['invalid_subsets'],
                'invalid_reasons': diagnostics['invalid_reasons'],
            }
            warnings.warn(
                f'layer {layer} dropped: every sampled subset produced an '
                'invalid RSA score', stacklevel=2)
            continue
        scores[layer] = layer_scores
        layer_logs[str(layer)] = {
            'seed': layer_seed,
            'n_syllables_in_population': len(syllable_population),
            'skipped': syllable_population.skipped,
            'invalid_subsets': diagnostics['invalid_subsets'],
            'invalid_reasons': diagnostics['invalid_reasons'],
            'syllable_keys': [_key_to_text(key)
                for key in syllable_population.keys],
        }

    if not scores:
        raise ValueError('every requested layer failed to fetch: '
            f'{sorted(failed_layers)}')

    log = _build_log(model_name, layers, echoframe_store, subset_size,
        n_subsets, collar, seed, ci, layer_logs, failed_layers)
    summary = summarize_rsa_scores(scores, ci=ci)
    for row in summary:
        row['subset_size'] = subset_size
        row['run_id'] = log['run_id']
    return summary, scores, log


def save_analysis(summary, scores, log, out):
    """
    Save summary, raw RSA scores, and the run log to a directory.

    summary: summary rows from run_analysis
    scores: raw scores per layer from run_analysis
    log: run log from run_analysis
    out: output directory
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / 'summary.csv', SUMMARY_COLUMNS, summary)
    _write_csv(out / 'rsa_scores.csv', SCORE_COLUMNS,
        _score_rows(scores, log['run_id'], log['layers']))
    with open(out / 'run_log.json', 'w') as fout:
        json.dump(log, fout, indent=2)


def display_analysis(summary, scores, n=10):
    """
    Print the summary table and a preview of raw RSA scores.

    summary: summary rows from run_analysis
    scores: raw scores per layer from run_analysis
    n: number of raw scores to preview per layer
    """
    columns = [name for name in SUMMARY_COLUMNS if name != 'run_id']
    _print_table(columns, summary)
    print()
    for layer in sorted(scores):
        preview = ' '.join(f'{score:.3f}' for score in scores[layer][:n])
        print(f'layer {layer} rsa: {preview}')


def log_sampled_keys(log, layer):
    """
    Recompute the syllable keys drawn in each subset of a logged run.

    log: run log dict (or path to a run_log.json file)
    layer: layer to replay
    """
    if not isinstance(log, dict):
        with open(log) as fin:
            log = json.load(fin)
    entry = log['layers'][str(layer)]
    return replay_sampled_keys(entry['syllable_keys'], entry['seed'],
        log['parameters']['subset_size'], log['parameters']['n_subsets'])


def _build_log(model_name, layers, echoframe_store, subset_size,
        n_subsets, collar, seed, ci, layer_logs, failed_layers):
    """
    Assemble the run log dict.

    layer_logs: per-layer seed, population keys, and skip counts
    failed_layers: per-layer seed and reason for each dropped layer
    """
    now = datetime.datetime.now().astimezone()
    return {
        'run_id': f'{now:%Y-%m-%dT%H-%M-%S}_{secrets.token_hex(3)}',
        'created_at': now.isoformat(),
        'package': {'name': 'sonority-rsa', 'version': _package_version()},
        'parameters': {
            'model_name': model_name,
            'layers': list(layers),
            'collar': collar,
            'subset_size': subset_size,
            'n_subsets': n_subsets,
            'ci': ci,
            'seed': seed,
        },
        'echoframe_store': str(getattr(echoframe_store, 'root',
            echoframe_store)),
        'key_encoding': 'hex for bytes keys, text otherwise',
        'sampling': ('per layer: rng = default_rng(layer seed); one '
            'rng.choice(n_population, size=subset_size, replace=False) '
            'draw per subset over syllable_keys order'),
        'layers': layer_logs,
        'failed_layers': failed_layers,
    }


def _key_to_text(key):
    """
    Make a phraser key JSON-serializable.

    key: bytes LMDB key (stored as hex) or any other key (stored as str)
    """
    if isinstance(key, bytes):
        return key.hex()
    return str(key)


def _score_rows(scores, run_id, layer_logs):
    """
    Flatten per-layer scores into rsa_scores.csv rows.

    scores: raw scores per layer from run_analysis
    run_id: run identifier from the run log
    layer_logs: per-layer log entries, including invalid-score reasons
    """
    rows = []
    for layer in sorted(scores):
        reasons = layer_logs[str(layer)]['invalid_reasons']
        for subset, (rsa, invalid_reason) in enumerate(zip(scores[layer],
                reasons)):
            rows.append({'run_id': run_id, 'layer': layer,
                'subset': subset, 'rsa': rsa,
                'invalid_reason': invalid_reason or ''})
    return rows


def _write_csv(path, columns, rows):
    """
    Write dict rows to a CSV file.

    path: output path
    columns: column order
    rows: list of dicts with the given columns
    """
    with open(path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(columns, rows):
    """
    Print dict rows as an aligned plain-text table.

    columns: column order
    rows: list of dicts with the given columns
    """
    cells = [[_format_cell(row.get(name)) for name in columns]
        for row in rows]
    widths = [max(len(name), *(len(line[i]) for line in cells))
        if cells else len(name) for i, name in enumerate(columns)]
    print('  '.join(name.rjust(width)
        for name, width in zip(columns, widths)))
    for line in cells:
        print('  '.join(cell.rjust(width)
            for cell, width in zip(line, widths)))


def _format_cell(value):
    """
    Format one table cell.

    value: cell value (floats get four decimals)
    """
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _package_version():
    """Return the installed sonority-rsa version, or 'unknown'."""
    try:
        return version('sonority-rsa')
    except PackageNotFoundError:
        return 'unknown'
