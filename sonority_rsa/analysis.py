"""Run subset-sampled RSA from phraser/echoframe stores and log the run."""

import csv
import datetime
import json
import secrets
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from sonority_rsa.intensity import (IntensityCache,
    add_centered_intensities)
from sonority_rsa.sampling import (_compute_rsa_results,
    _summarize_rsa_scores, _validate_ci, make_rng, replay_sampled_keys,
    summarize_rsa_scores)
from sonority_rsa.fetch import fetch_syllable_data

SUMMARY_COLUMNS = ['run_id', 'layer', 'mean_rsa', 'ci_lower', 'ci_upper',
    'n_subsets', 'n_subsets_valid', 'subset_size']
RANDOM_BASELINE_SUMMARY_COLUMNS = ['mean_random_baseline_rsa',
    'random_baseline_ci_lower', 'random_baseline_ci_upper',
    'mean_rsa_difference', 'rsa_difference_ci_lower',
    'rsa_difference_ci_upper']
SCORE_COLUMNS = ['run_id', 'layer', 'subset', 'rsa', 'invalid_reason']
RANDOM_BASELINE_SCORE_COLUMNS = ['random_baseline_rsa', 'rsa_difference']
INTENSITY_RESULT_NAMES = ['intensity_rsa',
    'sonority_intensity_correlation',
    'sonority_intensity_rdm_correlation',
    'sonority_partial_rsa',
    'intensity_partial_rsa']
INTENSITY_SUMMARY_COLUMNS = [column
    for name in INTENSITY_RESULT_NAMES
    for column in [f'mean_{name}', f'{name}_ci_lower', f'{name}_ci_upper']]
INTENSITY_SUMMARY_COLUMNS.append('n_subsets_partial_valid')
INTENSITY_SCORE_COLUMNS = INTENSITY_RESULT_NAMES + [
    'intensity_invalid_reason', 'partial_invalid_reason']
RANDOM_BASELINE_SALT = 42


def run_analysis(syllables, model_name, layers, echoframe_store,
        subset_size, n_subsets, collar=500, random_state=42, ci=0.95,
        verbose=False, compute_random_baseline=False,
        compute_intensity_baseline=False):
    """
    Fetch syllable populations per layer and run subset-sampled RSA.

    Returns (summary, results, log): summary rows per layer, raw results
    grouped first by metric and then by layer, and a run log that makes
    every sampled subset replayable (see replay_sampled_keys and
    log_sampled_keys). results always contains 'rsa'; when
    compute_random_baseline is true it also contains
    'random_baseline_rsa', with one shuffled-label score paired to each
    observed score from the same subset. When compute_intensity_baseline is
    true, results also contains intensity RSA and direct/RDM correlations
    between sonority and centered intensity for each subset, plus partial
    sonority and intensity RSA scores controlling for the other predictor.

    A layer with no usable data (no stored embedding for any phrase, or
    every syllable skipped) is dropped: it is left out of summary and
    results and recorded under the log's 'failed_layers' key with the
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
    compute_random_baseline: compute one shuffled-sonority RSA score for
        every sampled subset (off by default)
    compute_intensity_baseline: compute centered-intensity RSA, both
        sonority/intensity correlations, and both partial RSA directions
        for every subset (off by default)
    """
    layers = list(layers)
    _validate_ci(ci, warn=False)
    seed = int(random_state)
    rng = make_rng(seed)
    results = {'rsa': {}}
    if compute_random_baseline:
        results['random_baseline_rsa'] = {}
    if compute_intensity_baseline:
        results.update({name: {} for name in INTENSITY_RESULT_NAMES})
        intensity_cache = IntensityCache()
    else:
        intensity_cache = None
    layer_logs, failed_layers = {}, {}

    for layer in layers:
        layer_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        baseline_seed = (_random_baseline_seed(layer_seed)
            if compute_random_baseline else None)
        try:
            syllable_population = fetch_syllable_data(syllables, model_name,
                layer, echoframe_store, collar=collar, verbose=verbose)
        except ValueError as error:
            failed_layers[str(layer)] = {'seed': layer_seed,
                'reason': str(error)}
            if baseline_seed is not None:
                failed_layers[str(layer)]['random_baseline_seed'] = (
                    baseline_seed)
            warnings.warn(f'layer {layer} dropped: {error}', stacklevel=2)
            continue
        if compute_intensity_baseline:
            add_centered_intensities(syllable_population, intensity_cache)
        layer_results, diagnostics = _compute_rsa_results(
            syllable_population, subset_size, n_subsets,
            random_state=layer_seed, random_baseline_state=baseline_seed,
            compute_intensity_baseline=compute_intensity_baseline)
        if not np.isfinite(layer_results['rsa']).any():
            failed_layers[str(layer)] = {
                'seed': layer_seed,
                'reason': 'every sampled subset produced an invalid RSA '
                    'score',
                'invalid_subsets': diagnostics['invalid_subsets'],
                'invalid_reasons': diagnostics['invalid_reasons'],
            }
            if baseline_seed is not None:
                failed_layers[str(layer)]['random_baseline_seed'] = (
                    baseline_seed)
            warnings.warn(
                f'layer {layer} dropped: every sampled subset produced an '
                'invalid RSA score', stacklevel=2)
            continue
        for metric, metric_scores in layer_results.items():
            results[metric][layer] = metric_scores
        layer_logs[str(layer)] = {
            'seed': layer_seed,
            'n_syllables_in_population': len(syllable_population),
            'skipped': syllable_population.skipped,
            'invalid_subsets': diagnostics['invalid_subsets'],
            'invalid_reasons': diagnostics['invalid_reasons'],
            'syllable_keys': [_key_to_text(key)
                for key in syllable_population.keys],
        }
        if baseline_seed is not None:
            layer_logs[str(layer)]['random_baseline_seed'] = baseline_seed
        if compute_intensity_baseline:
            layer_logs[str(layer)].update({
                'intensity_invalid_subsets':
                    diagnostics['intensity_invalid_subsets'],
                'intensity_invalid_reasons':
                    diagnostics['intensity_invalid_reasons'],
                'partial_invalid_subsets':
                    diagnostics['partial_invalid_subsets'],
                'partial_invalid_reasons':
                    diagnostics['partial_invalid_reasons'],
            })

    if not results['rsa']:
        raise ValueError(f'every requested layer failed: '
            f'{sorted(failed_layers)}')

    log = _build_log(model_name, layers, echoframe_store, subset_size,
        n_subsets, collar, seed, ci, compute_random_baseline,
        compute_intensity_baseline, layer_logs, failed_layers)
    summary = summarize_rsa_scores(results['rsa'], ci=ci)
    if compute_random_baseline:
        _add_random_baseline_summary(summary, results, ci)
    if compute_intensity_baseline:
        _add_intensity_summary(summary, results, ci)
    for row in summary:
        row['subset_size'] = subset_size
        row['run_id'] = log['run_id']
    return summary, results, log


def save_analysis(summary, results, log, out):
    """
    Save summary, raw per-subset results, and the run log to a directory.

    summary: summary rows from run_analysis
    results: raw results by metric and layer from run_analysis
    log: run log from run_analysis
    out: output directory
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    has_baseline = 'random_baseline_rsa' in results
    has_intensity = 'intensity_rsa' in results
    _write_csv(out / 'summary.csv', _summary_columns(has_baseline,
        has_intensity), summary)
    _write_csv(out / 'rsa_scores.csv', _score_columns(has_baseline,
        has_intensity),
        _score_rows(results, log['run_id'], log['layers']))
    with open(out / 'run_log.json', 'w') as fout:
        json.dump(log, fout, indent=2)


def display_analysis(summary, results, n=10):
    """
    Print the summary table and a preview of raw per-subset results.

    summary: summary rows from run_analysis
    results: raw results by metric and layer from run_analysis
    n: number of raw scores to preview per layer
    """
    columns = [name for name in _summary_columns(
        'random_baseline_rsa' in results, 'intensity_rsa' in results)
        if name != 'run_id']
    _print_table(columns, summary)
    print()
    for metric, scores_by_layer in results.items():
        for layer in sorted(scores_by_layer):
            preview = ' '.join(f'{score:.3f}'
                for score in scores_by_layer[layer][:n])
            print(f'layer {layer} {metric}: {preview}')


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
        n_subsets, collar, seed, ci, compute_random_baseline,
        compute_intensity_baseline, layer_logs, failed_layers):
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
            'compute_random_baseline': compute_random_baseline,
            'compute_intensity_baseline': compute_intensity_baseline,
        },
        'echoframe_store': str(getattr(echoframe_store, 'root',
            echoframe_store)),
        'key_encoding': 'hex for bytes keys, text otherwise',
        'sampling': ('per layer: rng = default_rng(layer seed); one '
            'rng.choice(n_population, size=subset_size, replace=False) '
            'draw per subset over syllable_keys order'),
        'random_baseline': (
            'per subset: shuffle sampled sonority once with a separate rng; '
            'its seed is derived from SeedSequence([layer seed, 42]) so '
            'shuffling does not change subset sampling or replay'
            if compute_random_baseline else None),
        'intensity_baseline': (
            'per phone: 10 * log10(mean(signal ** 2) / 4e-10) over the '
            'complete phone interval; values are centered by subtracting '
            'the mean of usable phones from the same audio recording; per '
            'subset, intensity RSA and direct/RDM Spearman correlations '
            'with sonority reuse the observed model RDM'
            if compute_intensity_baseline else None),
        'partial_rsa': (
            'per subset: average-rank the upper triangles of the model, '
            'sonority, and intensity RDMs; residualize model and target '
            'ranks separately against control ranks with an intercept, '
            'then correlate the residuals; both directions are invalidated '
            'when either direction is undefined'
            if compute_intensity_baseline else None),
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


def _random_baseline_seed(layer_seed):
    """Derive a baseline seed without advancing the sampling generator."""
    seed_sequence = np.random.SeedSequence(
        [layer_seed, RANDOM_BASELINE_SALT])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _add_random_baseline_summary(summary, results, ci):
    """Add baseline and paired-difference statistics to summary rows."""
    baseline_rows = {
        row['layer']: row
        for row in _summarize_rsa_scores(
            results['random_baseline_rsa'], ci)
    }
    differences = {
        layer: (np.asarray(results['rsa'][layer], dtype=float)
            - np.asarray(results['random_baseline_rsa'][layer],
                dtype=float))
        for layer in results['rsa']
    }
    difference_rows = {
        row['layer']: row
        for row in _summarize_rsa_scores(differences, ci)
    }

    for row in summary:
        baseline = baseline_rows[row['layer']]
        difference = difference_rows[row['layer']]
        row.update({
            'mean_random_baseline_rsa': baseline['mean_rsa'],
            'random_baseline_ci_lower': baseline['ci_lower'],
            'random_baseline_ci_upper': baseline['ci_upper'],
            'mean_rsa_difference': difference['mean_rsa'],
            'rsa_difference_ci_lower': difference['ci_lower'],
            'rsa_difference_ci_upper': difference['ci_upper'],
        })


def _add_intensity_summary(summary, results, ci):
    """Add intensity RSA and correlation statistics to summary rows."""
    rows_by_metric = {
        metric: {
            row['layer']: row
            for row in _summarize_rsa_scores(results[metric], ci)
        }
        for metric in INTENSITY_RESULT_NAMES
    }
    for row in summary:
        layer = row['layer']
        for metric, metric_rows in rows_by_metric.items():
            metric_row = metric_rows[layer]
            row.update({
                f'mean_{metric}': metric_row['mean_rsa'],
                f'{metric}_ci_lower': metric_row['ci_lower'],
                f'{metric}_ci_upper': metric_row['ci_upper'],
            })
        row['n_subsets_partial_valid'] = rows_by_metric[
            'sonority_partial_rsa'][layer]['n_subsets_valid']


def _summary_columns(has_baseline, has_intensity):
    """Return summary columns, adding baseline fields when requested."""
    predictor_columns = []
    if has_baseline:
        predictor_columns.extend(RANDOM_BASELINE_SUMMARY_COLUMNS)
    if has_intensity:
        predictor_columns.extend(INTENSITY_SUMMARY_COLUMNS)
    return SUMMARY_COLUMNS[:5] + predictor_columns + SUMMARY_COLUMNS[5:]


def _score_columns(has_baseline, has_intensity):
    """Return raw-score columns, adding baseline fields when requested."""
    predictor_columns = []
    if has_baseline:
        predictor_columns.extend(RANDOM_BASELINE_SCORE_COLUMNS)
    if has_intensity:
        predictor_columns.extend(INTENSITY_SCORE_COLUMNS)
    return SCORE_COLUMNS[:4] + predictor_columns + SCORE_COLUMNS[4:]


def _score_rows(results, run_id, layer_logs):
    """
    Flatten per-layer scores into rsa_scores.csv rows.

    results: raw results by metric and layer from run_analysis
    run_id: run identifier from the run log
    layer_logs: per-layer log entries, including invalid-score reasons
    """
    rows = []
    scores = results['rsa']
    baseline_scores = results.get('random_baseline_rsa')
    intensity_scores = results.get('intensity_rsa')
    for layer in sorted(scores):
        reasons = layer_logs[str(layer)]['invalid_reasons']
        intensity_reasons = layer_logs[str(layer)].get(
            'intensity_invalid_reasons')
        partial_reasons = layer_logs[str(layer)].get(
            'partial_invalid_reasons')
        for subset, (rsa, invalid_reason) in enumerate(zip(scores[layer],
                reasons)):
            row = {'run_id': run_id, 'layer': layer,
                'subset': subset, 'rsa': rsa,
                'invalid_reason': invalid_reason or ''}
            if baseline_scores is not None:
                baseline = baseline_scores[layer][subset]
                row['random_baseline_rsa'] = baseline
                row['rsa_difference'] = rsa - baseline
            if intensity_scores is not None:
                for metric in INTENSITY_RESULT_NAMES:
                    row[metric] = results[metric][layer][subset]
                row['intensity_invalid_reason'] = (
                    intensity_reasons[subset] or '')
                row['partial_invalid_reason'] = (
                    partial_reasons[subset] or '')
            rows.append(row)
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
