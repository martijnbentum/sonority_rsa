"""Plot saved RSA scores and their per-layer summaries."""

import csv
import math
from pathlib import Path

PLOT_FILENAME = 'rsa_by_layer.png'
SUPPORTED_FILETYPES = {'png', 'pdf'}
SERIES = [
    {
        'label': 'Sonority',
        'score': 'rsa',
        'mean': 'mean_rsa',
        'lower': 'ci_lower',
        'upper': 'ci_upper',
        'color': 'C0',
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'label': 'Sonority (controlling intensity)',
        'score': 'sonority_partial_rsa',
        'mean': 'mean_sonority_partial_rsa',
        'lower': 'sonority_partial_rsa_ci_lower',
        'upper': 'sonority_partial_rsa_ci_upper',
        'color': 'C0',
        'linestyle': '--',
        'marker': 's',
    },
    {
        'label': 'Intensity',
        'score': 'intensity_rsa',
        'mean': 'mean_intensity_rsa',
        'lower': 'intensity_rsa_ci_lower',
        'upper': 'intensity_rsa_ci_upper',
        'color': 'C1',
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'label': 'Intensity (controlling sonority)',
        'score': 'intensity_partial_rsa',
        'mean': 'mean_intensity_partial_rsa',
        'lower': 'intensity_partial_rsa_ci_lower',
        'upper': 'intensity_partial_rsa_ci_upper',
        'color': 'C1',
        'linestyle': '--',
        'marker': 's',
    },
    {
        'label': 'Random baseline',
        'score': 'random_baseline_rsa',
        'mean': 'mean_random_baseline_rsa',
        'lower': 'random_baseline_ci_lower',
        'upper': 'random_baseline_ci_upper',
        'color': 'C2',
        'linestyle': '-',
        'marker': 'o',
    },
]
PARTIAL_SERIES = [SERIES[1], SERIES[3]]


def plot_analysis(output_dir, title, filetype='.png'):
    """
    Plot RSA by layer from a saved analysis directory.

    The subset-level values in ``rsa_scores.csv`` are shown as translucent
    points. The means and confidence intervals in ``summary.csv`` are shown
    as lines and shaded bands. Intensity and random-baseline RSA are added
    when their columns are available. Partial sonority and intensity RSA
    are shown as dashed lines when an intensity-enabled analysis includes
    them.

    output_dir: directory written by save_analysis
    title: plot title
    filetype: output format, ``.png`` or ``.pdf`` (default ``.png``)

    Returns the path to ``rsa_by_layer.<filetype>`` in output_dir.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    extension = _file_extension(filetype)
    summary_path = output_dir / 'summary.csv'
    scores_path = output_dir / 'rsa_scores.csv'
    summary_columns, summary_rows = _read_csv(summary_path)
    score_columns, score_rows = _read_csv(scores_path)
    active_series = _active_series(summary_columns, score_columns,
        summary_path, scores_path)
    layers = _summary_layers(summary_rows, summary_path)

    fig = Figure(figsize=(9, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    _plot_raw_scores(ax, layers, score_rows, active_series, scores_path)
    _plot_summaries(ax, layers, summary_rows, active_series, summary_path)
    ax.axhline(0, color='0.4', linewidth=0.8, linestyle='--')
    ax.set_title(str(title))
    ax.set_xlabel('Layer')
    ax.set_ylabel('RSA value')
    ax.set_xticks([layer['value'] for layer in layers],
        [layer['label'] for layer in layers])
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f'rsa_by_layer.{extension}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return output_path


def _file_extension(filetype):
    """Normalize and validate a requested plot file type."""
    extension = str(filetype).lower().removeprefix('.')
    if extension not in SUPPORTED_FILETYPES:
        supported = ', '.join(f'.{name}'
            for name in sorted(SUPPORTED_FILETYPES))
        raise ValueError(f'filetype must be one of: {supported}')
    return extension


def _read_csv(path):
    """Return the columns and rows from a CSV file."""
    with open(path, newline='') as fin:
        reader = csv.DictReader(fin)
        columns = set(reader.fieldnames or [])
        return columns, list(reader)


def _active_series(summary_columns, score_columns, summary_path,
        scores_path):
    """Validate required fields and select optional plot series."""
    partial_summary_fields = {field
        for series in PARTIAL_SERIES
        for field in (series['mean'], series['lower'], series['upper'])}
    partial_score_fields = {series['score'] for series in PARTIAL_SERIES}
    has_partial = (bool(partial_summary_fields & summary_columns)
        or bool(partial_score_fields & score_columns))
    if has_partial:
        _require_columns(summary_columns,
            {'layer'} | partial_summary_fields, summary_path)
        _require_columns(score_columns, {'layer'} | partial_score_fields,
            scores_path)

    active = []
    for index, series in enumerate(SERIES):
        summary_fields = {series['mean'], series['lower'], series['upper']}
        score_fields = {series['score']}
        is_present = (index == 0
            or bool(summary_fields & summary_columns)
            or bool(score_fields & score_columns))
        if not is_present:
            continue
        _require_columns(summary_columns, {'layer'} | summary_fields,
            summary_path)
        _require_columns(score_columns, {'layer'} | score_fields,
            scores_path)
        active.append(series)
    return active


def _require_columns(columns, required, path):
    """Raise when a plot input CSV lacks required columns."""
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f'{path} is missing required columns: '
            f'{", ".join(missing)}')


def _summary_layers(rows, path):
    """Read, validate, and numerically sort summary layer values."""
    if not rows:
        raise ValueError(f'{path} contains no data rows')
    layers = []
    seen = set()
    for row in rows:
        value = _float_value(row['layer'], 'layer', path)
        if value in seen:
            raise ValueError(f'{path} contains duplicate layer {row["layer"]}')
        seen.add(value)
        layers.append({'value': value, 'label': row['layer']})
    return sorted(layers, key=lambda layer: layer['value'])


def _plot_raw_scores(ax, layers, rows, active_series, path):
    """Plot finite per-subset RSA values as translucent points."""
    values_by_layer = {}
    for row in rows:
        layer = _float_value(row['layer'], 'layer', path)
        values_by_layer.setdefault(layer, []).append(row)

    offsets = _series_offsets(layers, len(active_series))
    for series, offset in zip(active_series, offsets):
        for layer in layers:
            values = [_optional_float(row[series['score']], series['score'],
                path) for row in values_by_layer.get(layer['value'], [])]
            values = [value for value in values if value is not None]
            if values:
                ax.scatter([layer['value'] + offset] * len(values), values,
                    color=series['color'], alpha=0.18, s=16,
                    edgecolors='none')


def _plot_summaries(ax, layers, rows, active_series, path):
    """Plot per-layer means and confidence intervals."""
    rows_by_layer = {
        _float_value(row['layer'], 'layer', path): row
        for row in rows
    }
    x_values = [layer['value'] for layer in layers]
    for series in active_series:
        means = [_number_value(rows_by_layer[layer][series['mean']],
            series['mean'], path) for layer in x_values]
        lower = [_number_value(rows_by_layer[layer][series['lower']],
            series['lower'], path) for layer in x_values]
        upper = [_number_value(rows_by_layer[layer][series['upper']],
            series['upper'], path) for layer in x_values]
        ax.fill_between(x_values, lower, upper, color=series['color'],
            alpha=0.14)
        ax.plot(x_values, means, color=series['color'],
            linestyle=series['linestyle'], marker=series['marker'],
            linewidth=2, label=series['label'])


def _series_offsets(layers, n_series):
    """Return small, centered horizontal offsets for raw score points."""
    if n_series == 1:
        return [0]
    values = [layer['value'] for layer in layers]
    gaps = [right - left for left, right in zip(values, values[1:])
        if right > left]
    spacing = min(gaps) if gaps else 1
    step = spacing * 0.04
    midpoint = (n_series - 1) / 2
    return [(index - midpoint) * step for index in range(n_series)]


def _float_value(value, column, path):
    """Parse one required finite float from a CSV cell."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path} has invalid {column} value: {value!r}') \
            from error
    if not math.isfinite(number):
        raise ValueError(f'{path} has non-finite {column} value: {value!r}')
    return number


def _number_value(value, column, path):
    """Parse one required numeric CSV cell, including a legitimate NaN."""
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path} has invalid {column} value: {value!r}') \
            from error


def _optional_float(value, column, path):
    """Parse an optional float, omitting blank and non-finite score cells."""
    if value in (None, ''):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path} has invalid {column} value: {value!r}') \
            from error
    return number if math.isfinite(number) else None
