"""Plot saved RSA scores and their per-layer statistics."""

import csv
import math
from pathlib import Path

from sonority_rsa.util_stats import score_statistics

PLOT_FILENAME = 'rsa_by_layer.png'
PANELS_PLOT_FILENAME = 'rsa_by_layer_panels.png'
SUPPORTED_FILETYPES = {'png', 'pdf'}
SERIES = [
    {
        'label': 'Sonority',
        'score': 'rsa',
        'color': 'C0',
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'label': 'Sonority (controlling intensity)',
        'score': 'sonority_partial_rsa',
        'color': 'C0',
        'linestyle': '--',
        'marker': 's',
    },
    {
        'label': 'Intensity',
        'score': 'intensity_rsa',
        'color': 'C1',
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'label': 'Intensity (controlling sonority)',
        'score': 'intensity_partial_rsa',
        'color': 'C1',
        'linestyle': '--',
        'marker': 's',
    },
    {
        'label': 'Random baseline',
        'score': 'random_baseline_rsa',
        'color': 'C2',
        'linestyle': '-',
        'marker': 'o',
    },
]
PARTIAL_SERIES = [SERIES[1], SERIES[3]]


def plot_analysis(output_dir, title, filetype='.png', show_plot=True,
        confidence=0.95):
    """
    Plot RSA by layer from a saved analysis directory.

    The finite subset-level values in ``rsa_scores.csv`` are shown as
    translucent points and used to compute the mean lines and shaded
    Student's t confidence intervals around those means. Intensity and random-
    baseline RSA are added when their raw-score columns are available. Partial
    sonority and intensity RSA are shown as dashed lines when an intensity-
    enabled analysis includes them.

    output_dir: directory written by save_analysis
    title: plot title
    filetype: output format, ``.png`` or ``.pdf`` (default ``.png``)
    show_plot: enable Matplotlib interactive mode and show the figure
    confidence: confidence level for Student's t intervals around means

    Returns the path to ``rsa_by_layer.<filetype>`` in output_dir.
    """
    from matplotlib import pyplot as plt

    output_dir = Path(output_dir)
    extension = _file_extension(filetype)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        _plot_analysis_on_ax(ax, output_dir, title, confidence=confidence)
        fig.tight_layout()
        output_path = output_dir / f'rsa_by_layer.{extension}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception:
        plt.close(fig)
        raise
    _show_figure(plt, fig, show_plot)
    return output_path


def plot_analyses(output_dirs, titles, filetype='.png', show_plot=True,
        confidence=0.95):
    """
    Plot saved analyses as panels in one side-by-side figure.

    output_dirs: directories written by save_analysis, one per panel
    titles: panel titles in the same order as output_dirs
    filetype: shared output format, ``.png`` or ``.pdf`` (default ``.png``)
    show_plot: enable Matplotlib interactive mode and show the figure
    confidence: confidence level for Student's t intervals around means

    The combined figure is saved as ``rsa_by_layer_panels.<filetype>`` in
    the parent directory of the first analysis directory. Returns that path.
    """
    from matplotlib import pyplot as plt

    output_dirs = _panel_values(output_dirs, 'output_dirs')
    titles = _panel_values(titles, 'titles')
    if len(output_dirs) != len(titles):
        raise ValueError('output_dirs and titles must have the same length')

    extension = _file_extension(filetype)
    figure_width = max(6 * len(output_dirs), 9)
    fig, axes = plt.subplots(1, len(output_dirs),
        figsize=(figure_width, 5.5), squeeze=False, sharey=True)
    try:
        for index, (ax, output_dir, title) in enumerate(
                zip(axes[0], output_dirs, titles)):
            _plot_analysis_on_ax(ax, Path(output_dir), title,
                show_legend=index == 0,
                show_ylabel=index == 0,
                legend_location='lower left' if index == 0 else None,
                confidence=confidence)
        fig.tight_layout()
        output_path = (Path(output_dirs[0]).parent
            / f'rsa_by_layer_panels.{extension}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception:
        plt.close(fig)
        raise
    _show_figure(plt, fig, show_plot)
    return output_path


def _plot_analysis_on_ax(ax, output_dir, title, show_legend=True,
        show_ylabel=True, legend_location=None, confidence=0.95):
    """Plot one saved analysis on an existing Matplotlib axes."""
    scores_path = output_dir / 'rsa_scores.csv'
    score_columns, score_rows = _read_csv(scores_path)
    active_series = _active_series(score_columns, scores_path)
    layers = _layers_from_score_rows(score_rows, scores_path)
    values_by_series = _series_values_by_layer(score_rows, active_series,
        scores_path)

    _plot_raw_scores(ax, layers, active_series, values_by_series)
    _plot_mean_confidence_intervals(ax, layers, active_series,
        values_by_series, confidence)
    ax.axhline(0, color='0.4', linewidth=0.8, linestyle='-')
    ax.set_title(str(title))
    ax.set_xlabel('Layer')
    if show_ylabel:
        ax.set_ylabel('ρ', rotation=-90)
    ax.set_xticks([layer['value'] for layer in layers],
        [layer['label'] for layer in layers])
    ax.grid(axis='y', alpha=0.25)
    if show_legend:
        legend_options = {'handlelength': 3.5}
        if legend_location:
            legend_options['loc'] = legend_location
        ax.legend(**legend_options)


def _panel_values(values, name):
    """Return a non-empty panel argument list and reject scalar strings."""
    if isinstance(values, (str, Path)):
        raise TypeError(f'{name} must be a sequence, not a single value')
    values = list(values)
    if not values:
        raise ValueError(f'{name} must contain at least one value')
    return values


def _show_figure(plt, fig, show_plot):
    """Show a figure interactively, or close it when display is suppressed."""
    if show_plot:
        plt.ion()
        plt.show(block=False)
    else:
        plt.close(fig)


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


def _active_series(score_columns, scores_path):
    """Validate raw-score fields and select optional plot series."""
    partial_score_fields = {series['score'] for series in PARTIAL_SERIES}
    has_partial = bool(partial_score_fields & score_columns)
    if has_partial:
        _require_columns(score_columns, {'layer'} | partial_score_fields,
            scores_path)

    active = []
    for index, series in enumerate(SERIES):
        is_present = index == 0 or series['score'] in score_columns
        if not is_present:
            continue
        _require_columns(score_columns, {'layer', series['score']},
            scores_path)
        active.append(series)
    return active


def _require_columns(columns, required, path):
    """Raise when a plot input CSV lacks required columns."""
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f'{path} is missing required columns: '
            f'{", ".join(missing)}')


def _layers_from_score_rows(rows, path):
    """Return the unique, numerically sorted layers in raw score rows."""
    if not rows:
        raise ValueError(f'{path} contains no data rows')
    labels_by_value = {}
    for row in rows:
        value = _float_value(row['layer'], 'layer', path)
        labels_by_value.setdefault(value, row['layer'])
    layers = [{'value': value, 'label': label}
        for value, label in labels_by_value.items()]
    return sorted(layers, key=lambda layer: layer['value'])


def _series_values_by_layer(rows, active_series, path):
    """Parse finite raw scores into a per-series, per-layer mapping."""
    values_by_series = {series['score']: {} for series in active_series}
    for row in rows:
        layer = _float_value(row['layer'], 'layer', path)
        for series in active_series:
            score = series['score']
            value = _optional_float(row[score], score, path)
            if value is not None:
                values_by_series[score].setdefault(layer, []).append(value)
    return values_by_series


def _plot_raw_scores(ax, layers, active_series, values_by_series):
    """Plot finite per-subset RSA values as translucent points."""
    offsets = _series_offsets(layers, len(active_series))
    for series, offset in zip(active_series, offsets):
        for layer in layers:
            values = values_by_series[series['score']].get(
                layer['value'], [])
            if values:
                ax.scatter([layer['value'] + offset] * len(values), values,
                    color=series['color'], alpha=0.18, s=16,
                    edgecolors='none')


def _plot_mean_confidence_intervals(ax, layers, active_series,
        values_by_series, confidence):
    """Plot raw-score means and confidence intervals around the means."""
    x_values = [layer['value'] for layer in layers]
    for series in active_series:
        statistics = [score_statistics(
            values_by_series[series['score']].get(layer, []),
            confidence=confidence) for layer in x_values]
        means = [statistic['mean'] for statistic in statistics]
        lower = [statistic['mean_ci_lower'] for statistic in statistics]
        upper = [statistic['mean_ci_upper'] for statistic in statistics]
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
