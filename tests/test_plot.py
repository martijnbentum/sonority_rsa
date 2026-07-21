import csv

import matplotlib.axes
import matplotlib.pyplot
import pytest

from sonority_rsa.plot import (PANELS_PLOT_FILENAME, PLOT_FILENAME,
    plot_analyses, plot_analysis)


SUMMARY_ROWS = [
    {'layer': 2, 'mean_rsa': 0.3, 'ci_lower': 0.2, 'ci_upper': 0.4},
    {'layer': 1, 'mean_rsa': 0.2, 'ci_lower': 0.1, 'ci_upper': 0.3},
]
SCORE_ROWS = [
    {'layer': 1, 'rsa': 0.1},
    {'layer': 1, 'rsa': 0.3},
    {'layer': 2, 'rsa': 0.2},
    {'layer': 2, 'rsa': 0.4},
]


def write_csv(path, rows):
    with open(path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot_inputs(output_dir, summary_rows=SUMMARY_ROWS,
        score_rows=SCORE_ROWS):
    write_csv(output_dir / 'summary.csv', summary_rows)
    write_csv(output_dir / 'rsa_scores.csv', score_rows)


def test_plot_analysis_saves_sonority_plot(tmp_path):
    write_plot_inputs(tmp_path)

    output_path = plot_analysis(tmp_path, 'Sonority by layer',
        show_plot=False)

    assert output_path == tmp_path / PLOT_FILENAME
    assert output_path.read_bytes().startswith(b'\x89PNG')


def test_plot_analysis_can_save_pdf(tmp_path):
    write_plot_inputs(tmp_path)

    output_path = plot_analysis(tmp_path, 'Sonority by layer',
        filetype='.pdf', show_plot=False)

    assert output_path == tmp_path / 'rsa_by_layer.pdf'
    assert output_path.read_bytes().startswith(b'%PDF')


@pytest.mark.parametrize('filetype', ['svg', '', None])
def test_plot_analysis_rejects_unsupported_filetype(tmp_path, filetype):
    write_plot_inputs(tmp_path)

    with pytest.raises(ValueError, match=r'\.pdf, \.png'):
        plot_analysis(tmp_path, 'Sonority by layer', filetype=filetype,
            show_plot=False)


def test_plot_analysis_shows_interactively_by_default(tmp_path,
        monkeypatch):
    write_plot_inputs(tmp_path)
    calls = []
    monkeypatch.setattr(matplotlib.pyplot, 'ion',
        lambda: calls.append('ion'))
    monkeypatch.setattr(matplotlib.pyplot, 'show',
        lambda **kwargs: calls.append(('show', kwargs)))

    plot_analysis(tmp_path, 'Sonority by layer')

    assert calls == ['ion', ('show', {'block': False})]


def test_plot_analysis_can_suppress_interactive_display(tmp_path,
        monkeypatch):
    write_plot_inputs(tmp_path)
    monkeypatch.setattr(matplotlib.pyplot, 'ion',
        lambda: pytest.fail('ion should not be called'))
    monkeypatch.setattr(matplotlib.pyplot, 'show',
        lambda **kwargs: pytest.fail('show should not be called'))

    plot_analysis(tmp_path, 'Sonority by layer', show_plot=False)


def test_plot_analysis_uses_solid_zero_line(tmp_path, monkeypatch):
    write_plot_inputs(tmp_path)
    line_styles = []
    original_axhline = matplotlib.axes.Axes.axhline

    def record_axhline(self, *args, **kwargs):
        line_styles.append(kwargs.get('linestyle'))
        return original_axhline(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'axhline', record_axhline)

    plot_analysis(tmp_path, 'Sonority by layer', show_plot=False)

    assert line_styles == ['-']


def test_plot_analysis_computes_mean_and_ci_from_raw_scores(tmp_path,
        monkeypatch):
    score_rows = [SCORE_ROWS[2], SCORE_ROWS[0], SCORE_ROWS[3], SCORE_ROWS[1]]
    write_csv(tmp_path / 'rsa_scores.csv', score_rows)
    plotted_layers = []
    plotted_means = []
    plotted_intervals = []
    original_plot = matplotlib.axes.Axes.plot
    original_fill_between = matplotlib.axes.Axes.fill_between

    def record_plot(self, x, y, *args, **kwargs):
        if kwargs.get('label') == 'Sonority':
            plotted_layers.extend(x)
            plotted_means.extend(y)
        return original_plot(self, x, y, *args, **kwargs)

    def record_fill_between(self, x, lower, upper, *args, **kwargs):
        plotted_intervals.append((lower, upper))
        return original_fill_between(self, x, lower, upper, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', record_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, 'fill_between',
        record_fill_between)

    plot_analysis(tmp_path, 'Raw score statistics', show_plot=False)

    assert plotted_layers == [1, 2]
    assert plotted_means == pytest.approx([0.2, 0.3])
    assert plotted_intervals[0][0] == pytest.approx(
        [-1.07062, -0.97062], abs=1e-5)
    assert plotted_intervals[0][1] == pytest.approx(
        [1.47062, 1.57062], abs=1e-5)


def test_plot_analyses_saves_side_by_side_panels(tmp_path, monkeypatch):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    write_plot_inputs(first)
    write_plot_inputs(second)
    titles = []
    legend_calls = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_legend = matplotlib.axes.Axes.legend

    def record_title(self, title, *args, **kwargs):
        titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    def record_legend(self, *args, **kwargs):
        legend_calls.append((self, kwargs))
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title', record_title)
    monkeypatch.setattr(matplotlib.axes.Axes, 'legend', record_legend)

    output_path = plot_analyses([first, second], ['First', 'Second'],
        show_plot=False)

    assert output_path == tmp_path / PANELS_PLOT_FILENAME
    assert output_path.read_bytes().startswith(b'\x89PNG')
    assert titles == ['First', 'Second']
    assert len(legend_calls) == 1
    assert legend_calls[0][1] == {'loc': 'lower left'}


def test_plot_analyses_requires_one_title_per_directory(tmp_path):
    with pytest.raises(ValueError, match='same length'):
        plot_analyses([tmp_path, tmp_path], ['Only one'], show_plot=False)


def test_plot_analysis_uses_raw_scores_and_optional_series(tmp_path,
        monkeypatch):
    score_rows = [{
        **row,
        'intensity_rsa': 0.15,
        'random_baseline_rsa': 0.01,
    } for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)
    labels = []
    scatter_values = []
    original_plot = matplotlib.axes.Axes.plot
    original_scatter = matplotlib.axes.Axes.scatter

    def record_plot(self, *args, **kwargs):
        if kwargs.get('label'):
            labels.append(kwargs['label'])
        return original_plot(self, *args, **kwargs)

    def record_scatter(self, x, y, *args, **kwargs):
        scatter_values.extend(y)
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', record_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, 'scatter', record_scatter)

    plot_analysis(tmp_path, 'All RSA predictors', show_plot=False)

    assert labels == ['Sonority', 'Intensity', 'Random baseline']
    assert scatter_values == pytest.approx([
        0.1, 0.3, 0.2, 0.4,
        0.15, 0.15, 0.15, 0.15,
        0.01, 0.01, 0.01, 0.01,
    ])


def test_plot_analysis_uses_dashed_partial_rsa_series(tmp_path, monkeypatch):
    score_rows = [{
        **row,
        'intensity_rsa': 0.15,
        'sonority_partial_rsa': 0.18,
        'intensity_partial_rsa': 0.12,
    } for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)
    styles = {}
    original_plot = matplotlib.axes.Axes.plot

    def record_plot(self, *args, **kwargs):
        if kwargs.get('label'):
            styles[kwargs['label']] = kwargs['linestyle']
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', record_plot)

    plot_analysis(tmp_path, 'Partial RSA', show_plot=False)

    assert styles == {
        'Sonority': '-',
        'Sonority (controlling intensity)': '--',
        'Intensity': '-',
        'Intensity (controlling sonority)': '--',
    }


def test_plot_analysis_uses_optional_raw_series_without_summary_values(
        tmp_path):
    score_rows = [{**row, 'intensity_rsa': 0.1} for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)

    output_path = plot_analysis(tmp_path, 'Raw intensity', show_plot=False)

    assert output_path == tmp_path / PLOT_FILENAME


def test_plot_analysis_rejects_unpaired_partial_rsa_series(tmp_path):
    score_rows = [{**row, 'sonority_partial_rsa': 0.1}
        for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)

    with pytest.raises(ValueError, match='intensity_partial_rsa'):
        plot_analysis(tmp_path, 'Incomplete partial data', show_plot=False)
