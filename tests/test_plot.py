import csv

import matplotlib.axes
import pytest

from sonority_rsa.plot import PLOT_FILENAME, plot_analysis


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

    output_path = plot_analysis(tmp_path, 'Sonority by layer')

    assert output_path == tmp_path / PLOT_FILENAME
    assert output_path.read_bytes().startswith(b'\x89PNG')


def test_plot_analysis_can_save_pdf(tmp_path):
    write_plot_inputs(tmp_path)

    output_path = plot_analysis(tmp_path, 'Sonority by layer',
        filetype='.pdf')

    assert output_path == tmp_path / 'rsa_by_layer.pdf'
    assert output_path.read_bytes().startswith(b'%PDF')


@pytest.mark.parametrize('filetype', ['svg', '', None])
def test_plot_analysis_rejects_unsupported_filetype(tmp_path, filetype):
    write_plot_inputs(tmp_path)

    with pytest.raises(ValueError, match=r'\.pdf, \.png'):
        plot_analysis(tmp_path, 'Sonority by layer', filetype=filetype)


def test_plot_analysis_uses_raw_scores_summary_and_optional_series(tmp_path,
        monkeypatch):
    summary_rows = [{
        **row,
        'mean_intensity_rsa': 0.15,
        'intensity_rsa_ci_lower': 0.05,
        'intensity_rsa_ci_upper': 0.25,
        'mean_random_baseline_rsa': 0.01,
        'random_baseline_ci_lower': -0.05,
        'random_baseline_ci_upper': 0.05,
    } for row in SUMMARY_ROWS]
    score_rows = [{
        **row,
        'intensity_rsa': 0.15,
        'random_baseline_rsa': 0.01,
    } for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, summary_rows, score_rows)
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

    plot_analysis(tmp_path, 'All RSA predictors')

    assert labels == ['Sonority', 'Intensity', 'Random baseline']
    assert scatter_values == pytest.approx([
        0.1, 0.3, 0.2, 0.4,
        0.15, 0.15, 0.15, 0.15,
        0.01, 0.01, 0.01, 0.01,
    ])


def test_plot_analysis_uses_dashed_partial_rsa_series(tmp_path, monkeypatch):
    summary_rows = [{
        **row,
        'mean_intensity_rsa': 0.15,
        'intensity_rsa_ci_lower': 0.05,
        'intensity_rsa_ci_upper': 0.25,
        'mean_sonority_partial_rsa': 0.18,
        'sonority_partial_rsa_ci_lower': 0.08,
        'sonority_partial_rsa_ci_upper': 0.28,
        'mean_intensity_partial_rsa': 0.12,
        'intensity_partial_rsa_ci_lower': 0.02,
        'intensity_partial_rsa_ci_upper': 0.22,
    } for row in SUMMARY_ROWS]
    score_rows = [{
        **row,
        'intensity_rsa': 0.15,
        'sonority_partial_rsa': 0.18,
        'intensity_partial_rsa': 0.12,
    } for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, summary_rows, score_rows)
    styles = {}
    original_plot = matplotlib.axes.Axes.plot

    def record_plot(self, *args, **kwargs):
        if kwargs.get('label'):
            styles[kwargs['label']] = kwargs['linestyle']
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', record_plot)

    plot_analysis(tmp_path, 'Partial RSA')

    assert styles == {
        'Sonority': '-',
        'Sonority (controlling intensity)': '--',
        'Intensity': '-',
        'Intensity (controlling sonority)': '--',
    }


def test_plot_analysis_rejects_incomplete_optional_series(tmp_path):
    score_rows = [{**row, 'intensity_rsa': 0.1} for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)

    with pytest.raises(ValueError, match='mean_intensity_rsa'):
        plot_analysis(tmp_path, 'Incomplete data')


def test_plot_analysis_rejects_unpaired_partial_rsa_series(tmp_path):
    score_rows = [{**row, 'sonority_partial_rsa': 0.1}
        for row in SCORE_ROWS]
    write_plot_inputs(tmp_path, score_rows=score_rows)

    with pytest.raises(ValueError, match='mean_intensity_partial_rsa'):
        plot_analysis(tmp_path, 'Incomplete partial data')
