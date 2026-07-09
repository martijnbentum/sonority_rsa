import pandas as pd

from sonority_rsa.analysis import run_analysis, save_analysis


def test_run_analysis_returns_summary_and_scores():
    summary, scores = run_analysis(
        'examples/toy_frames.csv',
        n_syllables=3,
        n_bootstraps=2,
        random_state=1,
    )

    assert sorted(summary['layer'].tolist()) == [0, 1]
    assert summary['n_syllables'].tolist() == [3, 3]
    assert len(scores) == 4


def test_save_analysis_writes_expected_files(tmp_path):
    summary = pd.DataFrame([{
        'layer': 0,
        'mean_rsa': 0.1,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'n_bootstraps': 2,
        'n_syllables': 3,
    }])
    scores = pd.DataFrame([{'layer': 0, 'bootstrap': 0, 'rsa': 0.1}])

    save_analysis(summary, scores, tmp_path)

    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'bootstrap_scores.csv').exists()
