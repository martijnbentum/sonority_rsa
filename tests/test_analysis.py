import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import LAYERS, MODEL_NAME

import sonority_rsa.intensity as intensity
from sonority_rsa.analysis import (display_analysis, log_sampled_keys,
    run_analysis, save_analysis)
from sonority_rsa.fetch import SyllableData, SyllablePopulation
from sonority_rsa.intensity import IntensityComputationError


def run_toy_analysis(corpus, **kwargs):
    settings = dict(model_name=MODEL_NAME, layers=LAYERS,
        echoframe_store=corpus.echoframe_store, subset_size=30,
        n_subsets=4, random_state=1)
    settings.update(kwargs)
    return run_analysis(corpus.analysis_syllables, **settings)


def test_run_analysis_returns_summary_results_and_log(corpus):
    summary, results, log = run_toy_analysis(corpus)

    assert [row['layer'] for row in summary] == LAYERS
    assert all(row['subset_size'] == 30 for row in summary)
    assert all(row['run_id'] == log['run_id'] for row in summary)
    assert list(results) == ['rsa']
    assert sorted(results['rsa']) == LAYERS
    assert all(len(layer_scores) == 4
        for layer_scores in results['rsa'].values())


def test_run_analysis_logs_population_and_seeds(corpus):
    summary, results, log = run_toy_analysis(corpus)

    expected_keys = [syllable.key.hex()
        for syllable in corpus.analysis_syllables]
    assert log['parameters']['seed'] == 1
    assert log['parameters']['model_name'] == MODEL_NAME
    assert log['echoframe_store'] == str(corpus.echoframe_store.root)
    for layer in LAYERS:
        entry = log['layers'][str(layer)]
        assert entry['syllable_keys'] == expected_keys
        assert entry['n_syllables_in_population'] == 90
        assert entry['invalid_subsets'] == {
            'single_sonority_class': 0,
            'undefined_vector_distance': 0,
        }
        assert entry['invalid_reasons'] == [None] * 4
        assert isinstance(entry['seed'], int)


def test_run_analysis_logs_generator_layers(corpus):
    _, _, log = run_toy_analysis(corpus, layers=(layer for layer in LAYERS))

    assert log['parameters']['layers'] == LAYERS


def test_run_analysis_drops_layer_without_data(corpus):
    # the corpus stores layers 0 and 1 only, so layer 9 has no embeddings
    with pytest.warns(UserWarning, match='layer 9 dropped'):
        summary, results, log = run_toy_analysis(corpus, layers=[0, 9])

    assert list(results['rsa']) == [0]
    assert [row['layer'] for row in summary] == [0]
    assert list(log['layers']) == ['0']
    assert '0' not in log['failed_layers']
    assert 'seed' in log['failed_layers']['9']
    assert log['failed_layers']['9']['reason']


def test_run_analysis_drops_layer_with_only_invalid_scores(corpus,
        monkeypatch):
    population = SyllablePopulation(MODEL_NAME, 0, 500, [
        SyllableData(f's{i}', ['p', 'a'], [3, 3], [[1.0, 0.0], [0.0, 1.0]])
        for i in range(30)
    ], skipped={})
    monkeypatch.setattr('sonority_rsa.analysis.fetch_syllable_data',
        lambda *args, **kwargs: population)

    with pytest.warns(UserWarning) as warnings_issued:
        with pytest.raises(ValueError, match='every requested layer failed'):
            run_toy_analysis(corpus, layers=[0], n_subsets=1)

    assert any('every sampled subset' in str(warning.message)
        for warning in warnings_issued)


def test_run_analysis_raises_when_every_layer_fails(corpus):
    with pytest.raises(ValueError, match='every requested layer failed'):
        run_toy_analysis(corpus, layers=[8, 9])


def test_run_analysis_rejects_invalid_confidence_level_before_fetch(corpus,
        monkeypatch):
    monkeypatch.setattr('sonority_rsa.analysis.fetch_syllable_data',
        lambda *args, **kwargs: pytest.fail('fetch should not be called'))

    with pytest.raises(ValueError, match='greater than 0 and less than 1'):
        run_toy_analysis(corpus, ci=1)


def test_run_analysis_defaults_seed_to_42(corpus):
    settings = dict(model_name=MODEL_NAME, layers=LAYERS,
        echoframe_store=corpus.echoframe_store, subset_size=30, n_subsets=4)
    _, _, log = run_analysis(corpus.analysis_syllables, **settings)

    assert log['parameters']['seed'] == 42


def test_run_analysis_is_deterministic_for_a_seed(corpus):
    _, first, _ = run_toy_analysis(corpus)
    _, second, _ = run_toy_analysis(corpus)

    assert first == second


def test_run_analysis_computes_paired_random_baseline(corpus):
    summary, results, log = run_toy_analysis(corpus,
        compute_random_baseline=True)

    assert list(results) == ['rsa', 'random_baseline_rsa']
    assert sorted(results['random_baseline_rsa']) == LAYERS
    assert all(len(layer_scores) == 4
        for layer_scores in results['random_baseline_rsa'].values())
    assert log['parameters']['compute_random_baseline'] is True
    for layer in LAYERS:
        observed = np.asarray(results['rsa'][layer])
        baseline = np.asarray(results['random_baseline_rsa'][layer])
        row = next(row for row in summary if row['layer'] == layer)
        assert isinstance(log['layers'][str(layer)]['random_baseline_seed'],
            int)
        assert row['mean_random_baseline_rsa'] == pytest.approx(
            np.mean(baseline))
        assert row['mean_rsa_difference'] == pytest.approx(
            np.mean(observed - baseline))


def test_random_baseline_does_not_change_observed_scores_or_draws(corpus):
    _, observed_only, observed_log = run_toy_analysis(corpus)
    _, with_baseline, baseline_log = run_toy_analysis(corpus,
        compute_random_baseline=True)

    assert observed_only['rsa'] == with_baseline['rsa']
    for layer in LAYERS:
        assert log_sampled_keys(observed_log, layer) == log_sampled_keys(
            baseline_log, layer)


def test_random_baseline_is_deterministic(corpus):
    _, first, _ = run_toy_analysis(corpus, compute_random_baseline=True)
    _, second, _ = run_toy_analysis(corpus, compute_random_baseline=True)

    assert first['random_baseline_rsa'] == second['random_baseline_rsa']


def test_run_analysis_computes_intensity_rsa_and_correlations(corpus):
    summary, results, log = run_toy_analysis(corpus,
        compute_intensity_baseline=True)

    assert list(results) == [
        'rsa',
        'intensity_rsa',
        'sonority_intensity_correlation',
        'sonority_intensity_rdm_correlation',
    ]
    assert log['parameters']['compute_intensity_baseline'] is True
    assert log['intensity_baseline']
    for metric in results:
        assert sorted(results[metric]) == LAYERS
        assert all(len(scores) == 4 for scores in results[metric].values())
    for row in summary:
        assert np.isfinite(row['mean_intensity_rsa'])
        assert np.isfinite(row['mean_sonority_intensity_correlation'])
        assert np.isfinite(
            row['mean_sonority_intensity_rdm_correlation'])
    for layer in LAYERS:
        entry = log['layers'][str(layer)]
        assert entry['intensity_invalid_subsets'] == {
            'single_intensity_value': 0,
            'undefined_intensity_rsa': 0,
        }
        assert entry['intensity_invalid_reasons'] == [None] * 4


def test_intensity_baseline_does_not_change_observed_scores(corpus):
    _, observed_only, _ = run_toy_analysis(corpus)
    _, with_intensity, _ = run_toy_analysis(corpus,
        compute_intensity_baseline=True)

    assert observed_only['rsa'] == with_intensity['rsa']


def test_intensity_audio_is_loaded_once_across_layers(corpus, monkeypatch):
    original = intensity._load_audio
    calls = []

    def counted_load_audio(filename, phone):
        calls.append(filename)
        return original(filename, phone)

    monkeypatch.setattr(intensity, '_load_audio', counted_load_audio)

    run_toy_analysis(corpus, compute_intensity_baseline=True)

    assert calls == [str(corpus.audio_filename)]


def test_intensity_is_not_computed_for_layer_without_vectors(corpus,
        monkeypatch):
    monkeypatch.setattr('sonority_rsa.analysis.add_centered_intensities',
        lambda *args: pytest.fail('intensity should not be computed'))

    with pytest.warns(UserWarning, match='layer 9 dropped'):
        with pytest.raises(ValueError, match='every requested layer failed'):
            run_toy_analysis(corpus, layers=[9],
                compute_intensity_baseline=True)


def test_intensity_failure_for_phone_with_vector_aborts_run(corpus,
        tmp_path, monkeypatch):
    filename = tmp_path / 'missing.wav'
    rng = np.random.default_rng(0)
    syllables = []
    for index in range(30):
        phone = SimpleNamespace(
            audio=SimpleNamespace(filename=str(filename)),
            start=index * 10, end=(index + 1) * 10, label='a')
        syllables.append(SyllableData(f's{index}', ['a'], [index % 2],
            rng.normal(size=(1, 4)), phone_segments=[phone]))
    population = SyllablePopulation(MODEL_NAME, 0, 500, syllables,
        skipped={})
    monkeypatch.setattr('sonority_rsa.analysis.fetch_syllable_data',
        lambda *args, **kwargs: population)

    with pytest.raises(IntensityComputationError) as error:
        run_toy_analysis(corpus, layers=[0], n_subsets=1,
            compute_intensity_baseline=True)

    message = str(error.value)
    assert str(filename) in message
    assert 'start=0 ms' in message
    assert 'end=10 ms' in message


def test_log_sampled_keys_replays_draws(corpus):
    summary, results, log = run_toy_analysis(corpus)

    draws = log_sampled_keys(log, layer=0)

    expected_keys = {syllable.key.hex()
        for syllable in corpus.analysis_syllables}
    assert len(draws) == 4
    assert all(len(draw) == 30 for draw in draws)
    assert all(key in expected_keys for draw in draws for key in draw)


def test_save_analysis_writes_results_and_log(corpus, tmp_path):
    summary, results, log = run_toy_analysis(corpus)

    save_analysis(summary, results, log, tmp_path)

    with open(tmp_path / 'run_log.json') as fin:
        saved_log = json.load(fin)
    assert saved_log['run_id'] == log['run_id']
    with open(tmp_path / 'rsa_scores.csv') as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 8
    assert rows[0]['run_id'] == log['run_id']
    assert rows[0]['invalid_reason'] == ''
    assert 'random_baseline_rsa' not in rows[0]
    assert (tmp_path / 'summary.csv').exists()

    draws = log_sampled_keys(tmp_path / 'run_log.json', layer=1)
    assert draws == log_sampled_keys(log, layer=1)


def test_save_analysis_writes_random_baseline_results(corpus, tmp_path):
    summary, results, log = run_toy_analysis(corpus,
        compute_random_baseline=True)

    save_analysis(summary, results, log, tmp_path)

    with open(tmp_path / 'summary.csv') as fin:
        summary_rows = list(csv.DictReader(fin))
    with open(tmp_path / 'rsa_scores.csv') as fin:
        score_rows = list(csv.DictReader(fin))
    assert summary_rows[0]['mean_random_baseline_rsa']
    assert summary_rows[0]['mean_rsa_difference']
    assert score_rows[0]['random_baseline_rsa']
    assert score_rows[0]['rsa_difference']


def test_save_analysis_writes_intensity_results(corpus, tmp_path):
    summary, results, log = run_toy_analysis(corpus,
        compute_intensity_baseline=True)

    save_analysis(summary, results, log, tmp_path)

    with open(tmp_path / 'summary.csv') as fin:
        summary_rows = list(csv.DictReader(fin))
    with open(tmp_path / 'rsa_scores.csv') as fin:
        score_rows = list(csv.DictReader(fin))
    assert summary_rows[0]['mean_intensity_rsa']
    assert summary_rows[0]['mean_sonority_intensity_correlation']
    assert summary_rows[0]['mean_sonority_intensity_rdm_correlation']
    assert score_rows[0]['intensity_rsa']
    assert score_rows[0]['sonority_intensity_correlation']
    assert score_rows[0]['sonority_intensity_rdm_correlation']
    assert score_rows[0]['intensity_invalid_reason'] == ''
    assert 'rsa_intensity_difference' not in score_rows[0]


def test_display_analysis_prints_summary(corpus, capsys):
    summary, results, log = run_toy_analysis(corpus)

    display_analysis(summary, results, n=2)

    output = capsys.readouterr().out
    assert 'mean_rsa' in output
    assert 'layer 1 rsa:' in output


def test_display_analysis_prints_random_baseline(corpus, capsys):
    summary, results, log = run_toy_analysis(corpus,
        compute_random_baseline=True)

    display_analysis(summary, results, n=2)

    output = capsys.readouterr().out
    assert 'mean_random_baseline_rsa' in output
    assert 'layer 1 random_baseline_rsa:' in output


def test_display_analysis_prints_intensity_results(corpus, capsys):
    summary, results, log = run_toy_analysis(corpus,
        compute_intensity_baseline=True)

    display_analysis(summary, results, n=2)

    output = capsys.readouterr().out
    assert 'mean_intensity_rsa' in output
    assert 'layer 1 intensity_rsa:' in output
    assert 'layer 1 sonority_intensity_correlation:' in output
    assert 'layer 1 sonority_intensity_rdm_correlation:' in output
