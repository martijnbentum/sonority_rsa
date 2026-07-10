import csv
import json

import pytest
from conftest import LAYERS, MODEL_NAME

from sonority_rsa.analysis import (display_analysis, log_sampled_keys,
    run_analysis, save_analysis)


def run_toy_analysis(corpus, **kwargs):
    settings = dict(model_name=MODEL_NAME, layers=LAYERS,
        echoframe_store=corpus.echoframe_store, subset_size=3,
        n_subsets=4, random_state=1)
    settings.update(kwargs)
    return run_analysis(corpus.syllables, **settings)


def test_run_analysis_returns_summary_scores_and_log(corpus):
    summary, scores, log = run_toy_analysis(corpus)

    assert [row['layer'] for row in summary] == LAYERS
    assert all(row['subset_size'] == 3 for row in summary)
    assert all(row['run_id'] == log['run_id'] for row in summary)
    assert sorted(scores) == LAYERS
    assert all(len(layer_scores) == 4 for layer_scores in scores.values())


def test_run_analysis_logs_population_and_seeds(corpus):
    summary, scores, log = run_toy_analysis(corpus)

    expected_keys = [syllable.key.hex() for syllable in corpus.syllables]
    assert log['parameters']['seed'] == 1
    assert log['parameters']['model_name'] == MODEL_NAME
    assert log['echoframe_store'] == str(corpus.echoframe_store.root)
    for layer in LAYERS:
        entry = log['layers'][str(layer)]
        assert entry['syllable_keys'] == expected_keys
        assert entry['n_syllables_in_population'] == 3
        assert isinstance(entry['seed'], int)


def test_run_analysis_drops_layer_without_data(corpus):
    # the corpus stores layers 0 and 1 only, so layer 9 has no embeddings
    with pytest.warns(UserWarning, match='layer 9 dropped'):
        summary, scores, log = run_toy_analysis(corpus, layers=[0, 9])

    assert list(scores) == [0]
    assert [row['layer'] for row in summary] == [0]
    assert list(log['layers']) == ['0']
    assert '0' not in log['failed_layers']
    assert 'seed' in log['failed_layers']['9']
    assert log['failed_layers']['9']['reason']


def test_run_analysis_raises_when_every_layer_fails(corpus):
    with pytest.raises(ValueError, match='every requested layer failed'):
        run_toy_analysis(corpus, layers=[8, 9])


def test_run_analysis_defaults_seed_to_42(corpus):
    settings = dict(model_name=MODEL_NAME, layers=LAYERS,
        echoframe_store=corpus.echoframe_store, subset_size=3, n_subsets=4)
    _, _, log = run_analysis(corpus.syllables, **settings)

    assert log['parameters']['seed'] == 42


def test_run_analysis_is_deterministic_for_a_seed(corpus):
    _, first, _ = run_toy_analysis(corpus)
    _, second, _ = run_toy_analysis(corpus)

    assert first == second


def test_log_sampled_keys_replays_draws(corpus):
    summary, scores, log = run_toy_analysis(corpus)

    draws = log_sampled_keys(log, layer=0)

    expected_keys = {syllable.key.hex() for syllable in corpus.syllables}
    assert len(draws) == 4
    assert all(len(draw) == 3 for draw in draws)
    assert all(key in expected_keys for draw in draws for key in draw)


def test_save_analysis_writes_results_and_log(corpus, tmp_path):
    summary, scores, log = run_toy_analysis(corpus)

    save_analysis(summary, scores, log, tmp_path)

    with open(tmp_path / 'run_log.json') as fin:
        saved_log = json.load(fin)
    assert saved_log['run_id'] == log['run_id']
    with open(tmp_path / 'rsa_scores.csv') as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 8
    assert rows[0]['run_id'] == log['run_id']
    assert (tmp_path / 'summary.csv').exists()

    draws = log_sampled_keys(tmp_path / 'run_log.json', layer=1)
    assert draws == log_sampled_keys(log, layer=1)


def test_display_analysis_prints_summary(corpus, capsys):
    summary, scores, log = run_toy_analysis(corpus)

    display_analysis(summary, scores, n=2)

    output = capsys.readouterr().out
    assert 'mean_rsa' in output
    assert 'layer 1 rsa:' in output
