import csv
import json

from sonority_rsa.analysis import (display_analysis, log_sampled_keys,
    run_analysis, save_analysis)


def run_toy_analysis(toy_syllables, toy_store, **kwargs):
    settings = dict(model_name='wav2vec2', layers=[0, 1],
        echoframe_store=toy_store, n_syllables=3, n_bootstraps=4,
        random_state=1)
    settings.update(kwargs)
    return run_analysis(toy_syllables, **settings)


def test_run_analysis_returns_summary_scores_and_log(toy_syllables,
        toy_store):
    summary, scores, log = run_toy_analysis(toy_syllables, toy_store)

    assert [row['layer'] for row in summary] == [0, 1]
    assert all(row['n_syllables'] == 3 for row in summary)
    assert all(row['run_id'] == log['run_id'] for row in summary)
    assert sorted(scores) == [0, 1]
    assert all(len(layer_scores) == 4 for layer_scores in scores.values())


def test_run_analysis_logs_population_and_seeds(toy_syllables, toy_store):
    summary, scores, log = run_toy_analysis(toy_syllables, toy_store)

    assert log['parameters']['seed'] == 1
    assert log['parameters']['model_name'] == 'wav2vec2'
    assert log['echoframe_store'] == 'fake-echoframe-store'
    for layer in ['0', '1']:
        entry = log['layers'][layer]
        assert entry['syllable_keys'] == ['s1', 's2', 's3']
        assert entry['n_syllables_in_population'] == 3
        assert isinstance(entry['seed'], int)


def test_run_analysis_is_deterministic_for_a_seed(toy_syllables, toy_store):
    _, first, _ = run_toy_analysis(toy_syllables, toy_store)
    _, second, _ = run_toy_analysis(toy_syllables, toy_store)

    assert first == second


def test_log_sampled_keys_replays_draws(toy_syllables, toy_store):
    summary, scores, log = run_toy_analysis(toy_syllables, toy_store)

    draws = log_sampled_keys(log, layer=0)

    assert len(draws) == 4
    assert all(len(draw) == 3 for draw in draws)
    assert all(key in {'s1', 's2', 's3'} for draw in draws for key in draw)


def test_save_analysis_writes_results_and_log(toy_syllables, toy_store,
        tmp_path):
    summary, scores, log = run_toy_analysis(toy_syllables, toy_store)

    save_analysis(summary, scores, log, tmp_path)

    with open(tmp_path / 'run_log.json') as fin:
        saved_log = json.load(fin)
    assert saved_log['run_id'] == log['run_id']
    with open(tmp_path / 'bootstrap_scores.csv') as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 8
    assert rows[0]['run_id'] == log['run_id']
    assert (tmp_path / 'summary.csv').exists()

    draws = log_sampled_keys(tmp_path / 'run_log.json', layer=1)
    assert draws == log_sampled_keys(log, layer=1)


def test_display_analysis_prints_summary(toy_syllables, toy_store, capsys):
    summary, scores, log = run_toy_analysis(toy_syllables, toy_store)

    display_analysis(summary, scores, n=2)

    output = capsys.readouterr().out
    assert 'mean_rsa' in output
    assert 'layer 1 rsa:' in output
