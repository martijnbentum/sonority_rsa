import numpy as np
import pytest

from sonority_rsa.fetch import fetch_syllable_data, phone_sonority
from conftest import FakePhone, FakeStore, FakeSyllable


def test_fetch_builds_a_population_per_layer(toy_syllables, toy_store):
    population = fetch_syllable_data(toy_syllables, 'wav2vec2', layer=0,
        echoframe_store=toy_store, verbose=False)

    assert len(population) == 3
    assert population.keys == ['s1', 's2', 's3']
    assert population.layer == 0
    assert population[0].sonority.tolist() == [0, 5]
    assert np.array_equal(population[0].vectors,
        np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_fetch_loads_one_embedding_per_phrase(toy_syllables, toy_store):
    fetch_syllable_data(toy_syllables, 'wav2vec2', layer=1,
        echoframe_store=toy_store, verbose=False)

    assert [call[0] for call in toy_store.calls] == ['ph1', 'ph2']


def test_fetch_skips_unusable_syllables_and_phones(toy_store):
    syllables = [
        FakeSyllable('s1', 'ph1', [
            FakePhone('a', {0: [0.0, 1.0]}),
            FakePhone('(..)', {0: [0.5, 0.5]}),
            FakePhone('l', {0: [0.5, 0.5]}, overlaps=False),
        ]),
        FakeSyllable('s2', None, [FakePhone('m', {0: [0.3, 0.7]})]),
        FakeSyllable('s3', 'ph-unstored', [FakePhone('p', {0: [1.0, 0.0]})]),
        FakeSyllable('s4', 'ph1', [FakePhone('(..)', {0: [0.5, 0.5]})]),
    ]

    population = fetch_syllable_data(syllables, 'wav2vec2', layer=0,
        echoframe_store=toy_store, verbose=False)

    assert population.keys == ['s1']
    assert population[0].phone_labels == ['a']
    assert population.skipped == {'no_phrase': 1, 'no_embedding': 1,
        'no_sonority': 2, 'no_frames': 1, 'no_phones': 1}


def test_fetch_raises_when_nothing_is_stored(toy_syllables):
    store = FakeStore(phrase_keys=[], layers=[0])

    with pytest.raises(ValueError, match='no syllables fetched'):
        fetch_syllable_data(toy_syllables, 'wav2vec2', layer=0,
            echoframe_store=store, verbose=False)


def test_phone_sonority_returns_none_for_unknown_labels():
    assert phone_sonority(FakePhone('a', {})) == 5
    assert phone_sonority(FakePhone('(..)', {})) is None
