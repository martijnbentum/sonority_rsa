import numpy as np
import pytest

from conftest import MODEL_NAME, payload
from phraser.models import Phone
from sonority_rsa.fetch import fetch_syllable_data, phone_sonority


def fetch(corpus, syllables, layer=0):
    return fetch_syllable_data(syllables, MODEL_NAME, layer=layer,
        echoframe_store=corpus.echoframe_store, verbose=False)


def test_fetch_builds_a_population_with_middle_frame_vectors(corpus):
    population = fetch(corpus, corpus.syllables)

    assert len(population) == 3
    assert population.keys == [corpus.s1.key, corpus.s2.key, corpus.s3.key]
    assert population[0].phone_labels == ['p', 'a']
    assert population[0].sonority.tolist() == [0, 5]
    assert population[1].sonority.tolist() == [1, 0]
    assert population[2].sonority.tolist() == [2, 3]
    assert np.array_equal(population[0].vectors, payload(0, 0)[[2, 6]])
    assert np.array_equal(population[1].vectors, payload(0, 0)[[11, 16]])
    assert np.array_equal(population[2].vectors, payload(1, 0)[[2, 6]])


def test_fetch_uses_the_requested_layer(corpus):
    population = fetch(corpus, corpus.syllables, layer=1)

    assert np.array_equal(population[0].vectors, payload(0, 1)[[2, 6]])
    assert np.array_equal(population[2].vectors, payload(1, 1)[[2, 6]])


def test_fetch_loads_one_embedding_per_phrase(corpus, monkeypatch):
    calls = []
    original = corpus.echoframe_store.phraser_key_to_embedding

    def counting(phraser_key, *args, **kwargs):
        calls.append(phraser_key)
        return original(phraser_key, *args, **kwargs)

    monkeypatch.setattr(corpus.echoframe_store, 'phraser_key_to_embedding',
        counting)

    fetch(corpus, corpus.syllables)

    assert calls == [corpus.ph1.key, corpus.ph2.key]


def test_fetch_skips_unusable_syllables_and_phones(corpus):
    syllables = [corpus.s1, corpus.s_unlinked, corpus.s_unstored,
        corpus.s_bad]

    population = fetch(corpus, syllables)

    assert population.keys == [corpus.s1.key]
    assert population.skipped == {'no_phrase': 1, 'no_embedding': 1,
        'no_sonority': 1, 'no_frames': 1, 'constant_vector': 0,
        'no_phones': 1}


def test_fetch_raises_when_nothing_is_stored(corpus):
    with pytest.raises(ValueError, match='no syllables fetched'):
        fetch(corpus, [corpus.s_unstored])


class Sub:
    def __init__(self, data):
        self.data = data


class ConstantForLabel:
    """Fake embedding: a constant vector for one phone label, else varied."""

    def __init__(self, constant_label):
        self.constant_label = constant_label

    def sub_embedding(self, phone, aggregate=None):
        if phone.label == self.constant_label:
            return Sub(np.ones(3))
        return Sub(np.array([1.0, 2.0, 3.0]))


def test_fetch_skips_phones_with_constant_vectors(corpus, monkeypatch):
    monkeypatch.setattr(corpus.echoframe_store, 'phraser_key_to_embedding',
        lambda *a, **k: ConstantForLabel('a'))

    population = fetch(corpus, corpus.syllables)

    # only s1 has an 'a' phone; it is dropped, the rest of s1 survives
    assert population.skipped['constant_vector'] == 1
    assert population[0].phone_labels == ['p']


def test_fetch_raises_when_population_has_one_sonority_class(corpus,
        monkeypatch):
    monkeypatch.setattr('sonority_rsa.fetch.phone_sonority', lambda phone: 5)

    with pytest.raises(ValueError, match='one sonority class'):
        fetch(corpus, corpus.syllables)


def test_phone_sonority_returns_none_for_unknown_labels():
    vowel = Phone(label='a', start=0, end=100, save=False, store=None)
    silence = Phone(label='(..)', start=0, end=100, save=False, store=None)

    assert phone_sonority(vowel) == 5
    assert phone_sonority(silence) is None
