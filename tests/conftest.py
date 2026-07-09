import numpy as np
import pytest


class FakePhone:

    def __init__(self, label, vectors, overlaps=True):
        self.label = label
        self.vectors = vectors
        self.overlaps = overlaps


class FakeSyllable:

    def __init__(self, key, phrase_key, phones):
        self.key = key
        self.phrase_key = phrase_key
        self.phones = phones


class FakeSlicedEmbedding:

    def __init__(self, data):
        self.data = data


class FakeEmbedding:

    def __init__(self, layer):
        self.layer = layer

    def sub_embedding(self, phone, aggregate=None):
        assert aggregate == 'middle'
        if not phone.overlaps:
            raise ValueError(f'no frames overlap Phone {phone.label!r}')
        return FakeSlicedEmbedding(
            np.asarray(phone.vectors[self.layer], dtype=float))


class FakeStore:

    root = 'fake-echoframe-store'

    def __init__(self, phrase_keys, layers):
        self.phrase_keys = set(phrase_keys)
        self.layers = set(layers)
        self.calls = []

    def phraser_key_to_embedding(self, phraser_key, model_name, layer,
            collar=500):
        self.calls.append((phraser_key, model_name, layer, collar))
        if phraser_key not in self.phrase_keys or layer not in self.layers:
            raise ValueError(f'nothing stored for {phraser_key!r}')
        return FakeEmbedding(layer)


@pytest.fixture
def toy_syllables():
    """Three syllables over two phrases, with two stored layers."""
    return [
        FakeSyllable('s1', 'ph1', [
            FakePhone('p', {0: [1.0, 0.0], 1: [0.0, 1.0]}),
            FakePhone('a', {0: [0.0, 1.0], 1: [1.0, 0.0]}),
        ]),
        FakeSyllable('s2', 'ph1', [
            FakePhone('s', {0: [0.8, 0.2], 1: [0.2, 0.8]}),
            FakePhone('t', {0: [0.9, 0.1], 1: [0.1, 0.9]}),
        ]),
        FakeSyllable('s3', 'ph2', [
            FakePhone('m', {0: [0.3, 0.7], 1: [0.7, 0.3]}),
            FakePhone('l', {0: [0.2, 0.8], 1: [0.8, 0.2]}),
        ]),
    ]


@pytest.fixture
def toy_store():
    return FakeStore(phrase_keys=['ph1', 'ph2'], layers=[0, 1])
