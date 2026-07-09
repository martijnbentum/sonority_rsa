"""Run the bootstrap RSA on a small fake phraser/echoframe setup.

Real analyses pass phraser Syllable objects and an echoframe Store;
this example fakes both so it runs without any LMDB data.
"""

import numpy as np

from sonority_rsa import display_analysis, run_analysis, save_analysis


class FakePhone:

    def __init__(self, label, vectors):
        self.label = label
        self.vectors = vectors


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
        return FakeSlicedEmbedding(
            np.asarray(phone.vectors[self.layer], dtype=float))


class FakeStore:

    root = 'fake-echoframe-store'

    def phraser_key_to_embedding(self, phraser_key, model_name, layer,
            collar=500):
        return FakeEmbedding(layer)


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


if __name__ == '__main__':
    summary, scores, log = run_analysis(
        toy_syllables(),
        model_name='wav2vec2',
        layers=[0, 1],
        echoframe_store=FakeStore(),
        n_syllables=3,
        n_bootstraps=100,
        random_state=1,
    )
    display_analysis(summary, scores)
    save_analysis(summary, scores, log, 'results/')
