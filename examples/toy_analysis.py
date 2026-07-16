"""Run the subset-sampled RSA on a small fake phraser/echoframe setup.

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
    """Sixty syllables over two phrases, with two stored layers."""
    syllables = []
    for index in range(60):
        value = index / 60
        label = 'a' if index % 2 else 'p'
        phone = FakePhone(label, {
            0: [value, 1 - value, 0.2 + value / 2],
            1: [1 - value, value, 0.3 + value / 3],
        })
        syllables.append(FakeSyllable(f's{index}',
            f'ph{1 + index % 2}', [phone]))
    return syllables


if __name__ == '__main__':
    summary, results, log = run_analysis(
        toy_syllables(),
        model_name='wav2vec2',
        layers=[0, 1],
        echoframe_store=FakeStore(),
        subset_size=30,
        n_subsets=100,
        random_state=1,
        compute_random_baseline=True,
    )
    display_analysis(summary, results)
    save_analysis(summary, results, log, 'results/')
