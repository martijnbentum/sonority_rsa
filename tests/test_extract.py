import numpy as np
import pytest

from sonority_rsa.extract import build_frame_table, phone_sonority


def test_build_frame_table_extracts_middle_frame_rows():
    phrase = FakePhrase([
        FakePhone('p', 'syl-1', [1.0, 0.0]),
        FakePhone('a', 'syl-1', [0.0, 1.0]),
    ])

    df = build_frame_table([phrase], 'wav2vec2', layers=[3, 7],
        verbose=False)

    assert len(df) == 4
    assert sorted(df['layer'].unique().tolist()) == [3, 7]
    assert df['syllable_id'].unique().tolist() == ['syl-1']
    assert df['sonority'].tolist()[:2] == [0, 5]
    assert np.array_equal(df['vector'].iloc[0], np.array([1.0, 0.0]))


def test_build_frame_table_skips_unusable_phones():
    phrase = FakePhrase([
        FakePhone('a', 'syl-1', [0.0, 1.0]),
        FakePhone('(..)', 'syl-1', [0.5, 0.5]),
        FakePhone('m', None, [0.5, 0.5]),
        FakePhone('l', 'syl-2', [0.5, 0.5], overlaps=False),
    ])

    df = build_frame_table([phrase], 'wav2vec2', layers=[0], verbose=False)

    assert df['phone'].tolist() == ['a']


def test_build_frame_table_skips_phrases_without_embeddings():
    stored = FakePhrase([FakePhone('a', 'syl-1', [0.0, 1.0])])
    unstored = FakePhrase([FakePhone('p', 'syl-2', [1.0, 0.0])],
        has_embedding=False)

    df = build_frame_table([stored, unstored], 'wav2vec2', layers=[0],
        verbose=False)

    assert df['syllable_id'].tolist() == ['syl-1']


def test_build_frame_table_raises_when_nothing_is_extracted():
    unstored = FakePhrase([FakePhone('a', 'syl-1', [0.0, 1.0])],
        has_embedding=False)

    with pytest.raises(ValueError, match='no phone frames extracted'):
        build_frame_table([unstored], 'wav2vec2', layers=[0], verbose=False)


def test_phone_sonority_returns_none_for_unknown_labels():
    assert phone_sonority(FakePhone('a', 'syl-1', [0.0])) == 5
    assert phone_sonority(FakePhone('(..)', 'syl-1', [0.0])) is None


class FakePhone:

    def __init__(self, label, syllable_key, vector, overlaps=True):
        self.label = label
        self.syllable = FakeSyllable(syllable_key) if syllable_key else None
        self.vector = np.array(vector)
        self.overlaps = overlaps


class FakeSyllable:

    def __init__(self, key):
        self.key = key


class FakePhrase:

    def __init__(self, phones, has_embedding=True):
        self.phones = phones
        self.has_embedding = has_embedding

    def embedding(self, model_name, layer, collar=500):
        if not self.has_embedding:
            raise ValueError('no stored embedding for Phrase')
        return FakeEmbedding(self)


class FakeEmbedding:

    def __init__(self, phrase):
        self.phrase = phrase

    def sub_embedding(self, phone, aggregate=None):
        assert aggregate == 'middle'
        if not phone.overlaps:
            raise ValueError(f'no frames overlap Phone {phone.label!r}')
        return FakeSlicedEmbedding(phone.vector)


class FakeSlicedEmbedding:

    def __init__(self, data):
        self.data = data
