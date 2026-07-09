import numpy as np
import pandas as pd
import pytest

from sonority_rsa.data import (load_frame_table, prepare_frame_table,
    save_frame_table)


def test_frame_table_round_trips_through_parquet(toy_frames, tmp_path):
    path = tmp_path / 'frames.parquet'

    save_frame_table(toy_frames, path)
    loaded = load_frame_table(path)

    assert len(loaded) == len(toy_frames)
    assert loaded['syllable_id'].tolist() == toy_frames[
        'syllable_id'].tolist()
    for original, restored in zip(toy_frames['vector'], loaded['vector']):
        assert isinstance(restored, np.ndarray)
        assert np.array_equal(original, restored)


def test_prepare_frame_table_rejects_missing_columns():
    df = pd.DataFrame({'syllable_id': ['s1'], 'vector': [[0.1]]})

    with pytest.raises(ValueError, match='missing required columns'):
        prepare_frame_table(df)


def test_prepare_frame_table_rejects_unequal_vector_lengths(toy_frames):
    broken = toy_frames.copy()
    broken.at[0, 'vector'] = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match='same length'):
        prepare_frame_table(broken)


def test_prepare_frame_table_rejects_two_dimensional_vectors(toy_frames):
    broken = toy_frames.copy()
    broken['vector'] = [np.zeros((2, 2))] * len(broken)

    with pytest.raises(ValueError, match='one-dimensional'):
        prepare_frame_table(broken)
