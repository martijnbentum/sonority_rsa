import numpy as np
import pandas as pd
import pytest

from sonority_rsa.data import prepare_frame_table


@pytest.fixture
def toy_frames():
    """A small two-layer frame table with three syllables."""
    rows = []
    vectors = {
        0: [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7],
            [0.2, 0.8]],
        1: [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3],
            [0.8, 0.2]],
    }
    phones = [
        ('s1', 'p', 0),
        ('s1', 'a', 5),
        ('s2', 's', 1),
        ('s2', 't', 0),
        ('s3', 'm', 2),
        ('s3', 'l', 3),
    ]
    for layer, layer_vectors in vectors.items():
        for (syllable_id, phone, sonority), vector in zip(phones,
                layer_vectors):
            rows.append({
                'syllable_id': syllable_id,
                'phone': phone,
                'sonority': sonority,
                'layer': layer,
                'vector': np.array(vector),
            })
    return prepare_frame_table(pd.DataFrame(rows))
