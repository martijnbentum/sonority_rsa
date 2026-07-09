"""Run the bootstrap RSA on a small hand-made frame table.

Real analyses build the frame table from phraser/echoframe stores with
build_frame_table; this example fakes that step with inline data.
"""

import numpy as np
import pandas as pd

from sonority_rsa import (display_analysis, prepare_frame_table,
    run_analysis, save_analysis)


def toy_frame_table():
    """Build a two-layer frame table with three toy syllables."""
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


if __name__ == '__main__':
    frames = toy_frame_table()
    summary, scores = run_analysis(frames, n_syllables=3, n_bootstraps=100,
        random_state=1)
    display_analysis(summary, scores)
    save_analysis(summary, scores, 'results/')
