"""Validate and cache phone-center frame tables."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ['syllable_id', 'phone', 'sonority', 'layer', 'vector']


def prepare_frame_table(df):
    """
    Validate and normalize a phone-center frame table.

    df: pandas DataFrame with required RSA input columns
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f'missing required columns: {missing}')

    prepared = df.copy()
    prepared['vector'] = prepared['vector'].map(_to_vector)
    prepared['sonority'] = pd.to_numeric(prepared['sonority'])

    vector_lengths = prepared['vector'].map(len).unique()
    if len(vector_lengths) != 1:
        raise ValueError('all vectors must have the same length')
    if vector_lengths[0] == 0:
        raise ValueError('vectors cannot be empty')
    if prepared['sonority'].isna().any():
        raise ValueError('sonority values cannot be missing')

    return prepared


def save_frame_table(df, path):
    """
    Save a frame table to a Parquet cache file.

    df: frame table DataFrame (e.g. from build_frame_table)
    path: output Parquet path
    """
    out = df.copy()
    out['vector'] = out['vector'].map(_to_list)
    out.to_parquet(Path(path), index=False)


def load_frame_table(path):
    """
    Load a frame table from a Parquet cache file.

    path: Parquet path written by save_frame_table
    """
    return prepare_frame_table(pd.read_parquet(Path(path)))


def _to_vector(value):
    """
    Convert one vector value to a one-dimensional float numpy array.

    value: numpy array or array-like sequence of floats
    """
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f'vector must be one-dimensional, got {vector.ndim}')
    return vector


def _to_list(value):
    """
    Convert one vector value to a plain list for Parquet storage.

    value: numpy array or array-like sequence of floats
    """
    return np.asarray(value, dtype=float).tolist()
