"""Load and validate phone-center frame tables."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ['syllable_id', 'phone', 'sonority', 'layer', 'vector']


def load_frame_table(path):
    """
    Load a CSV or Parquet phone-center frame table.

    path: input CSV or Parquet path
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix in {'.parquet', '.pq'}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f'unsupported input format: {path.suffix}')

    return prepare_frame_table(df)


def parse_vector(value):
    """
    Parse one vector value into a numpy array.

    value: array-like object, JSON-like list string, or space-separated string
    """
    if isinstance(value, np.ndarray):
        return value.astype(float)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)

    if pd.isna(value):
        raise ValueError('vector value cannot be missing')

    text = str(value).strip()
    if not text:
        raise ValueError('vector value cannot be empty')

    if text[0] in '[(':
        parsed = _parse_list_vector(text)
        return np.asarray(parsed, dtype=float)

    parts = text.replace(',', ' ').split()
    return np.asarray([float(part) for part in parts], dtype=float)


def prepare_frame_table(df):
    """
    Validate and normalize a phone-center frame table.

    df: pandas DataFrame with required RSA input columns
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f'missing required columns: {missing}')

    prepared = df.copy()
    prepared['vector'] = prepared['vector'].map(parse_vector)
    prepared['sonority'] = pd.to_numeric(prepared['sonority'])

    vector_lengths = prepared['vector'].map(len).unique()
    if len(vector_lengths) != 1:
        raise ValueError('all vectors must have the same length')
    if vector_lengths[0] == 0:
        raise ValueError('vectors cannot be empty')
    if prepared['sonority'].isna().any():
        raise ValueError('sonority values cannot be missing')

    return prepared


def _parse_list_vector(text):
    """
    Parse a vector string that looks like a Python or JSON list.

    text: string starting with '[' or '('
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)
