"""Bootstrap RSA for wav2vec phone-center frames and sonority."""

from sonority_rsa.analysis import display_analysis
from sonority_rsa.analysis import run_analysis
from sonority_rsa.analysis import save_analysis
from sonority_rsa.bootstrap import (
    compute_bootstrap_by_layer, compute_bootstrap_for_layer,
    sample_syllables, summarize_bootstrap)
from sonority_rsa.data import load_frame_table
from sonority_rsa.data import parse_vector
from sonority_rsa.data import prepare_frame_table
from sonority_rsa.rdm import (compute_sonority_rsa, correlation_rdm,
    sonority_rdm, spearman_rsa, upper_triangle)

__all__ = [
    'compute_bootstrap_by_layer',
    'compute_bootstrap_for_layer',
    'compute_sonority_rsa',
    'correlation_rdm',
    'display_analysis',
    'load_frame_table',
    'parse_vector',
    'prepare_frame_table',
    'run_analysis',
    'save_analysis',
    'sample_syllables',
    'sonority_rdm',
    'spearman_rsa',
    'summarize_bootstrap',
    'upper_triangle',
]
