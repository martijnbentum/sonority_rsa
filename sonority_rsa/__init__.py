"""Bootstrap RSA for wav2vec phone-center frames and sonority."""

from sonority_rsa.analysis import display_analysis
from sonority_rsa.analysis import run_analysis
from sonority_rsa.analysis import run_analysis_from_stores
from sonority_rsa.analysis import save_analysis
from sonority_rsa.bootstrap import (
    compute_bootstrap_by_layer, compute_bootstrap_for_layer,
    sample_syllables, summarize_bootstrap)
from sonority_rsa.data import load_frame_table
from sonority_rsa.data import prepare_frame_table
from sonority_rsa.data import save_frame_table
from sonority_rsa.extract import build_frame_table
from sonority_rsa.extract import phone_sonority
from sonority_rsa.rdm import (compute_sonority_rsa, correlation_rdm,
    sonority_rdm, spearman_rsa, upper_triangle)

__all__ = [
    'build_frame_table',
    'compute_bootstrap_by_layer',
    'compute_bootstrap_for_layer',
    'compute_sonority_rsa',
    'correlation_rdm',
    'display_analysis',
    'load_frame_table',
    'phone_sonority',
    'prepare_frame_table',
    'run_analysis',
    'run_analysis_from_stores',
    'save_analysis',
    'save_frame_table',
    'sample_syllables',
    'sonority_rdm',
    'spearman_rsa',
    'summarize_bootstrap',
    'upper_triangle',
]
