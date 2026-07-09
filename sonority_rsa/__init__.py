"""Bootstrap RSA for wav2vec phone middle frames and sonority."""

from sonority_rsa.analysis import display_analysis
from sonority_rsa.analysis import log_sampled_keys
from sonority_rsa.analysis import run_analysis
from sonority_rsa.analysis import save_analysis
from sonority_rsa.bootstrap import (
    compute_bootstrap, replay_sampled_keys, sample_syllables,
    summarize_bootstrap)
from sonority_rsa.fetch import SyllableData
from sonority_rsa.fetch import SyllablePopulation
from sonority_rsa.fetch import fetch_syllable_data
from sonority_rsa.fetch import phone_sonority
from sonority_rsa.rdm import (compute_sonority_rsa, correlation_rdm,
    sonority_rdm, spearman_rsa, upper_triangle)

__all__ = [
    'SyllableData',
    'SyllablePopulation',
    'compute_bootstrap',
    'compute_sonority_rsa',
    'correlation_rdm',
    'display_analysis',
    'fetch_syllable_data',
    'log_sampled_keys',
    'phone_sonority',
    'replay_sampled_keys',
    'run_analysis',
    'sample_syllables',
    'save_analysis',
    'sonority_rdm',
    'spearman_rsa',
    'summarize_bootstrap',
    'upper_triangle',
]
