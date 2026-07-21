"""Subset-sampled RSA for phone embeddings, sonority, and intensity."""

from sonority_rsa.analysis import display_analysis
from sonority_rsa.analysis import log_sampled_keys
from sonority_rsa.analysis import run_analysis
from sonority_rsa.analysis import save_analysis
from sonority_rsa.sampling import (
    compute_rsa_scores, replay_sampled_keys, sample_syllables,
    summarize_rsa_scores)
from sonority_rsa.fetch import SyllableData
from sonority_rsa.fetch import SyllablePopulation
from sonority_rsa.fetch import fetch_syllable_data
from sonority_rsa.fetch import phone_sonority
from sonority_rsa.intensity import (IntensityComputationError,
    compute_intensity, compute_phone_intensity, compute_praat_intensity)
from sonority_rsa.partial import partial_spearman_rsa
from sonority_rsa.plot import plot_analyses, plot_analysis
from sonority_rsa.rdm import (compute_intensity_rsa,
    compute_sonority_random_baseline, compute_sonority_rsa, correlation_rdm,
    intensity_rdm, sonority_rdm, spearman_rsa, upper_triangle)

__all__ = [
    'IntensityComputationError',
    'SyllableData',
    'SyllablePopulation',
    'compute_rsa_scores',
    'compute_intensity',
    'compute_intensity_rsa',
    'compute_phone_intensity',
    'compute_praat_intensity',
    'compute_sonority_random_baseline',
    'compute_sonority_rsa',
    'correlation_rdm',
    'display_analysis',
    'fetch_syllable_data',
    'log_sampled_keys',
    'intensity_rdm',
    'phone_sonority',
    'partial_spearman_rsa',
    'plot_analyses',
    'plot_analysis',
    'replay_sampled_keys',
    'run_analysis',
    'sample_syllables',
    'save_analysis',
    'sonority_rdm',
    'spearman_rsa',
    'summarize_rsa_scores',
    'upper_triangle',
]
