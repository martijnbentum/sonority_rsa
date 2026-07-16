"""Phone-level audio intensity computation."""

from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

REFERENCE_POWER = 4 * 10 ** -10


class IntensityComputationError(RuntimeError):
    """Raised when intensity cannot be computed for a usable phone."""


def compute_intensity(signal):
    """
    Compute the RMS-power intensity of an audio signal in dB.

    Uses the same formula as the referenced stress analysis:
    10 * log10(mean(signal ** 2) / 4e-10).

    signal: one-dimensional audio signal
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError('signal must be one-dimensional')
    if signal.size == 0:
        raise ValueError('signal is empty')
    if not np.isfinite(signal).all():
        raise ValueError('signal contains non-finite samples')
    power = float(np.mean(signal ** 2))
    if not np.isfinite(power) or power <= 0:
        raise ValueError(f'signal power must be finite and positive, got '
            f'{power}')
    return float(10 * np.log10(power / REFERENCE_POWER))


def compute_phone_intensity(phone):
    """
    Compute RMS-power intensity over a phraser Phone's complete interval.

    phone: phraser Phone with audio, start, and end attributes
    """
    filename = _phone_filename(phone)
    signal, sample_rate = _load_audio(filename, phone)
    return _intensity_from_loaded_audio(phone, signal, sample_rate)


def compute_praat_intensity(phone, minimum_pitch=100.0, time_step=None,
        subtract_mean=True):
    """
    Compute genuine Praat contour intensity averaged over a phone.

    Builds Praat's intensity contour over the complete recording, then
    returns its energy-domain average within the phone interval.

    phone: phraser Phone with audio, start, and end attributes
    minimum_pitch: Praat intensity pitch floor in Hz (default 100)
    time_step: contour time step in seconds; None uses Praat's default
    subtract_mean: remove local mean pressure before analysis (default true)
    """
    try:
        import parselmouth
    except ImportError as error:
        raise RuntimeError(
            'compute_praat_intensity requires praat-parselmouth') from error

    filename = _phone_filename(phone)
    _validate_filename(filename, phone)
    try:
        start, end = _phone_times_seconds(phone)
        sound = parselmouth.Sound(filename)
        sound_end = sound.get_end_time()
        if end > sound_end:
            raise ValueError(
                f'phone end ({end:.6f} s) exceeds audio duration '
                f'({sound_end:.6f} s)')
        contour = sound.to_intensity(minimum_pitch=minimum_pitch,
            time_step=time_step, subtract_mean=subtract_mean)
        value = contour.get_average(start, end,
            averaging_method=parselmouth.Intensity.AveragingMethod.ENERGY)
        if not np.isfinite(value):
            raise ValueError(f'Praat returned non-finite intensity {value}')
        return float(value)
    except IntensityComputationError:
        raise
    except Exception as error:
        raise _phone_error(phone, error) from error


class IntensityCache:
    """Cache raw RMS-power intensity values across model layers."""

    def __init__(self):
        self._values = {}

    def values_for_phones(self, phones):
        """Return raw intensity values in phone order."""
        phones = list(phones)
        missing_by_filename = defaultdict(list)
        for phone in phones:
            key = _phone_cache_key(phone)
            if key not in self._values:
                missing_by_filename[key[0]].append(phone)

        for filename, file_phones in missing_by_filename.items():
            signal, sample_rate = _load_audio(filename, file_phones[0])
            for phone in file_phones:
                key = _phone_cache_key(phone)
                self._values[key] = _intensity_from_loaded_audio(
                    phone, signal, sample_rate)

        return np.asarray([
            self._values[_phone_cache_key(phone)] for phone in phones
        ], dtype=float)


def add_centered_intensities(syllable_population, cache):
    """
    Attach per-recording-centered phone intensities to a population.

    Only phone rows that survived embedding/sonority validation are present
    in SyllableData.phone_segments, so audio failures for already skipped
    phones are never evaluated.

    syllable_population: fetched SyllablePopulation for one model layer
    cache: IntensityCache shared across layers
    """
    for syllable in syllable_population:
        if len(syllable.phone_segments) != len(syllable):
            raise RuntimeError(
                'cannot attach intensity: SyllableData phone_segments must '
                'match its vector rows')
    phones = [phone for syllable in syllable_population
        for phone in syllable.phone_segments]
    raw_values = cache.values_for_phones(phones)
    filenames = [_phone_filename(phone) for phone in phones]
    values_by_filename = defaultdict(list)
    for value, filename in zip(raw_values, filenames):
        values_by_filename[filename].append(value)
    means = {filename: float(np.mean(values))
        for filename, values in values_by_filename.items()}
    centered = np.asarray([
        value - means[filename]
        for value, filename in zip(raw_values, filenames)
    ], dtype=float)

    offset = 0
    for syllable in syllable_population:
        end = offset + len(syllable)
        syllable.intensity = centered[offset:end]
        offset = end
    return syllable_population


def _load_audio(filename, phone):
    """Load one recording at its native sample rate and convert it to mono."""
    _validate_filename(filename, phone)
    try:
        signal, sample_rate = librosa.load(filename, sr=None, mono=True)
    except Exception as error:
        raise _phone_error(phone, error) from error
    return np.asarray(signal, dtype=float), int(sample_rate)


def _intensity_from_loaded_audio(phone, signal, sample_rate):
    """Slice a loaded recording to one phone and compute its intensity."""
    try:
        start, end = _phone_times_seconds(phone)
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        if end_sample > signal.size:
            duration = signal.size / sample_rate
            raise ValueError(
                f'phone end ({end:.6f} s) exceeds audio duration '
                f'({duration:.6f} s)')
        return compute_intensity(signal[start_sample:end_sample])
    except IntensityComputationError:
        raise
    except Exception as error:
        raise _phone_error(phone, error) from error


def _phone_times_seconds(phone):
    """Validate and return phone start/end times in seconds."""
    start = int(phone.start)
    end = int(phone.end)
    if start < 0:
        raise ValueError(f'phone start must be non-negative, got {start} ms')
    if end <= start:
        raise ValueError(
            f'phone end must be greater than start, got {start}-{end} ms')
    return start / 1000, end / 1000


def _phone_filename(phone):
    """Return a phone's audio filename with contextual errors."""
    try:
        audio = phone.audio
        filename = getattr(audio, 'filename', None) if audio else None
    except Exception as error:
        raise _phone_error(phone, error, filename=None) from error
    if not filename:
        raise _phone_error(phone, 'phone has no audio filename',
            filename=filename)
    return str(filename)


def _validate_filename(filename, phone):
    """Raise a contextual error when an audio filename is not a file."""
    if not Path(filename).is_file():
        raise _phone_error(phone, 'audio file does not exist',
            filename=filename)


def _phone_cache_key(phone):
    """Return the stable audio interval key for a phone."""
    return (_phone_filename(phone), int(phone.start), int(phone.end))


def _phone_error(phone, error, filename=None):
    """Build an intensity error containing the requested phone context."""
    if filename is None:
        try:
            audio = phone.audio
            filename = getattr(audio, 'filename', None) if audio else None
        except Exception:
            filename = None
    label = getattr(phone, 'label', None)
    start = getattr(phone, 'start', None)
    end = getattr(phone, 'end', None)
    return IntensityComputationError(
        'failed to compute intensity: '
        f'filename={filename!r}, phone={label!r}, start={start} ms, '
        f'end={end} ms: {error}')
