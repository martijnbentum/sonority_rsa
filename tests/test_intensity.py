from types import SimpleNamespace

import numpy as np
import parselmouth
import pytest
import soundfile

from sonority_rsa.fetch import SyllableData, SyllablePopulation
from sonority_rsa.intensity import (IntensityCache,
    IntensityComputationError, add_centered_intensities, compute_intensity,
    compute_phone_intensity, compute_praat_intensity)


def make_phone(filename, start, end, label='a'):
    return SimpleNamespace(audio=SimpleNamespace(filename=str(filename)),
        start=start, end=end, label=label)


def write_audio(path, signal, sample_rate=1000):
    soundfile.write(path, signal, sample_rate, subtype='FLOAT')


def test_compute_intensity_uses_rms_power_db_formula():
    signal = np.array([-0.5, 0.5, -0.5, 0.5])

    result = compute_intensity(signal)

    expected = 10 * np.log10(np.mean(signal ** 2) / (4 * 10 ** -10))
    assert result == pytest.approx(expected)


@pytest.mark.parametrize('signal, message', [
    ([], 'empty'),
    ([0, 0], 'positive'),
    ([0, np.nan], 'non-finite'),
])
def test_compute_intensity_rejects_invalid_signal(signal, message):
    with pytest.raises(ValueError, match=message):
        compute_intensity(signal)


def test_compute_phone_intensity_uses_complete_phone_interval(tmp_path):
    filename = tmp_path / 'audio.wav'
    signal = np.concatenate([
        np.full(100, 0.1),
        np.full(100, 0.25),
        np.full(100, 0.5),
    ])
    write_audio(filename, signal)
    phone = make_phone(filename, start=100, end=200)

    result = compute_phone_intensity(phone)

    assert result == pytest.approx(compute_intensity(signal[100:200]),
        abs=1e-5)


def test_compute_praat_intensity_uses_energy_average(tmp_path):
    filename = tmp_path / 'audio.wav'
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 200 * time)
    write_audio(filename, signal, sample_rate)
    phone = make_phone(filename, start=200, end=800)

    result = compute_praat_intensity(phone)

    sound = parselmouth.Sound(str(filename))
    contour = sound.to_intensity(minimum_pitch=100, time_step=None,
        subtract_mean=True)
    expected = contour.get_average(0.2, 0.8,
        averaging_method=parselmouth.Intensity.AveragingMethod.ENERGY)
    assert result == pytest.approx(expected)


def test_centered_intensities_have_zero_mean_per_recording(tmp_path):
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    write_audio(first, np.concatenate([
        np.full(100, 0.1), np.full(100, 0.2)]))
    write_audio(second, np.concatenate([
        np.full(100, 0.3), np.full(100, 0.6)]))
    phones = [
        make_phone(first, 0, 100),
        make_phone(first, 100, 200),
        make_phone(second, 0, 100),
        make_phone(second, 100, 200),
    ]
    syllables = [
        SyllableData(f's{i}', [phone.label], [i % 2], [[i, i + 1]],
            phone_segments=[phone])
        for i, phone in enumerate(phones)
    ]
    population = SyllablePopulation('model', 0, 500, syllables, skipped={})

    add_centered_intensities(population, IntensityCache())

    assert np.mean([population[0].intensity[0],
        population[1].intensity[0]]) == pytest.approx(0)
    assert np.mean([population[2].intensity[0],
        population[3].intensity[0]]) == pytest.approx(0)


def test_intensity_error_lists_audio_and_phone_interval(tmp_path):
    filename = tmp_path / 'missing.wav'
    phone = make_phone(filename, start=1250, end=1320, label='a')

    with pytest.raises(IntensityComputationError) as error:
        compute_phone_intensity(phone)

    message = str(error.value)
    assert str(filename) in message
    assert "phone='a'" in message
    assert 'start=1250 ms' in message
    assert 'end=1320 ms' in message
    assert 'audio file does not exist' in message
